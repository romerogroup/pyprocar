"""Symmetry reduction of a Gamma-centred k-grid and the spin transform for each unfolded k-point.

Method. spglib gives the real-space point operations W of a cell (fractional, x' = W x).
Each acts on fractional k as R = inv(W).T and on Cartesian vectors as C = A.T @ W @ inv(A.T),
where the lattice A holds the direct vectors as rows in Angstrom. Spin is an axial vector,
so S(R k) = det(C) C S(k). Time reversal maps k to -k and S to -S. The grid is reduced to
one representative per orbit of the point group times {E, time reversal}, and every
full-grid k gets its source and the Cartesian matrix that carries the source spin to it.

Inputs: the cell (lattice (3, 3), positions (n_atoms, 3) fractional, numbers (n_atoms,)),
the fractional k-points (n_k, 3) of a full Gamma-centred grid in any order and wrapping,
and the grid shape (3,). The analytic case in ``validate`` is a periodic Rashba texture
plus a C3v warping term on the buckled P3m1 BiSb monolayer, 12x12x1.
"""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import spglib
from references.validation import require
from references.vasp import read_poscar

_ANALYTIC_TOL = 1e-12
"""A few 3x3 products in double precision on a field of size about 2."""
GRID_TOLERANCE = 1e-3
"""Largest distance from a grid node, in units of the grid step, that counts as on the grid."""


class ImageClass(IntEnum):
    """How a full-grid k is reached from its source: k, -k, R k or -R k (R not E)."""

    SOURCE = 0
    TIME_REVERSED = 1
    ROTATED = 2
    ROTATED_TIME_REVERSED = 3


IMAGE_CLASS_LABELS = ("source", "-k", "Rk", "-Rk")


@dataclass(frozen=True, slots=True)
class Cell:
    """lattice (3, 3) direct vectors as rows in Angstrom, positions (n_atoms, 3) fractional,
    numbers (n_atoms,) one integer per species."""

    lattice: np.ndarray
    positions: np.ndarray
    numbers: np.ndarray


@dataclass(frozen=True, slots=True)
class PointGroup:
    """The n_ops point operations of a cell, identity first.

    kspace (n_ops, 3, 3) integer inv(W).T for spglib's real-space W (the matrices a parser
    hands ``Structure(rotations=...)``), cartesian (n_ops, 3, 3) A.T @ W @ inv(A.T), and det
    (n_ops,) the integer determinant of each operation (-1 for a mirror).
    """

    kspace: np.ndarray
    cartesian: np.ndarray
    det: np.ndarray


@dataclass(frozen=True, slots=True)
class IbzReduction:
    """One full grid of n_k points reduced to n_ibz orbit representatives.

    ibz_indices (n_ibz,) full-grid row of each representative, ascending.
    source (n_k,) position in ``ibz_indices`` of the representative of each full-grid row.
    operation (n_k,) point-group index of the operation that reaches the row from its source.
    image_class (n_k,) ``ImageClass`` value of that operation combined with time reversal.
    spin_rotation (n_k, 3, 3) Cartesian matrix with S(row) = spin_rotation @ S(source row).
    orbit_size (n_ibz,) number of distinct full-grid rows in each orbit.
    group_order the number of distinct k-space actions, point group times {E, -E}.
    """

    ibz_indices: np.ndarray
    source: np.ndarray
    operation: np.ndarray
    image_class: np.ndarray
    spin_rotation: np.ndarray
    orbit_size: np.ndarray
    group_order: int


def read_cell(path: Path) -> Cell:
    """The cell of a POSCAR; species are numbered by name."""
    poscar = read_poscar(path)
    _, numbers = np.unique(np.asarray(poscar.species), return_inverse=True)
    return Cell(lattice=poscar.lattice, positions=poscar.positions, numbers=numbers + 1)


def point_group(cell: Cell, symprec: float = 1e-5) -> PointGroup:
    spglib_cell = (cell.lattice.tolist(), cell.positions.tolist(), cell.numbers.tolist())
    symmetry = spglib.get_symmetry(spglib_cell, symprec=symprec)
    if symmetry is None:
        raise ValueError("spglib found no symmetry for the cell")
    real_space = np.unique(np.asarray(symmetry["rotations"], dtype=int), axis=0)
    identity = np.flatnonzero(np.all(real_space == np.eye(3, dtype=int), axis=(1, 2)))
    order = np.concatenate([identity, np.setdiff1d(np.arange(len(real_space)), identity)])
    real_space = real_space[order]
    kspace = np.rint(np.linalg.inv(real_space).transpose(0, 2, 1)).astype(int)
    a_t = cell.lattice.T
    cartesian = a_t @ real_space @ np.linalg.inv(a_t)
    det = np.rint(np.linalg.det(cartesian)).astype(int)
    return PointGroup(kspace=kspace, cartesian=cartesian, det=det)


def grid_cells(kpoints: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    """Flat C-order index of the Gamma-centred grid node of each k-point, shape (n_k,)."""
    scaled = np.asarray(kpoints, dtype=float) * mesh
    nearest = np.rint(scaled)
    off = np.abs(scaled - nearest).max()
    if off > GRID_TOLERANCE:
        raise ValueError(f"a k-point is {off:.2e} grid steps from the {mesh.tolist()} grid")
    nodes = np.mod(nearest.astype(int), mesh)
    return np.ravel_multi_index((nodes[:, 0], nodes[:, 1], nodes[:, 2]), tuple(mesh.tolist()))


def kspace_actions(group: PointGroup) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The point group times {E, time reversal} in priority order E, -E, R, -R.

    Returns the operation index (n_g,), the time-reversal sign (n_g,) and the fractional
    k-space matrix sign * R (n_g, 3, 3).
    """
    n_ops = len(group.kspace)
    rest = np.arange(1, n_ops)
    operation = np.concatenate([[0, 0], rest, rest])
    sign = np.concatenate([[1, -1], np.ones(n_ops - 1, dtype=int), -np.ones(n_ops - 1, dtype=int)])
    return operation, sign, sign[:, np.newaxis, np.newaxis] * group.kspace[operation]


def reduce_grid(kpoints: np.ndarray, mesh: np.ndarray, group: PointGroup) -> IbzReduction:
    """Reduce a full grid; the first k-point of each orbit (in row order) is its representative."""
    mesh = np.asarray(mesh, dtype=int)
    n_k = len(kpoints)
    cells = grid_cells(kpoints, mesh)
    if n_k != np.prod(mesh) or len(np.unique(cells)) != n_k:
        raise ValueError(f"the k-points are not a full {mesh.tolist()} grid, each node once")
    row_of_cell = np.empty(n_k, dtype=int)
    row_of_cell[cells] = np.arange(n_k)

    operation, sign, actions = kspace_actions(group)
    images = np.stack([row_of_cell[grid_cells(kpoints @ m.T, mesh)] for m in actions])

    source_row = np.full(n_k, -1)
    action_of = np.zeros(n_k, dtype=int)
    representatives: list[int] = []
    orbit_size: list[int] = []
    for row in range(n_k):
        if source_row[row] >= 0:
            continue
        representatives.append(row)
        orbit_size.append(len(np.unique(images[:, row])))
        for action, image in enumerate(images[:, row]):
            if source_row[image] < 0:
                source_row[image] = row
                action_of[image] = action

    ibz_indices = np.array(representatives)
    position = np.empty(n_k, dtype=int)
    position[ibz_indices] = np.arange(len(ibz_indices))

    signs = sign * group.det[operation]
    spin_matrices = signs[:, np.newaxis, np.newaxis] * group.cartesian[operation]
    rotated = operation[action_of] != 0
    reversed_ = sign[action_of] < 0
    image_class = np.where(
        rotated,
        np.where(reversed_, ImageClass.ROTATED_TIME_REVERSED, ImageClass.ROTATED),
        np.where(reversed_, ImageClass.TIME_REVERSED, ImageClass.SOURCE),
    )
    return IbzReduction(
        ibz_indices=ibz_indices,
        source=position[source_row],
        operation=operation[action_of],
        image_class=image_class,
        spin_rotation=spin_matrices[action_of],
        orbit_size=np.array(orbit_size),
        group_order=len(np.unique(actions, axis=0)),
    )


def burnside_orbit_count(mesh: np.ndarray, group: PointGroup) -> int:
    """Orbit count by Burnside's lemma: the mean number of grid nodes each k-space action fixes."""
    mesh = np.asarray(mesh, dtype=int)
    nodes = np.stack(np.meshgrid(*[np.arange(n) for n in mesh], indexing="ij"), -1).reshape(-1, 3)
    kpoints = nodes / mesh
    actions = np.unique(kspace_actions(group)[2], axis=0)
    fixed = [
        int(np.sum(grid_cells(kpoints @ m.T, mesh) == grid_cells(kpoints, mesh))) for m in actions
    ]
    if sum(fixed) % len(actions):
        raise ValueError("Burnside's sum is not a multiple of the group order")
    return sum(fixed) // len(actions)


def rotate_spin(reduction: IbzReduction, ibz_spin: np.ndarray) -> np.ndarray:
    """Full-grid spin (n_k, ..., 3) from the spin at the representatives (n_ibz, ..., 3)."""
    return np.einsum("kij,k...j->k...i", reduction.spin_rotation, ibz_spin[reduction.source])


def bisb_cell() -> Cell:
    """The BiSb monolayer of data/examples/fermi2d/bisb_monolayer/POSCAR made exactly hexagonal."""
    a, c = 4.2552190129705005, 16.3717418484577628
    lattice = np.array([[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]])
    positions = np.array([[1 / 3, 2 / 3, 0.1005045352039531], [0.0, 0.0, 0.9975172177960445]])
    return Cell(lattice=lattice, positions=positions, numbers=np.array([1, 2]))


def rashba_texture(kpoints: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Periodic Rashba texture plus a C3v warping S_z at fractional k-points, shape (n_k, 3).

    With star = (a1, a2, -a1-a2), f(k) = sum_t (t/|t|) sin(2 pi k.t) is a polar in-plane
    vector under every lattice operation, so S = z_hat x f is axial and odd in k. The warping
    term S_z = sum_t sin(2 pi k.t) is invariant under C3 and odd under the P3m1 mirrors (they
    map the star onto its negative), which an axial S_z requires; a C2 about z would leave
    S_z even, so the field holds exactly C3v times time reversal.
    """
    star = np.array([lattice[0], lattice[1], -lattice[0] - lattice[1]])
    phases = 2 * np.pi * kpoints @ np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]])
    sines = np.sin(phases)
    in_plane = sines @ (star / np.linalg.norm(star, axis=1, keepdims=True))
    return np.stack([-in_plane[:, 1], in_plane[:, 0], 0.5 * sines.sum(axis=1)], axis=1)


def validate() -> dict[str, float | int | bool | str | list]:
    """Reduce the BiSb monolayer 12x12x1 grid and rebuild the analytic texture from the IBZ."""
    cell = bisb_cell()
    group = point_group(cell)
    n_mirrors = int(np.sum(group.det < 0))
    require(len(group.kspace) == 6 and n_mirrors == 3, "P3m1 has C3 and three mirrors")
    orthogonality = float(
        np.abs(group.cartesian @ group.cartesian.transpose(0, 2, 1) - np.eye(3)).max()
    )
    require(orthogonality < _ANALYTIC_TOL, "each Cartesian operation is orthogonal")

    mesh = np.array([12, 12, 1])
    nodes = np.stack(np.meshgrid(*[np.arange(n) for n in mesh], indexing="ij"), -1).reshape(-1, 3)
    kpoints = np.mod(nodes / mesh + 0.5, 1.0) - 0.5
    kpoints = kpoints[np.random.default_rng(287).permutation(len(kpoints))]
    reduction = reduce_grid(kpoints, mesh, group)

    n_ibz = len(reduction.ibz_indices)
    burnside = burnside_orbit_count(mesh, group)
    require(reduction.group_order == 12, "C3v times time reversal acts on k as a group of 12")
    by_hand = (144 + 4 + 2 * 3 + 2 * 1 + 6 * 12) // 12
    require(n_ibz == by_hand == burnside, f"{by_hand} orbits expected, got {n_ibz}, {burnside}")
    require(
        int(reduction.orbit_size.sum()) == len(kpoints), "the orbits cover the grid exactly once"
    )

    truth = rashba_texture(kpoints, cell.lattice)
    rebuilt = rotate_spin(reduction, truth[reduction.ibz_indices])
    error = np.abs(rebuilt - truth).max(axis=1)
    require(float(error.max()) < _ANALYTIC_TOL, f"axial rebuild error {error.max():.2e}")

    mirror = group.det[reduction.operation] < 0
    polar = reduction.spin_rotation * group.det[reduction.operation][:, np.newaxis, np.newaxis]
    source_spin = truth[reduction.ibz_indices][reduction.source]
    polar_error = np.abs(np.einsum("kij,kj->ki", polar, source_spin) - truth).max(axis=1)
    require(
        float(polar_error[~mirror].max()) < _ANALYTIC_TOL,
        "polar and axial agree on proper rotations",
    )
    require(float(polar_error[mirror].max()) > 0.5, "a polar spin must fail on the mirror images")

    no_flip = reduction.spin_rotation.copy()
    reversed_rows = np.isin(
        reduction.image_class, [ImageClass.TIME_REVERSED, ImageClass.ROTATED_TIME_REVERSED]
    )
    no_flip[reversed_rows] *= -1
    no_flip_error = np.abs(np.einsum("kij,kj->ki", no_flip, source_spin) - truth).max(axis=1)
    require(float(no_flip_error[reversed_rows].max()) > 0.5, "time reversal must reverse the spin")

    counts = np.bincount(reduction.image_class, minlength=len(ImageClass))
    return {
        "mesh": mesh.tolist(),
        "n_ops": len(group.kspace),
        "n_ibz": n_ibz,
        "burnside_orbits": burnside,
        "image_classes": list(IMAGE_CLASS_LABELS),
        "image_class_counts": counts.tolist(),
        "max_abs_spin": float(np.abs(truth).max()),
        "axial_max_error": float(error.max()),
        "cartesian_orthogonality_error": orthogonality,
        "negative_polar_mirror_max_error": float(polar_error[mirror].max()),
        "negative_polar_n_mirror_images": int(mirror.sum()),
        "negative_no_time_reversal_flip_max_error": float(no_flip_error[reversed_rows].max()),
    }
