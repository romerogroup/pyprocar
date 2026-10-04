"""Popescu-Zunger unfolding weights from raw VASP PROCAR phases and POSCAR, numpy only.

The weight of supercell band n at listed k-point K is
W_n(K) = (1/N) sum_t <c_n | T_t c_n> / <c_n | c_n>, where c_n is the (n_atoms, n_orbitals)
complex coefficient block from the PROCAR phase section (LORBIT = 12), T_t moves every atom
by the primitive translation t, and t runs over the N = |det M| primitive lattice points
inside the supercell for an integer transformation matrix M (supercell = M @ primitive,
row lattice vectors). The PROCAR phases are cell-periodic, so no exp(-2 pi i K.t) factor
enters. Arrays: `Procar.kpoints` (n_k, 3) supercell fractional, `Procar.energies`
(n_k, n_bands) eV, `Procar.phases` (n_k, n_bands, n_atoms, n_orbitals) complex,
`Poscar.positions` (n_atoms, 3) fractional, translations (N, 3) supercell fractional,
permutations (N, n_atoms) int. `validate()` builds supercells of a two-atom primitive cell,
fills them with plane-wave coefficient blocks c_{a,o} = v_o exp(2 pi i G.r_a), and checks
W = 1 for G in the primitive reciprocal lattice and W = 0 otherwise.
"""

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from references.validation import require

_FLOAT = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")
_POSITION_TOL = 1e-3
_ANALYTIC_TOL = 1e-12
"""A weight averages N <= 8 overlaps of about 10^3 double-precision terms, so rounding
stays near 1e-14, 11 decades below the 0/1 gap."""


@dataclass(frozen=True, slots=True)
class Procar:
    kpoints: np.ndarray
    energies: np.ndarray
    phases: np.ndarray


def _floats(line: str) -> list[float]:
    """Every decimal number in a line, including negatives printed with no separating space."""
    return [float(x) for x in _FLOAT.findall(line)]


def read_procar_phases(path: Path) -> Procar:
    """Parse a collinear, non-spin-polarized LORBIT = 12 PROCAR."""
    with path.open(encoding="utf-8") as handle:
        _title = handle.readline()
        header = handle.readline()
        counts = [int(x) for x in re.findall(r":\s*(\d+)", header)]
        n_k, n_bands, n_atoms = counts
        kpoints = np.full((n_k, 3), np.nan)
        energies = np.full((n_k, n_bands), np.nan)
        phases: np.ndarray | None = None
        ik = ib = -1
        for line in handle:
            if line.startswith(" k-point"):
                ik += 1
                if ik >= n_k:
                    raise ValueError(f"{path}: more than {n_k} k-point blocks (spin-polarized?)")
                kpoints[ik] = _floats(line.split(":", 1)[1].split("weight")[0])[:3]
            elif line.startswith("band"):
                ib = int(line.split()[1]) - 1
                energies[ik, ib] = _floats(line)[0]
            elif line.startswith("ion") and line.split()[-1] != "tot":
                n_orbitals = len(line.split()) - 1
                if phases is None:
                    phases = np.full((n_k, n_bands, n_atoms, n_orbitals), np.nan, dtype=complex)
                for ia in range(n_atoms):
                    values = np.array(_floats(next(handle))[: 2 * n_orbitals])
                    phases[ik, ib, ia] = values[0::2] + 1j * values[1::2]
    if phases is None:
        raise ValueError(f"{path}: no phase section; the run needs LORBIT = 12")
    if np.isnan(kpoints).any() or np.isnan(energies).any() or np.isnan(phases).any():
        raise ValueError(f"{path}: incomplete PROCAR (missing k-points, bands or phases)")
    return Procar(kpoints=kpoints, energies=energies, phases=phases)


def _unique_mod_one(points: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.round(points, 10), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=1e-9)] = 0.0
    return np.unique(np.round(wrapped, 9), axis=0)


def primitive_translations(matrix: np.ndarray) -> np.ndarray:
    """The N = |det M| primitive lattice points in the supercell, (N, 3) supercell fractional."""
    n_cells = round(abs(np.linalg.det(matrix)))
    box = np.array(list(itertools.product(range(n_cells), repeat=3)), dtype=float)
    translations = _unique_mod_one(box @ np.linalg.inv(matrix))
    if len(translations) != n_cells:
        raise ValueError(f"found {len(translations)} translations for |det M| = {n_cells}")
    return translations


def atom_permutations(
    positions: np.ndarray, species: Sequence[str], translations: np.ndarray
) -> np.ndarray:
    """perm[t, a] is the atom that sits at r_a + t (same species), (N, n_atoms) int."""
    labels = np.asarray(species)
    perms = np.empty((len(translations), len(positions)), dtype=int)
    for it, t in enumerate(translations):
        delta = positions[None, :, :] - (positions[:, None, :] + t)
        delta -= np.round(delta)
        hits = np.all(np.abs(delta) < _POSITION_TOL, axis=2) & (labels[None, :] == labels[:, None])
        if not np.all(hits.sum(axis=1) == 1):
            raise ValueError(f"translation {t} does not map the atoms one to one")
        perms[it] = hits.argmax(axis=1)
    return perms


def translation_overlaps(phases: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    """<c|T_t c> / <c|c> for coefficient blocks (..., n_atoms, n_orbitals), shape (..., N)."""
    norm = np.sum(np.abs(phases) ** 2, axis=(-2, -1))
    overlaps = [
        np.sum(np.conj(phases) * phases[..., perm, :], axis=(-2, -1)) for perm in permutations
    ]
    return np.stack(overlaps, axis=-1) / norm[..., None]


def unfolding_weights(phases: np.ndarray, permutations: np.ndarray) -> np.ndarray:
    return translation_overlaps(phases, permutations).mean(axis=-1).real


def _supercell_reciprocal_cosets(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One integer supercell reciprocal vector G per coset modulo the primitive reciprocal
    lattice (G = M m), shape (N, 3), and a bool (N,) marking the coset of the primitive lattice."""
    n_cells = round(abs(np.linalg.det(matrix)))
    inv = np.linalg.inv(matrix)
    seen: dict[tuple[float, ...], np.ndarray] = {}
    for g in itertools.product(range(n_cells), repeat=3):
        key = _unique_mod_one((inv @ np.array(g, dtype=float))[None, :])[0]
        seen.setdefault(tuple(key.tolist()), np.array(g, dtype=float))
    if len(seen) != n_cells:
        raise ValueError(f"found {len(seen)} reciprocal cosets for |det M| = {n_cells}")
    keys = np.array(list(seen.keys()))
    return np.array(list(seen.values())), np.all(keys == 0.0, axis=1)


def _analytic_supercell(
    matrix: np.ndarray, translations: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """A shuffled supercell of a two-atom primitive cell: positions (n, 3), species, and the
    primitive basis index (n,) of every atom."""
    basis = np.array([[0.0, 0.0, 0.0], [0.31, 0.62, 0.47]])
    inv = np.linalg.inv(matrix)
    positions = np.mod(np.concatenate([b @ inv + translations for b in basis]), 1.0)
    site = np.repeat(np.arange(len(basis)), len(translations))
    order = rng.permutation(len(positions))
    names = ("A", "B")
    return positions[order], tuple(names[s] for s in site[order]), site[order]


def _check_matrix(
    matrix: np.ndarray, rng: np.random.Generator, n_orbitals: int = 9
) -> dict[str, float | int | bool]:
    translations = primitive_translations(matrix)
    positions, species, site = _analytic_supercell(matrix, translations, rng)
    perms = atom_permutations(positions, species, translations)
    g_vectors, in_primitive = _supercell_reciprocal_cosets(matrix)
    orbitals = rng.normal(size=(2, n_orbitals)) + 1j * rng.normal(size=(2, n_orbitals))
    plane_waves = np.exp(2j * np.pi * g_vectors @ positions.T)[:, :, None] * orbitals[site]

    weights = unfolding_weights(plane_waves, perms)
    expected = in_primitive.astype(float)
    error = float(np.abs(weights - expected).max())
    require(error < _ANALYTIC_TOL, f"plane-wave weights off by {error:.2e} for M = {matrix}")

    flat = plane_waves.reshape(len(g_vectors), -1)
    gram = np.conj(flat) @ flat.T
    off_diagonal = float(np.abs(gram - np.diag(np.diag(gram))).max() / np.abs(np.diag(gram)).min())
    require(off_diagonal < _ANALYTIC_TOL, f"plane-wave states not orthogonal ({off_diagonal:.2e})")
    sum_error = float(abs(weights.sum() - 1.0))
    require(sum_error < _ANALYTIC_TOL, f"weights of the N states sum to {weights.sum()}")

    alpha, beta = 0.6 - 0.3j, 0.5 + 0.55j
    primitive_state = plane_waves[in_primitive][0]
    mixture_errors = [
        abs(
            float(unfolding_weights(alpha * primitive_state + beta * other, perms))
            - abs(alpha) ** 2 / (abs(alpha) ** 2 + abs(beta) ** 2)
        )
        for other in plane_waves[~in_primitive]
    ]
    mixture_error = max(mixture_errors)
    require(mixture_error < _ANALYTIC_TOL, f"mixture weight off by {mixture_error:.2e}")

    kpoint = np.array([0.23, 0.11, 0.37])
    phased = translation_overlaps(plane_waves, perms) * np.exp(-2j * np.pi * translations @ kpoint)
    control_error = float(np.abs(phased.mean(axis=-1).real - expected).max())
    require(control_error > 0.1, f"k-phase control passed the 0/1 test ({control_error:.2e})")

    return {
        "n_translations": len(translations),
        "max_weight_error": error,
        "max_orthogonality_error": off_diagonal,
        "weight_sum_error": sum_error,
        "max_mixture_error": mixture_error,
        "kphase_control_max_error": control_error,
    }


def validate() -> dict[str, float | int | bool | str | list]:
    """Plane-wave states in supercells of a two-atom cell must unfold to exactly 0 or 1."""
    rng = np.random.default_rng(287)
    matrices = {
        "diag_2_2_2": np.diag([2, 2, 2]),
        "shear_det2": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]]),
        "shear_det4": np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]),
        "non_normal_shear_det2": np.array([[1, 0, 0], [1, 2, 0], [0, 0, 1]]),
    }
    per_matrix = {name: _check_matrix(m, rng) for name, m in matrices.items()}
    run_together = _floats(" k-point  9 :    0.50000000-0.25000000-0.12500000     weight")
    require(run_together == [0.5, -0.25, -0.125], str(run_together))
    return {
        "tolerance": _ANALYTIC_TOL,
        "matrices": list(matrices),
        "n_translations": [r["n_translations"] for r in per_matrix.values()],
        "max_weight_error": max(float(r["max_weight_error"]) for r in per_matrix.values()),
        "max_orthogonality_error": max(
            float(r["max_orthogonality_error"]) for r in per_matrix.values()
        ),
        "max_weight_sum_error": max(float(r["weight_sum_error"]) for r in per_matrix.values()),
        "max_mixture_error": max(float(r["max_mixture_error"]) for r in per_matrix.values()),
        "kphase_control_min_error": min(
            float(r["kphase_control_max_error"]) for r in per_matrix.values()
        ),
    }
