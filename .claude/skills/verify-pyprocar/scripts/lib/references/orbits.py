"""Closed orbits of a plane through a periodic band isosurface, from a tiled marching cube.

Inputs are band energies ``energies`` (N1, N2, N3) at the fractional points (i/N1, j/N2,
k/N3) of a full Gamma-centred grid, an isovalue, the reciprocal lattice ``lattice`` (3, 3)
whose rows are b1, b2, b3 in 1/Angstrom without 2 pi (Cartesian k = frac @ lattice), and a
plane (Cartesian unit normal and origin). Marching cubes runs once on the periodic cell
[0, 1]^3 (N + 1 grid points per axis). Every lattice translate within ``reach`` cells whose
surface straddles the plane is sliced with VTK, the slice points are merged by distance
with a KD-tree, and the merged polylines split into closed loops and open curves. A loop
counts when a vertex lies in the first (Wigner-Seitz) zone, once per lattice-translation
class; an open curve counts when it meets the zone. The reach is 8 cells, and 16 when an
open curve through the zone reaches the tile edge. ``read_vasp_grid`` builds the grid from
a VASP IBZ run (EIGENVAL unfolded by the brute-force lattice point group). ``validate``
checks the areas against pi r^2 / cos t for a cylinder about the zone edge cut at tilt t,
pi (r^2 - d^2) for a Gamma sphere on an fcc reciprocal lattice, the same for a cut near the
X face of the fcc zone, and open curves for a plane along the cylinder.
"""

import itertools
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from references.validation import ValidationError, require
from references.vasp import read_eigenval, read_poscar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

REACHES = (8, 16)
MERGE_DISTANCE = 1e-10
"""1/Angstrom. Shared cut points from neighbouring translates differ by float rounding
(about 1e-15 at 16 cells); distinct cut points are grid-spacing apart."""
PLANE_NUDGE = 1e-9
"""1/Angstrom. The plane moves this far towards Gamma so that no mesh vertex lies on it (a
plane through Gamma along a lattice direction contains grid lines) and a plane on a zone face
stays in the zone. The areas change by about the perimeter times this."""
ZONE_SLACK = 1e-9
SHELL = np.array([s for s in itertools.product(range(-2, 3), repeat=3) if any(s)], float)


@dataclass(frozen=True, slots=True)
class Plane:
    """A plane n . k = n . origin in Cartesian k (1/Angstrom)."""

    normal: np.ndarray
    origin: np.ndarray

    @property
    def unit(self) -> np.ndarray:
        return self.normal / np.linalg.norm(self.normal)

    @property
    def offset(self) -> float:
        return float(self.unit @ self.origin)


@dataclass(frozen=True, slots=True)
class OrbitCut:
    """The orbits of one plane: one loop (M, 3) per counted closed orbit, with its area.

    ``n_open`` counts the open curves that meet the zone, ``open_at_edge`` says whether one
    of them ends on the tile edge at the final ``reach``, and ``n_defects`` counts open ends
    inside the tile plus branching points on curves through the zone (0 for a closed surface).
    """

    areas: tuple[float, ...]
    loops: tuple[np.ndarray, ...]
    n_open: int
    open_at_edge: bool
    reach: int
    n_defects: int


def cell_surface(energies: np.ndarray, isovalue: float, lattice: np.ndarray) -> pv.PolyData | None:
    """Marching cubes of one period of ``energies`` (N1, N2, N3), in Cartesian k."""
    n = energies.shape
    tile = np.pad(energies, [(0, 1)] * 3, mode="wrap")
    image = pv.ImageData(dimensions=tile.shape, spacing=tuple(1.0 / m for m in n))
    image.point_data["e"] = tile.reshape(-1, order="F")
    mesh = image.contour([isovalue], scalars="e", method="marching_cubes")
    assert isinstance(mesh, pv.PolyData)
    if mesh.n_cells == 0:
        return None
    return pv.PolyData(np.asarray(mesh.points, dtype=np.float64) @ lattice, mesh.faces)


def _segments(polylines: np.ndarray) -> np.ndarray:
    """Point-index pairs (S, 2) of a VTK lines array; a plane slice holds only 2-point lines."""
    cells = polylines.reshape(-1, 3)
    if not np.all(cells[:, 0] == 2):
        raise ValueError("the slice holds a polyline with more than two points")
    return cells[:, 1:]


def _zone_measure(points: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """max_G p . G / |G|^2 for each point (P, 3); at most 1/2 inside the first zone."""
    g = SHELL @ lattice
    return (points @ g.T / (g * g).sum(axis=1)).max(axis=1)


def _loop_order(component: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Nodes of a closed component (every node of degree 2) in walking order."""
    neighbours: dict[int, list[int]] = {int(c): [] for c in component}
    for a, b in pairs.tolist():
        neighbours[a].append(b)
        neighbours[b].append(a)
    start = int(component[0])
    order = [start, neighbours[start][0]]
    while len(order) < len(component):
        a, b = neighbours[order[-1]]
        order.append(b if a == order[-2] else a)
    return np.array(order)


def _area_and_centroid(loop: np.ndarray, normal: np.ndarray) -> tuple[float, np.ndarray]:
    """Area and centroid of a planar polygon (M, 3), from a fan about its vertex mean."""
    mean = loop.mean(axis=0)
    rel = loop - mean
    signed = np.cross(rel, np.roll(rel, -1, axis=0)) @ normal
    total = signed.sum()
    centroid = mean + (signed[:, None] * (rel + np.roll(rel, -1, axis=0))).sum(axis=0) / (3 * total)
    return 0.5 * abs(float(total)), centroid


def _tiled_cut(surface: pv.PolyData, lattice: np.ndarray, plane: Plane, reach: int) -> OrbitCut:
    """Orbits of one band's cut through the translates within ``reach`` cells."""
    normal = plane.unit
    offset = plane.offset - PLANE_NUDGE * (1.0 if plane.offset >= 0 else -1.0)
    heights = np.asarray(surface.points) @ normal
    steps = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), float)
    shift_heights = steps @ (lattice @ normal)
    keep = (offset - shift_heights >= heights.min()) & (offset - shift_heights <= heights.max())
    points, pairs, base = [], [], 0
    with pv.vtk_verbosity("error"):  # vtkmSlice warns on every PolyData slice, then falls back
        for step in steps[keep]:
            shift = step @ lattice
            cut = surface.slice(normal=normal, origin=(offset - shift @ normal) * normal)
            if cut.n_points == 0:
                continue
            points.append(np.asarray(cut.points, dtype=np.float64) + shift)
            pairs.append(_segments(np.asarray(cut.lines)) + base)
            base += cut.n_points
    if not points:
        return OrbitCut((), (), 0, False, reach, 0)
    p = np.concatenate(points)
    close = cKDTree(p).query_pairs(MERGE_DISTANCE, output_type="ndarray")
    graph = coo_matrix((np.ones(len(close)), (close[:, 0], close[:, 1])), shape=(len(p), len(p)))
    _, node = connected_components(graph, directed=False)
    seg = node[np.concatenate(pairs)]
    seg = np.unique(np.sort(seg[seg[:, 0] != seg[:, 1]], axis=1), axis=0)
    n_nodes = int(node.max()) + 1
    position = np.zeros((n_nodes, 3))
    position[node] = p
    degree = np.bincount(seg.reshape(-1), minlength=n_nodes)
    n_comp, label = connected_components(
        coo_matrix((np.ones(len(seg)), (seg[:, 0], seg[:, 1])), shape=(n_nodes, n_nodes)),
        directed=False,
    )
    inside = _zone_measure(position, lattice) <= 0.5 + ZONE_SLACK
    in_zone = np.bincount(label, weights=inside, minlength=n_comp) > 0
    to_frac = np.linalg.inv(lattice)
    frac = position @ to_frac
    on_edge = np.minimum(np.abs(frac + reach), np.abs(frac - reach - 1)).min(axis=1) < 1e-6
    areas: list[float] = []
    loops: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    n_open, open_at_edge, n_defects = 0, False, 0
    seg_label = label[seg[:, 0]]
    for c in np.flatnonzero(in_zone):
        members = np.flatnonzero(label == c)
        deg = degree[members]
        ends = deg == 1
        n_defects += int((ends & ~on_edge[members]).sum() + (deg > 2).sum())
        if (deg != 2).any():
            n_open += 1
            open_at_edge |= bool((ends & on_edge[members]).any())
            continue
        loop = position[_loop_order(members, seg[seg_label == c])]
        area, centroid = _area_and_centroid(loop, normal)
        centroid_frac = centroid @ to_frac
        shifts = [centroid_frac - f for f in centroids]
        if any(
            abs(area - a) <= 1e-6 * a and np.abs(s - np.rint(s)).max() < 1e-6
            for a, s in zip(areas, shifts, strict=True)
        ):
            continue
        areas.append(area)
        loops.append(loop)
        centroids.append(centroid_frac)
    return OrbitCut(tuple(areas), tuple(loops), n_open, open_at_edge, reach, n_defects)


def _grown_cut(
    surface: pv.PolyData, lattice: np.ndarray, plane: Plane, reaches: Sequence[int]
) -> OrbitCut:
    """The cut at the first reach whose open curves through the zone stay off the tile edge."""
    for reach in reaches[:-1]:
        cut = _tiled_cut(surface, lattice, plane, reach)
        if not cut.open_at_edge:
            return cut
    return _tiled_cut(surface, lattice, plane, reaches[-1])


def plane_orbits(
    surfaces: Sequence[pv.PolyData],
    lattice: np.ndarray,
    plane: Plane,
    reaches: Sequence[int] = REACHES,
) -> OrbitCut:
    """Orbits of ``plane`` through every band surface, each band grown on its own."""
    cuts = [_grown_cut(surface, lattice, plane, reaches) for surface in surfaces]
    pairs = sorted(
        ((area, loop) for cut in cuts for area, loop in zip(cut.areas, cut.loops, strict=True)),
        key=lambda pair: pair[0],
    )
    return OrbitCut(
        tuple(area for area, _ in pairs),
        tuple(loop for _, loop in pairs),
        sum(cut.n_open for cut in cuts),
        any(cut.open_at_edge for cut in cuts),
        max((cut.reach for cut in cuts), default=0),
        sum(cut.n_defects for cut in cuts),
    )


@dataclass(frozen=True, slots=True)
class BandGrid:
    """Band energies (n_bands, N1, N2, N3) on a full Gamma-centred grid of ``lattice``.

    ``real_lattice`` rows are a1, a2, a3 in Angstrom and ``lattice`` = inv(real_lattice).T.
    ``n_operations`` is the size of the lattice point group used to unfold the IBZ, and
    ``image_spread`` the largest energy difference between k-points that land on one grid
    point.
    """

    real_lattice: np.ndarray
    lattice: np.ndarray
    energies: np.ndarray
    fermi: float
    n_operations: int
    image_spread: float


def lattice_point_group(lattice: np.ndarray) -> np.ndarray:
    """Integer matrices (n_ops, 3, 3) with entries in {-1, 0, 1} that keep the metric of the
    basis ``lattice`` (rows), acting on fractional column vectors."""
    metric = lattice @ lattice.T
    entries = np.array(list(itertools.product((-1, 0, 1), repeat=9)), float).reshape(-1, 3, 3)
    kept = np.abs(np.einsum("oji,jk,okl->oil", entries, metric, entries) - metric).max(axis=(1, 2))
    return entries[kept < 1e-6 * np.abs(metric).max()].astype(np.int64)


def read_vasp_grid(calc: Path) -> BandGrid:
    """POSCAR, KPOINTS (Gamma-centred), OUTCAR's last E-fermi and EIGENVAL (ISPIN = 1), with
    the IBZ k-points unfolded by the lattice point group and time reversal."""
    real = read_poscar(calc / "POSCAR").lattice
    lattice = np.linalg.inv(real).T
    kpoints = (calc / "KPOINTS").read_text(encoding="utf-8").splitlines()
    if kpoints[2].strip()[0] not in "Gg":
        raise ValueError(f"KPOINTS mode {kpoints[2]!r} is not Gamma")
    n = np.array([int(x) for x in kpoints[3].split()[:3]])
    if np.any([float(x) for x in kpoints[4].split()[:3]]):
        raise ValueError("KPOINTS has a shift")
    fermi_lines = [
        line
        for line in (calc / "OUTCAR").read_text(encoding="utf-8").splitlines()
        if "E-fermi" in line
    ]
    fermi = float(fermi_lines[-1].split()[2])
    eigenval = read_eigenval(calc / "EIGENVAL")
    k_frac, energies = eigenval.kpoints, eigenval.energies
    n_k, n_bands = energies.shape
    ops = lattice_point_group(lattice)
    ops = np.unique(np.concatenate([ops, -ops]), axis=0)
    images = np.einsum("oij,kj->koi", ops, k_frac) * n
    if np.abs(images - np.rint(images)).max() >= 1e-4:
        raise ValueError("k-points are off the grid")
    index = np.rint(images).astype(np.int64) % n
    flat = np.ravel_multi_index((index[..., 0], index[..., 1], index[..., 2]), tuple(n))
    point = flat.reshape(-1)
    owner = np.repeat(np.arange(n_k), len(ops))
    size = int(np.prod(n))
    if np.bincount(point, minlength=size).min() == 0:
        raise ValueError("the unfolded k-points miss grid points")
    grid = np.empty((n_bands, size))
    spread = 0.0
    for band in range(n_bands):
        low = np.full(size, np.inf)
        high = np.full(size, -np.inf)
        np.minimum.at(low, point, energies[owner, band])
        np.maximum.at(high, point, energies[owner, band])
        grid[band] = low
        spread = max(spread, float((high - low).max()))
    return BandGrid(real, lattice, grid.reshape(n_bands, *n), fermi, len(ops), spread)


def _grid_points(n: int) -> np.ndarray:
    f = np.arange(n) / n
    return np.stack(np.meshgrid(f, f, f, indexing="ij"), axis=-1)


def cylinder_energies(n: int) -> np.ndarray:
    """(kx - 1/2)^2 + (ky - 1/2)^2 in fractional coordinates on an n^3 grid."""
    f = _grid_points(n)
    return (f[..., 0] - 0.5) ** 2 + (f[..., 1] - 0.5) ** 2


def sphere_energies(n: int, lattice: np.ndarray) -> np.ndarray:
    """min_G |k - G|^2 on an n^3 grid of ``lattice``."""
    k = _grid_points(n) @ lattice
    images = np.array(list(itertools.product(range(-2, 3), repeat=3)), float) @ lattice
    return np.min([((k - g) ** 2).sum(axis=-1) for g in images], axis=0)


def cylinder_orbit_count(radius: float, tilt: float, height: float) -> tuple[int, float]:
    """Translation classes of the cylinder ellipses that meet the cube zone |k_i| <= 1/2,
    and the smallest zone measure margin over all ellipses, for normal (sin t, 0, cos t).

    The plane holds (0, 1, 0), so cylinders that differ only in y are translates. The
    cylinders at x = -1/2 and 1/2 are translates when the plane holds some (1, 0, k), that
    is when tan t is an integer."""
    phi = np.linspace(0, 2 * np.pi, 20001)
    count, margin = 0, np.inf
    for centre in (-0.5, 0.5):
        x = centre + radius * np.cos(phi)
        y = 0.5 + radius * np.sin(phi)
        z = (height - x * np.sin(tilt)) / np.cos(tilt)
        measure = np.max(np.abs(np.stack([x, y, z])), axis=0)
        count += bool((measure <= 0.5).any())
        margin = min(margin, abs(float(measure.min()) - 0.5))
    tan = np.tan(tilt)
    return (min(count, 1) if abs(tan - np.rint(tan)) < 1e-9 else count), margin


@dataclass(frozen=True, slots=True)
class _Case:
    """An analytic cut: ``count`` orbits of area ``area`` and ``n_open`` open curves."""

    name: str
    surface: pv.PolyData
    lattice: np.ndarray
    plane: Plane
    area: float
    count: int
    n_open: int


FCC_RECIPROCAL = np.linalg.inv(np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])).T
CYLINDER_RADIUS = 0.3
CYLINDER_TILTS = (0.0, 0.5, 1.0, float(np.arctan(12.3)))
"""Radians. At tan t = 12.3 the ellipse is 3.7 cells long and reaches 3.7 cells along z."""
SPHERE_OFFSETS = (0.0, 0.4, 0.7)
SPHERE_NORMAL = np.array([0.3, -0.5, 0.81])
ZONE_SPHERE_FRACTION = 0.92
"""Of the inscribed radius of the fcc zone, so neighbouring Gamma spheres stay apart."""
ZONE_CUT_OFFSET = 0.55
"""1/Angstrom along x. The fcc zone is a truncated octahedron that reaches x = 1, and this cut
loop lies wholly inside it but at x > 1/2, outside the unit cube. A zone test that took the
fractional G for the Cartesian G finds no orbit here."""
GRIDS = (16, 32)
AREA_TOLERANCE = {16: 0.05, 32: 0.015}
"""Relative. Linear interpolation of a quadratic band along grid edges puts the marching-cubes
vertices O(h^2) inside the surface: measured at most 4.0% at N = 16 and 1.0% at N = 32,
worst for the small sphere cap at d = 0.7 r."""
MIN_REFINEMENT_GAIN = 3.0
"""The worst area error must fall at least 3x from N = 16 to N = 32 (O(h^2) gives 4x)."""


def _surface(energies: np.ndarray, isovalue: float, lattice: np.ndarray) -> pv.PolyData:
    surface = cell_surface(energies, isovalue, lattice)
    if surface is None:
        raise ValidationError(f"no isosurface at {isovalue}")
    return surface


def _cases(n: int) -> list[_Case]:
    cylinder = _surface(cylinder_energies(n), CYLINDER_RADIUS**2, np.eye(3))
    cases = []
    for tilt in CYLINDER_TILTS:
        normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
        origin = np.array([0.35, 0.0, 0.05])
        count, margin = cylinder_orbit_count(CYLINDER_RADIUS, tilt, float(normal @ origin))
        require(margin > 0.02, f"tilt {tilt}: an ellipse grazes the zone ({margin})")
        area = np.pi * CYLINDER_RADIUS**2 / np.cos(tilt)
        cases.append(
            _Case(
                f"cylinder_t{tilt:.3f}", cylinder, np.eye(3), Plane(normal, origin), area, count, 0
            )
        )
    cases.append(
        _Case(
            "cylinder_parallel",
            cylinder,
            np.eye(3),
            Plane(np.array([1.0, 0.0, 0.0]), np.array([0.4, 0.0, 0.0])),
            0.0,
            0,
            2,
        )
    )
    inscribed = 0.5 * float(np.linalg.norm(FCC_RECIPROCAL, axis=1).min())
    radius = 0.6 * inscribed
    fcc_energies = sphere_energies(n, FCC_RECIPROCAL)
    sphere = _surface(fcc_energies, radius**2, FCC_RECIPROCAL)
    unit = SPHERE_NORMAL / np.linalg.norm(SPHERE_NORMAL)
    for fraction in SPHERE_OFFSETS:
        d = fraction * radius
        area = np.pi * (radius**2 - d**2)
        cases.append(
            _Case(f"sphere_d{fraction}r", sphere, FCC_RECIPROCAL, Plane(unit, d * unit), area, 1, 0)
        )
    zone_radius = ZONE_SPHERE_FRACTION * inscribed
    x = np.array([1.0, 0.0, 0.0])
    cases.append(
        _Case(
            "fcc_zone_outside_unit_cube",
            _surface(fcc_energies, zone_radius**2, FCC_RECIPROCAL),
            FCC_RECIPROCAL,
            Plane(x, ZONE_CUT_OFFSET * x),
            np.pi * (zone_radius**2 - ZONE_CUT_OFFSET**2),
            1,
            0,
        )
    )
    return cases


def _area_error(case: _Case, cut: OrbitCut, tolerance: float) -> float:
    """Worst relative area error of a cut, after its orbit and open-curve counts check out."""
    require(len(cut.areas) == case.count, f"{case.name}: {len(cut.areas)} orbits, not {case.count}")
    require(cut.n_open == case.n_open, f"{case.name}: {cut.n_open} open curves, not {case.n_open}")
    require(cut.n_defects == 0, f"{case.name}: {cut.n_defects} open ends inside the tile")
    if case.n_open:
        require(cut.open_at_edge and cut.reach == REACHES[-1], f"{case.name}: open curve not grown")
    errors = [abs(a - case.area) / case.area for a in cut.areas]
    worst = max(errors, default=0.0)
    require(worst < tolerance, f"{case.name}: area error {worst:.3g} >= {tolerance}")
    return worst


def validate() -> dict[str, float | int | bool | str | list]:
    """Analytic cuts at N = 16 and 32, and the single-cell (untiled) negative control."""
    start = time.perf_counter()
    cases = {n: _cases(n) for n in GRIDS}
    errors = {
        n: [
            _area_error(
                case, plane_orbits([case.surface], case.lattice, case.plane), AREA_TOLERANCE[n]
            )
            for case in cases[n]
        ]
        for n in GRIDS
    }
    gain = max(errors[GRIDS[0]]) / max(errors[GRIDS[1]])
    require(gain >= MIN_REFINEMENT_GAIN, f"error fell only {gain:.2f}x from N=16 to N=32")
    steep = cases[GRIDS[-1]][len(CYLINDER_TILTS) - 1]
    untiled = plane_orbits([steep.surface], steep.lattice, steep.plane, reaches=(0,))
    try:
        _area_error(steep, untiled, AREA_TOLERANCE[GRIDS[-1]])
    except ValidationError as rejected:
        control = str(rejected)
    else:
        raise ValidationError("the single-cell cut passed the steep cylinder case")
    return {
        "cases": [case.name for case in cases[GRIDS[-1]]],
        "rel_area_error_n16": errors[GRIDS[0]],
        "rel_area_error_n32": errors[GRIDS[1]],
        "refinement_gain": gain,
        "tolerance_n16": AREA_TOLERANCE[GRIDS[0]],
        "tolerance_n32": AREA_TOLERANCE[GRIDS[1]],
        "untiled_control_rejected": control,
        "untiled_control_orbits": len(untiled.areas),
        "untiled_control_open": untiled.n_open,
        "seconds": time.perf_counter() - start,
    }
