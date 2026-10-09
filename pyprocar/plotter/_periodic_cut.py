"""Closed orbits of a plane through a periodic band isosurface.

Each band is marching-cubed once over one period: the (n+1)^3 periodic grid that spans the
cell of the tile lattice, a reduced basis of the k-grid (see ``PeriodicGrid``), in its
fractional coordinates. A mesh vertex on an upper cell face is the lower-face
vertex of the next cell, so every vertex gets a class and an integer cell. A point where
the plane cuts a mesh edge in some lattice translate of the cell is named by the edge's
class (its two vertex classes and their relative cell) and the cell of its lower-class
vertex, so the translates on both sides of a cell face name a shared point identically.
The cut curves join by these names, with no distance tolerance, and a curve closes or not
on its own.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import final

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import HalfspaceIntersection

from pyprocar.core._periodic_grid import CUT_TILE_BUDGET
from pyprocar.core.brillouin_zone import zone_face_steps

MAX_REACH = 16
"""Tile cells from the origin within which an orbit through the zone must close."""

_NEAR_VERTEX = np.array(list(itertools.product(range(-1, 3), repeat=3)))
"""Steps, relative to a vertex's cell, of the translates cut next at an open curve end.

The triangles at the vertex are in the translates 0 or 1 below; one more on each side
lets a curve grow two cells per round (0.6 s instead of 1.2 s on Au at 60^3)."""
_START_STEPS = np.array(list(itertools.product(range(-2, 3), repeat=3)))
_SLOTS = ((0, 1), (1, 2), (2, 0))
_PACK = 64
_PACK_OFFSET = 32
_CELLS = _PACK**3


@dataclass(frozen=True)
class PeriodicBand:
    """One period of a band isosurface in fractional coordinates.

    ``canon`` holds one position per vertex class. Each triangle has the class and cell of
    its vertices (``tri_cls``, ``tri_off``) and the class of its edges 01, 12 and 20
    (``tri_edges``); ``edge_step`` is the cell of an edge's higher-class vertex relative to
    its lower-class one.
    """

    canon: np.ndarray
    tri_cls: np.ndarray
    tri_off: np.ndarray
    tri_edges: np.ndarray
    edge_step: np.ndarray


def periodic_band(grid: np.ndarray, isovalue: float, shift: np.ndarray) -> PeriodicBand | None:
    """Marching cubes of ``grid`` (values at (i + shift) / N) over one period."""
    n = np.asarray(grid.shape)
    spacing, origin = 1.0 / n, shift / n
    tile = np.pad(grid, [(0, 1)] * 3, mode="wrap")
    image = pv.ImageData(dimensions=tile.shape, spacing=tuple(spacing), origin=tuple(origin))
    image.point_data["e"] = tile.reshape(-1, order="F")
    mesh = image.contour([isovalue], scalars="e", method="marching_cubes").triangulate()
    if mesh.n_cells == 0:
        return None
    points = np.asarray(mesh.points)
    low = origin.astype(points.dtype)
    high = (origin + n * spacing).astype(points.dtype)
    on_high = points == high
    reduced = np.where(on_high, low, points)
    _, first, cls = np.unique(reduced, axis=0, return_index=True, return_inverse=True)
    triangles = np.asarray(mesh.regular_faces)
    tri_cls = cls.reshape(-1)[triangles]
    tri_off = on_high.astype(np.int64)[triangles]
    a, b = [0, 1, 2], [1, 2, 0]
    flip = tri_cls[:, a] > tri_cls[:, b]
    step = np.where(flip[..., None], tri_off[:, a] - tri_off[:, b], tri_off[:, b] - tri_off[:, a])
    table = np.column_stack(
        [
            np.minimum(tri_cls[:, a], tri_cls[:, b]).reshape(-1),
            np.maximum(tri_cls[:, a], tri_cls[:, b]).reshape(-1),
            step.reshape(-1, 3),
        ]
    )
    edges, edge_ids = np.unique(table, axis=0, return_inverse=True)
    return PeriodicBand(
        canon=reduced[first].astype(np.float64),
        tri_cls=tri_cls,
        tri_off=tri_off,
        tri_edges=edge_ids.reshape(-1, 3),
        edge_step=edges[:, 2:],
    )


@dataclass(frozen=True)
class PeriodicBands:
    """One period of each band, and the two lattices it is only correct beside.

    The bands repeat by ``lattice``, the tile, which translates them and counts MAX_REACH.
    ``reciprocal_lattice`` is the crystal's: it gives the first zone and which orbits are
    translates of one another. They differ when the k-points are given in a sheared basis.
    """

    lattice: np.ndarray
    reciprocal_lattice: np.ndarray
    bands: list[PeriodicBand]


def periodic_bands(surface) -> PeriodicBands | None:
    """One period of each band of a FermiSurface, or None for any other mesh, a 2D grid,
    k-points that do not fill the uniform kgrid once, or a tile above CUT_TILE_BUDGET grids."""
    ebs = getattr(surface, "original_ebs", None)
    keys = getattr(surface, "band_isosurfaces", None)
    isovalue = getattr(surface, "isovalue", None)
    if ebs is None or keys is None or isovalue is None:
        return None
    grid = surface._periodic_grid
    if grid is None or min(grid.n) < 2 or grid.tile_multiple > CUT_TILE_BUDGET:
        return None
    energies = np.asarray(ebs.get_property("bands").value)
    bands = []
    for iband, ispin in keys:
        band = periodic_band(grid.tile(energies[:, iband, ispin]), float(isovalue), grid.shift)
        if band is not None:
            bands.append(band)
    return PeriodicBands(grid.lattice, grid.reciprocal_lattice, bands)


def _pack(cls, cells: np.ndarray) -> np.ndarray:
    c = cells + _PACK_OFFSET
    return ((cls * _PACK + c[..., 0]) * _PACK + c[..., 1]) * _PACK + c[..., 2]


def _unpack_cells(keys: np.ndarray) -> np.ndarray:
    cells = np.stack([keys // _PACK**2 % _PACK, keys // _PACK % _PACK, keys % _PACK], axis=-1)
    return cells - _PACK_OFFSET


@final
class _BandCut:
    """The plane n.k = d through lattice translates of one band's period.

    A cut point on a mesh edge is named edge class * _CELLS + packed cell of the edge's
    lower-class vertex; a cut exactly through a vertex is named -(packed class and cell + 1).
    """

    def __init__(self, band: PeriodicBand, lattice: np.ndarray, normal: np.ndarray, d: float):
        self.band, self.lattice, self.normal, self.d = band, lattice, normal, d
        self.w = lattice @ normal
        self.heights = band.canon @ self.w
        tri_h = self.heights[band.tri_cls] + band.tri_off @ self.w
        lo, hi = tri_h.min(axis=1), tri_h.max(axis=1)
        self.order = np.argsort(lo)
        self.lo_sorted = lo[self.order]
        self.extent = float((hi - lo).max())
        self.margin = 1e-6 * self.extent

    def segments(self, steps: np.ndarray):
        """Oriented segments of the cut through the translates by ``steps`` (cells): the
        names and positions of their two ends.

        Candidates come from the triangle heights of the period with a margin; the cut is
        decided by ``s``, the height of each vertex from its absolute cell, which is the same
        float for a vertex in every triangle and translate that holds it.
        """
        band = self.band
        t = self.d - steps @ self.w
        start = np.searchsorted(self.lo_sorted, t - self.extent - self.margin, side="left")
        counts = np.searchsorted(self.lo_sorted, t + self.margin, side="right") - start
        if counts.sum() == 0:
            return None
        which = np.repeat(np.arange(len(steps)), counts)
        pos = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        tri = self.order[np.repeat(start, counts) + pos]
        cells = band.tri_off[tri] + steps[which][:, None, :]
        cls = band.tri_cls[tri]
        s = self.heights[cls] + cells @ self.w - self.d
        above = s > 0
        straddle = above.any(axis=1) & ~above.all(axis=1)
        tri, cells, cls = tri[straddle], cells[straddle], cls[straddle]
        s, above = s[straddle], above[straddle]
        if len(tri) == 0:
            return None
        vertex_names = -(_pack(cls, cells) + 1)
        corners = (band.canon[cls] + cells) @ self.lattice
        rows = np.arange(len(tri))
        names, points, cuts = [], [], []
        for slot, (a, b) in enumerate(_SLOTS):
            low_first = cls[:, a] <= cls[:, b]
            ia, ib = np.where(low_first, a, b), np.where(low_first, b, a)
            s_lo, s_hi = s[rows, ia], s[rows, ib]
            cut = above[:, a] != above[:, b]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(cut, s_lo / (s_lo - s_hi), 0.0)
            edge_names = band.tri_edges[tri, slot] * _CELLS + _pack(0, cells[rows, ia])
            names.append(
                np.where(
                    frac == 0,
                    vertex_names[rows, ia],
                    np.where(frac == 1, vertex_names[rows, ib], edge_names),
                )
            )
            p_lo, p_hi = corners[rows, ia], corners[rows, ib]
            points.append(p_lo + frac[:, None] * (p_hi - p_lo))
            cuts.append(cut)
        names, points, has = np.stack(names, 1), np.stack(points, 1), np.stack(cuts, 1)
        first = np.argmax(has, axis=1)
        second = 2 - np.argmax(has[:, ::-1], axis=1)
        n1, n2 = names[rows, first], names[rows, second]
        q1, q2 = points[rows, first], points[rows, second]
        tri_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        flip = np.einsum("ij,ij->i", q2 - q1, np.cross(self.normal, tri_normal)) < 0
        n1, n2 = np.where(flip, n2, n1), np.where(flip, n1, n2)
        q1, q2 = np.where(flip[:, None], q2, q1), np.where(flip[:, None], q1, q2)
        return n1, n2, q1, q2

    def vertex_cells(self, names: np.ndarray) -> np.ndarray:
        """Cells of the mesh vertices at the cut points ``names``."""
        on_vertex = names < 0
        edges = names[~on_vertex]
        low = _unpack_cells(edges % _CELLS)
        high = low + self.band.edge_step[edges // _CELLS]
        return np.concatenate([_unpack_cells((-names[on_vertex] - 1) % _CELLS), low, high])


@dataclass(frozen=True)
class _Curves:
    """Cut segments joined into curves: each node's name and degree, each curve's label."""

    names: np.ndarray
    a: np.ndarray
    b: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    degree: np.ndarray
    label: np.ndarray
    meets_region: np.ndarray
    is_open: np.ndarray


def _meets(q1: np.ndarray, q2: np.ndarray, region: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Whether each segment q1 q2 has a point in the region normals @ k <= offsets.

    The segment is clipped by each half-space in turn (Liang-Barsky): it meets the region
    when it enters every half-space before it leaves any.
    """
    f1 = q1 @ region[0].T - region[1]
    f2 = q2 @ region[0].T - region[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = f1 / (f1 - f2)
    enter = np.where((f1 > 0) & (f2 <= 0), t, 0.0).max(axis=1)
    leave = np.where((f1 <= 0) & (f2 > 0), t, 1.0).min(axis=1)
    return ~((f1 > 0) & (f2 > 0)).any(axis=1) & (enter <= leave)


def _curves(found: list[tuple[np.ndarray, ...]]) -> _Curves:
    """Join the found segments (end names, end points, whether each meets the region)."""
    n1, n2, q1, q2, inside = (np.concatenate(part) for part in zip(*found, strict=True))
    names, ids = np.unique(np.concatenate([n1, n2]), return_inverse=True)
    a, b = ids[: len(n1)], ids[len(n1) :]
    real = a != b
    a, b, q1, q2, inside = a[real], b[real], q1[real], q2[real], inside[real]
    n_nodes = len(names)
    degree = np.bincount(np.concatenate([a, b]), minlength=n_nodes)
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n_nodes, n_nodes))
    n_comp, label = connected_components(graph, directed=False)
    meets_region = np.bincount(label[a[inside]], minlength=n_comp) > 0
    is_open = np.bincount(label[degree == 1], minlength=n_comp) > 0
    return _Curves(names, a, b, q1, q2, degree, label, meets_region, is_open)


def _start_steps(zone: np.ndarray, half: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """_START_STEPS, then the other translates whose period meets the zone's bounding box in
    the tile's fractional coordinates (``inverse``), which can reach beyond two cells."""
    halfspaces = np.column_stack([zone, -half])
    corners = HalfspaceIntersection(halfspaces, np.zeros(3)).intersections @ inverse
    low = np.floor(corners.min(axis=0)).astype(int) - 1
    high = np.floor(corners.max(axis=0)).astype(int)
    if max(-low.min(), high.max()) > MAX_REACH:
        raise ValueError(f"the first zone spans more than {MAX_REACH} cells of this basis")
    window = np.array(list(itertools.product(*map(range, low, high + 1))))
    return np.vstack([_START_STEPS, window[np.abs(window).max(axis=1) > 2]])


def _band_curves(
    cut: _BandCut, region: tuple[np.ndarray, np.ndarray], start: np.ndarray
) -> _Curves | None:
    """Curves of the cut through the ``start`` translates, extended through the translates at
    the open ends of curves that meet the region until those close or reach MAX_REACH cells."""
    found: list[tuple[np.ndarray, ...]] = []
    done = np.empty(0, dtype=np.int64)
    steps = start
    while True:
        pieces = cut.segments(steps)
        if pieces is not None:
            found.append((*pieces, _meets(pieces[2], pieces[3], region)))
        done = np.concatenate([done, _pack(0, steps)])
        if not found:
            return None
        curves = _curves(found)
        ends = curves.names[(curves.degree == 1) & curves.meets_region[curves.label]]
        vertex_cells = cut.vertex_cells(ends)
        steps = np.unique((vertex_cells[:, None, :] - _NEAR_VERTEX).reshape(-1, 3), axis=0)
        steps = steps[np.abs(steps).max(axis=1) <= MAX_REACH]
        steps = steps[~np.isin(_pack(0, steps), done)]
        if len(steps) == 0:
            return curves


def plane_orbits(
    periodic: PeriodicBands,
    normal,
    origin,
    box: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[float], int]:
    """Areas of the closed orbits that meet the first zone, and the open curves that do.

    With ``box`` = (normals, offsets), only orbits that also meet the box (normals @ k <=
    offsets) count, so the count follows the drawn, box-clipped slice. Orbits that are
    reciprocal lattice translates of one another count once: their area-weighted centroids
    differ by a reciprocal lattice vector.
    """
    lattice = np.asarray(periodic.lattice, dtype=np.float64)
    reciprocal = np.asarray(periodic.reciprocal_lattice, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    d = float(np.asarray(origin, dtype=np.float64) @ normal)
    zone = zone_face_steps(reciprocal) @ reciprocal
    half = 0.5 * (zone * zone).sum(axis=1)
    region = (zone, half * (1 + 2e-9))
    if box is not None:
        region = (np.vstack([region[0], box[0]]), np.concatenate([region[1], box[1]]))
    start = _start_steps(zone, half, np.linalg.inv(lattice))
    inverse = np.linalg.inv(reciprocal)
    areas: list[float] = []
    n_open = 0
    for band in periodic.bands:
        curves = _band_curves(_BandCut(band, lattice, normal, d), region, start)
        if curves is None:
            continue
        n_comp = len(curves.meets_region)
        label = curves.label[curves.a]
        signed = np.cross(curves.q1, curves.q2) @ normal
        total = np.bincount(label, weights=signed, minlength=n_comp)
        comp_area = 0.5 * np.abs(total)
        moment = np.zeros((n_comp, 3))
        np.add.at(moment, label, (curves.q1 + curves.q2) * signed[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            centre = moment / (3 * total[:, None]) @ inverse
        kept: list[tuple[float, np.ndarray]] = []
        for c in np.flatnonzero(curves.meets_region):
            if curves.is_open[c]:
                n_open += 1
                continue
            shifted = [centre[c] - other for _, other in kept]
            if any(
                abs(comp_area[c] - area) <= 1e-6 * area
                and np.abs(delta - np.rint(delta)).max() < 1e-6
                for (area, _), delta in zip(kept, shifted, strict=True)
            ):
                continue
            kept.append((float(comp_area[c]), centre[c]))
        areas += [area for area, _ in kept]
    return areas, n_open
