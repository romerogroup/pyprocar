"""PyVista plotter behaviour shared by FermiPlotter and BS2DPlotter."""

import itertools
import logging
import os
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyvista import ColorLike

from pyprocar.plotter._series import SurfaceSeries, finite_range
from pyprocar.plotter.fs_slice_plot import FermiSlicePlotter

logger = logging.getLogger(__name__)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


MATCH_TOL = 1e-5
"""Fractional distance within which two curve ends are the same point."""

SNAP_ANGLE = 3e-4
"""Radians within which a slice normal is replaced by a low-index lattice direction.

Rounding a lattice direction to 4 significant digits turns it by at most 7.4e-5 rad on
cubic, hexagonal and fcc cells (3 digits: 7.7e-4, which random normals reach too);
distinct directions with indices up to 4 are at least 2.4e-2 rad apart there.
"""

_DIRECTION_INDICES = np.array(
    [
        uvw
        for uvw in itertools.product(range(-4, 5), repeat=3)
        if any(uvw) and np.gcd.reduce(np.abs(uvw)) == 1
    ]
)


def snap_normal(
    normal, reciprocal_lattice: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    """The unit normal, or the lattice direction [u v w] within SNAP_ANGLE of it.

    Only along a real-space lattice vector t = u a1 + v a2 + w a3 do the plane's lattice
    translates sit at discrete offsets: for G = m1 b1 + m2 b2 + m3 b3, n . G =
    (u m1 + v m2 + w m3) / |t|. A normal typed with a few digits misses such a direction
    slightly and cuts an irrational plane, where near-copies of one orbit count
    separately. Returns the indices when the normal was changed, otherwise None.
    """
    normal = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    real = np.linalg.inv(np.asarray(reciprocal_lattice, dtype=np.float64)).T
    directions = _DIRECTION_INDICES @ real
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cosines = directions @ normal
    best = int(np.argmax(cosines))
    angle = float(np.arccos(min(cosines[best], 1.0)))
    if angle > SNAP_ANGLE or angle < 1e-12:
        return normal, None
    u, v, w = (int(i) for i in _DIRECTION_INDICES[best])
    return directions[best], (u, v, w)


def slice_loop_areas(slc: pv.PolyData) -> tuple[list[float], int]:
    """Areas of the closed loops in a planar slice, and the number of open curves."""
    areas, chains, n_branched = slice_loops(slc)
    return areas, n_branched + len(chains)


def cross_section_areas(
    mesh: pv.DataSet, normal, origin, reciprocal_lattice: np.ndarray | None = None
) -> tuple[list[float], int]:
    """Areas of the closed orbits through the plane's cut of ``mesh``, and its open curves.

    With ``reciprocal_lattice`` (rows are the b vectors), ``mesh`` is one period of a
    periodic surface, such as a Fermi surface clipped to the first zone. A curve that
    leaves the cut is followed through the cuts of the plane's lattice translates, and it
    is a closed orbit when it comes back with no net translation. Orbits of the plane
    that never pass through this cut are not counted. A normal within SNAP_ANGLE of a
    low-index lattice direction is first replaced by it (see ``snap_normal``).
    """
    normal = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    if reciprocal_lattice is not None:
        normal, _ = snap_normal(normal, reciprocal_lattice)
    origin = np.asarray(origin, dtype=np.float64)
    areas, chains, n_open = slice_loops(cast(pv.PolyData, mesh.slice(normal=normal, origin=origin)))
    if reciprocal_lattice is None or not chains:
        return areas, n_open + len(chains)
    lattice = np.asarray(reciprocal_lattice, dtype=np.float64)
    heights = np.asarray(mesh.points, dtype=np.float64) @ normal
    steps = np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    shifts = steps @ lattice
    offsets = (origin - shifts) @ normal
    tol = 1e-3 * float(np.linalg.norm(lattice, axis=1).min())
    same_plane = 1e-6 * tol
    crossing = (offsets >= heights.min() - tol) & (offsets <= heights.max() + tol)
    other = crossing & (np.abs(offsets - origin @ normal) > same_plane)
    _, first = np.unique(np.round(offsets[other] / same_plane), return_index=True)
    pieces = []
    for shift in shifts[other][first]:
        _, cut_chains, _ = slice_loops(
            cast(pv.PolyData, mesh.slice(normal=normal, origin=origin - shift))
        )
        pieces += [chain + shift for chain in cut_chains]
    orbit_areas, n_unjoined = join_across_zone(chains + pieces, lattice, n_seeds=len(chains))
    return areas + orbit_areas, n_open + n_unjoined


def slice_loops(slc: pv.PolyData) -> tuple[list[float], list[np.ndarray], int]:
    """Closed-loop areas, open chains (ordered points) and branched curves of a slice.

    Segments that touch a non-finite point are dropped first, so a curve broken by
    NaN energies counts as open.
    """
    cells = np.asarray(slc.lines)
    pairs = []
    i = 0
    while i < len(cells):
        ids = cells[i + 1 : i + 1 + cells[i]]
        pairs.append(np.column_stack([ids[:-1], ids[1:]]))
        i += 1 + cells[i]
    lines = np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=int)
    points = np.asarray(slc.points, dtype=np.float64)
    lines = lines[np.isfinite(points[lines]).all(axis=(1, 2))]
    if len(lines) == 0:
        return [], [], 0
    unique_points, merged = np.unique(np.round(points, 9), axis=0, return_inverse=True)
    lines = merged.reshape(-1)[lines]
    neighbours: dict[int, list[int]] = {}
    for a, b in lines[lines[:, 0] != lines[:, 1]].tolist():
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    areas: list[float] = []
    chains: list[np.ndarray] = []
    n_open = 0
    seen: set[int] = set()
    for start in neighbours:
        if start in seen:
            continue
        component = [start]
        seen.add(start)
        for node in component:
            for nxt in neighbours[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    component.append(nxt)
        if any(len(neighbours[node]) > 2 for node in component):
            n_open += 1
            continue
        ends = [node for node in component if len(neighbours[node]) == 1]
        path = [ends[0] if ends else start]
        path.append(neighbours[path[0]][0])
        while len(path) < len(component):
            a, b = neighbours[path[-1]]
            path.append(b if a == path[-2] else a)
        if ends:
            chains.append(unique_points[path])
        else:
            areas.append(polygon_area(unique_points[path]))
    return areas, chains, n_open


def polygon_area(corners: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(corners, np.roll(corners, -1, axis=0)).sum(axis=0)))


def polygon_centroid(corners: np.ndarray) -> np.ndarray:
    """Area centroid of a planar polygon, from a fan of triangles on its first corner."""
    edges = corners[1:] - corners[0]
    cross = np.cross(edges[:-1], edges[1:])
    weights = cross @ cross.sum(axis=0)
    centres = (corners[0] + corners[1:-1] + corners[2:]) / 3
    return weights @ centres / weights.sum()


def join_across_zone(
    chains: list[np.ndarray], reciprocal_lattice: np.ndarray, n_seeds: int | None = None
) -> tuple[list[float], int]:
    """Join open chains whose ends coincide, or coincide modulo a reciprocal lattice vector.

    Walks start from the first ``n_seeds`` chains (all by default); the other chains only
    complete them. An end continues into an end at the same point when there is one, and
    otherwise into the closest end modulo a lattice vector. An orbit is closed when the
    walk returns to its first chain with no net lattice translation, and orbits that are
    lattice translates of each other count once. Returns the closed orbits' areas and the
    number of curves left open: chains with an end that matches nothing, and orbits that
    run through the zone.
    """
    inverse = np.linalg.inv(reciprocal_lattice)
    ends = np.array([chain[i] for chain in chains for i in (0, -1)])
    frac = ends @ inverse
    partner: dict[int, tuple[int, np.ndarray]] = {}
    for a in range(len(ends)):
        shift = frac - frac[a]
        whole = np.rint(shift)
        residual = np.abs(shift - whole).max(axis=1)
        residual[a] = np.inf
        close = residual < MATCH_TOL
        coincident = close & ~whole.any(axis=1)
        pool = coincident if coincident.any() else close
        if pool.any():
            best = int(np.argmin(np.where(pool, residual, np.inf)))
            partner[a] = (best, whole[best] @ reciprocal_lattice)

    orbits: list[tuple[float, np.ndarray]] = []
    n_open = 0
    done: set[int] = set()
    for first in range(len(chains) if n_seeds is None else n_seeds):
        if first in done:
            continue
        entry, offset, pieces = 2 * first, np.zeros(3), []
        while True:
            done.add(entry // 2)
            forward = entry % 2 == 0
            pieces.append(chains[entry // 2][:: 1 if forward else -1] + offset)
            leave = entry + 1 if forward else entry - 1
            if leave not in partner:
                n_open += 1
                break
            entry, lattice_vector = partner[leave]
            offset = offset - lattice_vector
            if entry // 2 in done:
                if entry == 2 * first and np.allclose(offset, 0.0):
                    orbit = np.concatenate(pieces)
                    area, centre = polygon_area(orbit), polygon_centroid(orbit) @ inverse
                    if not any(same_orbit(area, centre, *seen) for seen in orbits):
                        orbits.append((area, centre))
                else:
                    n_open += 1
                break
    return [area for area, _ in orbits], n_open


def same_orbit(
    area: float, centre: np.ndarray, other_area: float, other_centre: np.ndarray
) -> bool:
    """Whether two orbits are lattice translates of each other: fractional centres within
    1e-4 of a lattice vector and areas within a relative 1e-6."""
    shift = centre - other_centre
    return bool(
        abs(area - other_area) <= 1e-6 * max(area, other_area)
        and np.abs(shift - np.rint(shift)).max() < 1e-4
    )


def open_curves_note(n_open: int) -> str:
    return f" ({n_open} open curve{'s' if n_open > 1 else ''} not counted)" if n_open else ""


def area_text(areas: list[float], n_open: int, scale: float = 1.0) -> str:
    return f"Cross sectional area : {sum(areas) * scale:.4f} Ang^-2" + open_curves_note(n_open)


def clip_to_zone(surface: pv.PolyData, zone: pv.PolyData) -> pv.PolyData:
    """Cut ``surface`` down to the part inside every face plane of ``zone``."""
    for normal, center in zip(zone.face_normals, zone.centers, strict=True):
        surface = cast(pv.PolyData, surface.clip(origin=center, normal=normal, inplace=False))
        if surface.points.shape[0] == 0:
            break
    return surface


def normalize_to_range(scalars, clim=(0, 1)):
    if clim is None:
        clim = (0, 1)
    return (scalars - scalars.min()) / (scalars.max() - scalars.min()) * (clim[1] - clim[0]) + clim[
        0
    ]


class SurfacePlotter(pv.Plotter):
    """Renders one mesh per (band, spin) series, with optional scalars and vector glyphs.

    Subclasses set ``glyph_scale``, the length of the longest arrow unless a plot passes
    ``glyph_length``.
    """

    glyph_scale: float = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._meshes: list[pv.PolyData] = []
        self.values_dict: dict[str, np.ndarray] = {}

    def add_plane_widget(self, callback, *, origin=None, bounds=None, **kwargs):
        """Grow ``bounds`` to hold ``origin``; VTK's plane widget rejects an origin outside them."""
        if origin is not None and bounds is not None:
            low = np.minimum(np.asarray(bounds)[::2], origin)
            high = np.maximum(np.asarray(bounds)[1::2], origin)
            bounds = tuple(np.column_stack([low, high]).ravel().tolist())
        return super().add_plane_widget(callback, origin=origin, bounds=bounds, **kwargs)

    def _plot_series(
        self,
        series_list: list[SurfaceSeries],
        scalars_mode: str,
        show_scalar_bar: bool,
        scalars_cmap: str,
        scalars_clim: tuple[float, float] | None,
        add_surface_kwargs: dict | None,
        add_texture_kwargs: dict | None,
        clip_to: pv.PolyData | None = None,
        glyph_length: float | None = None,
    ) -> dict[tuple[int, int], pv.PolyData]:
        has_scalars = any(s.scalars is not None for s in series_list)
        if scalars_clim is None and has_scalars and scalars_mode != "none":
            scalars_clim = finite_range(s.scalars for s in series_list)

        vector_norms = [
            np.linalg.norm(s.vectors, axis=-1) for s in series_list if s.vectors is not None
        ]
        longest = finite_range(vector_norms)[1]

        meshes: dict[tuple[int, int], pv.PolyData] = {}
        for i, series in enumerate(series_list):
            key = (series.band_index, series.spin_index)
            mesh = series.mesh.copy()

            if scalars_mode != "none" and series.scalars is not None:
                mesh.point_data["scalars"] = series.scalars
                mesh.set_active_scalars("scalars")

            if series.vectors is not None:
                mesh.point_data["vectors"] = series.vectors

            if clip_to is not None:
                mesh = clip_to_zone(mesh, clip_to)

            if series.vectors is not None and "vectors" in mesh.point_data:
                mesh.set_active_vectors("vectors")

            mesh_kwargs: dict[str, Any] = {
                "cmap": scalars_cmap,
                "clim": scalars_clim,
                "show_scalar_bar": show_scalar_bar and i == 0,
                "name": f"surface_{series.band_index}_{series.spin_index}",
                **(add_surface_kwargs or {}),
                **series.kwargs,
            }
            if show_scalar_bar and series.scalars_label:
                mesh_kwargs.setdefault("scalar_bar_args", {})
                mesh_kwargs["scalar_bar_args"]["title"] = series.scalars_label

            self.add_mesh(mesh, **mesh_kwargs)
            self._meshes.append(mesh)

            if series.vectors is not None:
                texture_kwargs: dict[str, Any] = {
                    "cmap": scalars_cmap,
                    "clim": scalars_clim,
                    "longest": longest,
                    "length": glyph_length,
                    **(add_texture_kwargs or {}),
                }
                texture_kwargs["add_mesh_args"] = {
                    "name": f"vectors_{series.band_index}_{series.spin_index}",
                    **(texture_kwargs.get("add_mesh_args") or {}),
                }
                self.add_texture(mesh, **texture_kwargs)

            meshes[key] = mesh

            prefix = f"band_{series.band_index}_spin_{series.spin_index}"
            self.values_dict[f"{prefix}_points"] = mesh.points
            if series.scalars is not None:
                rendered = mesh.point_data.get("scalars")
                self.values_dict[f"{prefix}_scalars"] = (
                    series.scalars if rendered is None else np.asarray(rendered)
                )
            if series.vectors is not None:
                self.values_dict[f"{prefix}_vectors"] = mesh.point_data["vectors"]

        return meshes

    def add_brillouin_zone(
        self,
        brillouin_zone: pv.PolyData = None,
        style: str = "wireframe",
        line_width: float = 2.0,
        color: ColorLike = "black",
        opacity: float = 1.0,
    ):
        self.add_mesh(
            brillouin_zone,
            style=style,
            line_width=line_width,
            color=color,
            opacity=opacity,
        )

    def add_texture(
        self,
        surface: pv.PolyData,
        vectors: str | bool = True,
        factor: float = 1.0,
        add_mesh_args: dict | None = None,
        glyph_args: dict | None = None,
        longest: float | None = None,
        length: float | None = None,
        **kwargs,
    ):
        """Draw ``surface``'s active vectors as arrows; the ``longest`` vector gets ``length``.

        ``longest`` defaults to this surface's longest vector and ``length`` to ``glyph_scale``.
        """
        active_vectors = surface.active_vectors
        if active_vectors is None:
            return None
        finite = np.isfinite(active_vectors).all(axis=1) & np.isfinite(surface.points).all(axis=1)
        if not finite.any():
            return None
        source = surface
        if not finite.all():
            source = pv.PolyData(surface.points[finite])
            for name in surface.point_data:
                source.point_data[name] = surface.point_data[name][finite]
            source.set_active_vectors(surface.active_vectors_name)

        if add_mesh_args is None:
            add_mesh_args = {}

        add_mesh_args["name"] = add_mesh_args.get("name", "vectors")
        add_mesh_args["show_scalar_bar"] = add_mesh_args.get("show_scalar_bar", False)
        add_mesh_args["scalar_bar_args"] = add_mesh_args.get("scalar_bar_args", {})
        add_mesh_args["cmap"] = add_mesh_args.get("cmap", "plasma")
        add_mesh_args["clim"] = add_mesh_args.get("clim")
        add_mesh_args["color"] = add_mesh_args.get("color")
        add_mesh_args.update(kwargs)

        if glyph_args is None:
            glyph_args = {}
        glyph_args["color_mode"] = glyph_args.get("color_mode", "vector")
        glyph_args["scale"] = glyph_args.get("scale", surface.active_vectors_name)
        glyph_args["orient"] = glyph_args.get("orient", vectors)

        if longest is None:
            longest = float(np.linalg.norm(active_vectors[finite], axis=1).max())
        if length is None:
            length = self.glyph_scale
        factor = length / longest * factor

        glyph_args["factor"] = factor
        glyph_args["indices"] = glyph_args.get("indices")

        # glyph(scale=<name>) makes that array the active scalars of the mesh it runs on.
        scalars_name = surface.point_data.active_scalars_name
        vectors_name = surface.point_data.active_vectors_name
        arrows = source.glyph(**glyph_args)
        surface.point_data.active_scalars_name = scalars_name
        surface.point_data.active_vectors_name = vectors_name
        self.add_mesh(arrows, **add_mesh_args)
        return arrows

    def export_data(self, filename: str) -> None:
        """Export recorded plot data to file.

        Supports VTK (.vtk, .vtp), PLY (.ply), STL (.stl), and NumPy (.npz) formats.

        Parameters
        ----------
        filename : str
            Output file path. Extension determines format:
            - .vtk/.vtp: VTK format (preserves all point data)
            - .ply: PLY format (geometry only)
            - .stl: STL format (geometry only)
            - .npz: NumPy archive (all recorded arrays)
        """
        ext = os.path.splitext(filename)[1].lower()

        if ext in [".vtk", ".vtp", ".ply", ".stl"]:
            if self._meshes:
                combined = self._meshes[0]
                for mesh in self._meshes[1:]:
                    combined = combined.merge(mesh)
                combined.save(filename)
        elif ext == ".npz":
            if self.values_dict:
                np.savez(filename, **self.values_dict)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .vtk, .vtp, .ply, .stl, or .npz")

    def savefig(self, filename, camera_position=None, **kwargs):
        logger.info("Saving plot")

        if camera_position:
            self.camera_position = camera_position
        else:
            self.view_isometric()

        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in [".pdf", ".eps", ".ps", ".tex", ".svg"]:
            self.save_graphic(filename)
        else:
            self.screenshot(filename)

    def check_can_save_2d(self, save_2d) -> None:
        if save_2d and not self.off_screen:
            raise ValueError(
                "save_2d needs a plotter created with off_screen=True, because PyVista "
                + "cannot screenshot an on-screen plotter before show()"
            )

    def save_slice_2d(self, surface, normal, origin, filename, cmap="plasma", clim=None):
        """Draw the cross section of ``surface`` at ``normal``/``origin`` with matplotlib."""
        slice_plotter = FermiSlicePlotter(surface, normal=normal, origin=origin)
        vectors_name = surface.active_vectors_name
        slice_plotter.plot(
            scalars_name=surface.active_scalars_name,
            vectors_name=vectors_name,
            scalars_cmap=cmap,
            scalars_clim=clim,
            vectors_cmap=cmap,
            plot_arrows=vectors_name is not None,
        )
        slice_plotter.savefig(filename)
        plt.close(slice_plotter.fig)
        return slice_plotter
