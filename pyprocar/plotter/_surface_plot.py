"""PyVista plotter behaviour shared by FermiPlotter and BS2DPlotter."""

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


def slice_loop_areas(slc: pv.PolyData) -> tuple[list[float], int]:
    """Areas of the closed loops in a planar slice, and the number of open curves.

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
        return [], 0
    unique_points, merged = np.unique(np.round(points, 9), axis=0, return_inverse=True)
    lines = merged.reshape(-1)[lines]
    neighbours: dict[int, list[int]] = {}
    for a, b in lines[lines[:, 0] != lines[:, 1]].tolist():
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    areas: list[float] = []
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
        if any(len(neighbours[node]) != 2 for node in component):
            n_open += 1
            continue
        loop = [start, neighbours[start][0]]
        while len(loop) < len(component):
            a, b = neighbours[loop[-1]]
            loop.append(b if a == loop[-2] else a)
        corners = unique_points[loop]
        areas.append(
            0.5 * float(np.linalg.norm(np.cross(corners, np.roll(corners, -1, axis=0)).sum(axis=0)))
        )
    return areas, n_open


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
        longest = max((float(n.max()) for n in vector_norms if n.size), default=0.0)

        meshes: dict[tuple[int, int], pv.PolyData] = {}
        for i, series in enumerate(series_list):
            key = (series.band_index, series.spin_index)
            mesh = series.mesh.copy()

            if scalars_mode != "none" and series.scalars is not None:
                mesh.point_data["scalars"] = series.scalars
                mesh.set_active_scalars("scalars")

            if series.vectors is not None:
                mesh.point_data["vectors"] = series.vectors
                mesh.set_active_vectors("vectors")

            if clip_to is not None:
                mesh = clip_to_zone(mesh, clip_to)

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
            longest = float(np.linalg.norm(active_vectors, axis=1).max())
        if length is None:
            length = self.glyph_scale
        factor = length / longest * factor

        glyph_args["factor"] = factor
        glyph_args["indices"] = glyph_args.get("indices")

        # glyph(scale=<name>) makes that array the active scalars of the mesh it runs on.
        scalars_name = surface.point_data.active_scalars_name
        vectors_name = surface.point_data.active_vectors_name
        arrows = surface.glyph(**glyph_args)
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
