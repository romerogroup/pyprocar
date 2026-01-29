import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import pyvista as pv
import vtk
from pyvista import ColorLike
from pyvista.core.filters import _get_output
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    set_algorithm_input,
)

logger = logging.getLogger(__name__)


@dataclass
class FermiSeries:
    """Container for Fermi surface plotting data.

    Similar to BandSeries/Series in other plotters, but for 3D surface data.
    Each FermiSeries represents one (band, spin) combination's isosurface.
    """

    mesh: pv.PolyData  # Surface geometry
    scalars: np.ndarray | None
    scalars_label: str | None
    scalars_unit: str | None
    scalars_lim: tuple[float, float] | None
    vectors: np.ndarray | None
    vectors_label: str | None
    vectors_unit: str | None
    vectors_lim: tuple[float, float] | None
    label: str | None  # legend/annotation label
    band_index: int  # which band
    spin_index: int  # which spin channel
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


BZ_SCALE_FACTOR = 0.01

FS_AREA_SCALE_FACTOR = (2 * np.pi) ** 2


def find_nearest(array: np.ndarray, value: float) -> int:
    array = np.asarray(array)
    idx: int = int((np.abs(array - value)).argmin())
    return idx


def normalize_to_range(scalars: np.ndarray, clim: tuple[float, float] | None = None) -> np.ndarray:
    if clim is None:
        clim = (0, 1)
    return (scalars - scalars.min()) / (scalars.max() - scalars.min()) * (clim[1] - clim[0]) + clim[
        0
    ]


def dHvA_frequency(A_max_angstrom2: float) -> float:
    hbar = 1.0546e-27  # erg·s
    e = 4.768e-10  # statcoulombs
    c = 3.0e10  # cm/s
    A_max_cm2: float = A_max_angstrom2 * 1e16  # cm^-2
    F_max_theory: float = (hbar * A_max_cm2 * c) / (2 * np.pi * e)  # Gauss
    return F_max_theory


class FermiPlotter(pv.Plotter):
    off_screen: bool
    camera_position: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | str
        | Any
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._meshes: list[pv.PolyData] = []
        self.values_dict: dict[str, np.ndarray] = {}
        self._scalar_bar: Any = None
        self.clipper: Any = None
        self.add_text_args: dict[str, Any] = {}

    def _to_series_list(
        self,
        fermi_surface: Any,
        scalars_data: Any | None = None,
        vectors_data: Any | None = None,
        **kwargs: Any,
    ) -> list[FermiSeries]:
        """Convert FermiSurface and Properties to list of FermiSeries.

        Parameters
        ----------
        fermi_surface : FermiSurface
            The Fermi surface to extract series from.
        scalars_data : Property | str | None
            Property for scalar coloring, or name to look up in point_set.
        vectors_data : Property | str | None
            Property for vector arrows, or name to look up in point_set.
        **kwargs
            Additional kwargs to distribute to series.

        Returns
        -------
        list[FermiSeries]
            One FermiSeries per (band, spin) isosurface.
        """
        # Resolve scalars from name or Property
        if isinstance(scalars_data, str):
            scalars_data = fermi_surface.point_set.get_property(scalars_data)

        # Resolve vectors from name or Property
        if isinstance(vectors_data, str):
            vectors_data = fermi_surface.point_set.get_property(vectors_data)

        # Extract scalar/vector metadata
        s_label = scalars_data.label if scalars_data else None
        s_unit = scalars_data.units if scalars_data else None
        s_lims = scalars_data.data_lim if scalars_data else None

        v_label = vectors_data.label if vectors_data else None
        v_unit = vectors_data.units if vectors_data else None
        v_lims = vectors_data.data_lim if vectors_data else None

        # Build one series per band/spin isosurface
        series_list: list[FermiSeries] = []

        for (iband, ispin), isosurface in fermi_surface.band_isosurfaces.items():
            # Get mask for this band/spin in the combined surface
            mask = fermi_surface.band_spin_mask[(iband, ispin)]

            # Extract scalars for this surface
            s = None
            s_lim = None
            if scalars_data is not None:
                full_scalars = scalars_data.to_array()
                s = full_scalars[mask] if full_scalars is not None else None
                s_lim = s_lims

            # Extract vectors for this surface
            v = None
            v_lim = None
            if vectors_data is not None:
                full_vectors = vectors_data.to_array()
                v = full_vectors[mask] if full_vectors is not None else None
                v_lim = v_lims

            # Build label
            label = self._build_series_label(iband, ispin, len(fermi_surface.band_isosurfaces))

            series_list.append(
                FermiSeries(
                    mesh=isosurface,
                    scalars=s,
                    scalars_label=s_label,
                    scalars_unit=s_unit,
                    scalars_lim=s_lim,
                    vectors=v,
                    vectors_label=v_label,
                    vectors_unit=v_unit,
                    vectors_lim=v_lim,
                    label=label,
                    band_index=iband,
                    spin_index=ispin,
                    additional_kwargs=kwargs.copy(),
                )
            )

        return series_list

    def _build_series_label(self, iband: int, ispin: int, n_surfaces: int) -> str | None:
        """Build label for a single Fermi series."""
        if n_surfaces > 1:
            spin_label = "↑" if ispin == 0 else "↓"
            return f"Band {iband} {spin_label}"
        return None

    def plot(
        self,
        fermi_surface: Any,
        scalars_data: Any | None = None,
        vectors_data: Any | None = None,
        scalars_mode: str = "surface",
        show_brillouin_zone: bool = True,
        show_scalar_bar: bool = True,
        scalars_cmap: str = "plasma",
        scalars_clim: tuple[float, float] | None = None,
        add_surface_kwargs: dict[str, Any] | None = None,
        add_texture_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[tuple[int, int], pv.PolyData]:
        """Plot Fermi surface from FermiSurface object with Property-based coloring.

        This is the unified entry point for Fermi surface plotting,
        following the same pattern as DOSPlotter.plot() and BandStructurePlotter.plot().

        Parameters
        ----------
        fermi_surface : FermiSurface
            The Fermi surface to plot.
        scalars_data : Property | str | None
            Property for scalar coloring, or property name to look up.
            If None, surfaces are rendered with default coloring.
        vectors_data : Property | str | None
            Property for vector arrows (e.g., spin texture).
        scalars_mode : str
            How to render scalar data:
            - "surface": Color surface by scalar values
            - "none": Plain surface (ignore scalars_data)
        show_brillouin_zone : bool
            Whether to display the Brillouin zone boundary.
        show_scalar_bar : bool
            Whether to show the scalar bar (colorbar).
        scalars_cmap : str
            Colormap for scalar coloring.
        scalars_clim : tuple of float, optional
            Color limits for scalars. None = auto from data.
        add_surface_kwargs : dict, optional
            Additional kwargs for add_mesh().
        add_texture_kwargs : dict, optional
            Additional kwargs for vector glyphs.
        **kwargs
            Additional kwargs passed to all rendering methods.

        Returns
        -------
        dict[tuple[int, int], pv.PolyData]
            Dict mapping (band_index, spin_index) to rendered meshes.
        """
        add_surface_kwargs = add_surface_kwargs or {}
        add_texture_kwargs = add_texture_kwargs or {}

        # Convert to series list
        series_list = self._to_series_list(fermi_surface, scalars_data, vectors_data, **kwargs)

        # Resolve global clim if not provided
        if scalars_clim is None and scalars_data is not None and scalars_mode != "none":
            scalars_clim = self._resolve_clim(series_list)

        # Add Brillouin zone if requested
        if show_brillouin_zone:
            self.add_brillouin_zone(fermi_surface.brillouin_zone)

        # Plot each series
        meshes: dict[tuple[int, int], pv.PolyData] = {}
        band_keys = list(fermi_surface.band_isosurfaces.keys())

        for series in series_list:
            key = (series.band_index, series.spin_index)

            # Prepare mesh with scalars/vectors
            mesh = series.mesh.copy()

            if scalars_mode != "none" and series.scalars is not None:
                mesh.point_data["scalars"] = series.scalars
                mesh.set_active_scalars("scalars")

            if series.vectors is not None:
                mesh.point_data["vectors"] = series.vectors
                mesh.set_active_vectors("vectors")

            # Build add_mesh kwargs - only show scalar bar for first surface
            is_first_surface = key == band_keys[0]
            mesh_kwargs: dict[str, Any] = {
                "cmap": scalars_cmap,
                "clim": scalars_clim,
                "show_scalar_bar": show_scalar_bar and is_first_surface,
                "name": f"surface_{series.band_index}_{series.spin_index}",
                **add_surface_kwargs,
                **series.additional_kwargs,
            }

            if show_scalar_bar and series.scalars_label:
                if "scalar_bar_args" not in mesh_kwargs:
                    mesh_kwargs["scalar_bar_args"] = {}
                mesh_kwargs["scalar_bar_args"]["title"] = series.scalars_label

            self.add_mesh(mesh, **mesh_kwargs)
            self._meshes.append(mesh)

            # Add vectors if present
            if series.vectors is not None:
                texture_kwargs: dict[str, Any] = {
                    "cmap": scalars_cmap,
                    "clim": scalars_clim,
                    **add_texture_kwargs,
                }
                self.add_texture(mesh, **texture_kwargs)

            meshes[key] = mesh

            # Record for export
            self._record_series_data(series)

        return meshes

    def _resolve_clim(
        self,
        series_list: list[FermiSeries],
    ) -> tuple[float, float]:
        """Resolve color limits from series data."""
        all_scalars = [s.scalars for s in series_list if s.scalars is not None]
        if not all_scalars:
            return (0.0, 1.0)

        combined = np.concatenate([s.ravel() for s in all_scalars])
        finite = combined[np.isfinite(combined)]
        if len(finite) == 0:
            return (0.0, 1.0)

        return (float(finite.min()), float(finite.max()))

    def _record_series_data(self, series: FermiSeries) -> None:
        """Record series data for export."""
        key_prefix = f"band_{series.band_index}_spin_{series.spin_index}"
        self.values_dict[f"{key_prefix}_points"] = series.mesh.points
        if series.scalars is not None:
            self.values_dict[f"{key_prefix}_scalars"] = series.scalars
        if series.vectors is not None:
            self.values_dict[f"{key_prefix}_vectors"] = series.vectors

    def add_brillouin_zone(
        self,
        brillouin_zone: pv.PolyData | None = None,
        style: str = "wireframe",
        line_width: float = 2.0,
        color: ColorLike = "black",
        opacity: float = 1.0,
    ) -> None:
        self.add_mesh(
            brillouin_zone,
            style=style,
            line_width=line_width,
            color=color,
            opacity=opacity,
        )

    def add_surface(
        self,
        fermi_surface: pv.PolyData,
        normalize: bool = False,
        add_texture_args: dict[str, Any] | None = None,
        add_active_vectors: bool = False,
        show_scalar_bar: bool = True,
        add_mesh_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        logger.info("____Adding Surface to Plotter____")

        if add_texture_args is None:
            add_texture_args = {}
        add_texture_args["name"] = add_texture_args.get("name", "vectors")

        if add_mesh_args is None:
            add_mesh_args = {}

        if show_scalar_bar and fermi_surface.active_scalars_name is not None:
            active_scalar_name = fermi_surface.active_scalars_name

            if "norm" in active_scalar_name:
                active_scalar_name = active_scalar_name.replace("-norm", "")
            add_mesh_args["show_scalar_bar"] = add_mesh_args.get("show_scalar_bar", True)
            add_mesh_args["scalar_bar_args"] = add_mesh_args.get("scalar_bar_args", {})
            add_mesh_args["scalar_bar_args"]["title"] = add_mesh_args.get(
                "scalar_bar_args", {}
            ).get("title", active_scalar_name)

        add_mesh_args["cmap"] = add_mesh_args.get("cmap", "plasma")
        add_mesh_args["clim"] = add_mesh_args.get("clim")
        add_mesh_args["name"] = add_mesh_args.get("name", "surface")
        add_mesh_args.update(kwargs)

        clim = add_mesh_args.get("clim")
        cmap = add_mesh_args.get("cmap", "plasma")

        if normalize and fermi_surface.active_scalars is not None:
            scalars = normalize_to_range(fermi_surface.active_scalars, clim=clim)
            add_mesh_args["scalars"] = scalars
        add_mesh_args["scalars"] = add_mesh_args.get("scalars")

        self.add_mesh(fermi_surface, **add_mesh_args)

        if add_active_vectors:
            # aligning the texture colors with the surface colors
            add_texture_args["cmap"] = add_texture_args.get("cmap", cmap)
            add_texture_args["clim"] = add_texture_args.get("clim", clim)

            self.add_texture(fermi_surface, **add_texture_args)

    def add_texture(
        self,
        fermi_surface: pv.PolyData,
        vectors: str | bool = True,
        factor: float = 1.0,
        add_mesh_args: dict[str, Any] | None = None,
        glyph_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        active_vectors = fermi_surface.active_vectors
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
        glyph_args["scale"] = glyph_args.get("scale", True)
        glyph_args["orient"] = glyph_args.get("orient", vectors)

        active_vector_magnitude: np.ndarray = np.linalg.norm(active_vectors, axis=1)
        vector_scale_factor: float = 1 / active_vector_magnitude.max()
        factor = vector_scale_factor * BZ_SCALE_FACTOR * factor

        glyph_args["factor"] = factor
        glyph_args["indices"] = glyph_args.get("indices")

        arrows = fermi_surface.glyph(**glyph_args)
        self.add_mesh(arrows, **add_mesh_args)
        return arrows

    def add_isoslider(
        self,
        e_surfaces: Any,
        energy_values: np.ndarray,
        add_slider_widget_args: dict[str, Any] | None = None,
        add_surface_args: dict[str, Any] | None = None,
        add_active_vectors: bool = False,
        add_texture_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if add_slider_widget_args is None:
            add_slider_widget_args = {
                "title": "Energy",
                "color": "black",
                "style": "modern",
            }

        if add_surface_args is None:
            add_surface_args = {}
        add_surface_args.update(kwargs)

        if add_texture_args is None:
            add_texture_args = {}

        self.add_slider_widget(
            partial(
                self._isoslider_callback,
                energy_values=energy_values,
                e_surfaces=e_surfaces,
                add_active_vectors=add_active_vectors,
                add_texture_args=add_texture_args,
                add_surface_args=add_surface_args,
            ),
            [np.amin(energy_values), np.amax(energy_values)],
            **add_slider_widget_args,
        )

    def _isoslider_callback(
        self,
        value: float,
        energy_values: np.ndarray | None = None,
        e_surfaces: Any | None = None,
        add_active_vectors: bool = False,
        add_texture_args: dict[str, Any] | None = None,
        add_surface_args: dict[str, Any] | None = None,
    ) -> None:
        if add_texture_args is None:
            add_texture_args = {}
        if add_surface_args is None:
            add_surface_args = {}

        add_texture_args["name"] = "vectors"

        if energy_values is None or e_surfaces is None:
            return

        res = float(value)
        closest_idx = find_nearest(energy_values, res)
        surface = e_surfaces[closest_idx]
        self.add_surface(
            surface,
            name="isosurface",
            add_active_vectors=add_active_vectors,
            add_texture_args=add_texture_args,
            **add_surface_args,
        )
        return

    def add_slicer(
        self,
        surface: pv.PolyData,
        normal: tuple[float, float, float] = (1, 0, 0),
        origin: tuple[float, float, float] = (0, 0, 0),
        show_van_alphen_frequency: bool = False,
        show_cross_section_area: bool = False,
        add_surface_args: dict[str, Any] | None = None,
        add_active_vectors: bool = False,
        add_plane_widget_args: dict[str, Any] | None = None,
    ) -> None:
        if add_surface_args is None:
            add_surface_args = {}
        if add_plane_widget_args is None:
            add_plane_widget_args = {}
        add_plane_widget_args["bounds"] = surface.bounds

        add_surface_args["add_active_vectors"] = add_surface_args.get(
            "add_active_vectors", add_active_vectors
        )

        self.add_plane_widget(
            partial(
                self._slice_callback,
                mesh=surface,
                add_surface_args=add_surface_args,
                show_van_alphen_frequency=show_van_alphen_frequency,
                show_cross_section_area=show_cross_section_area,
            ),
            normal,
            origin,
            **add_plane_widget_args,
        )

    def _slice_callback(
        self,
        normal: Any,
        origin: Any,
        mesh: pv.PolyData | None = None,
        add_surface_args: dict[str, Any] | None = None,
        add_text_args: dict[str, Any] | None = None,
        show_van_alphen_frequency: bool = False,
        show_cross_section_area: bool = False,
    ) -> Any:
        if add_surface_args is None:
            add_surface_args = {}

        if mesh is None:
            if not self._meshes:
                return None
            mesh = self._meshes[0]

        add_text_args = add_text_args or {}

        slc = mesh.slice(normal=normal, origin=origin)
        active_vector_name = slc.active_vectors_name

        is_empty_slice = slc.n_points == 0
        if is_empty_slice:
            return None

        if active_vector_name:
            add_surface_args["add_active_vectors"] = add_surface_args.get(
                "add_active_vectors", True
            )
            add_surface_args["add_texture_args"] = add_surface_args.get("add_texture_args", {})
            add_surface_args["add_texture_args"]["name"] = "vectors"
            slc.set_active_vectors(active_vector_name)

        self.add_surface(slc, name="slice", **add_surface_args)

        if show_van_alphen_frequency and show_cross_section_area:
            raise ValueError(
                "show_van_alphen_frequency and show_cross_section_area cannot be True at the same time"
            )

        if show_van_alphen_frequency:
            surface = slc.delaunay_2d()
            text = (
                f"Van Alphen Frequency : {dHvA_frequency(surface.area * FS_AREA_SCALE_FACTOR):.4f}"
                 " Gauss"
            )
            self.add_text(text, name="area_text", **add_text_args)
        elif show_cross_section_area:
            surface = slc.delaunay_2d()
            text = f"Cross sectional area : {surface.area * FS_AREA_SCALE_FACTOR:.4f}" + " Ang^-2"
            self.add_text(text, name="area_text", **add_text_args)

        return slc

    def add_isovalue_gif(
        self,
        e_surfaces: Any,
        save_gif: str,
        show_off_screen: bool = True,
        iter_reverse: bool = False,
        add_surface_args: dict[str, Any] | None = None,
        add_active_vectors: bool = False,
        add_texture_args: dict[str, Any] | None = None,
        add_text_args: dict[str, Any] | None = None,
    ) -> None:
        if add_surface_args is None:
            add_surface_args = {}
        add_surface_args["add_active_vectors"] = add_surface_args.get(
            "add_active_vectors", add_active_vectors
        )
        add_surface_args["add_texture_args"] = add_surface_args.get(
            "add_texture_args", add_texture_args
        )
        add_surface_args["name"] = add_surface_args.get("name", "surface")
        if add_text_args is None:
            add_text_args = {}
        add_text_args["color"] = add_text_args.get("color", "black")

        self.off_screen = show_off_screen
        self.open_gif(save_gif)

        self._iter_surfaces(
            e_surfaces,
            add_text_args=add_text_args,
            add_surface_args=add_surface_args,
        )

        if iter_reverse:
            self._iter_surfaces(
                e_surfaces,
                reverse=True,
                add_text_args=add_text_args,
                add_surface_args=add_surface_args,
            )

        if show_off_screen:
            self.close()

    def _iter_surfaces(
        self,
        e_surfaces: Any,
        reverse: bool = False,
        text_name: str = "energy_text",
        add_text_args: dict[str, Any] | None = None,
        add_surface_args: dict[str, Any] | None = None,
    ) -> None:
        if add_surface_args is None:
            add_surface_args = {}
        if add_text_args is None:
            add_text_args = {}

        if reverse:
            e_surfaces = e_surfaces[::-1]

        for e_surface in e_surfaces:
            energy = e_surface.fermi
            self.add_surface(e_surface, **add_surface_args)
            text = f"Energy Value : {energy:.4f} eV"
            self.add_text(text, name=text_name, **add_text_args)
            self.write_frame()

    def add_box_slicer(
        self,
        surface: pv.PolyData,
        normal: tuple[float, float, float] = (1, 0, 0),
        origin: tuple[float, float, float] = (0, 0, 0),
        add_surface_args: dict[str, Any] | None = None,
        add_active_vectors: bool = False,
        add_plane_widget_args: dict[str, Any] | None = None,
        add_text_args: dict[str, Any] | None = None,
        show_van_alphen_frequency: bool = False,
        show_cross_section_area: bool = False,
        _save_2d: str | None = None,
        _save_2d_slice: str | None = None,
        **kwargs: Any,
    ) -> None:
        if add_surface_args is None:
            add_surface_args = {}

        if add_plane_widget_args is None:
            add_plane_widget_args = {}

        add_surface_args["add_texture_args"] = add_surface_args.get("add_texture_args", {})
        add_surface_args["add_texture_args"]["name"] = "vectors"

        add_surface_args["add_active_vectors"] = add_surface_args.get(
            "add_active_vectors", add_active_vectors
        )
        add_surface_args.update(kwargs)

        if add_text_args is None:
            add_text_args = {}

        add_text_args["color"] = add_text_args.get("color", "black")

        self.add_text_args = add_text_args

        # Initialize clipper for surface
        mesh = pv.PolyData(surface)
        mesh, algo = algorithm_to_mesh_handler(
            add_ids_algorithm(mesh, point_ids=False, cell_ids=True)
        )

        self.clipper = vtk.vtkBoxClipDataSet()
        set_algorithm_input(self.clipper, algo)
        self.clipper.GenerateClippedOutputOn()

        # Initialize box widget

        self.add_box_widget(
            callback=partial(
                self._box_callback,
                port=0,
                add_surface_args=add_surface_args,
                show_van_alphen_frequency=show_van_alphen_frequency,
                show_cross_section_area=show_cross_section_area,
            ),
            bounds=surface.bounds,
            use_planes=True,
            interaction_event="end",
        )

        # Initialize plane widget. If mesh is not it uses self._meshes[0]
        self.add_plane_widget(
            partial(
                self._slice_callback,
                add_surface_args=add_surface_args,
                add_text_args=add_text_args,
                show_van_alphen_frequency=show_van_alphen_frequency,
                show_cross_section_area=show_cross_section_area,
            ),
            normal,
            origin,
            bounds=surface.bounds,
            **add_plane_widget_args,
        )

    def _box_callback(
        self,
        planes: Any,
        port: int = 0,
        add_surface_args: dict[str, Any] | None = None,
        show_van_alphen_frequency: bool = False,
        show_cross_section_area: bool = False,
    ) -> None:
        clip_bounds: list[Any] = []

        for i in range(planes.GetNumberOfPlanes()):
            plane = planes.GetPlane(i)
            clip_bounds.append(plane.GetNormal())
            clip_bounds.append(plane.GetOrigin())

        self.clipper.SetBoxClip(*clip_bounds)
        self.clipper.Update()

        clipped = _get_output(self.clipper, oport=port)
        clipped_poly = clipped if isinstance(clipped, pv.PolyData) else pv.PolyData(clipped)

        if len(self._meshes) == 0:
            self._meshes.append(clipped_poly)
        else:
            self._meshes[0] = clipped_poly

        # Update plane widget after updating box widget
        if self.plane_widgets:
            widget_origin = self.plane_widgets[0].GetOrigin()
            widget_normal = self.plane_widgets[0].GetNormal()
            self._slice_callback(
                normal=widget_normal,
                origin=widget_origin,
                mesh=self._meshes[0],
                add_surface_args=add_surface_args,
                add_text_args=self.add_text_args,
                show_van_alphen_frequency=show_van_alphen_frequency,
                show_cross_section_area=show_cross_section_area,
            )

    def set_scalar_bar_title(self, title: str, **_kwargs: Any) -> None:
        """Set the scalar bar title."""
        if hasattr(self, "_scalar_bar") and self._scalar_bar is not None:
            self._scalar_bar.SetTitle(title)

    def set_scalar_bar_label_font_size(self, size: int) -> None:
        """Set scalar bar label font size."""
        if hasattr(self, "_scalar_bar") and self._scalar_bar is not None:
            self._scalar_bar.GetLabelTextProperty().SetFontSize(size)

    def set_scalar_bar_position(self, position: tuple[float, float]) -> None:
        """Set scalar bar position (x, y) in normalized coordinates."""
        if hasattr(self, "_scalar_bar") and self._scalar_bar is not None:
            self._scalar_bar.SetPosition(position)

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

        if ext in [".vtk", ".vtp"]:
            # Export as VTK - merge all meshes
            if self._meshes:
                combined = self._meshes[0]
                for mesh in self._meshes[1:]:
                    combined = combined.merge(mesh)
                combined.save(filename)
        elif ext == ".ply" or ext == ".stl":
            if self._meshes:
                combined = self._meshes[0]
                for mesh in self._meshes[1:]:
                    combined = combined.merge(mesh)
                combined.save(filename)
        elif ext == ".npz":
            if self.values_dict:
                np.savez(filename, **self.values_dict)  # pyright: ignore[reportArgumentType] - numpy savez stub limitation with dict expansion
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .vtk, .vtp, .ply, .stl, or .npz")

    def savefig(self, filename: str, camera_position: Any | None = None, **_kwargs: Any) -> None:
        logger.info("Saving plot")

        if camera_position:
            self.camera_position = camera_position
        else:
            self.view_isometric()

        # Get the file extension
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in [".pdf", ".eps", ".ps", ".tex", ".svg"]:
            self.save_graphic(filename)
        else:
            self.screenshot(filename)
