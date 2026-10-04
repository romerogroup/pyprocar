import logging
import warnings
from functools import partial
from typing import cast

import numpy as np
import pyvista as pv
import vtk
from pyvista.core.filters import _get_output
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    set_algorithm_input,
)
from scipy.constants import elementary_charge, hbar

from pyprocar.plotter._periodic_cut import PeriodicBand, periodic_bands, plane_orbits
from pyprocar.plotter._series import SurfaceSeries, surface_series
from pyprocar.plotter._surface_plot import (
    SurfacePlotter,
    area_text,
    find_nearest,
    normalize_to_range,
    open_curves_note,
    slice_loop_areas,
    snap_normal,
)

logger = logging.getLogger(__name__)


BZ_SCALE_FACTOR = 0.01

ARROW_ZONE_FRACTION = 0.1
"""Longest arrow in FermiPlotter.plot as a fraction of the Brillouin zone's largest extent."""

FS_AREA_SCALE_FACTOR = (2 * np.pi) ** 2


def dHvA_frequency(A_max_angstrom2):
    """Onsager frequency F = hbar A / (2 pi e), in gauss, of an orbit of area A in 1/Angstrom^2."""
    tesla = hbar * A_max_angstrom2 * 1e20 / (2 * np.pi * elementary_charge)
    return tesla * 1e4


def periodic_source(surface) -> tuple[np.ndarray, list[PeriodicBand]] | None:
    """The reciprocal lattice and one period of each band of a FermiSurface, else None."""
    lattice = getattr(surface, "reciprocal_lattice", None)
    bands = periodic_bands(surface) if lattice is not None else None
    return (np.asarray(lattice, dtype=np.float64), bands) if bands else None


class FermiPlotter(SurfacePlotter):
    glyph_scale = BZ_SCALE_FACTOR

    def _to_series_list(
        self,
        fermi_surface,
        scalars_data=None,
        vectors_data=None,
        **kwargs,
    ) -> list[SurfaceSeries]:
        """Convert FermiSurface and Properties to list of SurfaceSeries.

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
        list[SurfaceSeries]
            One SurfaceSeries per (band, spin) isosurface.
        """
        # Resolve scalars from name or Property
        if isinstance(scalars_data, str):
            scalars_data = fermi_surface.get_property(scalars_data)

        # Resolve vectors from name or Property
        if isinstance(vectors_data, str):
            vectors_data = fermi_surface.get_property(vectors_data)

        return surface_series(
            fermi_surface.band_isosurfaces,
            fermi_surface.band_spin_mask,
            scalars_data,
            vectors_data,
            kwargs=kwargs,
        )

    def plot(
        self,
        fermi_surface,
        scalars_data=None,
        vectors_data=None,
        scalars_mode: str = "surface",
        spins: list[int] | None = None,
        show_brillouin_zone: bool = True,
        show_scalar_bar: bool = True,
        scalars_cmap: str = "plasma",
        scalars_clim: tuple[float, float] | None = None,
        add_surface_kwargs: dict | None = None,
        add_texture_kwargs: dict | None = None,
        **kwargs,
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
        spins : list of int, optional
            Draw only the surfaces of these spin channels. None draws every surface.
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
        series_list = self._to_series_list(fermi_surface, scalars_data, vectors_data, **kwargs)
        if spins is not None:
            series_list = [s for s in series_list if s.spin_index in spins]
            if not series_list:
                warnings.warn(
                    f"No Fermi surface found: no band of spin channel(s) {list(spins)} crosses"
                    + " the isovalue (Fermi energy + fermi_shift). Try another spin channel,"
                    + " a different fermi_shift, or check the Fermi energy.",
                    UserWarning,
                    stacklevel=2,
                )

        if show_brillouin_zone:
            self.add_brillouin_zone(fermi_surface.brillouin_zone)

        return self._plot_series(
            series_list,
            scalars_mode,
            show_scalar_bar,
            scalars_cmap,
            scalars_clim,
            add_surface_kwargs,
            add_texture_kwargs,
            glyph_length=ARROW_ZONE_FRACTION
            * float(np.max(np.ptp(fermi_surface.brillouin_zone.points, axis=0))),
        )

    def add_surface(
        self,
        fermi_surface: pv.PolyData,
        normalize: bool = False,
        add_texture_args: dict = None,
        add_active_vectors: bool = False,
        show_scalar_bar: bool = True,
        add_mesh_args: dict = None,
        **kwargs,
    ):
        logger.info("____Adding Surface to Plotter____")

        if add_texture_args is None:
            add_texture_args = {}
        add_texture_args["name"] = add_texture_args.get("name", "vectors")

        if add_mesh_args is None:
            add_mesh_args = {}

        if show_scalar_bar and fermi_surface.active_scalars_name is not None:
            active_scalar_name = fermi_surface.active_scalars_name

            if active_scalar_name is None:
                raise ValueError(
                    "No active scalar found for the Fermi surface. "
                    "Use the compute* methods on the FermiSurface object to compute the scalar data."
                )

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

        if normalize:
            scalars = normalize_to_range(fermi_surface.active_scalars, clim=clim)
            add_mesh_args["scalars"] = scalars
        add_mesh_args["scalars"] = add_mesh_args.get("scalars")

        self.add_mesh(fermi_surface, **add_mesh_args)

        if add_active_vectors:
            # aligning the texture colors with the surface colors
            add_texture_args["cmap"] = add_texture_args.get("cmap", cmap)
            add_texture_args["clim"] = add_texture_args.get("clim", clim)

            self.add_texture(fermi_surface, **add_texture_args)

    def add_texture(self, fermi_surface: pv.PolyData, *args, **kwargs):
        return super().add_texture(fermi_surface, *args, **kwargs)

    def add_isoslider(
        self,
        e_surfaces,
        energy_values,
        add_slider_widget_args=None,
        add_surface_args=None,
        add_active_vectors=False,
        add_texture_args=None,
        **kwargs,
    ):
        if add_slider_widget_args is None:
            add_slider_widget_args = {}

        if add_surface_args is None:
            add_surface_args = {}
        add_surface_args.update(kwargs)

        if add_texture_args is None:
            add_texture_args = {}

        if add_slider_widget_args is None:
            add_slider_widget_args["title"] = "Energy"
            add_slider_widget_args["color"] = "black"
            add_slider_widget_args["style"] = "modern"

        energy_values = energy_values
        e_surfaces = e_surfaces

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
        value,
        energy_values=None,
        e_surfaces=None,
        add_active_vectors=False,
        add_texture_args=None,
        add_surface_args=None,
    ):
        if add_texture_args is None:
            add_texture_args = {}

        add_texture_args["name"] = "vectors"

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
        return None

    def add_slicer(
        self,
        surface,
        normal=(1, 0, 0),
        origin=(0, 0, 0),
        show_van_alphen_frequency=False,
        show_cross_section_area=False,
        add_surface_args=None,
        add_active_vectors=False,
        add_plane_widget_args=None,
    ):
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
                periodic=periodic_source(surface),
            ),
            normal=normal,
            origin=origin,
            **add_plane_widget_args,
        )

    def _slice_callback(
        self,
        normal,
        origin,
        mesh=None,
        add_surface_args=None,
        add_text_args=None,
        show_van_alphen_frequency=False,
        show_cross_section_area=False,
        periodic=None,
    ):
        if add_surface_args is None:
            add_surface_args = {}

        if mesh is None:
            mesh = self._meshes[0]

        add_text_args = add_text_args or {}

        if show_van_alphen_frequency and show_cross_section_area:
            raise ValueError(
                "show_van_alphen_frequency and show_cross_section_area cannot be True at the same time"
            )

        snapped = None
        if periodic is not None:
            normal, snapped = snap_normal(normal, periodic[0])
        slc = mesh.slice(normal=normal, origin=origin)
        active_vector_name = slc.active_vectors_name

        is_empty_slice = slc.n_points == 0
        if is_empty_slice:
            self.renderer.remove_actor("slice")
            self.renderer.remove_actor("slice_vectors")
        else:
            if active_vector_name:
                add_surface_args["add_active_vectors"] = add_surface_args.get(
                    "add_active_vectors", True
                )
                add_surface_args["add_texture_args"] = add_surface_args.get("add_texture_args", {})
                add_surface_args["add_texture_args"]["name"] = "slice_vectors"
                slc.set_active_vectors(active_vector_name)
            self.add_surface(cast(pv.PolyData, slc), name="slice", **add_surface_args)

        if show_van_alphen_frequency or show_cross_section_area:
            areas, n_open = (
                plane_orbits(periodic[1], periodic[0], normal, origin)
                if periodic is not None
                else slice_loop_areas(cast(pv.PolyData, slc))
            )
            if show_van_alphen_frequency:
                frequency = (
                    f"{dHvA_frequency(max(areas) * FS_AREA_SCALE_FACTOR):.4f} Gauss"
                    if areas
                    else "no closed orbit through this cut"
                )
                text = f"Van Alphen Frequency : {frequency}" + open_curves_note(n_open)
            else:
                text = area_text(areas, n_open, scale=FS_AREA_SCALE_FACTOR)
            if snapped is not None:
                text += " (normal snapped to [{} {} {}])".format(*snapped)
            self.add_text(text, name="area_text", **add_text_args)

        return slc

    def add_isovalue_gif(
        self,
        e_surfaces,
        save_gif,
        show_off_screen=True,
        iter_reverse=False,
        add_surface_args=None,
        add_active_vectors=False,
        add_texture_args=None,
        add_text_args=None,
    ):
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
        e_surfaces,
        reverse=False,
        text_name="energy_text",
        add_text_args=None,
        add_surface_args=None,
    ):
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
        surface,
        normal=(1, 0, 0),
        origin=(0, 0, 0),
        add_surface_args=None,
        add_active_vectors=False,
        add_plane_widget_args=None,
        add_text_args=None,
        show_van_alphen_frequency=False,
        show_cross_section_area=False,
        save_2d=None,
        save_2d_slice=None,
        **kwargs,
    ):
        self.check_can_save_2d(save_2d)
        if add_surface_args is None:
            add_surface_args = {}

        if add_plane_widget_args is None:
            add_plane_widget_args = {}

        add_surface_args["add_texture_args"] = add_surface_args.get("add_texture_args", {})
        add_surface_args["add_texture_args"]["name"] = "slice_vectors"

        add_surface_args["add_active_vectors"] = add_surface_args.get(
            "add_active_vectors", add_active_vectors
        )
        add_surface_args.update(kwargs)

        if add_text_args is None:
            add_text_args = {}

        add_text_args["color"] = add_text_args.get("color", "black")

        periodic = periodic_source(surface)
        mesh = pv.PolyData(surface)
        mesh, algo = algorithm_to_mesh_handler(
            add_ids_algorithm(mesh, point_ids=False, cell_ids=True)
        )

        clipper = vtk.vtkBoxClipDataSet()
        set_algorithm_input(clipper, algo)
        clipper.GenerateClippedOutputOn()

        # Initialize box widget

        self.add_box_widget(
            callback=partial(
                self._box_callback,
                clipper=clipper,
                port=0,
                add_surface_args=add_surface_args,
                add_text_args=add_text_args,
                show_van_alphen_frequency=show_van_alphen_frequency,
                show_cross_section_area=show_cross_section_area,
                periodic=periodic,
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
                periodic=periodic,
            ),
            normal=normal,
            origin=origin,
            bounds=surface.bounds,
            **add_plane_widget_args,
        )

        if save_2d:
            self.savefig(save_2d)
        if save_2d_slice:
            self.save_slice_2d(
                surface,
                normal,
                origin,
                save_2d_slice,
                cmap=add_surface_args.get("cmap", "plasma"),
                clim=add_surface_args.get("clim"),
            )

    def _box_callback(
        self,
        planes,
        clipper,
        port=0,
        add_surface_args=None,
        add_text_args=None,
        show_van_alphen_frequency=False,
        show_cross_section_area=False,
        periodic=None,
    ):
        bounds = []

        for i in range(planes.GetNumberOfPlanes()):
            plane = planes.GetPlane(i)
            bounds.append(plane.GetNormal())
            bounds.append(plane.GetOrigin())

        clipper.SetBoxClip(*bounds)
        clipper.Update()

        clipped = _get_output(clipper, oport=port)

        if len(self._meshes) == 0:
            self._meshes.append(clipped)
        else:
            self._meshes[0] = clipped

        # Update plane widget after updating box widget
        if self.plane_widgets:
            widget_origin = self.plane_widgets[0].GetOrigin()
            widget_normal = self.plane_widgets[0].GetNormal()
            self._slice_callback(
                normal=widget_normal,
                origin=widget_origin,
                mesh=self._meshes[0],
                add_surface_args=add_surface_args,
                add_text_args=add_text_args,
                show_van_alphen_frequency=show_van_alphen_frequency,
                show_cross_section_area=show_cross_section_area,
                periodic=periodic,
            )

    def set_scalar_bar_title(self, title: str, **kwargs) -> None:
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
