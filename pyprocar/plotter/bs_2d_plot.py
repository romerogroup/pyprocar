import logging
from functools import partial

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

from pyprocar.plotter._series import SurfaceSeries, surface_series
from pyprocar.plotter._surface_plot import SurfacePlotter, clip_to_zone, normalize_to_range

logger = logging.getLogger(__name__)

# BZ_SCALE_FACTOR = 0.025
BZ_SCALE_FACTOR = 1


def get_uv_bands_grid(
    grid_interpolation: tuple[int, int],
    u_limits: tuple[float, float],
    v_limits: tuple[float, float],
):
    grid_u, grid_v = np.mgrid[
        u_limits[0] : u_limits[1] : complex(0, grid_interpolation[0]),
        v_limits[0] : v_limits[1] : complex(0, grid_interpolation[1]),
    ]
    return grid_u, grid_v


class BS2DPlotter(SurfacePlotter):
    glyph_scale = BZ_SCALE_FACTOR

    def __init__(self, bandstructure2d, **kwargs):
        super().__init__(**kwargs)
        self.bs2d = bandstructure2d
        self._brillouin_zone: pv.PolyData | None = None

    def _to_series_list(
        self,
        bandstructure2d,
        scalars_data=None,
        vectors_data=None,
        **kwargs,
    ) -> list[SurfaceSeries]:
        """Convert BandStructure2D and Properties to list of SurfaceSeries.

        Parameters
        ----------
        bandstructure2d : BandStructure2D
            The 2D band structure surface to extract series from.
        scalars_data : Property | str | None
            Property for scalar coloring, or name to look up in point_set.
        vectors_data : Property | str | None
            Property for vector arrows, or name to look up in point_set.
        **kwargs
            Additional kwargs to distribute to series.

        Returns
        -------
        list[SurfaceSeries]
            One SurfaceSeries per (band, spin) surface.
        """
        # Resolve scalars from name or Property
        if isinstance(scalars_data, str):
            scalars_data = bandstructure2d.get_property(scalars_data)

        # Resolve vectors from name or Property
        if isinstance(vectors_data, str):
            vectors_data = bandstructure2d.get_property(vectors_data)

        return surface_series(
            bandstructure2d.band_surfaces,
            bandstructure2d.band_spin_mask,
            scalars_data,
            vectors_data,
            kwargs=kwargs,
        )

    def plot(
        self,
        bandstructure2d=None,
        scalars_data=None,
        vectors_data=None,
        scalars_mode: str = "surface",
        show_brillouin_zone: bool = True,
        clip_brillouin_zone: bool = False,
        show_scalar_bar: bool = True,
        scalars_cmap: str = "plasma",
        scalars_clim: tuple[float, float] | None = None,
        add_surface_kwargs: dict | None = None,
        add_texture_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[tuple[int, int], pv.PolyData]:
        """Plot 2D band structure from BandStructure2D with Property-based coloring.

        This is the unified entry point for 2D band structure plotting,
        following the same pattern as FermiPlotter.plot().

        Parameters
        ----------
        bandstructure2d : BandStructure2D | None
            The 2D band structure to plot. If None, uses self.bs2d.
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
            Whether to display the 2D Brillouin zone boundary.
        clip_brillouin_zone : bool
            Whether to cut each surface to the part inside the Brillouin zone.
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
        bs2d = bandstructure2d if bandstructure2d is not None else self.bs2d

        series_list = self._to_series_list(bs2d, scalars_data, vectors_data, **kwargs)

        bz = None
        if (show_brillouin_zone or clip_brillouin_zone) and hasattr(bs2d, "get_2d_brillouin_zone"):
            z_coords = bs2d.points[:, 2]
            e_min, e_max = float(np.nanmin(z_coords)), float(np.nanmax(z_coords))
            bz = bs2d.get_2d_brillouin_zone(e_min=e_min, e_max=e_max)
        if show_brillouin_zone and bz is not None:
            self.add_brillouin_zone(bz)

        return self._plot_series(
            series_list,
            scalars_mode,
            show_scalar_bar,
            scalars_cmap,
            scalars_clim,
            add_surface_kwargs,
            add_texture_kwargs,
            clip_to=bz if clip_brillouin_zone else None,
        )

    def add_brillouin_zone(
        self,
        brillouin_zone: pv.PolyData = None,
        style: str = "wireframe",
        line_width: float = 2.0,
        color: ColorLike = "black",
        opacity: float = 1.0,
    ):
        self._brillouin_zone = brillouin_zone
        super().add_brillouin_zone(brillouin_zone, style, line_width, color, opacity)

    def add_surface(
        self,
        surface: pv.PolyData,
        normalize: bool = False,
        clip_surface: bool = False,
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

        if show_scalar_bar:
            active_scalar_name = surface.active_scalars_name
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
            scalars = normalize_to_range(surface.active_scalars, clim=clim)
            add_mesh_args["scalars"] = scalars
        add_mesh_args["scalars"] = add_mesh_args.get("scalars")

        if clip_surface and self._brillouin_zone is not None:
            surface = self.clip_surface(surface, self._brillouin_zone)

        self.add_mesh(surface, **add_mesh_args)

        if add_active_vectors:
            # aligning the texture colors with the surface colors
            add_texture_args["cmap"] = add_texture_args.get("cmap", cmap)
            add_texture_args["clim"] = add_texture_args.get("clim", clim)

            self.add_texture(surface, **add_texture_args)

    def clip_surface(self, surface: pv.PolyData, brillouin_zone: pv.PolyData):
        return clip_to_zone(surface, brillouin_zone)

    def add_slicer(
        self,
        surface,
        normal=(1, 0, 0),
        origin=(0, 0, 0),
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
        cross_section_area=False,
    ):
        if add_surface_args is None:
            add_surface_args = {}

        if mesh is None:
            mesh = self._meshes[0]

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

        if cross_section_area:
            surface = slc.delaunay_2d()
            text = f"Cross sectional area : {surface.area:.4f}" + " Ang^-2"
            self.add_text(text, name="area_text", **add_text_args)

        return slc

    def add_box_slicer(
        self,
        surface,
        normal=(1, 0, 0),
        origin=(0, 0, 0),
        add_surface_args=None,
        add_active_vectors=False,
        add_plane_widget_args=None,
        add_text_args=None,
        cross_section_area=False,
        save_2d=None,
        save_2d_slice=None,
        **kwargs,
    ):
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

        # Initialize clipper for surface
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
                cross_section_area=cross_section_area,
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
                cross_section_area=cross_section_area,
            ),
            normal=normal,
            origin=origin,
            bounds=surface.bounds,
            **add_plane_widget_args,
        )

    def _box_callback(
        self,
        planes,
        clipper,
        port=0,
        add_surface_args=None,
        add_text_args=None,
        cross_section_area=False,
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
                cross_section_area=cross_section_area,
                add_text_args=add_text_args,
            )
