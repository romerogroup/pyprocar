import logging
import os
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

logger = logging.getLogger(__name__)

BZ_SCALE_FACTOR = 0.01

FS_AREA_SCALE_FACTOR = (2 * np.pi) ** 2


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def normalize_to_range(scalars, clim=(0, 1)):
    if clim is None:
        clim = (0, 1)
    return (scalars - scalars.min()) / (scalars.max() - scalars.min()) * (clim[1] - clim[0]) + clim[
        0
    ]


def dHvA_frequency(A_max_angstrom2):
    hbar = 1.0546e-27  # erg·s
    e = 4.768e-10  # statcoulombs
    c = 3.0e10  # cm/s
    A_max_cm2 = A_max_angstrom2 * 1e16  # cm^-2
    F_max_theory = (hbar * A_max_cm2 * c) / (2 * np.pi * e)  # Gauss
    return F_max_theory


class FermiPlotter(pv.Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._meshes = []

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

    def add_texture(
        self,
        fermi_surface: pv.PolyData,
        vectors: str | bool = True,
        factor: float = 1.0,
        add_mesh_args: dict = None,
        glyph_args: dict = None,
        **kwargs,
    ):
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

        active_vector_magnitude = np.linalg.norm(fermi_surface.active_vectors, axis=1)
        vector_scale_factor = 1 / active_vector_magnitude.max()
        factor = vector_scale_factor * BZ_SCALE_FACTOR * factor

        glyph_args["factor"] = factor
        glyph_args["indices"] = glyph_args.get("indices")

        arrows = fermi_surface.glyph(**glyph_args)
        self.add_mesh(arrows, **add_mesh_args)
        return arrows

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
            ),
            normal,
            origin,
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
    ):
        if add_surface_args is None:
            add_surface_args = {}

        if mesh is None:
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
                + " Gauss"
            )
            self.add_text(text, name="area_text", **add_text_args)
        elif show_cross_section_area:
            surface = slc.delaunay_2d()
            text = f"Cross sectional area : {surface.area * FS_AREA_SCALE_FACTOR:.4f}" + " Ang^-2"
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
        if add_surface_args is None:
            add_surface_args = {}

        if add_plane_widget_args is None:
            add_plane_widget_args = {}

        origin = np.array(origin)
        normal = np.array(normal)

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
        planes,
        port=0,
        add_surface_args=None,
        show_van_alphen_frequency=False,
        show_cross_section_area=False,
    ):
        bounds = []

        for i in range(planes.GetNumberOfPlanes()):
            plane = planes.GetPlane(i)
            bounds.append(plane.GetNormal())
            bounds.append(plane.GetOrigin())

        self.clipper.SetBoxClip(*bounds)
        self.clipper.Update()

        clipped = _get_output(self.clipper, oport=port)

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
                add_text_args=self.add_text_args,
                show_van_alphen_frequency=show_van_alphen_frequency,
                show_cross_section_area=show_cross_section_area,
            )

    def savefig(self, filename, camera_position=None, **kwargs):
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
