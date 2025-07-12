import copy
import logging
import os
from functools import partial
from typing import List, Tuple, Union

import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib import colors as mpcolors

from pyprocar.core import BandStructure2D

logger = logging.getLogger(__name__)

from pyvista import ColorLike
from pyvista.core.filters import _get_output
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    set_algorithm_input,
)

BZ_SCALE_FACTOR = 0.025


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def normalize_to_range(scalars, clim=(0, 1)):
    if clim is None:
        clim = (0, 1)
    return (scalars - scalars.min()) / (scalars.max() - scalars.min()) * (
        clim[1] - clim[0]
    ) + clim[0]


class BS2DPlotter(pv.Plotter):

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
        logger.info(f"____Adding Surface to Plotter____")

        if add_texture_args is None:
            add_texture_args = {}
        add_texture_args["name"] = add_texture_args.get("name", "vectors")

        if add_mesh_args is None:
            add_mesh_args = {}

        if show_scalar_bar:
            active_scalar_name = fermi_surface.active_scalars_name
            if "norm" in active_scalar_name:
                active_scalar_name = active_scalar_name.replace("-norm", "")
            add_mesh_args["show_scalar_bar"] = add_mesh_args.get(
                "show_scalar_bar", True
            )
            add_mesh_args["scalar_bar_args"] = add_mesh_args.get("scalar_bar_args", {})
            add_mesh_args["scalar_bar_args"]["title"] = add_mesh_args.get(
                "scalar_bar_args", {}
            ).get("title", active_scalar_name)

        add_mesh_args["cmap"] = add_mesh_args.get("cmap", "plasma")
        add_mesh_args["clim"] = add_mesh_args.get("clim", None)
        add_mesh_args["name"] = add_mesh_args.get("name", "surface")
        add_mesh_args.update(kwargs)

        clim = add_mesh_args.get("clim", None)
        cmap = add_mesh_args.get("cmap", "plasma")

        if normalize:
            scalars = normalize_to_range(fermi_surface.active_scalars, clim=clim)
            add_mesh_args["scalars"] = scalars
        add_mesh_args["scalars"] = add_mesh_args.get("scalars", None)

        self.add_mesh(fermi_surface, **add_mesh_args)

        if add_active_vectors:
            # aligning the texture colors with the surface colors
            add_texture_args["cmap"] = add_texture_args.get("cmap", cmap)
            add_texture_args["clim"] = add_texture_args.get("clim", clim)

            self.add_texture(fermi_surface, **add_texture_args)

    def add_texture(
        self,
        fermi_surface: pv.PolyData,
        vectors: Union[str, bool] = True,
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
        add_mesh_args["clim"] = add_mesh_args.get("clim", None)
        add_mesh_args["color"] = add_mesh_args.get("color", None)
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
        glyph_args["indices"] = glyph_args.get("indices", None)

        arrows = fermi_surface.glyph(**glyph_args)
        self.add_mesh(arrows, **add_mesh_args)
        return arrows

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
            add_surface_args["add_texture_args"] = add_surface_args.get(
                "add_texture_args", {}
            )
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
        self.cross_section_area = cross_section_area
        if add_surface_args is None:
            add_surface_args = {}

        if add_plane_widget_args is None:
            add_plane_widget_args = {}

        add_surface_args["add_texture_args"] = add_surface_args.get(
            "add_texture_args", {}
        )
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
                self._box_callback, port=0, add_surface_args=add_surface_args
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
                cross_section_area=self.cross_section_area,
            ),
            normal,
            origin,
            bounds=surface.bounds,
            **add_plane_widget_args,
        )

    def _box_callback(self, planes, port=0, add_surface_args=None):
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
                cross_section_area=self.cross_section_area,
                add_text_args=self.add_text_args,
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
