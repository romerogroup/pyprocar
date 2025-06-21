import copy
import logging
import os
from collections.abc import Sequence
from functools import partial
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib import cm
from matplotlib import colors as mpcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from PIL import Image
from pyvista import ColorLike
from pyvista.core.filters import _get_output
from pyvista.core.utilities import (
    NORMALS,
    assert_empty_kwargs,
    generate_plane,
    get_array,
    get_array_association,
    try_callback,
)
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    crinkle_algorithm,
    outline_algorithm,
    pointset_to_polydata_algorithm,
    set_algorithm_input,
)
from scipy.interpolate import griddata

from pyprocar.core.fermisurface3Dnew import FermiSurface
from pyprocar.utils import ROOT

logger = logging.getLogger(__name__)


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


class FermiSlicePlotter:
    """
    A plotter class for 2D Fermi surface slices using matplotlib.

    This class takes sliced data from a 3D FermiSurface object and plots it
    in 2D using matplotlib, similar to the functionality in fermisurface.py
    but adapted for PyVista slice objects.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (8, 6)
    dpi : int, optional
        Figure resolution in dots per inch, by default 100
    """

    def __init__(self, figsize=(8, 6), dpi=100, ax=None):
        self.figsize = figsize
        self.dpi = dpi

        self.ax = ax
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)

        self.slice_data = None
        self.points = None
        self.lines = None
        self.scalars = None
        self.vectors = None

        self.set_plot_settings()

    def set_plot_settings(self, origin: List[float] = [0, 0, 0], **kwargs):
        self.ax.set_xlabel("$k_x$")
        self.ax.set_ylabel("$k_y$")
        self.ax.set_title(f"Fermi Surface Slice at $k_z$ = {origin[2]}")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle="--", alpha=0.6)

    def _prepare_slice_data(
        self, fermi_surface: pv.PolyData, normal: List[float], origin: List[float]
    ):
        """Slices the Fermi surface and stores the resulting data."""
        self.slice_data = fermi_surface.slice(normal=normal, origin=origin)
        print(self.slice_data.point_data)
        self.points = self.slice_data.points
        self.lines = self.slice_data.lines

        scalars = self.slice_data.active_scalars
        if scalars.shape[-1] == 3:
            self.scalars = np.linalg.norm(scalars, axis=1)
            self.vectors = scalars
        else:
            self.scalars = scalars
            self.vectors = None

        return self.lines, self.points, self.scalars, self.vectors

    def _iter_segments(self, lines: List[int]):
        """
        A generator that yields the start and end indices of each line segment.

        This is the core of the loop abstraction. It handles the parsing of the
        PyVista `lines` array, correctly processing polylines.
        """
        if lines is None:
            raise ValueError(
                "Slice data is not prepared. Call a plotting method first."
            )

        i = 0
        while i < len(lines):
            num_points_in_line = lines[i]
            # The indices for points in this line start at the next position
            line_connectivity_start = i + 1

            # Iterate through the pairs of points that form the line's segments
            for j in range(num_points_in_line - 1):
                start_idx = lines[line_connectivity_start + j]
                end_idx = lines[line_connectivity_start + j + 1]
                yield start_idx, end_idx

            # Move the main index to the beginning of the next line definition
            i += num_points_in_line + 1

    def plot_lines(
        self,
        lines: List[int],
        points: List[List[float]],
        scalars: List[float] = None,
        cmap: str = "plasma",
        **kwargs,
    ):
        """
        Plots the line segments of the slice.

        This method uses the `_iter_segments` generator and is focused only on
        drawing the lines. It uses LineCollection for high performance.

        kwargs are passed to the matplotlib LineCollection.
        """
        if points is None or scalars is None:
            raise ValueError("Slice data is not available for plotting lines.")

        line_segments = []
        colors = []

        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=scalars.min(), vmax=scalars.max())
        # Use the generator to iterate
        for start_idx, end_idx in self._iter_segments(lines):
            # Create the line segment
            p1 = points[start_idx]
            p2 = points[end_idx]
            line_segments.append([(p1[0], p1[1]), (p2[0], p2[1])])

            # Determine color from scalar data
            if scalars is not None:
                avg_scalar = (scalars[start_idx] + scalars[end_idx]) / 2.0
                colors.append(cmap(norm(avg_scalar)))

        if len(colors) == 0:
            colors = None

        # Use a LineCollection for much better performance than plotting one by one
        lc = LineCollection(line_segments, colors=colors, **kwargs)
        self.ax.add_collection(lc)
        self.ax.autoscale_view()  # Important after adding a collection

    def plot_points(
        self,
        points: List[List[float]],
        scalars: List[float] = None,
        cmap: str = "plasma",
        **kwargs,
    ):
        if points is None:
            raise ValueError("Slice data is not available for plotting lines.")

        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=scalars.min(), vmax=scalars.max())

        self.ax.scatter(
            points[:, 0], points[:, 1], c=scalars, cmap=cmap, norm=norm, **kwargs
        )

    def plot_arrows(
        self,
        lines: List[int],
        points: List[List[float]],
        vectors: List[List[float]],
        factor: float = 1.0,
        cmap: str = "plasma",
        **arrow_kwargs,
    ):
        """
        Plots vectors as arrows on the slice.

        This method also uses the `_iter_segments` generator and is focused
        only on drawing arrows.

        Parameters
        ----------
        factor : float, optional
            A scaling factor for the length of the arrows.
        **arrow_kwargs
            Additional keyword arguments passed to `ax.arrow`.
        """
        if points is None or vectors is None:
            raise ValueError("Vector data is not available for plotting arrows.")

        # Use the generator to iterate
        vector_magnitude = np.linalg.norm(vectors, axis=1)
        vector_scale_factor = 1 / vector_magnitude.max()
        # factor = vector_scale_factor * BZ_SCALE_FACTOR * factor
        factor = vector_scale_factor * 0.01 * factor

        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vector_magnitude.min(), vmax=vector_magnitude.max())
        plt.quiver(
            points[:, 0],  # Arrow position x-component
            points[:, 1],  # Arrow position y-component
            vectors[:, 0],  # Arrow direction x-component
            vectors[:, 1],  # Arrow direction y-component
            vector_magnitude,
            cmap=cmap,
            norm=norm,
        )

    def plot(
        self,
        fermi_surface: pv.PolyData,
        normal: List[float] = [0, 0, 1],
        origin: List[float] = [0, 0, 0],
        plot_arrows: bool = True,
        arrow_factor: float = 1.0,
        cmap: str = "plasma",
        **line_kwargs,
    ):
        """
        A high-level method to generate a complete 2D slice plot.

        This method orchestrates the slicing and calling of the modular
        plotting functions.
        """
        # 1. Slice the data
        lines, points, scalars, vectors = self._prepare_slice_data(
            fermi_surface, normal, origin
        )
        # 2. Plot the contour lines
        self.plot_lines(lines, points, scalars, **line_kwargs)

        # 3. Optionally plot the vector arrows
        if plot_arrows:
            self.plot_arrows(lines, points, vectors, factor=arrow_factor)

    def scatter(
        self,
        fermi_surface: pv.PolyData,
        normal: List[float] = [0, 0, 1],
        origin: List[float] = [0, 0, 0],
        plot_arrows: bool = True,
        arrow_factor: float = 1.0,
        cmap: str = "plasma",
        **line_kwargs,
    ):
        """
        A high-level method to generate a complete 2D slice plot.

        This method orchestrates the slicing and calling of the modular
        plotting functions.
        """
        # 1. Slice the data
        lines, points, scalars, vectors = self._prepare_slice_data(
            fermi_surface, normal, origin
        )
        # 2. Plot the contour lines
        self.plot_points(points, scalars, cmap=cmap, **line_kwargs)

        # 3. Optionally plot the vector arrows
        if plot_arrows:
            self.plot_arrows(lines, points, vectors, factor=arrow_factor, cmap=cmap)

    def show(self):
        plt.show()
