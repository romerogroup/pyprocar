import logging
import os
from functools import partial
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib.collections import LineCollection
from pyvista import ColorLike
from pyvista.core.filters import _get_output
from pyvista.plotting.utilities.algorithms import (
    add_ids_algorithm,
    algorithm_to_mesh_handler,
    set_algorithm_input,
)

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

BZ_SCALE_FACTOR = 0.01

FS_AREA_SCALE_FACTOR = (2*np.pi)**2


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


def dHvA_frequency(A_max_angstrom2):
    hbar = 1.0546e-27    # erg·s
    e = 4.768e-10        # statcoulombs
    c = 3.0e10           # cm/s
    A_max_cm2 = A_max_angstrom2 * 1e16          # cm^-2
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
        logger.info(f"____Adding Surface to Plotter____")

        if add_texture_args is None:
            add_texture_args = {}
        add_texture_args["name"] = add_texture_args.get("name", "vectors")

        if add_mesh_args is None:
            add_mesh_args = {}

        if show_scalar_bar:
            active_scalar_name = fermi_surface.active_scalars_name
            
            if active_scalar_name is None:
                raise ValueError("No active scalar found for the Fermi surface. "
                                 "Use the compute* methods on the FermiSurface object to compute the scalar data.")
            
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
            add_surface_args["add_texture_args"] = add_surface_args.get(
                "add_texture_args", {}
            )
            add_surface_args["add_texture_args"]["name"] = "vectors"
            slc.set_active_vectors(active_vector_name)

        self.add_surface(slc, name="slice", **add_surface_args)

        if show_van_alphen_frequency and show_cross_section_area:
            raise ValueError("show_van_alphen_frequency and show_cross_section_area cannot be True at the same time")
        
        if show_van_alphen_frequency:
            surface = slc.delaunay_2d()
            text = f"Van Alphen Frequency : {dHvA_frequency(surface.area*FS_AREA_SCALE_FACTOR):.4f}" + " Gauss"
            self.add_text(text, name="area_text", **add_text_args)
        elif show_cross_section_area:
            surface = slc.delaunay_2d()
            text = f"Cross sectional area : {surface.area*FS_AREA_SCALE_FACTOR:.4f}" + " Ang^-2"
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
                self._box_callback, port=0, add_surface_args=add_surface_args, 
                show_van_alphen_frequency=show_van_alphen_frequency, 
                show_cross_section_area=show_cross_section_area
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

    def _box_callback(self, planes, port=0, add_surface_args=None, show_van_alphen_frequency=False, show_cross_section_area=False):
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

    def __init__(self, normal, origin=None, figsize=(8, 6), dpi=100, ax=None):
        self.figsize = figsize
        self.dpi = dpi

        self.ax = ax
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)

        self.normal = normal
        if origin is None:
            self.origin = np.array([0, 0, 0])
        else:
            self.origin = origin

        self.set_default_settings()
        
    def get_orthonormal_basis(self):
        if np.abs(np.dot(self.normal, [0, 0, 1])) < 0.99:
            v_temp = np.array([0, 0, 1])  # Not parallel to normal
        else:
            v_temp = np.array([0, 1, 0])  # Not parallel to normal
            
        # u = np.cross(self.normal, v_temp).astype(np.float32)
        u = np.cross(v_temp,self.normal).astype(np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u).astype(np.float32)
        v /= np.linalg.norm(v)  # Ensure normalization
        return u, v

    def set_default_settings(self):
        if np.isclose(self.origin, np.array([0, 0, 0])).all() and np.isclose(self.normal, np.array([0, 0, 1])).all():
            x_label = "$k_x$ (1/$\AA$)"
            y_label = "$k_y$ (1/$\AA$)"
            title=f"Fermi Surface Slice at $k_z$ = {self.origin[2]:.2f} (1/$\AA$)"
        elif np.isclose(self.origin, np.array([0, 0, 0])).all() and np.isclose(self.normal, np.array([0, 1, 0])).all():
            x_label = "$k_x$ (1/$\AA$)"
            y_label = "$k_z$ (1/$\AA$)"
            title=f"Fermi Surface Slice at $k_y$ = {self.origin[1]:.2f} (1/$\AA$)"
        elif np.isclose(self.origin, np.array([0, 0, 0])).all() and np.isclose(self.normal, np.array([1, 0, 0])).all():
            x_label = "$k_y$ (1/$\AA$)"
            y_label = "$k_z$ (1/$\AA$)"
            title=f"Fermi Surface Slice at $k_x$ = {self.origin[0]:.2f} (1/$\AA$)"
        else:
            u,v = self.get_orthonormal_basis()
            x_label = "$k_u$ (1/$\AA$)"
            y_label = "$k_v$ (1/$\AA$)"
            title=f"Fermi Surface Slice (origin={self.origin}, normal={self.normal}, u={u}, v={v})"
        
        self.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.set_title(title)
        self.set_aspect("equal", adjustable="box")
        self.set_grid(visible=True, linestyle="--", alpha=0.6)
    
        
    def set_xlabel(self, label:str, **kwargs):
        self.ax.set_xlabel(label, **kwargs)
        
    def set_ylabel(self, label:str, **kwargs):
        self.ax.set_ylabel(label, **kwargs)
        
    def set_title(self, label:str, **kwargs):
        self.ax.set_title(label, **kwargs)
        
    def set_grid(self, visible:bool=True, **kwargs):
        self.ax.grid(visible, **kwargs)
        
    def set_aspect(self, aspect:str, **kwargs):
        self.ax.set_aspect(aspect, **kwargs)

    def _prepare_slice_data(
        self, fermi_surface: pv.PolyData, scalars_name:str=None, vectors_name:str=None
    ):
        """Slices the Fermi surface and stores the resulting data."""
        
        slice_data = fermi_surface.slice(normal=self.normal, origin=self.origin)
        points = slice_data.points
        lines = slice_data.lines
        
        scalars = slice_data.active_scalars
        vectors = slice_data.active_vectors
        active_scalars_name = slice_data.active_scalars_name
        active_vectors_name = slice_data.active_vectors_name
        if vectors is not None and scalars is not None:
            scalars = scalars
            vectors = vectors
        elif scalars.shape[-1] == 3:
            scalars = np.linalg.norm(scalars, axis=1)
            vectors = scalars
        else:
            scalars = scalars
            vectors = None
            
        if scalars_name is not None and scalars_name in slice_data.point_data:
            scalars = slice_data.point_data[scalars_name]
        elif scalars_name is not None and scalars_name not in slice_data.point_data:
            msg = f"Scalars name {scalars_name} not found in slice data."
            msg += f"Using active scalars ({active_scalars_name}) instead."
            user_logger.warning(msg)
            
        if vectors_name is not None and vectors_name in slice_data.point_data:
            vectors = slice_data.point_data[vectors_name]
        elif vectors_name is not None and vectors_name not in slice_data.point_data:
            msg = f"Vectors name {vectors_name} not found in slice data."
            msg += f"Using active vectors ({active_vectors_name}) instead."
            user_logger.warning(msg)

        return lines, points, scalars, vectors

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
        fermi_surface:pv.PolyData,
        scalars_name:str=None,
        vectors_name:str=None,
        cmap: str = "plasma",
        **kwargs,
    ):
        """
        Plots the line segments of the slice.

        This method uses the `_iter_segments` generator and is focused only on
        drawing the lines. It uses LineCollection for high performance.

        kwargs are passed to the matplotlib LineCollection.
        """
        if vectors_name is not None:
            self.vector_name = vectors_name
        elif fermi_surface.active_vectors_name is not None:
            self.vector_name = fermi_surface.active_vectors_name
        else:
            self.vector_name = "vector"
            
        if scalars_name is not None:
            self.scalar_name = scalars_name
        elif fermi_surface.active_scalars_name is not None:
            self.scalar_name = fermi_surface.active_scalars_name
        else:
            self.scalar_name = "scalar"

        lines, points, scalars, vectors = self._prepare_slice_data(fermi_surface, scalars_name, vectors_name)
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
        self.scalar_plot = self.ax.add_collection(lc)
        self.ax.autoscale_view()  # Important after adding a collection

    def plot_points(
        self,
        fermi_surface:pv.PolyData,
        scalars_name:str=None,
        vectors_name:str=None,
        cmap: str = "plasma",
        **kwargs,
    ):
        if vectors_name is not None:
            self.vector_name = vectors_name
        elif fermi_surface.active_vectors_name is not None:
            self.vector_name = fermi_surface.active_vectors_name
        else:
            self.vector_name = "vector"
            
        if scalars_name is not None:
            self.scalar_name = scalars_name
        elif fermi_surface.active_scalars_name is not None:
            self.scalar_name = fermi_surface.active_scalars_name
        else:
            self.scalar_name = "scalar"

        lines, points, scalars, vectors = self._prepare_slice_data(fermi_surface, scalars_name, vectors_name)
        if points is None:
            raise ValueError("Slice data is not available for plotting lines.")

        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=scalars.min(), vmax=scalars.max())

        self.scalar_plot = self.ax.scatter(
            points[:, 0], points[:, 1], c=scalars, cmap=cmap, norm=norm, **kwargs
        )

    def plot_arrows(
        self,
        fermi_surface:pv.PolyData,
        scalars_name:str=None,
        vectors_name:str=None,
        angles:str='uv',
        scale=None,
        arrow_length_factor:float=1.0,
        scale_units:str='inches',
        units:str='inches',
        color=None,
        cmap: str = "plasma",
        clim:Tuple[float, float]=None,
        **kwargs,
    ):
        """
        Plots vectors as arrows on the slice.

        This method also uses the `_iter_segments` generator and is focused
        only on drawing arrows.

        Parameters
        ----------
        factor : float, optional
            A scaling factor for the length of the arrows.
        
        """
        if vectors_name is not None:
            self.vector_name = vectors_name
        elif fermi_surface.active_vectors_name is not None:
            self.vector_name = fermi_surface.active_vectors_name
            self.vector_name = "vector"
            
        if scalars_name is not None:
            self.scalar_name = scalars_name
        elif fermi_surface.active_scalars_name is not None:
            self.scalar_name = fermi_surface.active_scalars_name
        else:
            self.scalar_name = "scalar"

        lines, points, scalars, vectors = self._prepare_slice_data(fermi_surface, scalars_name, vectors_name)
        if points is None or vectors is None:
            raise ValueError("Vector data is not available for plotting arrows.")

        # Use the generator to iterate
        vector_magnitude = np.linalg.norm(vectors, axis=1)
        
        if scale is None:
            scale = vector_magnitude.max()*3
        scale = scale / arrow_length_factor
        
        quiver_args = [points[:, 0], points[:, 1], vectors[:, 0], vectors[:, 1]]
        if color is not None:
            quiver_args.append(vector_magnitude)
        
        cmap = plt.get_cmap(cmap)
        if clim is not None:
            norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
        else:
            norm=plt.Normalize(vmin=vector_magnitude.min(), vmax=vector_magnitude.max())
        
        
        self.vector_plot = self.ax.quiver(
            points[:, 0],  # Arrow position x-component
            points[:, 1],  # Arrow position y-component
            vectors[:, 0],  # Arrow direction x-component
            vectors[:, 1],  # Arrow direction y-component
            vector_magnitude,
            angles=angles,
            scale=scale,
            scale_units=scale_units,
            units = units,
            color=color,
            cmap=cmap,
            norm=norm,
            **kwargs
        )

    def plot(
        self,
        fermi_surface: pv.PolyData,
        vectors_name:str=None,
        scalars_name:str=None,
        plot_arrows: bool = False,
        plot_arrows_args: dict = None,
        **line_kwargs,
    ):
        """
        A high-level method to generate a complete 2D slice plot.

        This method orchestrates the slicing and calling of the modular
        plotting functions.
        """
        lines, points, scalars, vectors = self._prepare_slice_data(fermi_surface, scalars_name, vectors_name)

        # 2. Plot the contour lines
        self.plot_lines(fermi_surface, scalars_name, vectors_name, **line_kwargs)

        # 3. Optionally plot the vector arrows
        plot_arrows_args = plot_arrows_args or {}
        if plot_arrows and vectors is not None:
            self.plot_arrows(fermi_surface, scalars_name, vectors_name, **plot_arrows_args)

    def scatter(
        self,
        fermi_surface: pv.PolyData,
        scalars_name:str=None,
        vectors_name:str=None,
        plot_arrows: bool = False,
        plot_arrows_args: dict = None,
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
            fermi_surface, scalars_name, vectors_name
        )
        # 2. Plot the contour lines
        self.plot_points(fermi_surface, scalars_name, vectors_name, cmap=cmap, **line_kwargs)

        # 3. Optionally plot the vector arrows
        plot_arrows_args = plot_arrows_args or {}
        if plot_arrows and vectors is not None:
            self.plot_arrows(fermi_surface, scalars_name, vectors_name, factor=arrow_factor, cmap=cmap)

    def show_colorbar(self, 
                      show_vectors:bool=False,
                      show_scalars:bool=False,
                      label:str="",
                      vector_label:str="",
                      scalar_label:str="",
                      vector_colorbar_args:dict=None,
                      scalar_colorbar_args:dict=None,
                      **kwargs):
        plot_handles = []
        labels = []
        colorbar_args_list = []
        vector_colorbar_args = vector_colorbar_args if vector_colorbar_args is not None else {}
        scalar_colorbar_args = scalar_colorbar_args if scalar_colorbar_args is not None else {}
        
        
        if show_vectors and show_scalars:
            plot_handles = [self.scalar_plot, self.vector_plot]
            labels = [scalar_label or f"{self.scalar_name}", vector_label or f"{self.vector_name}"]
            colorbar_args_list = []
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(scalar_colorbar_args)
            colorbar_args_list.append(tmp_colorbar_args)
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(vector_colorbar_args)
            colorbar_args_list.append(tmp_colorbar_args)
        elif show_vectors and hasattr(self, "vector_plot"):
            plot_handles = [self.vector_plot]
            labels = [label or f"{self.vector_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(vector_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        elif show_scalars and hasattr(self, "scalar_plot"):
            plot_handles = [self.scalar_plot]
            labels = [label or f"{self.scalar_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(scalar_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        elif not show_vectors and not show_scalars and hasattr(self, "scalar_plot"):
            plot_handles = [self.scalar_plot]
            labels = [label or f"{self.scalar_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(scalar_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        elif not show_vectors and not show_scalars and hasattr(self, "vector_plot"):
            plot_handles = [self.vector_plot]
            labels = [label or f"{self.vector_name}"]
            tmp_colorbar_args = kwargs.copy()
            tmp_colorbar_args.update(vector_colorbar_args)
            colorbar_args_list = [tmp_colorbar_args]
        else:
            raise ValueError("No plot to show colorbar for")
        
        
        self.colorbars = []
        for plot_handle, label, colorbar_args in zip(plot_handles, labels, colorbar_args_list):
            self.colorbars.append(self.fig.colorbar(plot_handle, label=label, **colorbar_args))
        
        
    def show(self):
        plt.show()
