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

# BZ_SCALE_FACTOR = 0.025
BZ_SCALE_FACTOR = 1


@dataclass
class BS2DSeries:
    """Container for 2D band structure surface plotting data.

    Similar to FermiSeries, but for 2D band structure surfaces where
    z-axis represents energy. Each BS2DSeries represents one (band, spin)
    combination's surface.
    """

    mesh: pv.PolyData  # Surface geometry (x=u, y=v, z=energy)
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


class BS2DPlotter(pv.Plotter):
    def __init__(self, bandstructure2d, **kwargs):
        super().__init__(**kwargs)
        self.bs2d = bandstructure2d
        self._meshes: list[pv.PolyData] = []
        self.values_dict: dict[str, np.ndarray] = {}

    def _to_series_list(
        self,
        bandstructure2d,
        scalars_data=None,
        vectors_data=None,
        **kwargs,
    ) -> list[BS2DSeries]:
        """Convert BandStructure2D and Properties to list of BS2DSeries.

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
        list[BS2DSeries]
            One BS2DSeries per (band, spin) surface.
        """
        # Resolve scalars from name or Property
        if isinstance(scalars_data, str):
            scalars_data = bandstructure2d.point_set.get_property(scalars_data)

        # Resolve vectors from name or Property
        if isinstance(vectors_data, str):
            vectors_data = bandstructure2d.point_set.get_property(vectors_data)

        # Extract scalar/vector metadata
        s_label = scalars_data.label if scalars_data else None
        s_unit = scalars_data.units if scalars_data else None
        s_lims = scalars_data.data_lim if scalars_data else None

        v_label = vectors_data.label if vectors_data else None
        v_unit = vectors_data.units if vectors_data else None
        v_lims = vectors_data.data_lim if vectors_data else None

        # Build one series per band/spin surface
        series_list: list[BS2DSeries] = []

        for (iband, ispin), surface in bandstructure2d.band_surfaces.items():
            # Get mask for this band/spin in the combined surface
            mask = bandstructure2d.band_spin_mask[(iband, ispin)]

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
            label = self._build_series_label(iband, ispin, len(bandstructure2d.band_surfaces))

            series_list.append(
                BS2DSeries(
                    mesh=surface,
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
        """Build label for a single BS2D series."""
        if n_surfaces > 1:
            spin_label = "up" if ispin == 0 else "down"
            return f"Band {iband} {spin_label}"
        return None

    def plot(
        self,
        bandstructure2d=None,
        scalars_data=None,
        vectors_data=None,
        scalars_mode: str = "surface",
        show_brillouin_zone: bool = True,
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
        add_surface_kwargs = add_surface_kwargs or {}
        add_texture_kwargs = add_texture_kwargs or {}

        # Convert to series list
        series_list = self._to_series_list(bs2d, scalars_data, vectors_data, **kwargs)

        # Resolve global clim if not provided
        if scalars_clim is None and scalars_data is not None and scalars_mode != "none":
            scalars_clim = self._resolve_clim(series_list)

        # Add Brillouin zone if requested
        if show_brillouin_zone and hasattr(bs2d, "get_2d_brillouin_zone"):
            # Get energy range from surfaces
            z_coords = bs2d.points[:, 2]
            e_min, e_max = float(z_coords.min()), float(z_coords.max())
            bz = bs2d.get_2d_brillouin_zone(e_min=e_min, e_max=e_max)
            self.add_brillouin_zone(bz)

        # Plot each series
        meshes: dict[tuple[int, int], pv.PolyData] = {}
        band_keys = list(bs2d.band_surfaces.keys())

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
            mesh_kwargs = {
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
                mesh_kwargs["scalar_bar_args"]["title"] = series.scalars_label  # type: ignore[index]

            self.add_mesh(mesh, **mesh_kwargs)
            self._meshes.append(mesh)

            # Add vectors if present
            if series.vectors is not None:
                texture_kwargs = {
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
        series_list: list[BS2DSeries],
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

    def _record_series_data(self, series: BS2DSeries) -> None:
        """Record series data for export."""
        key_prefix = f"band_{series.band_index}_spin_{series.spin_index}"
        self.values_dict[f"{key_prefix}_points"] = series.mesh.points
        if series.scalars is not None:
            self.values_dict[f"{key_prefix}_scalars"] = series.scalars
        if series.vectors is not None:
            self.values_dict[f"{key_prefix}_vectors"] = series.vectors

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
            # Export as mesh format - merge all meshes
            if self._meshes:
                combined = self._meshes[0]
                for mesh in self._meshes[1:]:
                    combined = combined.merge(mesh)
                combined.save(filename)
        elif ext == ".npz":
            if self.values_dict:
                np.savez(filename, **self.values_dict)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. Use .vtk, .vtp, .ply, .stl, or .npz"
            )

    def add_brillouin_zone(
        self,
        brillouin_zone: pv.PolyData = None,
        style: str = "wireframe",
        line_width: float = 2.0,
        color: ColorLike = "black",
        opacity: float = 1.0,
    ):
        self.brillouin_zone = brillouin_zone
        self.add_mesh(
            brillouin_zone,
            style=style,
            line_width=line_width,
            color=color,
            opacity=opacity,
        )

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

        if clip_surface:
            surface = self.clip_surface(surface, self.brillouin_zone)

        self.add_mesh(surface, **add_mesh_args)

        if add_active_vectors:
            # aligning the texture colors with the surface colors
            add_texture_args["cmap"] = add_texture_args.get("cmap", cmap)
            add_texture_args["clim"] = add_texture_args.get("clim", clim)

            self.add_texture(surface, **add_texture_args)

    def clip_surface(self, surface: pv.PolyData, brillouin_zone: pv.PolyData):
        for normal, center in zip(brillouin_zone.face_normals, brillouin_zone.centers):
            surface = surface.clip(origin=center, normal=normal, inplace=False)
            if surface.points.shape[0] == 0:
                break

        return surface

    def add_texture(
        self,
        surface: pv.PolyData,
        vectors: str | bool = True,
        factor: float = 1.0,
        add_mesh_args: dict = None,
        glyph_args: dict = None,
        **kwargs,
    ):
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
        glyph_args["scale"] = glyph_args.get("scale", True)
        glyph_args["orient"] = glyph_args.get("orient", vectors)

        active_vector_magnitude = np.linalg.norm(surface.active_vectors, axis=1)
        vector_scale_factor = 1 / active_vector_magnitude.max()
        factor = vector_scale_factor * BZ_SCALE_FACTOR * factor

        glyph_args["factor"] = factor
        glyph_args["indices"] = glyph_args.get("indices")

        arrows = surface.glyph(**glyph_args)
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
        self.cross_section_area = cross_section_area
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
            callback=partial(self._box_callback, port=0, add_surface_args=add_surface_args),
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
