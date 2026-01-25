from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from pyprocar.core.property_store import Property
from pyprocar.plotter.dos_plot import ShowColorbar
from pyprocar.core.bandstructure2D import (
    find_plane_limits,
    get_orthonormal_basis,
    get_uv_grid,
    get_uv_grid_points,
    transform_points_to_uv,
)

logger = logging.getLogger(__name__)


class PlaneScalarsMode(Enum):
    """Rendering modes for scalar data on plane slices."""

    PCOLORMESH = "pcolormesh"
    CONTOUR = "contour"
    CONTOURF = "contourf"

    @classmethod
    def from_string(cls, string: str | PlaneScalarsMode) -> PlaneScalarsMode:
        if isinstance(string, PlaneScalarsMode):
            return string
        string = string.lower()
        mode_map = {
            "pcolormesh": cls.PCOLORMESH,
            "contour": cls.CONTOUR,
            "contourf": cls.CONTOURF,
        }
        if string in mode_map:
            return mode_map[string]
        valid = ", ".join(mode_map.keys())
        raise ValueError(f"Invalid scalars mode: {string}. Valid modes: {valid}")


@dataclass
class PlaneSeries:
    """Container for a plane slice's plotting data.

    Similar to DOSPlotter's Series, but for 2D plane slices.
    Each PlaneSeries represents the interpolated grid data ready for rendering.
    """

    u_grid: np.ndarray  # 2D grid of u coordinates
    v_grid: np.ndarray  # 2D grid of v coordinates
    scalars: np.ndarray | None  # Interpolated scalar values on grid
    scalars_label: str | None
    scalars_unit: str | None
    scalars_lim: tuple[float, float] | None
    vectors_u: np.ndarray | None  # Vector u-components on grid
    vectors_v: np.ndarray | None  # Vector v-components on grid
    vectors_magnitude: np.ndarray | None  # Vector magnitudes for coloring
    vectors_label: str | None
    vectors_unit: str | None
    vectors_lim: tuple[float, float] | None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


class EBSPlanePlotter:
    def __init__(
        self,
        ebs_mesh,
        normal=(0, 0, 1),
        origin=(0, 0, 0),
        grid_interpolation=(20, 20),
        ax=None,
        figsize=(8, 7),
        dpi=100,
    ):
        self.ebs_mesh = ebs_mesh
        self.normal = normal
        self.origin = origin
        self.grid_interpolation = grid_interpolation

        slice = ebs_mesh.slice(normal=normal, origin=origin)

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.fig = ax.get_figure()
            self.ax = ax

        self.values_dict = {}

        self.u, self.v = get_orthonormal_basis(normal=normal)
        self.plane_points = transform_points_to_uv(slice.points, self.u, self.v)
        u_limits, v_limits = find_plane_limits(self.plane_points)

        self.u_grid, self.v_grid = get_uv_grid(
            grid_interpolation=grid_interpolation, u_limits=u_limits, v_limits=v_limits
        )

        self.uv_grid_points = get_uv_grid_points(self.u_grid, self.v_grid)

        self.n_points = self.uv_grid_points.shape[0]

    def interpolate_values(self, values: np.ndarray):
        if values.shape[-1] != 3:
            new_values = np.zeros(self.n_points)
            interpolator = LinearNDInterpolator(self.plane_points, values)
            new_values = interpolator(self.uv_grid_points)
        else:
            new_values = np.zeros((self.n_points, values.shape[-1]))
            for icoord in range(values.shape[-1]):
                interpolator = LinearNDInterpolator(self.plane_points, values[..., icoord])
                new_values[..., icoord] = interpolator(self.uv_grid_points)
        return new_values

    def points_to_grid(self, points: np.ndarray):
        return points.reshape(self.u_grid.shape)

    def project_vector_to_plane(self, vectors: np.ndarray):
        velocity_u = np.dot(vectors, self.u)
        velocity_v = np.dot(vectors, self.v)
        return velocity_u, velocity_v

    def _to_series(
        self,
        scalars_data: Property | tuple[str, np.ndarray] | None = None,
        vectors_data: Property | tuple[str, np.ndarray] | None = None,
        **kwargs,
    ) -> PlaneSeries:
        """Convert Property objects or tuples to PlaneSeries for plotting.

        This method handles the interpolation from slice points to the UV grid.

        Args:
            scalars_data: Property or (name, array) tuple for scalar coloring
            vectors_data: Property or (name, array) tuple for vector arrows
            **kwargs: Additional kwargs to include in series

        Returns:
            PlaneSeries with interpolated data ready for rendering
        """
        # Extract scalars
        scalars_grid = None
        s_label = None
        s_unit = None
        s_lim = None

        if scalars_data is not None:
            if isinstance(scalars_data, Property):
                scalars_values = scalars_data.to_array()
                s_label = scalars_data.label
                s_unit = scalars_data.units
                # rounded_data_lim may return an array, convert to tuple
                raw_lim = getattr(scalars_data, "rounded_data_lim", None)
                if raw_lim is not None and hasattr(raw_lim, "ravel"):
                    flat = raw_lim.ravel()
                    s_lim = (float(flat[0]), float(flat[1])) if len(flat) >= 2 else None
                else:
                    s_lim = raw_lim
            else:
                # Legacy tuple: (name, array)
                s_label, scalars_values = scalars_data
                s_unit = None
                s_lim = None

            # Interpolate scalars to grid
            scalars_grid_flat = self.interpolate_values(scalars_values)
            scalars_grid = self.points_to_grid(scalars_grid_flat)

        # Extract vectors
        vectors_u_grid = None
        vectors_v_grid = None
        vectors_magnitude_grid = None
        v_label = None
        v_unit = None
        v_lim = None

        if vectors_data is not None:
            if isinstance(vectors_data, Property):
                vectors_values = vectors_data.to_array()
                v_label = vectors_data.label
                v_unit = vectors_data.units
                # rounded_data_lim may return an array, convert to tuple
                raw_lim = getattr(vectors_data, "rounded_data_lim", None)
                if raw_lim is not None and hasattr(raw_lim, "ravel"):
                    flat = raw_lim.ravel()
                    v_lim = (float(flat[0]), float(flat[1])) if len(flat) >= 2 else None
                else:
                    v_lim = raw_lim
            else:
                # Legacy tuple: (name, array)
                v_label, vectors_values = vectors_data
                v_unit = None
                v_lim = None

            # Interpolate vectors to grid
            vectors_grid_flat = self.interpolate_values(vectors_values)

            # Project to plane coordinates
            velocity_u, velocity_v = self.project_vector_to_plane(vectors_grid_flat)
            vectors_u_grid = self.points_to_grid(velocity_u)
            vectors_v_grid = self.points_to_grid(velocity_v)

            # Compute magnitude for coloring
            magnitude_flat = np.sqrt(velocity_u**2 + velocity_v**2)
            vectors_magnitude_grid = self.points_to_grid(magnitude_flat)

        return PlaneSeries(
            u_grid=self.u_grid,
            v_grid=self.v_grid,
            scalars=scalars_grid,
            scalars_label=s_label,
            scalars_unit=s_unit,
            scalars_lim=s_lim,
            vectors_u=vectors_u_grid,
            vectors_v=vectors_v_grid,
            vectors_magnitude=vectors_magnitude_grid,
            vectors_label=v_label,
            vectors_unit=v_unit,
            vectors_lim=v_lim,
            additional_kwargs=kwargs.copy(),
        )

    def plot(
        self,
        scalars_data: Property | tuple[str, np.ndarray] | None = None,
        vectors_data: Property | tuple[str, np.ndarray] | None = None,
        scalars_mode: str | PlaneScalarsMode = "pcolormesh",
        scalars_cmap: str = "plasma",
        scalars_clim: tuple[float, float] | None = None,
        scalars_alpha: float = 0.7,
        scalars_show_colorbar: str | ShowColorbar = "single",
        vectors_cmap: str = "plasma",
        vectors_clim: tuple[float, float] | None = None,
        vectors_show_colorbar: str | ShowColorbar = "none",
        vectors_skip: int = 1,
        vectors_scale: float | None = None,
        vectors_arrow_length_factor: float = 1.0,
        contour_levels: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Plot plane slice data from Property objects or legacy tuples.

        This is the unified entry point for plane slice plotting,
        following the same pattern as DOSPlotter.plot().

        Parameters
        ----------
        scalars_data : Property or tuple, optional
            Property or (name, array) tuple for scalar coloring.
            Array should be 1D with shape (n_slice_points,).
        vectors_data : Property or tuple, optional
            Property or (name, array) tuple for vector arrows.
            Array should be 2D with shape (n_slice_points, 3).
        scalars_mode : str
            How to render scalar data:
            - "pcolormesh": Heatmap (default)
            - "contour": Line contours
            - "contourf": Filled contours
        scalars_cmap : str
            Colormap for scalar coloring. Default "plasma".
        scalars_clim : tuple, optional
            Color limits (vmin, vmax) for scalars. None = auto.
        scalars_alpha : float
            Transparency for scalar plot. Default 0.7.
        scalars_show_colorbar : str
            "single", "none". Default "single".
        vectors_cmap : str
            Colormap for vector magnitude coloring. Default "plasma".
        vectors_clim : tuple, optional
            Color limits for vector magnitudes. None = auto.
        vectors_show_colorbar : str
            "single", "none". Default "none".
        vectors_skip : int
            Plot every Nth arrow to reduce clutter. Default 1.
        vectors_scale : float, optional
            Quiver scale factor. None = auto.
        vectors_arrow_length_factor : float
            Multiplier for arrow length. Default 1.0.
        contour_levels : int
            Number of contour levels for contour/contourf modes. Default 10.
        **kwargs
            Additional kwargs passed to render methods.

        Returns
        -------
        dict[str, Any]
            Dict with keys "scalars", "vectors", "colorbar" mapping to artists.
        """
        scalars_mode = PlaneScalarsMode.from_string(scalars_mode)
        scalars_show_colorbar = ShowColorbar.from_string(scalars_show_colorbar)
        vectors_show_colorbar = ShowColorbar.from_string(vectors_show_colorbar)

        # Convert to series
        series = self._to_series(scalars_data, vectors_data, **kwargs)

        # Resolve color limits
        if scalars_clim is None and series.scalars is not None:
            if series.scalars_lim is not None:
                scalars_clim = series.scalars_lim
            else:
                finite = series.scalars[np.isfinite(series.scalars)]
                scalars_clim = (
                    (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0.0, 1.0)
                )

        if vectors_clim is None and series.vectors_magnitude is not None:
            if series.vectors_lim is not None:
                vectors_clim = series.vectors_lim
            else:
                finite = series.vectors_magnitude[np.isfinite(series.vectors_magnitude)]
                vectors_clim = (
                    (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0.0, 1.0)
                )

        artists: dict[str, Any] = {}

        # Render scalars
        if series.scalars is not None:
            if scalars_mode is PlaneScalarsMode.PCOLORMESH:
                artists["scalars"] = self._add_pcolormesh(
                    series, scalars_cmap, scalars_clim, scalars_alpha, **kwargs
                )
            elif scalars_mode is PlaneScalarsMode.CONTOUR:
                artists["scalars"] = self._add_contour(
                    series, scalars_cmap, scalars_clim, contour_levels, **kwargs
                )
            elif scalars_mode is PlaneScalarsMode.CONTOURF:
                artists["scalars"] = self._add_contourf(
                    series, scalars_cmap, scalars_clim, scalars_alpha, contour_levels, **kwargs
                )

            # Store for export
            self.scalar_name = series.scalars_label or "scalars"
            self.scalar_plot = artists["scalars"]

        # Render vectors
        if series.vectors_u is not None and series.vectors_v is not None:
            artists["vectors"] = self._add_quiver(
                series,
                vectors_cmap,
                vectors_clim,
                vectors_skip,
                vectors_scale,
                vectors_arrow_length_factor,
                **kwargs,
            )

            # Store for export
            self.vector_name = series.vectors_label or "vectors"
            self.vector_plot = artists["vectors"]

        # Add colorbars
        if scalars_show_colorbar is ShowColorbar.SINGLE and series.scalars is not None:
            label = series.scalars_label or ""
            if series.scalars_unit:
                label = f"{label} ({series.scalars_unit})"
            self._add_colorbar(artists.get("scalars"), label, scalars_cmap, scalars_clim)

        if vectors_show_colorbar is ShowColorbar.SINGLE and series.vectors_magnitude is not None:
            label = series.vectors_label or "Magnitude"
            if series.vectors_unit:
                label = f"{label} ({series.vectors_unit})"
            # Only add if no scalar colorbar was added
            if scalars_show_colorbar is ShowColorbar.NONE:
                self._add_colorbar(artists.get("vectors"), label, vectors_cmap, vectors_clim)

        # Record for export
        self._record_exports(series)

        return artists

    def _add_pcolormesh(
        self,
        series: PlaneSeries,
        cmap: str,
        clim: tuple[float, float] | None,
        alpha: float,
        shading: str = "gouraud",
        **kwargs,
    ):
        """Add pcolormesh plot for scalar data."""
        return self.ax.pcolormesh(
            series.u_grid,
            series.v_grid,
            series.scalars,
            cmap=cmap,
            clim=clim,
            alpha=alpha,
            shading=shading,  # type: ignore[arg-type]
            **kwargs,
        )

    def _add_contour(
        self,
        series: PlaneSeries,
        cmap: str,
        clim: tuple[float, float] | None,
        levels: int,
        **kwargs,
    ):
        """Add contour line plot for scalar data."""
        vmin = clim[0] if clim is not None else None
        vmax = clim[1] if clim is not None else None
        return self.ax.contour(
            series.u_grid,
            series.v_grid,
            series.scalars,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    def _add_contourf(
        self,
        series: PlaneSeries,
        cmap: str,
        clim: tuple[float, float] | None,
        alpha: float,
        levels: int,
        **kwargs,
    ):
        """Add filled contour plot for scalar data."""
        vmin = clim[0] if clim is not None else None
        vmax = clim[1] if clim is not None else None
        return self.ax.contourf(
            series.u_grid,
            series.v_grid,
            series.scalars,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            **kwargs,
        )

    def _add_quiver(
        self,
        series: PlaneSeries,
        cmap: str,
        clim: tuple[float, float] | None,
        skip: int,
        scale: float | None,
        arrow_length_factor: float,
        angles: str = "uv",
        scale_units: str = "inches",
        units: str = "inches",
        **kwargs,
    ):
        """Add quiver plot for vector data."""
        if series.vectors_u is None or series.vectors_v is None or series.vectors_magnitude is None:
            raise ValueError("Series must have vector data for quiver plot")

        u_grid = series.u_grid[::skip, ::skip]
        v_grid = series.v_grid[::skip, ::skip]
        vec_u = series.vectors_u[::skip, ::skip]
        vec_v = series.vectors_v[::skip, ::skip]
        magnitude = series.vectors_magnitude[::skip, ::skip]

        # Auto-scale if not provided
        if scale is None:
            scale = magnitude.max() * 3 if magnitude.max() > 0 else 1.0
        scale = scale / arrow_length_factor

        vmin = clim[0] if clim is not None else magnitude.min()
        vmax = clim[1] if clim is not None else magnitude.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)  # type: ignore[attr-defined]
        cmap_obj = plt.get_cmap(cmap)

        return self.ax.quiver(
            u_grid,
            v_grid,
            vec_u,
            vec_v,
            magnitude,
            angles=angles,
            scale=scale,
            scale_units=scale_units,
            units=units,
            cmap=cmap_obj,
            norm=norm,
            **kwargs,
        )

    def _add_colorbar(
        self,
        mappable,
        label: str,
        cmap: str,
        clim: tuple[float, float] | None,
        **kwargs,
    ):
        """Add colorbar for a plot."""
        if mappable is None and clim is not None:
            # Create ScalarMappable for colorbar
            norm = plt.Normalize(vmin=clim[0], vmax=clim[1])  # type: ignore[attr-defined]
            sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
            sm.set_array([])
            mappable = sm

        if not hasattr(self, "colorbars"):
            self.colorbars = []

        if mappable is not None:
            cb = self.fig.colorbar(mappable, ax=self.ax, label=label, **kwargs)
            self.colorbars.append(cb)
            return cb
        return None

    def _record_exports(self, series: PlaneSeries) -> None:
        """Record series data for export."""
        self.values_dict["u_grid"] = series.u_grid.ravel()
        self.values_dict["v_grid"] = series.v_grid.ravel()

        if series.scalars is not None:
            self.values_dict["scalars"] = series.scalars.ravel()

        if series.vectors_u is not None and series.vectors_v is not None:
            self.values_dict["vectors_u"] = series.vectors_u.ravel()
            self.values_dict["vectors_v"] = series.vectors_v.ravel()
            if series.vectors_magnitude is not None:
                self.values_dict["vectors_magnitude"] = series.vectors_magnitude.ravel()

    def export_data(self, filename: str) -> None:
        """Export recorded plot arrays to CSV/TXT/JSON/DAT.

        Parameters
        ----------
        filename : str
            Output path; extension defines format.
        """
        import json

        import pandas as pd

        possible_file_types = ["csv", "txt", "json", "dat"]
        file_type = filename.split(".")[-1]
        if file_type not in possible_file_types:
            raise ValueError(f"File type must be one of {possible_file_types}")
        if not self.values_dict:
            raise ValueError("No values recorded. Plot first before exporting.")

        values = {k: v for k, v in self.values_dict.items() if v is not None}

        if file_type in ["csv", "txt", "dat"]:
            df = pd.DataFrame(values)
            sep = {"csv": ",", "txt": "\t", "dat": " "}[file_type]
            df.to_csv(filename, sep=sep, index=False)
        else:  # json
            serializable = {k: np.asarray(v).tolist() for k, v in values.items()}
            with open(filename, "w") as f:
                json.dump(serializable, f)

    def set_xlim(self, xlim: tuple[float, float] | None = None) -> None:
        """Set x-axis (u-coordinate) limits."""
        if xlim is None:
            xlim = (self.u_grid.min(), self.u_grid.max())
        self.ax.set_xlim(xlim)

    def set_ylim(self, ylim: tuple[float, float] | None = None) -> None:
        """Set y-axis (v-coordinate) limits."""
        if ylim is None:
            ylim = (self.v_grid.min(), self.v_grid.max())
        self.ax.set_ylim(ylim)

    def set_xlabel(self, label: str = r"k$_u$ (1/$\AA$)", **kwargs) -> None:
        """Set x-axis label."""
        self.ax.set_xlabel(label, **kwargs)

    def set_ylabel(self, label: str = r"k$_v$ (1/$\AA$)", **kwargs) -> None:
        """Set y-axis label."""
        self.ax.set_ylabel(label, **kwargs)

    def set_aspect(self, aspect: str = "equal") -> None:
        """Set axis aspect ratio."""
        self.ax.set_aspect(aspect)  # type: ignore[arg-type]

    def draw_origin(
        self,
        marker: str = "+",
        color: str = "white",
        markersize: int = 10,
        **kwargs,
    ) -> None:
        """Draw marker at origin (0, 0)."""
        self.ax.plot(0, 0, marker=marker, color=color, markersize=markersize, **kwargs)

    def grid(
        self,
        enabled: bool = True,
        color: str = "#cccccc",
        linestyle: str = ":",
        linewidth: float = 0.5,
        **kwargs,
    ) -> None:
        """Configure grid display."""
        self.ax.grid(enabled, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

    def plot_scalars(
        self,
        scalars: tuple[str, np.ndarray] | Property | None = None,
        grid_points: np.ndarray | None = None,
        name: str = "",
        cmap: str = "plasma",
        clim: tuple[float, float] | None = None,
        shading: str = "gouraud",
        alpha: float = 0.7,
        **kwargs,
    ):
        """Plot scalar field on the plane slice.

        This is the legacy API method. For the modern API, use plot().

        Parameters
        ----------
        scalars : tuple or Property, optional
            Either (name, array) tuple or Property object.
        grid_points : np.ndarray, optional
            Pre-interpolated grid points (bypasses interpolation).
        name : str
            Name for the scalar field (used if grid_points provided).
        cmap : str
            Colormap name. Default "plasma".
        clim : tuple, optional
            Color limits (vmin, vmax).
        shading : str
            Shading mode for pcolormesh. Default "gouraud".
        alpha : float
            Transparency. Default 0.7.
        **kwargs
            Additional kwargs passed to pcolormesh.
        """
        if grid_points is not None:
            # Legacy path: direct grid points
            self.scalar_name = name
            scalars_grid = self.points_to_grid(grid_points)
            self.scalar_plot = self.ax.pcolormesh(
                self.u_grid,
                self.v_grid,
                scalars_grid,
                shading=shading,  # type: ignore[arg-type]
                cmap=cmap,
                alpha=alpha,
                clim=clim,
                **kwargs,
            )
        else:
            # Use new _to_series path
            series = self._to_series(scalars_data=scalars)

            resolved_clim = clim
            if resolved_clim is None and series.scalars is not None:
                if series.scalars_lim is not None:
                    resolved_clim = series.scalars_lim
                else:
                    finite = series.scalars[np.isfinite(series.scalars)]
                    resolved_clim = (
                        (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0.0, 1.0)
                    )

            self.scalar_name = series.scalars_label or name
            self.scalar_plot = self._add_pcolormesh(
                series, cmap, resolved_clim, alpha, shading=shading, **kwargs
            )

            # Record for export
            self._record_exports(series)

    def plot_vectors_quiver(
        self,
        vectors: tuple[str, np.ndarray] | Property | None = None,
        grid_points: np.ndarray | None = None,
        name: str = "",
        scalar_name: str = "",
        plot_scalar: bool = False,
        plot_scalar_args: dict | None = None,
        angles: str = "uv",
        scale: float | None = None,
        arrow_length_factor: float = 1.0,
        arrow_skip: int = 1,
        scale_units: str = "inches",
        units: str = "inches",
        color=None,
        cmap: str = "plasma",
        clim: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot vector field on the plane slice.

        This is the legacy API method. For the modern API, use plot().

        Parameters
        ----------
        vectors : tuple or Property, optional
            Either (name, array) tuple or Property object.
        grid_points : np.ndarray, optional
            Pre-interpolated vector grid points (bypasses interpolation).
        name : str
            Name for the vector field.
        scalar_name : str
            Name for scalar background if plot_scalar=True.
        plot_scalar : bool
            Whether to plot vector magnitude as scalar background.
        plot_scalar_args : dict, optional
            Arguments for the scalar plot.
        angles, scale, scale_units, units : quiver parameters
        arrow_length_factor : float
            Multiplier for arrow length.
        arrow_skip : int
            Plot every Nth arrow.
        color : optional
            Fixed color for arrows (disables cmap).
        cmap : str
            Colormap for magnitude coloring.
        clim : tuple, optional
            Color limits.
        **kwargs
            Additional kwargs passed to quiver.
        """
        if grid_points is not None:
            # Legacy path with pre-computed grid points
            self.vector_name = name

            velocity_u, velocity_v = self.project_vector_to_plane(grid_points)
            magnitude_grid_points = np.sqrt(velocity_u**2 + velocity_v**2)

            grid_u_vec = self.points_to_grid(velocity_u)
            grid_v_vec = self.points_to_grid(velocity_v)

            quiver_args = []
            quiver_args.append(self.u_grid[::arrow_skip, ::arrow_skip])
            quiver_args.append(self.v_grid[::arrow_skip, ::arrow_skip])
            quiver_args.append(grid_u_vec[::arrow_skip, ::arrow_skip])
            quiver_args.append(grid_v_vec[::arrow_skip, ::arrow_skip])

            if color is None:
                quiver_args.append(magnitude_grid_points)

            if scale is None:
                scale = magnitude_grid_points.max() * 3
            scale = scale / arrow_length_factor

            cmap_obj = plt.get_cmap(cmap)
            if clim is not None:
                norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
            else:
                norm = plt.Normalize(
                    vmin=magnitude_grid_points.min(), vmax=magnitude_grid_points.max()
                )

            if plot_scalar and not hasattr(self, "scalar_plot"):
                scalars_name = scalar_name if scalar_name else self.vector_name + "_magnitude"
                plot_scalar_args = plot_scalar_args if plot_scalar_args is not None else {}
                plot_scalar_args["cmap"] = plot_scalar_args.get("cmap", cmap_obj)
                plot_scalar_args["clim"] = plot_scalar_args.get("clim", clim)
                self.plot_scalars(
                    name=scalars_name, grid_points=magnitude_grid_points, **plot_scalar_args
                )

            self.vector_plot = self.ax.quiver(
                *quiver_args,
                angles=angles,
                scale=scale,
                scale_units=scale_units,
                units=units,
                color=color,
                cmap=cmap_obj,
                norm=norm,
                **kwargs,
            )
        else:
            # Use new _to_series path
            series = self._to_series(vectors_data=vectors)
            self.vector_name = series.vectors_label or name

            # Handle plot_scalar option
            if plot_scalar and not hasattr(self, "scalar_plot"):
                scalar_n = scalar_name if scalar_name else self.vector_name + "_magnitude"
                plot_scalar_args = plot_scalar_args or {}
                plot_scalar_args.setdefault("cmap", cmap)
                plot_scalar_args.setdefault("clim", clim)

                # Create scalar series from magnitude
                magnitude_series = PlaneSeries(
                    u_grid=series.u_grid,
                    v_grid=series.v_grid,
                    scalars=series.vectors_magnitude,
                    scalars_label=scalar_n,
                    scalars_unit=series.vectors_unit,
                    scalars_lim=series.vectors_lim,
                    vectors_u=None,
                    vectors_v=None,
                    vectors_magnitude=None,
                    vectors_label=None,
                    vectors_unit=None,
                    vectors_lim=None,
                )
                resolved_clim = plot_scalar_args.get("clim")
                if resolved_clim is None and magnitude_series.scalars is not None:
                    finite = magnitude_series.scalars[np.isfinite(magnitude_series.scalars)]
                    resolved_clim = (
                        (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0.0, 1.0)
                    )

                self.scalar_name = scalar_n
                self.scalar_plot = self._add_pcolormesh(
                    magnitude_series,
                    plot_scalar_args.get("cmap", cmap),
                    resolved_clim,
                    plot_scalar_args.get("alpha", 0.7),
                )

            # Resolve clim for vectors
            resolved_clim = clim
            if resolved_clim is None and series.vectors_magnitude is not None:
                if series.vectors_lim is not None:
                    resolved_clim = series.vectors_lim
                else:
                    finite = series.vectors_magnitude[np.isfinite(series.vectors_magnitude)]
                    resolved_clim = (
                        (float(finite.min()), float(finite.max())) if len(finite) > 0 else (0.0, 1.0)
                    )

            self.vector_plot = self._add_quiver(
                series,
                cmap,
                resolved_clim,
                arrow_skip,
                scale,
                arrow_length_factor,
                angles=angles,
                scale_units=scale_units,
                units=units,
                **kwargs,
            )

            # Record for export
            self._record_exports(series)

    def show_colorbar(
        self,
        show_vectors: bool = False,
        show_scalars: bool = False,
        label: str = "",
        vector_label: str = "",
        scalar_label: str = "",
        vector_colorbar_args: dict = None,
        scalar_colorbar_args: dict = None,
        **kwargs,
    ):
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
        elif (
            show_scalars
            and hasattr(self, "scalar_plot")
            or not show_vectors
            and not show_scalars
            and hasattr(self, "scalar_plot")
        ):
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

    def set_xaxis(self, label: str = "k$_u$ (1/Å)", fontsize: int = 12, **kwargs):
        self.ax.set_xlabel(label, fontsize=fontsize, **kwargs)

    def set_yaxis(self, label: str = "k$_v$ (1/Å)", fontsize: int = 12, **kwargs):
        self.ax.set_ylabel(label, fontsize=fontsize, **kwargs)

    def set_title(self, title: str = None, fontsize: int = 12, **kwargs):
        if title is not None:
            return self.ax.set_title(title, fontsize=fontsize, **kwargs)

        if hasattr(self, "scalar_plot"):
            title = f"Scalar {self.scalar_name} Contour Plot"
        elif hasattr(self, "vector_plot"):
            title = f"Vector {self.vector_name} Field Plot"
        elif hasattr(self, "scalar_plot") and hasattr(self, "vector_plot"):
            title = f"Scalar {self.scalar_name} and Vector {self.vector_name} Field Plot"
        else:
            title = ""
        title = title.replace("  ", " ")
        return self.ax.set_title(title, fontsize=fontsize, **kwargs)

    def set_default_params(self):
        self.set_xaxis()
        self.set_yaxis()
        self.set_title()

    def show(self, **kwargs):
        plt.show(**kwargs)

    def savefig(self, filename: str, **kwargs):
        plt.savefig(filename, **kwargs)

    def close(self, **kwargs):
        plt.close(**kwargs)

    def __str__(self):
        return f"EBSPlanePlotter(ebs_mesh={self.ebs_mesh})"

    def __repr__(self):
        return f"EBSPlanePlotter(ebs_mesh={self.ebs_mesh})"
