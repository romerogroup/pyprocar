"""2D Fermi surface slice plotter using matplotlib."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from pyprocar.core.property_store import Property
from pyprocar.plotter.dos_plot import ShowColorbar

if TYPE_CHECKING:
    from matplotlib.quiver import Quiver

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


@dataclass
class FermiSliceSeries:
    """Container for a single Fermi slice's plotting data.

    Similar to DOSPlotter's Series, but for 2D Fermi surface slice data.
    """

    points_2d: np.ndarray  # UV-projected slice points (N, 2)
    lines: np.ndarray  # Line connectivity array from PyVista
    scalars: np.ndarray | None  # Scalar values per point
    scalars_label: str | None
    scalars_unit: str | None
    scalars_lim: tuple[float, float] | None
    vectors: np.ndarray | None  # Vector values per point (N, 2 or N, 3)
    vectors_label: str | None
    vectors_unit: str | None
    vectors_lim: tuple[float, float] | None
    label: str | None
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


class FermiSlicePlotter:
    """A plotter class for 2D Fermi surface slices using matplotlib.

    This class takes sliced data from a 3D FermiSurface object and plots it
    in 2D using matplotlib.

    Parameters
    ----------
    fermi_surface : pv.PolyData
        The FermiSurface to slice.
    normal : array-like, optional
        Normal vector for the slice plane. Default is [0, 0, 1].
    origin : array-like, optional
        Origin point for the slice plane. Default is centroid.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
    dpi : int, optional
        Figure resolution. Default is 100.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, creates new figure/axes.
    """

    figsize: tuple[int, int]
    dpi: int
    fig: Figure
    ax: Axes
    fermi_surface: pv.PolyData
    origin: np.ndarray
    normal: np.ndarray
    values_dict: dict[str, np.ndarray]
    _scalar_plot: LineCollection | Any | None
    _vector_plot: Quiver | None
    _colorbar: Colorbar | None

    def __init__(
        self,
        fermi_surface: pv.PolyData,
        normal: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        figsize: tuple[int, int] = (8, 6),
        dpi: int = 100,
        ax: Axes | None = None,
    ):
        self.figsize = figsize
        self.dpi = dpi

        if ax is None:
            fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            assert isinstance(fig, Figure)
            self.fig = fig
        else:
            self.ax = ax
            fig = ax.get_figure()
            if fig is None:
                raise ValueError("Provided axes has no associated figure")
            assert isinstance(fig, Figure)
            self.fig = fig

        self.fermi_surface = fermi_surface

        centroid = self.fermi_surface.points.mean(axis=0)
        self.origin = centroid if origin is None else np.asarray(origin)
        self.normal = np.array([0, 0, 1]) if normal is None else np.asarray(normal)

        # Handle 2D Fermi surfaces (is2d/ebs are dynamic attributes added at runtime)
        if hasattr(fermi_surface, "is2d") and fermi_surface.is2d:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            self.origin = centroid
            n_kx: int = fermi_surface.ebs.n_kx  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
            n_ky: int = fermi_surface.ebs.n_ky  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
            n_kz: int = fermi_surface.ebs.n_kz  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
            if n_kz == 1:
                self.normal = np.array([0, 0, 1])
            elif n_ky == 1:
                self.normal = np.array([0, 1, 0])
            elif n_kx == 1:
                self.normal = np.array([1, 0, 0])

        # Data export storage
        self.values_dict = {}

        # Plot handles for colorbar
        self._scalar_plot = None
        self._vector_plot = None
        self._colorbar = None

    # ------------------------------------------------------------------
    # Core utility methods (Phase 3)
    # ------------------------------------------------------------------

    def get_orthonormal_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute orthonormal basis vectors for the slice plane."""
        if np.abs(np.dot(self.normal, [0, 0, 1])) < 0.99:
            v_temp = np.array([0, 0, 1])
        else:
            v_temp = np.array([0, 1, 0])

        u = np.cross(v_temp, self.normal).astype(np.float32)
        u /= np.linalg.norm(u)
        v = np.cross(self.normal, u).astype(np.float32)
        v /= np.linalg.norm(v)
        return u, v

    def _prepare_slice_data(
        self,
        fermi_surface: pv.PolyData,
        scalars_name: str | None = None,
        vectors_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Slice the Fermi surface and extract data.

        Returns
        -------
        tuple
            (lines, points, scalars, vectors)
        """
        slice_data = fermi_surface.slice(normal=self.normal, origin=self.origin)
        points = slice_data.points
        lines = slice_data.lines

        logger.debug(f"Slice Data: \n {slice_data}")
        logger.debug(f"Slice Lines Shape: {slice_data.lines.shape}")

        scalars = slice_data.active_scalars
        vectors = slice_data.active_vectors
        active_scalars_name = slice_data.active_scalars_name
        active_vectors_name = slice_data.active_vectors_name

        if vectors is not None and scalars is not None:
            pass  # Use as-is
        elif scalars is not None and scalars.shape[-1] == 3:
            vectors = scalars
            scalars = np.linalg.norm(scalars, axis=1)
        else:
            vectors = None

        if scalars_name is not None and scalars_name in slice_data.point_data:
            scalars = slice_data.point_data[scalars_name]
        elif scalars_name is not None and scalars_name not in slice_data.point_data:
            msg = f"Scalars name {scalars_name} not found in slice data."
            msg += f" Using active scalars ({active_scalars_name}) instead."
            user_logger.warning(msg)

        if vectors_name is not None and vectors_name in slice_data.point_data:
            vectors = slice_data.point_data[vectors_name]
        elif vectors_name is not None and vectors_name not in slice_data.point_data:
            msg = f"Vectors name {vectors_name} not found in slice data."
            msg += f" Using active vectors ({active_vectors_name}) instead."
            user_logger.warning(msg)

        return lines, points, scalars, vectors

    def _iter_segments(self, lines: np.ndarray) -> Iterator[tuple[int, int]]:
        """Yield (start_idx, end_idx) for each line segment.

        Handles PyVista's line connectivity array format.
        """
        i = 0
        while i < len(lines):
            num_points_in_line = lines[i]
            line_connectivity_start = i + 1

            for j in range(num_points_in_line - 1):
                start_idx = lines[line_connectivity_start + j]
                end_idx = lines[line_connectivity_start + j + 1]
                yield start_idx, end_idx

            i += num_points_in_line + 1

    @staticmethod
    def _extract_lim(raw: Any) -> tuple[float, float] | None:
        """Extract a (min, max) limit tuple from a raw value.

        Handles nested lists/tuples like [[vmin, vmax]] or flat [vmin, vmax].
        """
        if raw is None:
            return None
        try:
            if len(raw) == 0:
                return None
            first: Any = raw[0]
            if hasattr(first, "__len__") and len(first) >= 2:
                return (float(first[0]), float(first[1]))
            if len(raw) >= 2:
                return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError, IndexError):
            return None
        return None

    # ------------------------------------------------------------------
    # Property-based _to_series method (Phase 4)
    # ------------------------------------------------------------------

    def _to_series(
        self,
        fermi_surface: pv.PolyData | None = None,
        scalars_data: Property | None = None,
        vectors_data: Property | None = None,
        scalars_name: str | None = None,
        vectors_name: str | None = None,
        **kwargs: Any,
    ) -> FermiSliceSeries:
        """Convert Property objects or FermiSurface data to FermiSliceSeries.

        Parameters
        ----------
        fermi_surface : pv.PolyData, optional
            FermiSurface to slice. Uses self.fermi_surface if None.
        scalars_data : Property, optional
            Property object for scalar coloring.
        vectors_data : Property, optional
            Property object for vector arrows.
        scalars_name : str, optional
            Fallback: name of scalar field in PyVista point_data.
        vectors_name : str, optional
            Fallback: name of vector field in PyVista point_data.
        **kwargs
            Additional kwargs to pass to series.

        Returns
        -------
        FermiSliceSeries
            Normalized data container for plotting.
        """
        if fermi_surface is None:
            fermi_surface = self.fermi_surface

        # Resolve scalar/vector names from Property objects if provided
        resolved_scalars_name = scalars_name
        resolved_vectors_name = vectors_name

        if scalars_data is not None:
            resolved_scalars_name = scalars_data.name
        elif fermi_surface.active_scalars_name is not None:
            resolved_scalars_name = fermi_surface.active_scalars_name

        if vectors_data is not None:
            resolved_vectors_name = vectors_data.name
        elif fermi_surface.active_vectors_name is not None:
            resolved_vectors_name = fermi_surface.active_vectors_name

        # Prepare slice data
        lines, points, scalars, vectors = self._prepare_slice_data(
            fermi_surface, resolved_scalars_name, resolved_vectors_name
        )

        # Extract metadata from Property objects
        s_label = scalars_data.label if scalars_data else resolved_scalars_name
        s_unit = scalars_data.units if scalars_data else None
        s_lim_raw: Any = getattr(scalars_data, "rounded_data_lim", None) if scalars_data else None
        s_lim = self._extract_lim(s_lim_raw)

        v_label = vectors_data.label if vectors_data else resolved_vectors_name
        v_unit = vectors_data.units if vectors_data else None
        v_lim_raw: Any = getattr(vectors_data, "rounded_data_lim", None) if vectors_data else None
        v_lim = self._extract_lim(v_lim_raw)

        return FermiSliceSeries(
            points_2d=points[:, :2],
            lines=lines,
            scalars=scalars,
            scalars_label=s_label,
            scalars_unit=s_unit,
            scalars_lim=s_lim,
            vectors=vectors,
            vectors_label=v_label,
            vectors_unit=v_unit,
            vectors_lim=v_lim,
            label=None,
            additional_kwargs=kwargs,
        )

    # ------------------------------------------------------------------
    # Unified plot() method with ShowColorbar (Phase 5)
    # ------------------------------------------------------------------

    def plot(
        self,
        fermi_surface: pv.PolyData | None = None,
        scalars_data: Property | None = None,
        vectors_data: Property | None = None,
        scalars_name: str | None = None,
        vectors_name: str | None = None,
        scalars_mode: str = "lines",  # "lines", "scatter"
        vectors_mode: str = "quiver",  # "quiver" - reserved for future use
        scalars_cmap: str = "plasma",
        scalars_clim: tuple[float, float] | None = None,
        scalars_show_colorbar: ShowColorbar | str = ShowColorbar.SINGLE,
        vectors_cmap: str = "plasma",
        vectors_clim: tuple[float, float] | None = None,
        vectors_show_colorbar: ShowColorbar | str = ShowColorbar.NONE,
        plot_arrows: bool = False,
        line_kwargs: dict[str, Any] | None = None,
        scatter_kwargs: dict[str, Any] | None = None,
        quiver_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Plot Fermi surface slice with Property-based API.

        Parameters
        ----------
        fermi_surface : pv.PolyData, optional
            FermiSurface to slice. Uses self.fermi_surface if None.
        scalars_data : Property, optional
            Property object for scalar coloring.
        vectors_data : Property, optional
            Property object for vector arrows.
        scalars_name : str, optional
            Fallback: name of scalar field in PyVista point_data.
        vectors_name : str, optional
            Fallback: name of vector field in PyVista point_data.
        scalars_mode : str
            How to render scalars: "lines" or "scatter".
        vectors_mode : str
            How to render vectors: "quiver".
        scalars_cmap : str
            Colormap for scalar coloring.
        scalars_clim : tuple, optional
            Color limits for scalars.
        scalars_show_colorbar : ShowColorbar or str
            Colorbar mode: "single", "none".
        vectors_cmap : str
            Colormap for vector coloring.
        vectors_clim : tuple, optional
            Color limits for vectors.
        vectors_show_colorbar : ShowColorbar or str
            Colorbar mode for vectors.
        plot_arrows : bool
            Whether to plot vector arrows.
        line_kwargs : dict, optional
            kwargs for LineCollection.
        scatter_kwargs : dict, optional
            kwargs for scatter.
        quiver_kwargs : dict, optional
            kwargs for quiver.
        **kwargs
            Additional kwargs.

        Returns
        -------
        dict
            Dictionary of artist handles.
        """
        del vectors_mode  # Reserved for future use
        # Convert string colorbar modes to enum
        scalars_show_colorbar = ShowColorbar.from_string(scalars_show_colorbar)
        vectors_show_colorbar = ShowColorbar.from_string(vectors_show_colorbar)

        if fermi_surface is None:
            fermi_surface = self.fermi_surface

        # Convert to series
        series = self._to_series(
            fermi_surface=fermi_surface,
            scalars_data=scalars_data,
            vectors_data=vectors_data,
            scalars_name=scalars_name,
            vectors_name=vectors_name,
            **kwargs,
        )

        artists: dict[str, Any] = {}

        # Resolve color limits
        if scalars_clim is None and series.scalars is not None:
            scalars_clim = (float(series.scalars.min()), float(series.scalars.max()))
        if vectors_clim is None and series.vectors is not None:
            vec_mag = np.linalg.norm(series.vectors, axis=-1)
            vectors_clim = (float(vec_mag.min()), float(vec_mag.max()))

        # Plot scalars
        if series.scalars is not None:
            if scalars_mode == "lines":
                artists["scalars"] = self._add_lines(
                    series, scalars_cmap, scalars_clim, line_kwargs or {}
                )
            elif scalars_mode == "scatter":
                artists["scalars"] = self._add_scatter(
                    series, scalars_cmap, scalars_clim, scatter_kwargs or {}
                )

        # Plot vectors
        if plot_arrows and series.vectors is not None:
            artists["vectors"] = self._add_quiver(
                series, vectors_cmap, vectors_clim, quiver_kwargs or {}
            )

        # Add colorbars based on ShowColorbar enum
        if scalars_show_colorbar is ShowColorbar.SINGLE and series.scalars is not None:
            label = series.scalars_label or ""
            if series.scalars_unit:
                label = f"{label} ({series.scalars_unit})"
            self._add_colorbar(artists.get("scalars"), label, scalars_cmap, scalars_clim, "scalars")

        if vectors_show_colorbar is ShowColorbar.SINGLE and series.vectors is not None:
            label = series.vectors_label or "Magnitude"
            if series.vectors_unit:
                label = f"{label} ({series.vectors_unit})"
            # Only add if no scalar colorbar
            if scalars_show_colorbar is ShowColorbar.NONE:
                self._add_colorbar(
                    artists.get("vectors"), label, vectors_cmap, vectors_clim, "vectors"
                )

        # Record data for export
        self._record_values(series)

        # Auto-scale view
        self.ax.autoscale_view()

        return artists

    # ------------------------------------------------------------------
    # Rendering methods (Phase 6)
    # ------------------------------------------------------------------

    def _add_lines(
        self,
        series: FermiSliceSeries,
        cmap: str,
        clim: tuple[float, float] | None,
        line_kwargs: dict[str, Any],
    ) -> LineCollection:
        """Add line segments colored by scalar values."""
        line_segments: list[list[tuple[Any, Any]]] = []
        colors_list: list[Any] = []

        cmap_obj = plt.get_cmap(cmap)
        if clim is not None:
            norm = Normalize(vmin=clim[0], vmax=clim[1])
        elif series.scalars is not None:
            norm = Normalize(vmin=series.scalars.min(), vmax=series.scalars.max())
        else:
            norm = Normalize(vmin=0, vmax=1)

        points = series.points_2d
        for start_idx, end_idx in self._iter_segments(series.lines):
            p1 = points[start_idx]
            p2 = points[end_idx]
            line_segments.append([(p1[0], p1[1]), (p2[0], p2[1])])

            if series.scalars is not None:
                avg_scalar = (series.scalars[start_idx] + series.scalars[end_idx]) / 2.0
                colors_list.append(avg_scalar)

        colors: list[Any] | None = colors_list if len(colors_list) > 0 else None

        merged_kwargs = {**series.additional_kwargs, **line_kwargs}
        lc = LineCollection(line_segments, array=colors, cmap=cmap_obj, norm=norm, **merged_kwargs)
        self._scalar_plot = self.ax.add_collection(lc)
        return lc

    def _add_scatter(
        self,
        series: FermiSliceSeries,
        cmap: str,
        clim: tuple[float, float] | None,
        scatter_kwargs: dict[str, Any],
    ) -> Any:
        """Add scatter plot with scalar coloring."""
        merged_kwargs = {**series.additional_kwargs, **scatter_kwargs}

        vmin, vmax = clim if clim else (None, None)
        self._scalar_plot = self.ax.scatter(
            series.points_2d[:, 0],
            series.points_2d[:, 1],
            c=series.scalars,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **merged_kwargs,
        )
        return self._scalar_plot

    def _add_quiver(
        self,
        series: FermiSliceSeries,
        cmap: str,
        clim: tuple[float, float] | None,
        quiver_kwargs: dict[str, Any],
    ) -> Quiver | None:
        """Add vector arrows."""
        vectors = series.vectors
        if vectors is None:
            return None

        vector_magnitude = np.linalg.norm(vectors, axis=-1)

        merged_kwargs = {**series.additional_kwargs, **quiver_kwargs}
        merged_kwargs.setdefault("angles", "uv")
        merged_kwargs.setdefault("scale_units", "inches")
        merged_kwargs.setdefault("units", "inches")

        if "scale" not in merged_kwargs:
            merged_kwargs["scale"] = vector_magnitude.max() * 3

        vmin, vmax = clim if clim else (vector_magnitude.min(), vector_magnitude.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Use first 2 components for 2D plot
        u = vectors[:, 0]
        v = vectors[:, 1] if vectors.shape[-1] > 1 else np.zeros_like(u)

        self._vector_plot = self.ax.quiver(
            series.points_2d[:, 0],
            series.points_2d[:, 1],
            u,
            v,
            vector_magnitude,
            cmap=cmap,
            norm=norm,
            **merged_kwargs,
        )
        return self._vector_plot

    def _add_colorbar(
        self,
        mappable: Any,
        label: str,
        cmap: str,
        clim: tuple[float, float] | None,
        kind: str,
    ) -> None:
        """Add colorbar to the plot."""
        del cmap, clim, kind  # Unused but kept for API consistency
        if mappable is None:
            return

        self._colorbar = self.fig.colorbar(mappable, ax=self.ax, label=label)

    def _record_values(self, series: FermiSliceSeries) -> None:
        """Record series data for export."""
        self.values_dict["points_u"] = series.points_2d[:, 0]
        self.values_dict["points_v"] = series.points_2d[:, 1]
        if series.scalars is not None:
            label = series.scalars_label or "scalars"
            self.values_dict[label] = series.scalars
        if series.vectors is not None:
            label = series.vectors_label or "vectors"
            self.values_dict[f"{label}_magnitude"] = np.linalg.norm(series.vectors, axis=-1)

    # ------------------------------------------------------------------
    # Axis configuration methods (Phase 7)
    # ------------------------------------------------------------------

    def set_xlabel(self, label: str, **kwargs: Any) -> None:
        self.ax.set_xlabel(label, **kwargs)

    def set_ylabel(self, label: str, **kwargs: Any) -> None:
        self.ax.set_ylabel(label, **kwargs)

    def set_title(self, title: str, **kwargs: Any) -> None:
        self.ax.set_title(title, **kwargs)

    def set_xlim(self, lim: tuple[float, float] | None = None, **kwargs: Any) -> None:
        if lim is not None:
            self.ax.set_xlim(lim, **kwargs)

    def set_ylim(self, lim: tuple[float, float] | None = None, **kwargs: Any) -> None:
        if lim is not None:
            self.ax.set_ylim(lim, **kwargs)

    def set_aspect(self, aspect: float | Literal["auto", "equal"] = "equal", **kwargs: Any) -> None:
        self.ax.set_aspect(aspect, **kwargs)

    def set_grid(self, visible: bool = True, **kwargs: Any) -> None:
        self.ax.grid(visible, **kwargs)

    def set_default_labels(self) -> None:
        """Set default axis labels based on slice orientation."""
        if (
            np.isclose(self.origin, np.array([0, 0, 0])).all()
            and np.isclose(self.normal, np.array([0, 0, 1])).all()
        ):
            x_label = r"$k_x$ (1/$\AA$)"
            y_label = r"$k_y$ (1/$\AA$)"
            title = f"Fermi Surface Slice at $k_z$ = {self.origin[2]:.2f} (1/$\\AA$)"
        elif (
            np.isclose(self.origin, np.array([0, 0, 0])).all()
            and np.isclose(self.normal, np.array([0, 1, 0])).all()
        ):
            x_label = r"$k_x$ (1/$\AA$)"
            y_label = r"$k_z$ (1/$\AA$)"
            title = f"Fermi Surface Slice at $k_y$ = {self.origin[1]:.2f} (1/$\\AA$)"
        elif (
            np.isclose(self.origin, np.array([0, 0, 0])).all()
            and np.isclose(self.normal, np.array([1, 0, 0])).all()
        ):
            x_label = r"$k_y$ (1/$\AA$)"
            y_label = r"$k_z$ (1/$\AA$)"
            title = f"Fermi Surface Slice at $k_x$ = {self.origin[0]:.2f} (1/$\\AA$)"
        else:
            self.get_orthonormal_basis()  # Compute basis (currently unused for generic slices)
            x_label = r"$k_u$ (1/$\AA$)"
            y_label = r"$k_v$ (1/$\AA$)"
            title = f"Fermi Surface Slice (origin={self.origin}, normal={self.normal})"

        self.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.set_title(title)
        self.set_aspect("equal", adjustable="box")
        self.set_grid(visible=True, linestyle="--", alpha=0.6)

    # ------------------------------------------------------------------
    # Export methods (Phase 7)
    # ------------------------------------------------------------------

    def export_data(self, filename: str) -> None:
        """Export recorded plot arrays to file.

        Parameters
        ----------
        filename : str
            Output path. Extension determines format (csv, txt, json, dat).
        """
        import json

        import pandas as pd

        possible_file_types = ["csv", "txt", "json", "dat"]
        file_type = filename.split(".")[-1]
        if file_type not in possible_file_types:
            raise ValueError(f"File type must be one of {possible_file_types}")
        if not self.values_dict:
            raise ValueError("No values recorded. Plot first before exporting.")

        values: dict[str, np.ndarray] = {}
        for key, value in self.values_dict.items():
            arr = np.atleast_1d(value)
            if arr.size > 0:
                values[key] = arr

        if file_type in ["csv", "txt", "dat"]:
            df = pd.DataFrame(values)
            sep = "," if file_type == "csv" else "\t" if file_type == "txt" else " "
            df.to_csv(filename, sep=sep, index=False)
        else:  # json
            serializable: dict[str, list[Any]] = {
                k: np.asarray(v).tolist() for k, v in values.items()
            }
            with open(filename, "w") as f:
                json.dump(serializable, f)

    def show(self) -> None:
        """Display the plot."""
        plt.show()

    def savefig(self, filename: str, **kwargs: Any) -> None:
        """Save figure to file."""
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("dpi", self.dpi)
        self.fig.savefig(filename, **kwargs)

    @property
    def colorbar(self) -> Colorbar | None:
        """Return the colorbar if one exists."""
        return self._colorbar

    # ------------------------------------------------------------------
    # Legacy API compatibility methods (Phase 8)
    # ------------------------------------------------------------------

    def plot_lines(
        self,
        fermi_surface: pv.PolyData | None = None,
        scalars_name: str | None = None,
        vectors_name: str | None = None,
        cmap: str = "plasma",
        **kwargs: Any,
    ) -> LineCollection | None:
        """Plot line segments (legacy API).

        Delegates to plot() with scalars_mode="lines".
        """
        artists = self.plot(
            fermi_surface=fermi_surface,
            scalars_name=scalars_name,
            vectors_name=vectors_name,
            scalars_mode="lines",
            scalars_cmap=cmap,
            scalars_show_colorbar=ShowColorbar.NONE,
            line_kwargs=kwargs,
        )
        return artists.get("scalars")

    def plot_points(
        self,
        fermi_surface: pv.PolyData | None = None,
        scalars_name: str | None = None,
        vectors_name: str | None = None,
        cmap: str = "plasma",
        **kwargs: Any,
    ) -> Any:
        """Plot scatter points (legacy API).

        Delegates to plot() with scalars_mode="scatter".
        """
        artists = self.plot(
            fermi_surface=fermi_surface,
            scalars_name=scalars_name,
            vectors_name=vectors_name,
            scalars_mode="scatter",
            scalars_cmap=cmap,
            scalars_show_colorbar=ShowColorbar.NONE,
            scatter_kwargs=kwargs,
        )
        return artists.get("scalars")

    def plot_arrows(
        self,
        fermi_surface: pv.PolyData | None = None,
        scalars_name: str | None = None,
        vectors_name: str | None = None,
        cmap: str = "plasma",
        clim: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot vector arrows (legacy API).

        Delegates to plot() with plot_arrows=True.
        """
        artists = self.plot(
            fermi_surface=fermi_surface,
            scalars_name=scalars_name,
            vectors_name=vectors_name,
            scalars_mode="lines",
            vectors_cmap=cmap,
            vectors_clim=clim,
            scalars_show_colorbar=ShowColorbar.NONE,
            vectors_show_colorbar=ShowColorbar.NONE,
            plot_arrows=True,
            quiver_kwargs=kwargs,
        )
        return artists.get("vectors")

    def show_colorbar(
        self,
        show_vectors: bool = False,
        show_scalars: bool = False,
        label: str = "",
        **kwargs: Any,
    ) -> None:
        """Show colorbar (legacy API).

        For new code, use scalars_show_colorbar/vectors_show_colorbar
        parameters in plot() instead.
        """
        if show_scalars and self._scalar_plot is not None:
            self.fig.colorbar(self._scalar_plot, ax=self.ax, label=label, **kwargs)
        elif show_vectors and self._vector_plot is not None:
            self.fig.colorbar(self._vector_plot, ax=self.ax, label=label, **kwargs)
        elif self._scalar_plot is not None:
            self.fig.colorbar(self._scalar_plot, ax=self.ax, label=label, **kwargs)
        elif self._vector_plot is not None:
            self.fig.colorbar(self._vector_plot, ax=self.ax, label=label, **kwargs)
