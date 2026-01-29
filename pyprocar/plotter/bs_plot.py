from __future__ import annotations

__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, cast

import matplotlib.colors as mpcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from pyprocar.core import KPath

logger = logging.getLogger(__name__)

# Removed legacy style/strategy scaffolding in favor of simpler API.

# Simplified main plotter class


@dataclass
class PlainBandStyle:
    color: str = "black"
    linestyle: str = "-"
    linewidth: float = 1.0
    alpha: float = 1.0
    label: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, style_dict: dict[str, Any]) -> PlainBandStyle:
        return cls(**style_dict)

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class BandSeries:
    """Container for a single band's plotting data.

    Similar to DOSPlotter's Series, but for band structure data.
    Each BandSeries represents one (band, spin) combination.
    """

    x: np.ndarray  # k-path distances
    y: np.ndarray  # band energies
    scalars: np.ndarray | None  # optional scalar coloring data
    scalars_label: str | None
    scalars_unit: str | None
    scalars_lim: tuple[float, float] | None
    vectors: np.ndarray | None  # optional vector data (spin texture)
    vectors_label: str | None
    vectors_unit: str | None
    vectors_lim: tuple[float, float] | None
    label: str | None  # legend label
    band_index: int  # which band this is
    spin_index: int  # which spin channel
    additional_kwargs: dict[str, Any] = field(default_factory=dict)


class BandStructurePlotter:
    """Visualizer for band-structure arrays from a model layer.

    Parameters
    ----------
    figsize : tuple of float, optional
        Figure size (width, height). Default is (8, 6).
    dpi : int, optional
        Dots-per-inch for the figure. Default is 100.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure/axes are created.
    """

    def __init__(self, figsize: tuple[int, int] = (8, 6), dpi: int = 100, ax: Axes | None = None) -> None:
        self.figsize: tuple[int, int] = figsize
        self.dpi: int = dpi

        self.fig: Figure
        self.ax: Axes
        if ax is None:
            fig, created_ax = plt.subplots(figsize=figsize, dpi=dpi)
            assert isinstance(created_ax, Axes)
            self.fig = fig
            self.ax = created_ax
        else:
            self.fig = plt.gcf()
            self.ax = ax

        self.data_store: dict[str, Any] = {}
        self.values_dict: dict[str, Any] = {}
        self.cb: Colorbar | None = None
        self._legend_handles: list[mpatches.Patch] = []

        self.x: np.ndarray | None = None
        self.kpath: KPath | None = None

        # Phase 2/3: Attributes for Property-based plotting
        self._tick_positions: list[int] = []
        self._tick_names: list[str] = []
        self._k_distances: np.ndarray | None = None
        self.colorbar: Colorbar | None = None

    def _to_series_list(
        self,
        point_data: Any,
        scalars_data: Any,
        vectors_data: Any,
        channel_mode: str = "normal",
        **kwargs: object,
    ) -> list[BandSeries]:
        """Convert Property objects to list of BandSeries for plotting.

        Args:
            point_data: Property containing bands with kpath metadata
            scalars_data: Optional Property for scalar coloring
            vectors_data: Optional Property for vector arrows
            channel_mode: "normal" or "flip" for spin channel handling
            **kwargs: Additional kwargs to distribute to series

        Returns
        -------
            List of BandSeries, one per (band, spin) combination
        """
        # Extract kpath metadata
        kpath_meta = point_data.metadata.get("kpath", {})
        x_data = kpath_meta.get("k_distances")
        if x_data is None:
            raise ValueError("point_data must have kpath metadata with k_distances")

        # Extract bands array: shape (n_kpoints, n_bands, n_spins)
        bands = point_data.to_array()
        if bands.ndim == 2:
            bands = bands[:, :, np.newaxis]  # Add spin dimension if missing
        _, n_bands, n_spins = bands.shape

        # Extract scalars if provided
        scalars = scalars_data.to_array() if scalars_data is not None else None
        s_label = scalars_data.label if scalars_data else None
        s_unit = scalars_data.units if scalars_data else None
        s_lims_raw = getattr(scalars_data, "rounded_data_lim", None) if scalars_data else None

        # Compute global scalar limits per spin channel
        # Property.data_lim returns shape (n_spins, n_bands * 2) for 3D data
        # where first n_bands values are mins and last n_bands are maxs.
        # For 2D data (like DOS), shape is (n_spins, 2) with [:, 0]=min, [:, 1]=max.
        s_lims_per_spin: list[tuple[float, float] | None] = [None] * n_spins
        if s_lims_raw is not None:
            s_lims_arr = np.asarray(s_lims_raw)
            if s_lims_arr.ndim == 2 and s_lims_arr.shape[1] > 2:
                # Shape: (n_spins, n_bands * 2) -> compute global (min, max) per spin
                # First half are mins, second half are maxs
                half = s_lims_arr.shape[1] // 2
                for ispin in range(min(n_spins, s_lims_arr.shape[0])):
                    mins_per_band = s_lims_arr[ispin, :half]
                    maxs_per_band = s_lims_arr[ispin, half:]
                    global_min = float(np.min(mins_per_band))
                    global_max = float(np.max(maxs_per_band))
                    s_lims_per_spin[ispin] = (global_min, global_max)
            elif s_lims_arr.ndim == 2 and s_lims_arr.shape[1] == 2:
                # Shape: (n_spins, 2) -> use directly (min, max per spin)
                for ispin in range(min(n_spins, s_lims_arr.shape[0])):
                    s_lims_per_spin[ispin] = (float(s_lims_arr[ispin, 0]), float(s_lims_arr[ispin, 1]))

        # Extract vectors if provided
        vectors = vectors_data.to_array() if vectors_data is not None else None
        v_label = vectors_data.label if vectors_data else None
        v_unit = vectors_data.units if vectors_data else None
        v_lims_raw = getattr(vectors_data, "rounded_data_lim", None) if vectors_data else None

        # Compute global vector limits per spin channel (same logic as scalars)
        v_lims_per_spin: list[tuple[float, float] | None] = [None] * n_spins
        if v_lims_raw is not None:
            v_lims_arr = np.asarray(v_lims_raw)
            if v_lims_arr.ndim == 2 and v_lims_arr.shape[1] > 2:
                # Shape: (n_spins, n_bands * 2) -> compute global (min, max) per spin
                half = v_lims_arr.shape[1] // 2
                for ispin in range(min(n_spins, v_lims_arr.shape[0])):
                    mins_per_band = v_lims_arr[ispin, :half]
                    maxs_per_band = v_lims_arr[ispin, half:]
                    global_min = float(np.min(mins_per_band))
                    global_max = float(np.max(maxs_per_band))
                    v_lims_per_spin[ispin] = (global_min, global_max)
            elif v_lims_arr.ndim == 2 and v_lims_arr.shape[1] == 2:
                # Shape: (n_spins, 2) -> use directly
                for ispin in range(min(n_spins, v_lims_arr.shape[0])):
                    v_lims_per_spin[ispin] = (float(v_lims_arr[ispin, 0]), float(v_lims_arr[ispin, 1]))

        # Build kwargs per channel
        kwargs_per_channel = self._distribute_kwargs(kwargs, n_spins)

        # Build series list
        series_list: list[BandSeries] = []
        for iband in range(n_bands):
            for ispin in range(n_spins):
                y = bands[:, iband, ispin].copy()

                # Apply channel mode (flip second spin)
                if channel_mode == "flip" and ispin != 0:
                    y *= -1.0

                # Extract scalar slice for this band/spin
                s = None
                if scalars is not None:
                    if scalars.ndim == 3:
                        s = scalars[:, iband, ispin]
                    elif scalars.ndim == 2:
                        s = scalars[:, iband]
                    else:
                        s = scalars

                # Use pre-computed global limit for this spin channel
                s_lim = s_lims_per_spin[ispin]

                # Extract vector slice for this band/spin
                v = None
                if vectors is not None:
                    if vectors.ndim == 3:
                        v = vectors[:, iband, ispin]
                    elif vectors.ndim == 2:
                        v = vectors[:, iband]
                    else:
                        v = vectors

                # Use pre-computed global limit for this spin channel
                v_lim = v_lims_per_spin[ispin]

                # Build label
                label = self._build_series_label(point_data, iband, ispin, n_bands, n_spins)

                series_list.append(
                    BandSeries(
                        x=x_data,
                        y=y,
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
                        additional_kwargs=kwargs_per_channel[ispin],
                    )
                )

        return series_list

    def _distribute_kwargs(self, kwargs: dict[str, Any], n_channels: int) -> list[dict[str, Any]]:
        """Distribute kwargs to channels, handling list values."""
        kwargs_per_channel: list[dict[str, Any]] = []
        for i_channel in range(n_channels):
            channel_kwargs: dict[str, Any] = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(cast("list[Any]", value)) == n_channels:
                    channel_kwargs[key] = value[i_channel]
                else:
                    channel_kwargs[key] = value
            kwargs_per_channel.append(channel_kwargs)
        return kwargs_per_channel

    def _build_series_label(
        self,
        point_data: Any,
        iband: int,
        ispin: int,
        _n_bands: int,
        n_spins: int,
    ) -> str | None:
        """Build label for a single band series."""
        # Check for per-channel labels in metadata
        labels: list[str] | None = point_data.metadata.get("label")
        if labels and isinstance(labels, list) and len(labels) > ispin:
            return str(labels[ispin])

        # Default labeling
        if n_spins > 1:
            spin_label = "↑" if ispin == 0 else "↓"
            return f"Band {iband} {spin_label}"
        return None  # Don't label single-spin bands by default

    def plot(
        self,
        point_data: Any,
        scalars_data: Any | None = None,
        vectors_data: Any | None = None,
        scalars_mode: str = "none",  # "none", "scatter", "parametric"
        channel_mode: str = "normal",  # "normal", "flip"
        scalars_cmap: str = "plasma",
        scalars_clim: tuple[float, float] | None = None,
        scalars_show_colorbar: str = "single",  # "single", "none"
        line_kwargs: dict[str, Any] | None = None,
        scatter_kwargs: dict[str, Any] | None = None,
        collection_kwargs: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> dict[tuple[int, int], Any]:
        """Plot band structure from Property objects.

        This is the unified entry point for band structure plotting,
        following the same pattern as DOSPlotter.plot().

        Parameters
        ----------
        point_data : Property
            Property containing bands with kpath metadata.
            Must have metadata["kpath"] with k_distances, tick_positions, tick_names.
        scalars_data : Property, optional
            Property for scalar coloring (e.g., projections).
        vectors_data : Property, optional
            Property for vector arrows (e.g., spin texture).
        scalars_mode : str
            How to render scalar data:
            - "none": Plain line plot (ignore scalars_data)
            - "scatter": Scatter plot with scalar coloring
            - "parametric": LineCollection with segment coloring
        channel_mode : str
            How to handle spin channels:
            - "normal": All channels positive
            - "flip": Second channel negated (for spin visualization)
        scalars_cmap : str
            Colormap for scalar coloring.
        scalars_clim : tuple of float, optional
            Color limits for scalars. None = auto.
        scalars_show_colorbar : str
            "single" or "none".
        line_kwargs : dict, optional
            kwargs for line plots.
        scatter_kwargs : dict, optional
            kwargs for scatter plots.
        collection_kwargs : dict, optional
            kwargs for LineCollection.
        **kwargs
            Additional kwargs passed to all plot methods.

        Returns
        -------
        dict[tuple[int, int], Any]
            Dict mapping (band_index, spin_index) to matplotlib artists.
        """
        # Store kpath metadata for axis configuration
        kpath_meta = point_data.metadata.get("kpath", {})
        self._tick_positions = kpath_meta.get("tick_positions", [])
        self._tick_names = kpath_meta.get("tick_names", [])
        self._k_distances = kpath_meta.get("k_distances")

        # Convert Property objects to series list
        series_list = self._to_series_list(
            point_data, scalars_data, vectors_data, channel_mode, **kwargs
        )

        # Resolve color scaling across all series
        if scalars_data is not None and scalars_mode != "none":
            # Get global clim from series (uses rounded_data_lim via _to_series_list)
            if scalars_clim is not None:
                clim = scalars_clim
            else:
                # Compute global clim from all series scalars_lim (already rounded)
                all_lims = [s.scalars_lim for s in series_list if s.scalars_lim is not None]
                if all_lims:
                    clim = (
                        min(lim[0] for lim in all_lims),
                        max(lim[1] for lim in all_lims),
                    )
                else:
                    clim = self._resolve_clim(series_list, scalars_clim)
            cmap = scalars_cmap
        else:
            clim = None
            cmap = None

        # Validate scalars_mode early
        valid_modes = ("none", "scatter", "parametric")
        if scalars_mode not in valid_modes:
            raise ValueError(f"Unknown scalars_mode: {scalars_mode}. Must be one of {valid_modes}")

        # Plot each series
        artists: dict[tuple[int, int], Any] = {}
        for series in series_list:
            key = (series.band_index, series.spin_index)

            if scalars_mode == "none" or series.scalars is None:
                # Plain line plot
                artist = self._add_line(series, line_kwargs or {})
            elif scalars_mode == "scatter":
                # Scatter with scalar coloring (cmap/clim are set when scalars_mode != "none")
                assert cmap is not None and clim is not None
                artist = self._add_scatter(series, cmap, clim, scatter_kwargs or {})
            else:  # scalars_mode == "parametric"
                # LineCollection with segment coloring
                assert cmap is not None and clim is not None
                artist = self._add_line_collection(series, cmap, clim, collection_kwargs or {})

            artists[key] = artist

        # Store x data for axis methods
        if self._k_distances is not None:
            self.x = self._k_distances

        # Add colorbar if requested
        if scalars_mode != "none" and scalars_show_colorbar == "single" and clim is not None:
            assert cmap is not None  # cmap is set when scalars_mode != "none" and scalars_data exists
            self._add_colorbar(cmap, clim, scalars_data.label if scalars_data else None)

        # Draw vertical lines at high-symmetry points
        self._draw_high_symmetry_lines()

        # Record exportable data
        bands = point_data.to_array()
        if bands.ndim == 2:
            bands = bands[:, :, np.newaxis]
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                key_str = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[key_str] = bands[:, iband, ispin]
        if self._k_distances is not None:
            self._record_kpath_metadata_exports()

        # Configure axes (following dos_plot.py:325-334 pattern)
        self.set_xlim()
        self.set_ylim()
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()

        return artists

    def _resolve_clim(
        self,
        series_list: list[BandSeries],
        user_clim: tuple[float, float] | None,
    ) -> tuple[float, float]:
        """Resolve color limits from series data or user override."""
        if user_clim is not None:
            return user_clim

        # Compute global min/max from all series scalars
        all_scalars = [s.scalars for s in series_list if s.scalars is not None]
        if not all_scalars:
            return (0.0, 1.0)

        combined = np.concatenate([s.ravel() for s in all_scalars])
        finite = combined[np.isfinite(combined)]
        if len(finite) == 0:
            return (0.0, 1.0)

        return (float(finite.min()), float(finite.max()))

    def _add_line(self, series: BandSeries, line_kwargs: dict[str, Any]) -> Line2D:
        """Add a simple line plot for one band."""
        merged_kwargs: dict[str, Any] = {**series.additional_kwargs, **line_kwargs}
        lines = self.ax.plot(series.x, series.y, **merged_kwargs)
        return lines[0]

    def _add_scatter(
        self,
        series: BandSeries,
        cmap: str,
        clim: tuple[float, float],
        scatter_kwargs: dict[str, Any],
    ) -> PathCollection:
        """Add scatter plot with scalar coloring for one band."""
        merged_kwargs: dict[str, Any] = {**series.additional_kwargs, **scatter_kwargs}
        merged_kwargs.setdefault("s", 10)  # default marker size

        scatter = self.ax.scatter(
            series.x,
            series.y,
            c=series.scalars,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            **merged_kwargs,
        )
        return scatter

    def _add_line_collection(
        self,
        series: BandSeries,
        cmap: str,
        clim: tuple[float, float],
        collection_kwargs: dict[str, Any],
    ) -> LineCollection:
        """Add LineCollection with segment coloring for one band."""
        # Create segments from consecutive point pairs
        points = np.array([series.x, series.y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Use midpoint scalars for segment colors
        if series.scalars is not None:
            segment_scalars = (series.scalars[:-1] + series.scalars[1:]) / 2
        else:
            segment_scalars = None

        merged_kwargs: dict[str, Any] = {**series.additional_kwargs, **collection_kwargs}
        merged_kwargs.setdefault("linewidth", 2.0)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=plt.Normalize(clim[0], clim[1]),
            **merged_kwargs,
        )
        if segment_scalars is not None:
            lc.set_array(segment_scalars)

        self.ax.add_collection(lc)
        return lc

    def _add_colorbar(
        self,
        cmap: str,
        clim: tuple[float, float],
        label: str | None,
    ) -> None:
        """Add colorbar to the plot."""
        sm = cm.ScalarMappable(cmap=cmap, norm=mpcolors.Normalize(clim[0], clim[1]))
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax)
        self.colorbar = cbar
        if label:
            self.set_colorbar_label(label)

    def set_colorbar_label(self, label: str, rotation: int = 270, labelpad: int = 12, **kwargs: object) -> None:
        """Set colorbar label with proper orientation.

        Following dos_plot.py pattern, uses rotation=270 for top-to-bottom readability.
        """
        if self.colorbar is None:
            return
        self.colorbar.ax.set_ylabel(label, rotation=rotation, labelpad=labelpad, **kwargs)

    def _draw_high_symmetry_lines(self) -> None:
        """Draw vertical lines at high-symmetry k-points."""
        if self._k_distances is None or not self._tick_positions:
            return

        for pos in self._tick_positions:
            if 0 <= pos < len(self._k_distances):
                x = self._k_distances[pos]
                self.ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    def _record_kpath_metadata_exports(self) -> None:
        """Record kpath metadata for export."""
        if self._k_distances is None:
            return
        k_distances = self._k_distances
        self.values_dict["kpath_values"] = k_distances
        tick_names: list[str] = []
        for i, _x in enumerate(k_distances):
            name = ""
            for i_tick, pos in enumerate(self._tick_positions):
                if i == pos:
                    name = self._tick_names[i_tick] if i_tick < len(self._tick_names) else ""
                    break
            tick_names.append(name)
        self.values_dict["kpath_tick_names"] = tick_names

    # ---- Property wrapper helpers for legacy API compatibility ----

    def _wrap_as_property(self, kpath: KPath, bands: np.ndarray) -> Any:
        """Wrap arrays and KPath into a Property object.

        This helper enables the legacy array-based API to use the new
        Property-based plot() method internally.

        Parameters
        ----------
        kpath : KPath
            K-path object with high-symmetry point information.
        bands : np.ndarray
            Band energies with shape (n_kpoints, n_bands) or (n_kpoints, n_bands, n_spins).

        Returns
        -------
        Property
            Property containing bands with kpath metadata.
        """
        from pyprocar.core.property_store import Property

        # Ensure 3D shape
        if bands.ndim == 2:
            bands = bands[:, :, np.newaxis]

        kpath_metadata: dict[str, Any] = {
            "k_distances": kpath.get_distances(as_segments=False),
            "tick_positions": list(kpath.tick_positions),
            "tick_names": list(kpath.tick_names),
            "tick_names_latex": list(kpath.tick_names_latex),
        }

        return Property(
            name="bands",
            value=bands,
            units="eV",
            label="Energy",
            # cast: metadata contains ndarray values which don't match MetadataValue
            metadata=cast("dict[str, Any]", {"kpath": kpath_metadata}),
        )

    def _wrap_scalars_as_property(self, scalars: np.ndarray, label: str = "Projection") -> Any:
        """Wrap scalar array into a Property object.

        Parameters
        ----------
        scalars : np.ndarray
            Scalar data (e.g., projections) with shape matching bands.
        label : str, optional
            Label for the scalars. Default is "Projection".

        Returns
        -------
        Property
            Property containing scalar data.
        """
        from pyprocar.core.property_store import Property

        return Property(
            name="scalars",
            value=scalars,
            units="",
            label=label,
            metadata={},
        )

    def plot_plain(self, kpath: KPath, bands: np.ndarray, line_kwargs: dict[str, Any] | None = None, **kwargs: object) -> dict[tuple[int, int], Line2D]:
        """Plot plain band structure lines (legacy array-based interface).

        Parameters
        ----------
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        line_kwargs : dict, optional
            Keyword arguments forwarded to `Axes.plot` for line styling.
        **kwargs
            Additional matplotlib line keywords (fallback, merged under
            the hood; explicit `line_kwargs` takes precedence).
        """
        self.kpath = kpath
        self.x = np.asarray(kpath.get_distances(as_segments=False))

        bands, _, _ = self._validate_data(bands=bands)

        # Merge kwargs: generic kwargs as fallback, line_kwargs override
        merged_line_kwargs: dict[str, Any] = {}

        if line_kwargs:
            merged_line_kwargs.update(kwargs)
        # Preserve previous default linestyle if user didn't supply one
        if "linestyle" not in merged_line_kwargs:
            merged_line_kwargs["linestyle"] = "-"

        created_lines: dict[tuple[int, int], Line2D] = {}
        n_bands = bands.shape[1]
        n_spin_channels = bands.shape[-1]
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                ret = self.ax.plot(self.x, bands[:, iband, ispin], **merged_line_kwargs)
                created_lines[(iband, ispin)] = ret[0]

        self.set_xlim()
        ymin = float(bands.min())
        ymax = float(bands.max())
        elimit = (ymin, ymax)
        self.set_ylim(elimit)
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()

        # Record exportable data
        for ispin in range(n_spin_channels):
            for iband in range(bands.shape[1]):
                key = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[key] = bands[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_lines

    def plot_scatter(
        self,
        kpath: KPath,
        bands: np.ndarray,
        scalars: np.ndarray | None = None,
        scatter_kwargs: dict[str, Any] | None = None,
        cmap: str | mpcolors.Colormap = "plasma",
        norm: str | mpcolors.Normalize | type | None = "auto",
        _clim: tuple[float | None, float | None] | None = None,  # reserved for future use
        show_colorbar: bool | None = True,
        colorbar_kwargs: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> dict[tuple[int, int], Any]:
        """Plot band energies as scatter, optionally colored by `scalars`.

        Parameters
        ----------
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        scalars : ndarray, optional
            Colormap scalars with the same shape as `bands`.
        scatter_kwargs : dict, optional
            Keyword arguments forwarded to `Axes.scatter`.
        **kwargs
            Additional matplotlib scatter keywords (fallback, merged under
            the hood; explicit `scatter_kwargs` takes precedence).
        """
        self.kpath = kpath
        self.x = np.asarray(kpath.get_distances(as_segments=False))

        # Validate data
        bands, scalars, _ = self._validate_data(bands, scalars)

        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=cmap,
            norm=norm,
        )

        # Merge kwargs for scatter
        merged_scatter_kwargs: dict[str, Any] = {}
        if kwargs:
            merged_scatter_kwargs.update(kwargs)
        if scatter_kwargs:
            merged_scatter_kwargs.update(scatter_kwargs)

        merged_scatter_kwargs["norm"] = resolved_norm
        merged_scatter_kwargs["cmap"] = resolved_cmap

        created_collections: dict[tuple[int, int], PathCollection] = {}

        # Plot scatter
        assert self.x is not None
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin in range(n_spin_channels):
            data = None
            if scalars is not None:
                data = scalars[..., ispin]

            for iband in range(n_bands):
                y = bands[:, iband, ispin]
                c_vals = None if data is None else data[:, iband]
                coll = self.ax.scatter(self.x, y, c=c_vals, **merged_scatter_kwargs)
                created_collections[(iband, ispin)] = coll

        # Add colorbar if requested
        if scalars is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)

        self.set_xlim()
        ymin = float(bands.min())
        ymax = float(bands.max())
        elimit = (ymin, ymax)
        self.set_ylim(elimit)
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()

        # Record exportable data
        for ispin in range(n_spin_channels):
            for iband in range(bands.shape[1]):
                bkey = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[bkey] = bands[:, iband, ispin]
                if scalars is not None:
                    pkey = f"projections__scatter__band-{iband}_spinChannel-{ispin}"
                    self.values_dict[pkey] = scalars[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_collections

    def plot_parametric(
        self,
        kpath: KPath,
        bands: np.ndarray,
        scalars: np.ndarray | None = None,
        cmap: str | mpcolors.Colormap = "plasma",
        norm: str | mpcolors.Normalize | type | None = "auto",
        clim: tuple[float | None, float | None] | None = None,
        linewidth: float = 2.0,
        collection_kwargs: dict[str, Any] | None = None,
        show_colorbar: bool | None = True,
        colorbar_kwargs: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> dict[tuple[int, int], LineCollection]:
        """Plot parametric bands colored by `scalars` via LineCollection.

        Parameters
        ----------
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        scalars : ndarray, optional
            Colormap scalars with the same shape as `bands`.
        cmap : str, optional
            Colormap name. Default is "plasma".
        norm : matplotlib.colors.Normalize, optional
            Normalization instance or class. If class, built with `clim`.
        clim : tuple, optional
            (vmin, vmax). Defaults to (None, None).
        linewidth : float, optional
            Base linewidth for segments.
        collection_kwargs : dict, optional
            Keyword arguments forwarded to `LineCollection` construction.
        colorbar_kwargs : dict, optional
            Keyword arguments forwarded to `Figure.colorbar` when a
            colorbar is added.
        **kwargs
            Additional LineCollection kwargs (fallback, merged under the
            hood; explicit `collection_kwargs` takes precedence).
        """
        self.kpath = kpath
        self.x = np.asarray(kpath.get_distances(as_segments=False))

        # Validate data
        bands, scalars, _ = self._validate_data(bands=bands, scalars=scalars)

        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=cmap,
            norm=norm,
            clim=clim,
        )

        # Prepare data
        width_weights = np.ones_like(bands)
        mbands = np.ma.masked_array(bands, False)
        # if width_weights is not None:
        #     logger.info(f"___Applying width mask___")
        #     mbands = np.ma.masked_array(
        #         bands,
        #         np.abs(width_weights) < width_mask,
        #     )
        # if color_mask is not None:
        #     logger.info(f"___Applying color mask___")
        #     mbands = np.ma.masked_array(
        #         self.ebs.bands,
        #         np.abs(color_weights) < color_mask,
        #     )

        # Merge kwargs for LineCollection
        merged_collection_kwargs: dict[str, Any] = {}
        if kwargs:
            merged_collection_kwargs.update(kwargs)
        if collection_kwargs:
            merged_collection_kwargs.update(collection_kwargs)

        # Plot parametric bands
        assert self.x is not None
        created_collections: dict[tuple[int, int], LineCollection] = {}

        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin_channel in range(n_spin_channels):
            for iband in range(n_bands):
                points = np.array([self.x, mbands[:, iband, ispin_channel]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments, **merged_collection_kwargs)

                # Handle colors
                if scalars is not None:
                    lc.set_array(scalars[:, iband, ispin_channel])
                    lc.set_cmap(resolved_cmap)
                    lc.set_norm(resolved_norm)

                lc.set_linewidth(width_weights[:, iband, ispin_channel] * linewidth)
                self.ax.add_collection(lc)
                created_collections[(iband, ispin_channel)] = lc

        # Add colorbar if requested
        if scalars is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)

        # Set default plot parameters
        self.set_xlim()
        ymin = float(bands.min())
        ymax = float(bands.max())
        elimit: tuple[float, float] = (ymin, ymax)
        self.set_ylim(elimit)
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()

        # Record exportable data
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                bkey = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[bkey] = bands[:, iband, ispin]
                if scalars is not None:
                    pkey = f"projections__parametric__band-{iband}_spinChannel-{ispin}"
                    self.values_dict[pkey] = scalars[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_collections

    def plot_quiver(
        self,
        kpath: KPath,
        bands: np.ndarray,
        vectors: np.ndarray | None = None,
        skip: int = 1,
        angles: str = "uv",
        scale: float | None = None,
        scale_units: str = "inches",
        units: str = "inches",
        color: str | None = None,
        cmap: str | mpcolors.Colormap = "plasma",
        norm: str | mpcolors.Normalize | type | None = "auto",
        clim: tuple[float | None, float | None] | None = None,
        quiver_kwargs: dict[str, Any] | None = None,
        show_colorbar: bool | None = True,
        colorbar_kwargs: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> dict[tuple[int, int], object]:
        """Plot vector-valued data (e.g., velocities) as arrows along bands.

        Parameters
        ----------
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        vectors : ndarray
            Vector magnitudes aligned with `bands` (same shape). These values
            are drawn vertically; arrows are oriented horizontally with unit x.
        skip : int, optional
            Plot every `skip`-th point to reduce clutter. Default is 1.
        angles : str, optional
            Quiver angles mode. Default is 'uv'.
        scale, scale_units, units, color : optional
            Passed through to `Axes.quiver`.
        **kwargs
            Additional `Axes.quiver` kwargs.
        """
        if vectors is None:
            raise ValueError("vectors must be provided for plot_quiver")

        self.kpath = kpath
        self.x = np.asarray(kpath.get_distances(as_segments=False))

        # Validate data
        bands, _, vectors = self._validate_data(bands=bands, vectors=vectors)
        assert vectors is not None, "vectors unexpectedly None after validation"

        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=vectors,
            cmap=cmap,
            norm=norm,
            clim=clim,
        )

        # Merge kwargs and quiver_kwargs
        merged_collection_kwargs: dict[str, Any] = {}
        if kwargs:
            merged_collection_kwargs.update(kwargs)
        if quiver_kwargs:
            merged_collection_kwargs.update(quiver_kwargs)

        merged_collection_kwargs["norm"] = resolved_norm
        merged_collection_kwargs["cmap"] = resolved_cmap

        # Plot quivers
        created_quivers: dict[tuple[int, int], object] = {}
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin_channel in range(n_spin_channels):
            u = vectors[..., ispin_channel]  # Arrow y-component
            v = np.ones_like(vectors[..., ispin_channel])  # Arrow x-component
            vector_norms = vectors[..., ispin_channel]
            current_bands = bands[..., ispin_channel]

            for iband in range(n_bands):
                band_u = u[..., iband]
                band_v = v[..., iband]
                band_current_bands = current_bands[..., iband]

                quiver_args: list[np.ndarray] = []
                quiver_args.append(self.x[::skip])
                quiver_args.append(band_current_bands[::skip])
                quiver_args.append(band_u[::skip])
                quiver_args.append(band_v[::skip])
                if color is None:
                    quiver_args.append(vector_norms[..., iband])

                qv = self.ax.quiver(
                    *quiver_args,
                    angles=angles,
                    scale=scale,
                    scale_units=scale_units,
                    units=units,
                    color=color,
                    **merged_collection_kwargs,
                )
                created_quivers[(iband, ispin_channel)] = qv

        # Add colorbar if requested
        if show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)

        # Record exportable bands
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                key = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[key] = bands[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_quivers

    def plot_atomic_levels(
        self,
        bands: np.ndarray,
        elimit: tuple[float, float] | None = None,
        labels_prefix: str = "s",
        scalars: np.ndarray | None = None,
        cmap: str = "plasma",
        norm: mpcolors.Normalize | type | None = None,
        clim: tuple[float | None, float | None] = (None, None),
        show_colorbar: bool | None = True,
        colorbar_kwargs: dict[str, Any] | None = None,
        _linewidth: float | None = None,
        line_collection_kwargs: dict[str, Any] | None = None,
        show_text: bool = True,
    ) -> None:
        """Plot atomic-like energy levels for a single k-point input.

        This duplicates the single k-point to draw short horizontal segments
        and annotates them with band indices, attempting to avoid label overlap.
        When `scalars` is provided, segments are colored similarly to
        `plot_parametric` using a LineCollection and a colorbar is added.

        Parameters
        ----------
        bands : ndarray
            Energies with shape (1, n_bands, n_spins) or (1, n_bands).
        elimit : tuple of float, optional
            y-limits (ymin, ymax). If None, inferred from data.
        labels_prefix : str, optional
            Prefix for the spin label in text. Default is 's'.
        scalars : ndarray, optional
            Scalar values compatible with bands shape (1, n_bands, n_spins) or
            broadcastable; used to color each level segment.
        cmap : str, optional
            Colormap name. Default 'plasma'.
        norm : Normalize or type, optional
            Normalization instance or class; if class, constructed with `clim`.
        clim : tuple of float, optional
            (vmin, vmax) for normalization. Defaults to (None, None).
        linewidth : float, optional
            Line width for colored segments. Default 2.0.
        show_text : bool, optional
            Whether to draw text labels near levels. Default True.
        """
        # Fake 2-point x-axis for drawing horizontal segments
        self.kpath = None
        self.x = np.array([0.0, 1.0])

        # Validate data
        bands, scalars, _ = self._validate_data(bands=bands, scalars=scalars)
        if bands.shape[0] != 1:
            raise ValueError("plot_atomic_levels requires a single k-point (n_k=1)")

        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=cmap,
            norm=norm,
            clim=clim,
        )

        # Merge kwargs and line_collection_kwargs
        merged_collection_kwargs: dict[str, Any] = {}
        if line_collection_kwargs:
            merged_collection_kwargs.update(line_collection_kwargs)
        merged_collection_kwargs["norm"] = resolved_norm
        merged_collection_kwargs["cmap"] = resolved_cmap

        # Remove x ticks for atomic levels and compute text bbox in data units
        self.ax.xaxis.set_major_locator(plt.NullLocator())

        # Determine a representative text bbox in data coordinates
        n_bands = bands.shape[1]
        sample_text = f"{labels_prefix}-0 : b-{n_bands}"
        tmp_txt = self.ax.text(self.x[0], float(bands.min()), sample_text)
        try:
            bbox = tmp_txt.get_window_extent()
            bbox_data = self.ax.transData.inverted().transform_bbox(bbox)
            w, h = bbox_data.width, bbox_data.height
        except Exception as e:
            logger.error(f"Error getting text bbox: {e}")
            # Fallback small sizes if renderer not ready
            w, h = 0.05, 0.05
        tmp_txt.remove()

        # Ensure enough x-range to accommodate lateral shifts and keep labels inside
        x_base = self.x[0] + 0.2 * w
        self.set_xlim((self.x[0], self.x[-1]))

        # Plot atomic levels
        # Sort energies to manage label overlap; alternate lateral shifts based on bbox h
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin in range(n_spin_channels):
            energies = bands[0, :, ispin]
            # order = np.argsort(energies)
            last_y = None
            shift_state = 0  # 0: first column near left edge, 1: second column to the right
            for iband in range(n_bands):
                y = float(energies[iband])

                pts = np.array([[self.x[0], y], [self.x[1], y]])
                segments = np.array([pts])
                lc = LineCollection(segments, **merged_collection_kwargs)
                if scalars is not None:
                    level_scalar = float(np.asarray(scalars[0, iband, ispin]))
                    lc.set_array(np.array([level_scalar]))
                self.ax.add_collection(lc)

                if show_text:
                    # if vertical overlap, toggle lateral shift
                    if last_y is not None and y < (last_y + h):
                        shift_state = 1 - shift_state
                    else:
                        shift_state = 0
                    x_pos = x_base + (2.0 * w if shift_state == 1 else 0.0)
                    # Clamp inside current xlim
                    xmin, xmax = self.ax.get_xlim()
                    x_pos = min(max(x_pos, xmin + 0.05 * w), xmax - 0.05 * w)
                    self.ax.text(x_pos, y, f"{labels_prefix}-{ispin} : b-{iband + 1}")
                    last_y = y

        # Add colorbar if requested
        if scalars is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)

        # Determine limits
        if elimit is None:
            ymin = float(bands.min())
            ymax = float(bands.max())
            elimit = (ymin, ymax)
        self.set_ylim(elimit)

        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()

        # Export
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                key = f"bands__band-{iband}_spinChannel-{ispin}"
                level = float(bands[0, iband, ispin])
                self.values_dict[key] = np.array([level, level])
        self.values_dict["kpath_values"] = self.x
        self.values_dict["kpath_tick_names"] = ["", ""]

    def plot_overlay(
        self,
        kpath: KPath,
        bands: np.ndarray,
        weights: list[np.ndarray],
        labels: list[str] | None = None,
        colors: list[Any] | None = None,
        norm: str | mpcolors.Normalize | type | None = "auto",
        clim: tuple[float | None, float | None] | None = None,
        _cmap: str | mpcolors.Colormap = "plasma",
        cmap_list: list[str] | None = None,
        fill_alpha: float = 0.3,
        linewidth: float = 1.0,
        fill_between_kwargs: dict[str, Any] | None = None,
        show_colorbar: bool | None = None,
        colorbar_kwargs: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        """Plot overlays by filling band envelopes using provided weights.

        Parameters
        ----------
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        weights : list of ndarray
            Each weight array must match `bands` shape; it defines a thickness
            around each band: [E - w/2, E + w/2] is filled.
        colors : list of str, optional
            Colors per weight set. If None, matplotlib cycles colors.
        fill_alpha : float, optional
            Alpha for filled region. Default is 0.3.
        edge_alpha : float, optional
            Alpha for the edges. Default is 0.8.
        linewidth : float, optional
            Edge line width. Default is 1.0.
        **kwargs
            Additional `Axes.fill_between` kwargs.
        """
        self.kpath = kpath
        self.x = np.asarray(kpath.get_distances(as_segments=False))

        # Validate data
        bands, _, _ = self._validate_data(bands=bands)
        for i, weight in enumerate(weights):
            _, validated_weight, _ = self._validate_data(bands=bands, scalars=weight)
            if validated_weight is not None:
                weights[i] = validated_weight

        # Default colormap names per overlay, fallback if explicit colors not given
        if cmap_list is None:
            cmap_list = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
        use_colormaps = colors is None
        if colors is None:
            # Sample mid tone from each colormap to derive RGBA colors
            colors = []
            for i in range(len(weights)):
                cmap_name = cmap_list[i % len(cmap_list)]
                cmap_obj = plt.get_cmap(cmap_name)
                colors.append(cmap_obj(0.7))

        cmaps: list[Any] = []
        norms: list[Any] = []
        scalar_mappables: list[Any] = []
        if use_colormaps:
            for i in range(len(weights)):
                resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
                    data=weights[i],
                    cmap=cmap_list[i % len(cmap_list)],
                    norm=norm,
                    clim=clim,
                )
                cmaps.append(resolved_cmap)
                norms.append(resolved_norm)
                scalar_mappables.append(scalar_mappable)

        # Merge kwargs and fill_between_kwargs
        merged_collection_kwargs: dict[str, Any] = {}
        if kwargs:
            merged_collection_kwargs.update(kwargs)
        if fill_between_kwargs:
            merged_collection_kwargs.update(fill_between_kwargs)

        merged_collection_kwargs["alpha"] = fill_alpha
        merged_collection_kwargs["linewidth"] = linewidth

        # Iterate over weights and fill between
        legend_handles: list[mpatches.Patch] = []
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for widx, w in enumerate(weights):
            if w.ndim == 2:
                w = w[..., np.newaxis]
            if w.shape != bands.shape:
                raise ValueError("Each weight must have the same shape as bands")
            color = colors[widx]
            for ispin in range(n_spin_channels):
                for iband in range(n_bands):
                    y = bands[:, iband, ispin]
                    width_arr = w[:, iband, ispin]

                    fill_cmap: Any = cmaps[widx] if use_colormaps and cmaps else None
                    fill_norm: Any = norms[widx] if use_colormaps and norms else None

                    self.ax.fill_between(
                        self.x,
                        y - width_arr / 2.0,
                        y + width_arr / 2.0,
                        color=color,
                        cmap=fill_cmap,
                        norm=fill_norm,
                        **merged_collection_kwargs,
                    )
            legend_label = labels[widx] if labels and widx < len(labels) else f"overlay-{widx + 1}"
            legend_handles.append(mpatches.Patch(color=color, label=legend_label, alpha=fill_alpha))

        # Add colorbar if requested
        if use_colormaps and show_colorbar and scalar_mappables:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappables[0], ax=self.ax, **colorbar_kwargs)

        # Export
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                key = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[key] = bands[:, iband, ispin]
        self._record_kpath_exports(kpath)
        self._legend_handles = legend_handles

        self.set_xlim()
        self.set_ylim()
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()
        self.legend()

    def set_xlim(self, xlim: tuple[float, float] | list[float] | None = None, **kwargs: Any) -> None:
        if xlim is None:
            assert self.x is not None, "x not initialized; call a plotting method first"
            xlim = (float(self.x[0]), float(self.x[-1]))

        self.ax.set_xlim(xlim, **kwargs)

    def set_ylim(self, ylim: tuple[float, float] | list[float] | None = None, **kwargs: Any) -> None:
        """Set y-axis limits, inferring from recorded bands if not provided."""
        if ylim is None:
            bands_cols = [v for k, v in self.values_dict.items() if k.startswith("bands__")]
            if bands_cols:
                all_vals = np.concatenate([np.atleast_1d(v).ravel() for v in bands_cols])
                ymin = float(all_vals.min())
                ymax = float(all_vals.max())
                pad = 0.1 * max(abs(ymin), abs(ymax))
                ylim = (ymin - pad, ymax + pad)
            else:
                raise ValueError("Cannot infer ylim; pass explicit limits or plot first")
        self.ax.set_ylim(ylim, **kwargs)

    def set_xticks(
        self, tick_positions: list[int] | None = None, tick_names: list[str] | None = None, color: str = "black"
    ) -> None:
        """Set high-symmetry tick marks and labels using the current k-path.

        Parameters
        ----------
        tick_positions : list of int, optional
            Indices into the k-path where separator lines and ticks are placed.
            If None and a `KPath` was used, defaults to `kpath.tick_positions`.
        tick_names : list of str, optional
            Labels for the ticks. If None and a `KPath` was used, defaults to
            `kpath.tick_names`.
        color : str, optional
            Color of the vertical separator lines.
        """
        if self.x is None:
            raise ValueError("x not initialized; call a plotting method first")
        x = self.x
        # First try self.kpath (legacy methods), then self._tick_positions (new plot() method)
        if tick_positions is None and hasattr(self, "kpath") and self.kpath is not None:
            tick_positions = self.kpath.tick_positions
        elif tick_positions is None and self._tick_positions:
            tick_positions = self._tick_positions
        if tick_names is None and hasattr(self, "kpath") and self.kpath is not None:
            tick_names = self.kpath.tick_names
        elif tick_names is None and self._tick_names:
            tick_names = self._tick_names

        if tick_positions is not None:
            for ipos in tick_positions:
                if 0 <= ipos < len(x):
                    self.ax.axvline(x[ipos], color=color)
            self.ax.set_xticks(x[tick_positions])
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)

    def set_yticks(self, major: float | None = None, minor: float | None = None, interval: tuple[float, float] | list[float] | None = None) -> None:
        """Set y-axis tick locators using heuristics if not provided.

        Parameters
        ----------
        major : float, optional
            Spacing for major ticks.
        minor : float, optional
            Spacing for minor ticks.
        interval : list of float, optional
            Range (ymin, ymax) used to infer sensible tick spacing.
        """
        if interval is None:
            bands_cols = [v for k, v in self.values_dict.items() if k.startswith("bands__")]
            if bands_cols:
                all_vals = np.concatenate([np.atleast_1d(v).ravel() for v in bands_cols])
                interval = (float(all_vals.min()), float(all_vals.max()))
            else:
                interval = (-10.0, 10.0)
        width = abs(interval[1] - interval[0])
        if major is None or minor is None:
            if 20 <= width < 30:
                major, minor = 5, 1
            elif 10 <= width < 20:
                major, minor = 4, 0.5
            elif 5 <= width < 10:
                major, minor = 2, 0.2
            elif 3 <= width < 5:
                major, minor = 1, 0.1
            elif 1 <= width < 3:
                major, minor = 0.5, 0.1
        if major is not None:
            self.ax.yaxis.set_major_locator(MultipleLocator(major))
        if minor is not None:
            self.ax.yaxis.set_minor_locator(MultipleLocator(minor))

    def set_xlabel(self, label: str = "K vector", **kwargs: object) -> None:
        """Set x-axis label.

        Parameters
        ----------
        label : str, optional
            Axis label.
        """
        self.ax.set_xlabel(label, **kwargs)

    def set_ylabel(self, label: str = r"E (eV)", **kwargs: object) -> None:
        """Set y-axis label.

        Parameters
        ----------
        label : str, optional
            Axis label.
        """
        self.ax.set_ylabel(label, **kwargs)

    def set_title(self, title: str = "Band Structure", **kwargs: object) -> None:
        """Set plot title."""
        self.ax.set_title(title, **kwargs)

    def set_colorbar_title(self, title: str = "Atomic Orbital Projections", **kwargs: Any) -> None:
        """Set colorbar title if a colorbar exists."""
        if self.cb is not None:
            self.cb.ax.tick_params(labelsize=kwargs.pop("labelsize", None))
            self.cb.set_label(title, **kwargs)

    def draw_fermi(
        self,
        fermi_level: float = 0.0,
        color: str = "k",
        linestyle: str = "--",
        linewidth: float = 1.0,
    ) -> None:
        """Draw a horizontal Fermi level line."""
        self.ax.axhline(y=fermi_level, color=color, linestyle=linestyle, linewidth=linewidth)

    def grid(
        self,
        enabled: bool = True,
        which: str = "both",
        color: str = "#cccccc",
        linestyle: str = ":",
        linewidth: float = 0.8,
    ) -> None:
        """Configure grid display."""
        if enabled:
            self.ax.grid(
                enabled, which=which, color=color, linestyle=linestyle, linewidth=linewidth
            )

    def legend(self, labels: list[str] | None = None, **kwargs: Any) -> None:
        """Show legend; uses stored handles when available.

        Parameters
        ----------
        labels : list of str, optional
            If provided and stored legend handles exist, these labels will
            override the handle labels.
        """
        if self._legend_handles:
            if labels is not None and len(labels) == len(self._legend_handles):
                for h, lab in zip(self._legend_handles, labels):
                    h.set_label(lab)
            self.ax.legend(handles=self._legend_handles, **kwargs)
        else:
            self.ax.legend(labels, **kwargs)

    def save(self, filename: str = "bands.pdf", dpi: int | None = None, bbox_inches: str = "tight") -> None:
        """Save the current figure to disk."""
        plt.savefig(filename, dpi=(dpi or self.dpi), bbox_inches=bbox_inches)
        plt.clf()

    def export_data(self, filename: str) -> None:
        """Export recorded plot arrays to CSV/TXT/JSON/DAT.

        Parameters
        ----------
        filename : str
            Output path; extension defines format.
        """
        possible_file_types = ["csv", "txt", "json", "dat"]
        file_type = filename.split(".")[-1]
        if file_type not in possible_file_types:
            raise ValueError(f"The file type must be {possible_file_types}")
        if not self.values_dict:
            raise ValueError("No values recorded. Plot first before exporting.")

        values: dict[str, np.ndarray] = {}
        for key, value in self.values_dict.items():
            if value is None:
                continue
            arr = np.atleast_1d(value)
            if arr.size > 0:
                values[key] = arr

        column_names: list[str] = list(values.keys())
        sorted_columns: list[str] = []
        for key in ["kpath_values", "kpath_tick_names", "k_current"]:
            if key in column_names:
                sorted_columns.append(key)
        for ispin in range(2):
            for name in column_names:
                if name.startswith("bands__") and name.endswith(f"spinChannel-{ispin}"):
                    sorted_columns.append(name)
        for name in sorted(column_names):
            if name not in sorted_columns:
                sorted_columns.append(name)

        if file_type in ["csv", "txt", "dat"]:
            df = pd.DataFrame(values)
            if file_type == "csv":
                df.to_csv(filename, columns=sorted_columns, index=False)
            elif file_type == "txt":
                df.to_csv(filename, columns=sorted_columns, sep="\t", index=False)
            else:
                df.to_csv(filename, columns=sorted_columns, sep=" ", index=False)
        else:
            serializable: dict[str, Any] = {k: np.asarray(v).tolist() for k, v in values.items()}
            with open(filename, "w") as outfile:
                json.dump(serializable, outfile)

    # ---- helpers ----
    def _record_kpath_exports(self, kpath: KPath) -> None:
        self.values_dict["kpath_values"] = self.x
        tick_names: list[str] = []
        if self.x is not None:
            for i, _x in enumerate(self.x):
                name = ""
                for i_tick, pos in enumerate(kpath.tick_positions):
                    if i == pos:
                        name = kpath.tick_names[i_tick]
                        break
                tick_names.append(name)
        self.values_dict["kpath_tick_names"] = tick_names

    def show(self) -> None:
        plt.show()

    def _validate_data(
        self,
        bands: np.ndarray,
        scalars: np.ndarray | None = None,
        vectors: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        if bands.ndim == 2:
            bands = bands[..., np.newaxis]
        n_spin_channels = bands.shape[-1]

        if scalars is not None and scalars.ndim == 2:
            scalars = scalars[..., np.newaxis]
        if scalars is not None and scalars.ndim != bands.ndim:
            error_message = "scalars must have the same number of dimensions as bands"
            error_message += (
                f"bands has {bands.ndim} dimensions, scalars has {scalars.ndim} dimensions"
            )
            error_message += (
                "Use a built in method in ElectronicBandStructurePath to get the scalars"
            )
            raise ValueError(error_message)
        if (
            scalars is not None
            and scalars.ndim == bands.ndim
            and scalars.shape[-1] != n_spin_channels
        ):
            error_message = "scalars must have the same number of spin channels as bands."
            error_message += f"bands has {n_spin_channels} spin channels, scalars has {scalars.shape[-1]} spin channels\n"
            error_message += "This error is likely due to a non-colinear calculation where the scalars can have spin components\n"
            raise ValueError(error_message)

        if vectors is not None and vectors.ndim == 2:
            vectors = vectors[..., np.newaxis]
        if vectors is not None and vectors.ndim != bands.ndim:
            error_message = "vectors must have the same number of dimensions as bands"
            error_message += (
                f"bands has {bands.ndim} dimensions, vectors has {vectors.ndim} dimensions"
            )
            error_message += (
                "Use a built in method in ElectronicBandStructurePath to get the scalars"
            )
            raise ValueError(error_message)
        if (
            vectors is not None
            and vectors.ndim == bands.ndim
            and vectors.shape[-1] != n_spin_channels
        ):
            error_message = "vectors must have the same number of spin channels as bands."
            error_message += f"bands has {n_spin_channels} spin channels, vectors has {vectors.shape[1]} spin channels\n"
            error_message += "This error is likely due to a non-colinear calculation where the vectors can have spin components\n"
            raise ValueError(error_message)

        return bands, scalars, vectors

    # ---- color mapping resolver ----
    def _resolve_colormap(
        self,
        data: np.ndarray | None,
        cmap: str | mpcolors.Colormap = "plasma",
        norm: str | mpcolors.Normalize | type | None = "auto",
        clim: tuple[float | None, float | None] | None = None,
    ) -> tuple[mpcolors.Normalize | None, str | mpcolors.Colormap | None, cm.ScalarMappable | None]:
        """Resolve a Normalize and Colormap for the given data.

        - norm can be:
          - 'auto' or None: build Normalize using data/clim
          - a Normalize instance: use as is
          - a Normalize subclass: instantiate with vmin/vmax from data/clim
        - clim can be (vmin, vmax) with Nones to fill from data
        - If vmin/vmax are inferred from data, snap to visually pleasant bounds:
          prefer +/-0.5 when close, otherwise expand to integer floor/ceil.
        Returns (norm_obj, cmap_obj, scalar_mappable)
        """
        if data is None:
            return None, None, None

        vmin: float | None = None
        vmax: float | None = None
        if clim is not None:
            vmin, vmax = clim
        # Compute from data if needed
        data_min: float | None = None
        data_max: float | None = None
        try:
            data_min = float(np.nanmin(np.asarray(data)))
            data_max = float(np.nanmax(np.asarray(data)))
        except Exception:
            data_min, data_max = None, None
        # Only infer (and possibly snap) when not explicitly provided via
        # `clim`.
        infer_vmin = vmin is None
        infer_vmax = vmax is None
        if infer_vmin:
            vmin = data_min
        if infer_vmax:
            vmax = data_max

        # Snap heuristics: prefer +/-0.5 when near; otherwise use
        # integer floor/ceil to create clean colorbar bounds.
        half_tol = 0.05  # how close to 0.5/-0.5 to snap
        int_pad = 0.0  # extra pad after floor/ceil

        def _snap_min(value: float) -> float:
            if np.isfinite(value):
                if abs(value + 0.5) <= half_tol:
                    return -0.5
                if abs(value - 0.5) <= half_tol:
                    return 0.5
                return float(np.floor(value - int_pad))
            return value

        def _snap_max(value: float) -> float:
            if np.isfinite(value):
                if abs(value - 0.5) <= half_tol:
                    return 0.5
                if abs(value + 0.5) <= half_tol:
                    return -0.5
                return float(np.ceil(value + int_pad))
            return value

        if infer_vmin and vmin is not None:
            vmin = _snap_min(vmin)
        if infer_vmax and vmax is not None:
            vmax = _snap_max(vmax)

        # Ensure vmin < vmax; if equal after snapping, expand slightly
        if vmin is not None and vmax is not None and vmin >= vmax:
            eps = 1e-8
            if infer_vmin:
                vmin = vmin - eps
            else:
                vmax = vmax + eps

        # Resolve Normalize
        norm_obj: mpcolors.Normalize
        if isinstance(norm, mpcolors.Normalize):
            norm_obj = norm
        elif isinstance(norm, type) and issubclass(norm, mpcolors.Normalize):
            norm_obj = norm(vmin=vmin, vmax=vmax)
        else:
            norm_obj = mpcolors.Normalize(vmin=vmin, vmax=vmax)

        # Resolve cmap
        cmap_obj: str | mpcolors.Colormap = cmap
        # Build a ScalarMappable for colorbar convenience
        scalar_mappable = cm.ScalarMappable(norm=norm_obj, cmap=cmap_obj)
        scalar_mappable.set_array(np.asarray(data).ravel())

        return norm_obj, cmap_obj, scalar_mappable
