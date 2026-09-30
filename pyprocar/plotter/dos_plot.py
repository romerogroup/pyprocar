"""Shared utilities for density of states plotting backends."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import patches
from matplotlib.collections import LineCollection

from pyprocar.core.property_store import Property
from pyprocar.plotter._series import (
    ShowColorbar,
    channel_lims,
    line_series,
    resolve_clim,
)
from pyprocar.utils.func_utils import (
    keep_func_kwargs,
)

logger = logging.getLogger(__name__)


class ScalarsMode(Enum):
    LINE = "line"
    FILL = "fill"

    @classmethod
    def from_string(cls, string: str) -> ScalarsMode:
        if string == "line":
            return cls.LINE
        elif string == "fill":
            return cls.FILL
        else:
            raise ValueError(f"Invalid scalars mode: {string}")


class AxesOrientation(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

    @classmethod
    def from_string(cls, string: str | AxesOrientation) -> AxesOrientation:
        if isinstance(string, AxesOrientation):
            return string
        lower_string = string.lower()
        if lower_string[0] == "h":
            return cls.HORIZONTAL
        elif lower_string[0] == "v":
            return cls.VERTICAL
        else:
            raise ValueError(f"Invalid axes orientation: {string}")


class Axis(Enum):
    X = "x"
    Y = "y"
    BOTH = "both"

    @classmethod
    def from_string(cls, string: str) -> Axis:
        if string == "x":
            return cls.X
        elif string == "y":
            return cls.Y
        elif string == "both":
            return cls.BOTH
        else:
            raise ValueError(f"Invalid axis: {string}")


@dataclass
class DOSPlotter:
    """Lightweight wrapper around a matplotlib axis for DOS plots."""

    orientation: str = AxesOrientation.HORIZONTAL
    figsize: tuple[int, int] = (6, 4)
    dpi: int = 100
    ax: plt.Axes | None = None
    dos_lim: tuple[float, float] = None
    energy_lim: tuple[float, float] = None
    _handles: list[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.ax is None:
            self._fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            self.ax = self.ax
            self._fig = self.ax.get_figure()

        self.orientation = AxesOrientation.from_string(self.orientation)

    @property
    def fig(self) -> plt.Figure:
        return self._fig

    @property
    def colorbar(self) -> cm.ScalarMappable:
        if not hasattr(self, "_cb"):
            return None
        return self._cb

    @property
    def colorbar_axes(self) -> plt.Axes:
        return self.colorbar.ax

    @property
    def colorbar_orientation(self) -> str | None:
        if not hasattr(self, "_cb_orientation"):
            return None
        self._cb_orientation = AxesOrientation.from_string(self._cb_orientation)
        return self._cb_orientation

    @property
    def colorbar_location(self) -> str | None:
        if not hasattr(self, "_cb_location"):
            return None
        return self._cb_location

    # ------------------------------------------------------------------
    # High level orchestration
    # ------------------------------------------------------------------

    def plot(
        self,
        point_data: Property,
        scalars_data: Property | None = None,
        vectors_data: Property | None = None,
        scalars_mode: str = "line",
        channel_mode: str = "flip",
        plot_total: bool = True,
        vectors_cmap: str | mcolors.Colormap = "plasma",
        vectors_norm: str | mcolors.Normalize = None,
        vectors_clim: tuple[float | None, float | None] | None = None,
        vectors_show_colorbar: ShowColorbar | str = ShowColorbar.NONE,
        scalars_cmap: str | mcolors.Colormap = "plasma",
        scalars_norm: str | mcolors.Normalize = None,
        scalars_clim: tuple[float | None, float | None] | None = None,
        scalars_show_colorbar: ShowColorbar | str = ShowColorbar.SINGLE,
        plot_kwargs: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        scalars_mode = ScalarsMode.from_string(scalars_mode)

        x, y = self.orient_data(point_data.points, point_data.to_array())
        y = _as_channels(y)
        n_channels = y.shape[2]
        channel_labels = point_data.metadata.get("label")
        series_list = line_series(
            x,
            y,
            _as_channels(scalars_data.to_array()) if scalars_data is not None else None,
            _as_channels(vectors_data.to_array()) if vectors_data is not None else None,
            channel_mode,
            kwargs,
            lambda _, c: channel_labels[c] if channel_labels else point_data.label,
        )

        cmap_s, norm_s, clim_s, scalars_show_colorbar = _resolve_scaling(
            [s.scalars for s in series_list],
            channel_lims(getattr(scalars_data, "rounded_data_lim", None), n_channels),
            show_colorbar=scalars_show_colorbar,
            cmap=scalars_cmap,
            norm=scalars_norm,
            clim=scalars_clim,
        )
        cmap_v, norm_v, clim_v, vectors_show_colorbar = _resolve_scaling(
            [s.vectors for s in series_list],
            channel_lims(getattr(vectors_data, "rounded_data_lim", None), n_channels),
            show_colorbar=vectors_show_colorbar,
            cmap=vectors_cmap,
            norm=vectors_norm,
            clim=vectors_clim,
        )

        xlim = (0, 0)
        ylim = (0, 0)
        for i_channel, series in enumerate(series_list):
            xlim = (min(xlim[0], series.x.min()), max(xlim[1], series.x.max()))
            ylim = (min(ylim[0], series.y.min()), max(ylim[1], series.y.max()))

            if plot_kwargs is not None:
                plot_kwargs_channel = plot_kwargs[i_channel]
            else:
                plot_kwargs_channel = {}

            add_scalar_args = {
                "x": series.x,
                "y": series.y,
                "scalars": series.scalars,
                "label": scalars_data.label if scalars_data else None,
                "clim": clim_s[i_channel],
                "cmap": cmap_s[i_channel],
                "norm": norm_s[i_channel],
            }
            add_scalar_args.update(plot_kwargs_channel)
            add_line_args = {
                "x": series.x,
                "y": series.y,
                "label": series.label,
            }
            add_line_args.update(plot_kwargs_channel)
            add_vectors_args = {
                "x": series.x,
                "y": series.y,
                "vectors": series.vectors,
                "label": vectors_data.label if vectors_data else None,
                "clim": clim_v[i_channel],
                "norm": norm_v[i_channel],
                "cmap": cmap_v[i_channel],
            }
            add_vectors_args.update(plot_kwargs_channel)

            # add_scalar_args.update(series.additional_kwargs)
            if scalars_data and scalars_mode == ScalarsMode.LINE:
                artist = self.add_scalar_line(**add_scalar_args, **series.kwargs)
            elif scalars_data and scalars_mode == ScalarsMode.FILL:
                add_scalar_args["plot_total"] = plot_total
                artist = self.add_scalar_fill(**add_scalar_args, **series.kwargs)
            else:
                add_line_args
                artist = self.add_line(**add_line_args, **series.kwargs)

            if vectors_data:
                self.add_vectors(**add_vectors_args, **series.kwargs)

        if scalars_data and scalars_show_colorbar is not ShowColorbar.NONE:
            lab = scalars_data.label
            if scalars_data.units:
                lab = f"{lab} ({scalars_data.units})"
            n_bars = 1 if scalars_show_colorbar is ShowColorbar.SINGLE else len(series_list)
            for i in range(n_bars):
                self.plot_colorbar(label=lab, cmap=cmap_s[i], norm=norm_s[i])

        # vectors colorbar is only shown if you chose to
        # if vectors_data and vectors_show_colorbar is not ShowColorbar.NONE:
        #     # If color_src is SCALARS and scal_show already drew a colorbar, you may want to skip here.
        #     if not (VectorColorSource.SCALARS and scalars_show_colorbar is not ShowColorbar.NONE):
        #         # Draw per policy
        #         vlabel = series_list[0].vectors_label if vectors_show_colorbar is ShowColorbar.SINGLE else None
        #         if vectors_show_colorbar is ShowColorbar.SINGLE:
        #             self.plot_colorbar(label=vlabel or "", cmap=cmap_v[0], norm=norm_v[0])
        #         else:
        #             for i, s in enumerate(series_list):
        #                 self.plot_colorbar(label=s.vectors_label or "", cmap=cmap_v[i], norm=norm_v[i])

        self.set_energy_label(point_data.points_label, unit_label=point_data.points_units)
        self.set_energy_tick_params()

        self.set_energy_label(point_data.points_label, unit_label=point_data.points_units)
        self.set_dos_label(point_data.label, unit_label=point_data.units)
        self.set_ylim(ylim)
        self.set_xlim(xlim)
        self.set_dos_tick_params()

        self.draw_baseline(value=0.0)

    def add_line(self, x: np.ndarray, y: np.ndarray, label: str | None = None, **kwargs):
        handle = self.ax.plot(x, y, label=label, **keep_func_kwargs(kwargs, self.ax.plot))
        return handle

    def add_scalar_line(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scalars: np.ndarray,
        label: str | None = None,
        clim: tuple[float | None, float | None] | None = None,
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        linewidth: float = 1.5,
        linestyle: str = "-",
        alpha: float = 1.0,
        **kwargs,
    ):
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            array=scalars,
            label=label,
            clim=clim,
            cmap=cmap,
            norm=norm,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            **keep_func_kwargs(kwargs, LineCollection),
        )
        handle = self.ax.add_collection(lc)
        return handle

    def add_scalar_fill(
        self,
        x: np.ndarray,
        y: np.ndarray,
        scalars: np.ndarray,
        label: str | None = None,
        clim: tuple[float | None, float | None] | None = None,
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        baseline: float | None = 0.0,
        **kwargs,
    ):
        im = self.fill_between_image(
            x,
            y,
            scalars,
            orientation=self.orientation,
            baseline=baseline,
            label=label,
            cmap=cmap,
            norm=norm,
            clim=clim,
            **kwargs,
        )
        return im

    def add_vectors(
        self,
        x: np.ndarray,
        y: np.ndarray,
        vectors: np.ndarray,
        label: str | None = None,
        skip: int = 1,
        angles: str = "uv",
        scale: float = 100.0,
        scale_units: str = "inches",
        units: str = "inches",
        color=None,
        clim: tuple[float | None, float | None] | None = None,
        norm: mcolors.Normalize | str | None = None,
        cmap: str | mcolors.Colormap = "plasma",
        **kwargs,
    ):
        u = vectors  # Arrow x-component
        v = np.zeros_like(vectors)  # Arrow y-component
        vector_norms = vectors

        quiver_args = []
        quiver_args.append(x[::skip])
        quiver_args.append(y[::skip])
        quiver_args.append(u[::skip])
        quiver_args.append(v[::skip])
        if color is None:
            quiver_args.append(vector_norms[::skip])

        qv = self.ax.quiver(
            *quiver_args,
            angles=angles,
            scale=scale,
            scale_units=scale_units,
            units=units,
            color=color,
            cmap=cmap,
            norm=norm,
            **keep_func_kwargs(kwargs, self.ax.quiver),
        )

    def plot_colorbar(
        self,
        label: str,
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        pad: float = 0.02,
        shrink: float = 0.8,
        orientation: str = "vertical",
        location: str = "right",
        set_colorbar_label_kwargs: dict | None = None,
        set_colorbar_tick_params_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        kwargs.update(
            {
                "pad": pad,
                "shrink": shrink,
                "orientation": orientation,
                "location": location,
            }
        )

        self._cb_orientation = orientation
        self._cb_location = location
        self._cb = self.fig.colorbar(sm, ax=self.ax, **kwargs)

        set_colorbar_label_kwargs = (
            set_colorbar_label_kwargs if set_colorbar_label_kwargs is not None else {}
        )

        self.set_colorbar_label(label, **set_colorbar_label_kwargs)

        set_colorbar_tick_params_kwargs = (
            set_colorbar_tick_params_kwargs if set_colorbar_tick_params_kwargs is not None else {}
        )
        self.set_colorbar_tick_params(**set_colorbar_tick_params_kwargs)

    def set_colorbar_label(self, label: str, rotation=270, labelpad=12, **kwargs):
        self._validate_colorbar()
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            self.colorbar_axes.set_ylabel(label, rotation=rotation, labelpad=labelpad, **kwargs)
        else:
            self.colorbar_axes.set_xlabel(label, rotation=rotation, labelpad=labelpad, **kwargs)

    def set_colorbar_tick_params(self, **kwargs):
        self._validate_colorbar()
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            self.colorbar_axes.tick_params(axis="y", **kwargs)
        else:
            self.colorbar_axes.tick_params(axis="x", **kwargs)

    def set_colorbar_ticklabels(
        self,
        n_ticks: int = 5,
        clim: tuple[float, float] = None,
        labels: Sequence[str] = None,
        **kwargs,
    ):
        self._validate_colorbar()
        if (clim is None and labels is None) or (clim is not None and labels is not None):
            raise ValueError("Either clim or labels must be provided")
        elif clim is not None and labels is None:
            labels = [f"{x:.2f}" for x in np.linspace(clim[0], clim[1], n_ticks)]

        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            self.colorbar_axes.set_yticklabels(labels, **kwargs)
        else:
            self.colorbar_axes.set_xticklabels(labels, **kwargs)

    def set_colorbar_ticks(
        self,
        ticks: Sequence[float] | ticker.Locator | None = None,
        labels: Sequence[str] = None,
        n_ticks: int = 5,
        **kwargs,
    ):
        self._validate_ticks(ticks, labels)

        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            self.colorbar_axes.set_yticks(ticks, labels, **kwargs)
        else:
            self.colorbar_axes.set_xticks(ticks, labels, **kwargs)

    def get_colorbar_ticks(self):
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            return self.colorbar_axes.get_yticks()
        else:
            return self.colorbar_axes.get_xticks()

    def get_colorbar_lim(self):
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            return self.colorbar_axes.get_ylim()
        else:
            return self.colorbar_axes.get_xlim()

    def get_colorbar_ticklabels(self):
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            return self.colorbar_axes.get_yticklabels()
        else:
            return self.colorbar_axes.get_xticklabels()

    def _validate_ticks(self, ticks: Sequence[float] | ticker.Locator, labels: Sequence[str]):
        if isinstance(ticks, ticker.Locator):
            ticks = ticks.get_ticks()
        if isinstance(labels, ticker.Locator):
            labels = labels.get_ticklabels()
        if len(ticks) != len(labels):
            raise ValueError(
                f"Ticks and labels must have the same length: {len(ticks)} != {len(labels)}"
            )

    def _validate_colorbar(self):
        if not hasattr(self, "colorbar"):
            raise ValueError(
                "There is no colorbar for this plotter. call colorbar() or plot() with show_colorbar=True"
            )

    # ------------------------------------------------------------------
    # Orientation utilities
    # ------------------------------------------------------------------

    def orient_data(
        self, energies: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        values = np.asarray(values, dtype=np.float64)

        logger.debug(f"Plot orientation: {self.orientation}")
        if self.orientation is AxesOrientation.HORIZONTAL:
            return energies, values
        return values, energies

    def fill_between(
        self,
        energies: Iterable[float],
        values: Iterable[float],
        baseline: float | None = 0.0,
        **kwargs,
    ):
        # energies = np.asarray(list(energies), dtype=np.float64)
        # values = np.asarray(list(values), dtype=np.float64)
        # values = values.squeeze()

        if self.orientation is AxesOrientation.HORIZONTAL:
            return self.ax.fill_between(energies, values, baseline, **kwargs)
        return self.ax.fill_betweenx(energies, baseline, values, **kwargs)

    def fill_between_image(
        self,
        x: np.ndarray,
        y: np.ndarray,
        values: np.ndarray,
        orientation: AxesOrientation = AxesOrientation.HORIZONTAL,
        baseline: float | None = 0.0,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        clim: tuple[float | None, float | None] | None = None,
        label: str | None = None,
        plot_total: bool = True,
        **kwargs,
    ):
        if baseline is None:
            # match Matplotlib's default semantics: fill to 0 if not given
            baseline_arr = np.zeros_like(values)
        else:
            baseline_arr = (
                np.asarray(baseline, dtype=float) if np.ndim(baseline) else float(baseline)
            )
            if np.ndim(baseline_arr) == 0:
                baseline_arr = np.full_like(values, baseline_arr)

        y = y.squeeze()
        x = x.squeeze()
        baseline_arr = baseline_arr.squeeze()

        if orientation is AxesOrientation.HORIZONTAL:
            x_shift = x
            y_shift = y + baseline_arr
            img = values[np.newaxis, :]  # shape (1, N)

            x_poly_coords = np.r_[x_shift, x_shift[::-1]]
            y_poly_coords = np.r_[y_shift, baseline_arr[::-1]]
        else:
            img = values[:, np.newaxis]  # shape (1, N)

            x_shift = x + baseline_arr
            y_shift = y

            x_poly_coords = np.r_[x_shift, baseline_arr[::-1]]
            y_poly_coords = np.r_[y_shift, y_shift[::-1]]

        ylo = np.nanmin(np.c_[y_shift, baseline_arr])
        yhi = np.nanmax(np.c_[y_shift, baseline_arr])
        xlo = np.nanmin(np.c_[x_shift, baseline_arr])
        xhi = np.nanmax(np.c_[x_shift, baseline_arr])

        im = self.ax.imshow(
            img,
            extent=[xlo, xhi, ylo, yhi],
            origin=origin,
            aspect=aspect,
            interpolation=interpolation,
            zorder=zorder,
            cmap=cmap,
            norm=norm,
            clim=clim,
            **keep_func_kwargs(kwargs, self.ax.imshow),
        )

        # clip polygon under/over the curve (y between v and b)
        x_poly_coords = np.r_[x_shift, baseline_arr[::-1]]
        y_poly_coords = np.r_[y_shift, baseline_arr[::-1]]
        poly_xy = np.column_stack([x_poly_coords, y_poly_coords])

        patch = patches.Polygon(poly_xy, closed=True, facecolor="none", edgecolor="none")
        self.ax.add_patch(patch)
        im.set_clip_path(patch)

        return im

    # ------------------------------------------------------------------
    # Axis utilities
    # ------------------------------------------------------------------

    def set_dos_label(self, label: str = "DOS", unit_label: str = None):
        if unit_label is not None:
            label = f"{label} ({unit_label})"
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_ylabel(label)
        else:
            self.set_xlabel(label)

    def set_dos_lim(self, lim: tuple[float, float] = None, point_data: Property | None = None):
        if point_data is not None:
            lim = self._infer_point_dat_lim(point_data)

        if self.dos_lim is not None:
            self.dos_lim = (min(self.dos_lim[0], lim[0]), max(self.dos_lim[1], lim[1]))
        else:
            self.dos_lim = lim

        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_ylim(self.dos_lim)
        else:
            self.set_xlim(self.dos_lim)

    def set_dos_ticklabel(self, labels: Sequence[str] = None, positions: Sequence[float] = None):
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_yticklabel(labels, positions)
        else:
            self.set_xticklabel(labels, positions)

    def set_dos_tick_params(self, which: str = "major", **kwargs):
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_ytick_params(which=which, **kwargs)
        else:
            self.set_xtick_params(which=which, **kwargs)

    def set_energy_label(self, label: str = "Energy", unit_label: str = None):
        if unit_label is not None:
            label = f"{label} ({unit_label})"
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_xlabel(label)
        else:
            self.set_ylabel(label)

    def set_energy_lim(self, lim: tuple[float, float] = None, point_data: Property | None = None):
        if point_data is not None:
            lim = self._infer_points_lim(point_data)

        if self.energy_lim is not None:
            self.energy_lim = (min(self.energy_lim[0], lim[0]), max(self.energy_lim[1], lim[1]))
        else:
            self.energy_lim = lim

        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_xlim(self.energy_lim)
        else:
            self.set_ylim(self.energy_lim)

    def set_energy_ticklabel(self, labels: Sequence[str] = None, positions: Sequence[float] = None):
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_xticklabel(labels, positions)
        else:
            self.set_yticklabel(labels, positions)

    def set_energy_tick_params(self, which: str = "major", **kwargs):
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.set_xtick_params(which=which, **kwargs)
        else:
            self.set_ytick_params(which=which, **kwargs)

    def _infer_points_lim(self, point_data: Property) -> tuple[float, float]:
        point_data_array = point_data.points
        return point_data_array.min(), point_data_array.max()

    def _infer_point_dat_lim(self, point_data: Property) -> tuple[float, float]:
        point_data_array = point_data.to_array()
        return point_data_array.min(), point_data_array.max()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def draw_baseline(
        self, value: float, color="black", linewidth=0.8, linestyle="--", **kwargs
    ) -> None:
        all_kwargs = dict(color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.ax.axhline(value, **all_kwargs)
        else:
            self.ax.axvline(value, **all_kwargs)

    def draw_fermi(
        self, value: float, color="tab:red", linewidth=1.0, linestyle="--", **kwargs
    ) -> None:
        all_kwargs = dict(color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.ax.axvline(value, **all_kwargs)
        else:
            self.ax.axhline(value, **all_kwargs)

    def tight_layout(self):
        plt.tight_layout()

    def show(self, tight_layout: bool = True):
        if tight_layout:
            plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Public axis utilities
    # ------------------------------------------------------------------
    def set_title(self, title: str | None, **kwargs) -> None:
        if title is not None:
            self.ax.set_title(title, **kwargs)

    def set_xlim(self, limits: tuple[float, float] | None, **kwargs) -> None:
        if limits is None:
            return
        self.ax.set_xlim(limits, **kwargs)

    def set_ylim(self, limits: tuple[float, float] | None, **kwargs) -> None:
        if limits is None:
            return
        self.ax.set_ylim(limits, **kwargs)

    def set_xlabel(self, label: str | None, **kwargs) -> None:
        label_to_use = label if label is not None else ""
        self.ax.set_xlabel(label_to_use, **kwargs)

    def set_ylabel(self, label: str | None, **kwargs) -> None:
        label_to_use = label if label is not None else ""
        self.ax.set_ylabel(label_to_use, **kwargs)

    def set_xticklabel(
        self,
        labels: Sequence[str] | None,
        positions: Sequence[float] | None = None,
        **kwargs,
    ) -> None:
        if positions is not None:
            self.ax.set_xticks(positions)
        if labels is not None:
            self.ax.set_xticklabels(labels, **kwargs)

    def set_yticklabel(
        self,
        labels: Sequence[str] | None,
        positions: Sequence[float] | None = None,
        **kwargs,
    ) -> None:
        if positions is not None:
            self.ax.set_yticks(positions)
        if labels is not None:
            self.ax.set_yticklabels(labels, **kwargs)

    def set_tick_params(self, axis: str = "both", which: str = "major", **kwargs) -> None:
        self.ax.tick_params(axis=axis, **kwargs)

    def set_xtick_params(self, which: str = "major", **kwargs) -> None:
        self.set_tick_params(axis="x", which=which, **kwargs)

    def set_ytick_params(self, which: str = "major", **kwargs) -> None:
        self.set_tick_params(axis="y", which=which, **kwargs)

    def set_footnote(
        self,
        footnote: str | None,
        xy: tuple[float, float] = (0.0, -0.22),
        fontsize: str = "small",
        color: str = "0.4",
        textcoords: str = "offset points",
        xytext: tuple[float, float] = (0, 0.0),
        ha: str = "left",
        va: str = "bottom",
        annotation_clip: bool = False,
        xycoords: tuple[str, str] = ("axes fraction", "axes fraction"),
        **kwargs,
    ) -> None:
        footnote = f"footnote: {footnote}"
        if footnote is not None:
            self.ax.annotate(
                footnote,
                xy=xy,
                xycoords=xycoords,
                xytext=xytext,
                textcoords=textcoords,
                ha=ha,
                va=va,
                fontsize=fontsize,
                color=color,
                annotation_clip=annotation_clip,
                **kwargs,
            )

    def legend(
        self,
        handles: Sequence[Any] | None = None,
        labels: Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        if handles is not None or labels is not None:
            self.ax.legend(handles, labels, **kwargs)
        else:
            self.ax.legend(**kwargs)

    # ------------------------------------------------------------------
    # Data capture helpers
    # ------------------------------------------------------------------
    def store_arrays(self, mapping: Mapping[str, np.ndarray]) -> None:
        self._values.update({key: np.asarray(value) for key, value in mapping.items()})

    @property
    def values_dict(self) -> dict[str, np.ndarray]:
        return dict(self._values)

    def _resolve_channel_param(self, param: Iterable[Any] | None) -> Sequence[Any]:
        if len(param) > 1:
            return param
        return param[0]


def _as_channels(values: np.ndarray) -> np.ndarray:
    """Lay DOS data out as (n_points, 1 band, n_channels) for ``line_series``."""
    return values.reshape(values.shape[0], 1, -1)


def _resolve_scaling(
    values: list[np.ndarray | None],
    lims: list[tuple[float, float] | None],
    *,
    show_colorbar: ShowColorbar | str | None,
    cmap: str | mcolors.Colormap | None,
    norm: str | mcolors.Normalize | None,
    clim: tuple[float | None, float | None] | None,
):
    """Per-series (cmap, norm, clim) plus the parsed colorbar mode.

    A single colorbar shares the user's clim (default 0 to 1) across series;
    otherwise each channel resolves its own limits.
    """
    show_colorbar = ShowColorbar.from_string(show_colorbar)
    n = len(values)
    if show_colorbar is ShowColorbar.SINGLE:
        shared = _ensure_norm(norm, clim or (0.0, 1.0))
        return [_ensure_cmap(cmap)] * n, [shared] * n, [clim] * n, show_colorbar
    per_clim = [resolve_clim(clim, [lims[i]], [values[i]]) for i in range(n)]
    per_norm = [_ensure_norm(norm, c) for c in per_clim]
    per_cmap = [_ensure_cmap(cmap) if cmap is not None else None] * n
    return per_cmap, per_norm, per_clim, show_colorbar


def _ensure_cmap(cmap):
    if cmap is None:
        return plt.get_cmap("plasma")
    return plt.get_cmap(cmap) if isinstance(cmap, str) else cmap


def _ensure_norm(norm, clim):
    if isinstance(norm, mcolors.Normalize):
        return norm
    return mcolors.Normalize(*clim, clip=True)
