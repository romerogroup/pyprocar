"""Parametric density of states plotting."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Dict, Iterable, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
from matplotlib.collections import LineCollection

from pyprocar.plotter.dos_plots.base import BasePlotter

logger = logging.getLogger(__name__)


def _prepare_parametric_inputs(
    energies: Iterable[float],
    dos_values: Iterable[Iterable[float]] | np.ndarray,
    scalars: Iterable[Iterable[float]] | np.ndarray | None,
    labels: Iterable[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
    energies_array = np.asarray(list(energies), dtype=np.float64).reshape(-1)
    if energies_array.size < 2:
        raise ValueError("Parametric plots require at least two energy samples.")

    dos_array = np.asarray(dos_values, dtype=np.float64)
    if dos_array.ndim == 1:
        dos_array = dos_array[np.newaxis, :]
    if dos_array.shape[1] != energies_array.size:
        raise ValueError("dos_values must align with the provided energies.")

    if scalars is None:
        scalars_array = dos_array
    else:
        scalars_array = np.asarray(scalars, dtype=np.float64)
        if scalars_array.ndim == 1:
            scalars_array = scalars_array[np.newaxis, :]
        if scalars_array.shape != dos_array.shape:
            raise ValueError("scalars must have the same shape as dos_values.")

    labels_tuple: Tuple[str, ...] = tuple(labels or ())
    return energies_array, dos_array, scalars_array, labels_tuple


class ParametricPlot(BasePlotter):
    """Render a DOS where the fill color encodes projection weights."""

    cmap: str | mcolors.Colormap = "plasma"
    clim: tuple[float | None, float | None] | None = None
    show_colorbar: bool = True
    colorbar_kwargs: Dict[str, object] | None = None
    plot_total: bool = True
    total_line_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None
    fill_kwargs: Mapping[str, object] | None = None

    def _plot(
        self,
        energies: Iterable[float],
        dos_values: Iterable[Iterable[float]] | np.ndarray,
        scalars: Iterable[Iterable[float]] | np.ndarray,
        *,
        labels: Iterable[str] | None = None,
        scale: bool = False,
        plot_total: bool | None = None,
        colorbar: bool | None = None,
        total_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        fill_kwargs: Mapping[str, object] | None = None,
    ):
        energies_array, dos_array, scalar_array, labels_tuple = _prepare_parametric_inputs(
            energies, dos_values, scalars, labels
        )

        line_values = dos_array.copy()
        mirror = self.mirror_spins and line_values.shape[0] == 2
        if mirror:
            line_values[1:, :] *= -1.0

        colour_weights = scalar_array if not scale else scalar_array * line_values

        vmin, vmax = self._resolve_clim(colour_weights)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        try:
            cmap_obj = colormaps.get_cmap(self.cmap)
        except AttributeError:  # pragma: no cover - backward compatibility
            cmap_obj = cm.get_cmap(self.cmap)

        fill_cfg = dict(alpha=0.7)
        fill_cfg.update(self.fill_kwargs or {})
        fill_cfg.update(fill_kwargs or {})

        stored: Dict[str, np.ndarray] = {"energies": energies_array}

        for index in range(line_values.shape[0]):
            label = labels_tuple[index] if index < len(labels_tuple) else None
            self._plot_spin_series(
                energies_array,
                line_values[index],
                colour_weights[index],
                cmap=cmap_obj,
                norm=norm,
                fill_kwargs=fill_cfg,
            )

            if (plot_total if plot_total is not None else self.plot_total):
                kwargs = self._resolve_line_kwargs(
                    index,
                    overrides=total_kwargs,
                )
                x_data, y_data = self.orient_line(energies_array, line_values[index])
                if label is not None:
                    kwargs.setdefault("label", label)
                self.ax.plot(x_data, y_data, **kwargs)

            stored[f"dos_total_{index}"] = line_values[index]
            stored[f"dos_weight_{index}"] = colour_weights[index]

        if colorbar if colorbar is not None else self.show_colorbar:
            self._draw_colorbar(norm, cmap_obj)

        self.store_arrays(stored)
        self.finalize_axes(energies_array, line_values)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _plot_spin_series(
        self,
        energies: np.ndarray,
        totals: np.ndarray,
        colour_values: np.ndarray,
        *,
        cmap,
        norm,
        fill_kwargs: Mapping[str, object],
    ) -> None:
        energies = np.asarray(energies, dtype=np.float64)
        totals = np.asarray(totals, dtype=np.float64)
        colour_values = np.asarray(colour_values, dtype=np.float64)

        for idx in range(energies.size - 1):
            segment_color = cmap(norm(colour_values[idx]))
            x_segment = energies[idx : idx + 2]
            y_segment = totals[idx : idx + 2]
            self.fill_between(x_segment, y_segment, **fill_kwargs, color=segment_color)

    def _resolve_clim(self, colour_values: np.ndarray) -> tuple[float, float]:
        provided = self.clim
        if provided is not None:
            vmin, vmax = provided
            if vmin is None:
                vmin = float(np.nanmin(colour_values))
            if vmax is None:
                vmax = float(np.nanmax(colour_values))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            return float(vmin), float(vmax)

        finite = np.isfinite(colour_values)
        if not finite.any():
            return 0.0, 1.0
        vmin = float(np.nanmin(colour_values[finite]))
        vmax = float(np.nanmax(colour_values[finite]))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return vmin, vmax

    def _draw_colorbar(self, norm, cmap) -> None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        kwargs = dict(pad=0.02, shrink=0.8)
        kwargs.update(self.colorbar_kwargs or {})
        cb = self.fig.colorbar(sm, ax=self.ax, **kwargs)
        cb.ax.set_ylabel("Projection weight", rotation=270, labelpad=12)

    def _resolve_line_kwargs(
        self,
        index: int,
        overrides: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
    ) -> Dict[str, object]:
        base = dict(color="black", linewidth=1.2)

        configured = self.total_line_kwargs
        if isinstance(configured, Sequence) and not isinstance(configured, (dict, str)):
            if configured:
                idx = index % len(configured)
                base.update(dict(configured[idx]))
        elif isinstance(configured, Mapping):
            base.update(dict(configured))

        if isinstance(overrides, Sequence) and not isinstance(overrides, (dict, str)):
            if overrides:
                idx = index % len(overrides)
                base.update(dict(overrides[idx]))
        elif isinstance(overrides, Mapping):
            base.update(dict(overrides))

        return base


class ParametricLinePlot(ParametricPlot):
    """Render a DOS using coloured line segments instead of filled regions."""

    line_collection_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None

    def _plot(
        self,
        energies: Iterable[float],
        dos_values: Iterable[Iterable[float]] | np.ndarray,
        scalars: Iterable[Iterable[float]] | np.ndarray | None,
        *,
        labels: Iterable[str] | None = None,
        scale: bool = False,
        plot_total: bool | None = None,
        colorbar: bool | None = None,
        total_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        line_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    ):
        energies_array, dos_array, scalar_array, labels_tuple = _prepare_parametric_inputs(
            energies, dos_values, scalars, labels
        )

        line_values = dos_array.copy()
        mirror = self.mirror_spins and line_values.shape[0] == 2
        if mirror:
            line_values[1:, :] *= -1.0

        colour_weights = scalar_array if not scale else scalar_array * line_values

        vmin, vmax = self._resolve_clim(colour_weights)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        try:
            cmap_obj = colormaps.get_cmap(self.cmap)
        except AttributeError:  # pragma: no cover - backward compatibility
            cmap_obj = cm.get_cmap(self.cmap)

        stored: Dict[str, np.ndarray] = {"energies": energies_array}

        for index in range(line_values.shape[0]):
            label = labels_tuple[index] if index < len(labels_tuple) else None
            x_data, y_data = self.orient_line(energies_array, line_values[index])

            points = np.column_stack([x_data, y_data]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments)
            lc.set_cmap(cmap_obj)
            lc.set_norm(norm)

            segment_weights = 0.5 * (
                colour_weights[index, :-1] + colour_weights[index, 1:]
            )
            lc.set_array(segment_weights)

            kwargs = self._resolve_line_collection_kwargs(
                index,
                overrides=line_kwargs,
            )
            lc.set_linewidth(kwargs.get("linewidth", 1.5))
            lc.set_linestyle(kwargs.get("linestyle", "-"))
            lc.set_alpha(kwargs.get("alpha", 1.0))

            self.ax.add_collection(lc)

            stored[f"dos_total_{index}"] = line_values[index]
            stored[f"dos_weight_{index}"] = colour_weights[index]

        if colorbar if colorbar is not None else self.show_colorbar:
            self._draw_colorbar(norm, cmap_obj)

        self.store_arrays(stored)
        self.finalize_axes(energies_array, line_values)

    def _resolve_line_collection_kwargs(
        self,
        index: int,
        overrides: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
    ) -> Dict[str, object]:
        base: Dict[str, object] = {}

        configured = self.line_collection_kwargs
        if isinstance(configured, Sequence) and not isinstance(configured, (dict, str)):
            if configured:
                idx = index % len(configured)
                base.update(dict(configured[idx]))
        elif isinstance(configured, Mapping):
            base.update(dict(configured))

        if isinstance(overrides, Sequence) and not isinstance(overrides, (dict, str)):
            if overrides:
                idx = index % len(overrides)
                base.update(dict(overrides[idx]))
        elif isinstance(overrides, Mapping):
            base.update(dict(overrides))

        return base
