"""Shared utilities for density of states plotting backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _is_property(obj: Any) -> bool:
    return isinstance(obj, property)


def get_class_attributes(cls) -> Dict[str, Any]:
    """Return the public data attributes defined on ``cls``."""

    attributes: Dict[str, Any] = {}
    for name, value in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if callable(value) or _is_property(value):
            continue
        attributes[name] = value
    return attributes



class BasePlotter(ABC):
    """Lightweight wrapper around a matplotlib axis for DOS plots."""

    orientation: str = "horizontal"

    def __init__(self, figsize=(6, 4), dpi: int = 100, ax=None, **kwargs) -> None:
        self.figsize = figsize
        self.dpi = dpi

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.ax = ax
            self.fig = ax.get_figure()

        self._values: Dict[str, np.ndarray] = {}
        self.update_instance_params(**kwargs)
        self._validate_orientation()

    # ------------------------------------------------------------------
    # High level orchestration
    # ------------------------------------------------------------------
    def plot(self, *args, **kwargs):
        """Update instance parameters and delegate plotting to subclasses."""
        self.update_instance_params(**kwargs)
        return self._plot(*args, **kwargs)

    @abstractmethod
    def _plot(self, 
              energies: Iterable[float], 
              dos_values: Iterable[Iterable[float]] | np.ndarray,
              **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    @property
    def class_plot_params(self) -> Dict[str, Any]:
        return get_class_attributes(self.__class__)

    @property
    def instance_plot_params(self) -> Dict[str, Any]:
        attrs = {}
        for name in self.class_plot_params:
            attrs[name] = getattr(self, name, None)
        return attrs

    def update_instance_params(self, **kwargs) -> Dict[str, Any]:
        for key, value in kwargs.items():
            if key == "legend":
                self.legend_enabled = bool(value)
                continue
            if key in self.class_plot_params:
                setattr(self, key, value)
        self._validate_orientation()
        return self.instance_plot_params

    # ------------------------------------------------------------------
    # Orientation utilities
    # ------------------------------------------------------------------
    def _validate_orientation(self) -> None:
        if self.orientation not in {"horizontal", "vertical"}:
            raise ValueError(
                "orientation must be either 'horizontal' or 'vertical', "
                f"got {self.orientation!r}"
            )

    def orient_data(self, energies: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.orientation == "horizontal":
            return energies, values
        return values, energies

    def fill_between(
        self,
        energies: Iterable[float],
        values: Iterable[float],
        baseline: float | None = 0.0,
        **kwargs,
    ):
        energies = np.asarray(list(energies), dtype=np.float64)
        values = np.asarray(list(values), dtype=np.float64)

        if self.orientation == "horizontal":
            return self.ax.fill_between(energies, values, baseline, **kwargs)
        return self.ax.fill_betweenx(energies, baseline, values, **kwargs)

    def set_dos_label(self, label: str):
        if self.orientation == "horizontal":    
            self.set_ylabel(label)
        else:
            self.set_xlabel(label)
            
    def set_dos_lim(self, lim: tuple[float, float]):
        if self.orientation == "horizontal":
            self.set_ylim(lim)
        else:
            self.set_xlim(lim)

    def set_energy_label(self, label: str):
        self.ax.set_xlabel(label)

    # ------------------------------------------------------------------
    # Axis helpers
    # ------------------------------------------------------------------
    # def finalize_axes(
    #     self,
    #     energies: np.ndarray,
    #     values: np.ndarray,
    # ) -> None:
    #     energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    #     values = np.asarray(values, dtype=np.float64)
    #     if values.ndim > 1:
    #         values = values.reshape(-1)


    #     finite_mask = np.isfinite(values)
    #     if not finite_mask.any():
    #         values = np.zeros(1)
    #     else:
    #         values = values[finite_mask]
            
    #     x_data, y_data = self.orient_data(energies, values)

        
    #     if self.orientation == "horizontal":
    #         self.set_xlabel(self.energy_label)
    #         self.set_ylabel(self.dos_label)
    #         self.set_xlim((float(np.min(energies)), float(np.max(energies))))
    #         ymin = float(np.min(values))
    #         ymax = float(np.max(values))
    #         if np.isclose(ymin, ymax):
    #             pad = abs(ymin) if ymin != 0 else 1.0
    #             ymin -= pad
    #             ymax += pad
    #         self.set_ylim((ymin, ymax))
    #     else:
    #         self.set_ylabel(self.energy_label)
    #         self.set_xlabel(self.dos_label)
    #         ymin = float(np.min(energies))
    #         ymax = float(np.max(energies))
    #         self.set_ylim((ymin, ymax))
    #         xmin = float(np.min(values))
    #         xmax = float(np.max(values))
    #         if np.isclose(xmin, xmax):
    #             pad = abs(xmin) if xmin != 0 else 1.0
    #             xmin -= pad
    #             xmax += pad
    #         self.set_xlim((xmin, xmax))

    #     if self.legend_enabled:
    #         self.legend()

    #     self.draw_baseline()

    #     if self.show_fermi and self.fermi_energy is not None:
    #         self.draw_fermi(self.fermi_energy)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_baseline(self, value: float) -> None:
        kwargs = dict(color="black", linewidth=0.8, linestyle="--")
        kwargs.update(self.baseline_kwargs or {})
        if self.orientation == "horizontal":
            self.ax.axhline(value, **kwargs)
        else:
            self.ax.axvline(value, **kwargs)

    def _draw_fermi(self, value: float) -> None:
        kwargs = dict(color="tab:red", linewidth=1.0, linestyle="--")
        kwargs.update(self.fermi_line_kwargs or {})
        if self.orientation == "horizontal":
            self.ax.axvline(value, **kwargs)
        else:
            self.ax.axhline(value, **kwargs)
            
    def show(self):
        plt.show()

    # ------------------------------------------------------------------
    # Public axis utilities
    # ------------------------------------------------------------------
    def set_xlim(self, limits: Tuple[float, float] | None, **kwargs) -> None:
        if limits is None:
            return
        self.ax.set_xlim(limits, **kwargs)

    def set_ylim(self, limits: Tuple[float, float] | None, **kwargs) -> None:
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

    def set_tick_params(self, axis: str = "both", **kwargs) -> None:
        self.ax.tick_params(axis=axis, **kwargs)

    def legend(
        self,
        handles: Sequence[Any] | None = None,
        labels: Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        legend_kwargs = dict(self.legend_kwargs or {})
        legend_kwargs.update(kwargs)
        if handles is not None or labels is not None:
            self.ax.legend(handles, labels, **legend_kwargs)
        else:
            self.ax.legend(**legend_kwargs)

    def draw_fermi(self, value: float | None = None) -> None:
        if value is None:
            value = self.fermi_energy
        if value is None:
            return
        self._draw_fermi(value)

    def draw_baseline(self, value: float | None = None) -> None:
        if value is None:
            return
        self._draw_baseline(value)

    # ------------------------------------------------------------------
    # Data capture helpers
    # ------------------------------------------------------------------
    def store_arrays(self, mapping: Mapping[str, np.ndarray]) -> None:
        self._values.update({key: np.asarray(value) for key, value in mapping.items()})

    @property
    def values_dict(self) -> Dict[str, np.ndarray]:
        return dict(self._values)
