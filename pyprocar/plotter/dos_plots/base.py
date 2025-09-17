"""Shared utilities for density of states plotting backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping

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
    energy_label: str = "Energy (eV)"
    dos_label: str = "Density of states"
    baseline: float = 0.0
    baseline_kwargs: Dict[str, Any] | None = None
    show_baseline: bool = False
    show_fermi: bool = True
    fermi_energy: float | None = None
    fermi_line_kwargs: Dict[str, Any] | None = None
    legend: bool = False
    legend_kwargs: Dict[str, Any] | None = None
    mirror_spins: bool = True

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

    def orient_line(self, energies: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.orientation == "horizontal":
            return energies, values
        return values, energies

    def fill_between(
        self,
        energies: Iterable[float],
        values: Iterable[float],
        baseline: float | None = None,
        **kwargs,
    ):
        energies = np.asarray(list(energies), dtype=np.float64)
        values = np.asarray(list(values), dtype=np.float64)
        base_value = self.baseline if baseline is None else baseline

        if self.orientation == "horizontal":
            return self.ax.fill_between(energies, values, base_value, **kwargs)
        return self.ax.fill_betweenx(energies, base_value, values, **kwargs)

    # ------------------------------------------------------------------
    # Axis helpers
    # ------------------------------------------------------------------
    def finalize_axes(
        self,
        energies: np.ndarray,
        values: np.ndarray,
        *,
        baseline: float | None = None,
    ) -> None:
        energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim > 1:
            values = values.reshape(-1)

        if baseline is None:
            baseline = self.baseline

        if baseline is not None:
            values = np.concatenate([values, np.atleast_1d(baseline)])

        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            values = np.zeros(1)
        else:
            values = values[finite_mask]

        if self.orientation == "horizontal":
            self.ax.set_xlabel(self.energy_label)
            self.ax.set_ylabel(self.dos_label)
            self.ax.set_xlim(float(np.min(energies)), float(np.max(energies)))
            ymin = float(np.min(values))
            ymax = float(np.max(values))
            if np.isclose(ymin, ymax):
                pad = abs(ymin) if ymin != 0 else 1.0
                ymin -= pad
                ymax += pad
            self.ax.set_ylim(ymin, ymax)
        else:
            self.ax.set_ylabel(self.energy_label)
            self.ax.set_xlabel(self.dos_label)
            ymin = float(np.min(energies))
            ymax = float(np.max(energies))
            self.ax.set_ylim(ymin, ymax)
            xmin = float(np.min(values))
            xmax = float(np.max(values))
            if np.isclose(xmin, xmax):
                pad = abs(xmin) if xmin != 0 else 1.0
                xmin -= pad
                xmax += pad
            self.ax.set_xlim(xmin, xmax)

        if self.legend:
            self.ax.legend(**(self.legend_kwargs or {}))

        if self.show_baseline and baseline is not None:
            self._draw_baseline(baseline)

        if self.show_fermi and self.fermi_energy is not None:
            self._draw_fermi(self.fermi_energy)

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

    # ------------------------------------------------------------------
    # Data capture helpers
    # ------------------------------------------------------------------
    def store_arrays(self, mapping: Mapping[str, np.ndarray]) -> None:
        self._values.update({key: np.asarray(value) for key, value in mapping.items()})

    @property
    def values_dict(self) -> Dict[str, np.ndarray]:
        return dict(self._values)
