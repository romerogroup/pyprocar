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
    mirror_spins: bool = False
    legend_enabled: bool = True

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

    def set_dos_label(self, label: str = "DOS"):
        if self.orientation == "horizontal":    
            self.set_ylabel(label)
        else:
            self.set_xlabel(label)
            
    def set_dos_lim(self, lim: tuple[float, float] = None, data: np.ndarray = None):
        if data is not None:
            lim = (data.min(), data.max())
            
        if self.orientation == "horizontal":
            self.set_ylim(lim)
        else:
            self.set_xlim(lim)
            
    def set_dos_ticklabel(self, labels: Sequence[str] = None, positions: Sequence[float] = None):
        if self.orientation == "horizontal":
            self.set_yticklabel(labels, positions)
        else:
            self.set_xticklabel(labels, positions)
            
    def set_dos_tick_params(self, which: str = "major", **kwargs):
        if self.orientation == "horizontal":
            self.set_ytick_params(which=which, **kwargs)
        else:
            self.set_xtick_params(which=which, **kwargs)
    
    def set_energy_label(self, label: str = "Energy"):
        if self.orientation == "horizontal":
            self.set_xlabel(label)
        else:
            self.set_ylabel(label)
            
    def set_energy_lim(self, lim: tuple[float, float] = None, data: np.ndarray = None):
        if data is not None:
            lim = (data.min(), data.max())
            
        if self.orientation == "horizontal":
            self.set_xlim(lim)
        else:
            self.set_ylim(lim)
            
    def set_energy_ticklabel(self, labels: Sequence[str] = None, positions: Sequence[float] = None):
        if self.orientation == "horizontal":
            self.set_xticklabel(labels, positions)
        else:
            self.set_yticklabel(labels, positions)
            
    def set_energy_tick_params(self, which: str = "major", **kwargs):
        if self.orientation == "horizontal":
            self.set_xtick_params(which=which, **kwargs)
        else:
            self.set_ytick_params(which=which, **kwargs)

    # ------------------------------------------------------------------
    # Axis helpers
    # ------------------------------------------------------------------
    def finalize_axes(
        self,
        energies: np.ndarray,
        dos_values: np.ndarray,
    ) -> None:
        energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        dos_values = np.asarray(dos_values, dtype=np.float64)
        if dos_values.ndim > 1:
            dos_values = dos_values.reshape(-1)


        finite_mask = np.isfinite(dos_values)
        if not finite_mask.any():
            dos_values = np.zeros(1)
        else:
            dos_values = dos_values[finite_mask]

        self.set_energy_label()
        self.set_energy_lim(data=energies)
        self.set_energy_tick_params()
        
        self.set_dos_label()
        self.set_dos_lim(data=dos_values)
        self.set_dos_tick_params()


    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def draw_baseline(self, value: float, 
                      color="black", 
                      linewidth=0.8, 
                      linestyle="--", 
                      **kwargs) -> None:
        
        all_kwargs = dict(color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)
        if self.orientation == "horizontal":
            self.ax.axhline(value, **all_kwargs)
        else:
            self.ax.axvline(value, **all_kwargs)

    def draw_fermi(self, value: float,
                   color="tab:red", 
                   linewidth=1.0, 
                   linestyle="--", 
                   **kwargs) -> None:
        all_kwargs = dict(color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)
        if self.orientation == "horizontal":
            self.ax.axvline(value, **all_kwargs)
        else:
            self.ax.axhline(value, **all_kwargs)
            
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

    def set_tick_params(self, axis: str = "both", which: str = "major", **kwargs) -> None:
        self.ax.tick_params(axis=axis, **kwargs)

    def set_xtick_params(self, which: str = "major", **kwargs) -> None:
        self.set_tick_params(axis="x", which=which, **kwargs)

    def set_ytick_params(self, which: str = "major", **kwargs) -> None:
        self.set_tick_params(axis="y", which=which, **kwargs)

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
    def values_dict(self) -> Dict[str, np.ndarray]:
        return dict(self._values)
