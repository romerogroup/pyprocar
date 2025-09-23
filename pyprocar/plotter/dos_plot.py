"""Shared utilities for density of states plotting backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from pyprocar.core.property_store import Property
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection

from pyprocar.utils.inspect_utils import keep_func_kwargs


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



class ScalarsMode(Enum):
    LINE = "line"
    FILL = "fill"
    
    @classmethod
    def from_string(cls, string: str) -> "ScalarsMode":
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
    def from_string(cls, string: str) -> "AxesOrientation":
        if string == "horizontal":
            return cls.HORIZONTAL
        elif string == "vertical":
            return cls.VERTICAL
        else:
            raise ValueError(f"Invalid axes orientation: {string}")
        
class Axis(Enum):
    X = "x"
    Y = "y"
    BOTH = "both"
    
    @classmethod
    def from_string(cls, string: str) -> "Axis":
        if string == "x":
            return cls.X
        elif string == "y":
            return cls.Y
        elif string == "both":
            return cls.BOTH
        else:
            raise ValueError(f"Invalid axis: {string}")
    

class DOSPlotter:
    """Lightweight wrapper around a matplotlib axis for DOS plots."""

    orientation: str = "horizontal"
    mirror_spins: bool = False
    legend_enabled: bool = True

    def __init__(self, 
                 figsize=(6, 4), 
                 dpi: int = 100, 
                 ax=None,
                 orientation:str = "horizontal") -> None:
        self.figsize = figsize
        self.dpi = dpi

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.ax = ax
            self.fig = ax.get_figure()

        self._values: Dict[str, np.ndarray] = {}
        self._validate_orientation()
        
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
        scalars_mode: str = "line",
        **kwargs
    ):
        scalars_mode = ScalarsMode.from_string(scalars_mode)
        if scalars_data and scalars_mode == ScalarsMode.LINE:
            self.plot_scalar_line(point_data, scalars_data)
        elif scalars_data and scalars_mode == ScalarsMode.FILL:
            self.plot_scalar_fill(point_data, scalars_data)
        else:
            self.plot_line(point_data, **kwargs)
                
        
    def plot_line(
        self,
        point_data: Property,
        **kwargs
    ):
        energy_array = point_data.points
        energy_label = point_data.points_label
        energy_units = point_data.points_units
        
        data_array = point_data.to_array()
        data_label = point_data.label
        data_units = point_data.units
        
        if data_array.ndim == 1:
            n_channels = 1
        else:
            n_channels = data_array.shape[1]
            
        for i_channel in range(n_channels):
            
            x_data, y_data = self.orient_data(energy_array, data_array[:,i_channel])
            plt.plot(x_data, y_data, **kwargs)
        
        self.set_dos_label(data_label, unit_label=data_units)
        self.set_energy_label(energy_label, unit_label=energy_units)
        
    def plot_scalar_line(self,
        point_data: Property,
        scalars_data: Property,
        scale:bool = False, 
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        clim: tuple[float | None, float | None] | None = None,
        linewidth: float = 1.5,
        linestyle: str = "-",
        alpha: float = 1.0,
        show_colorbar: bool = True,
        **kwargs
    ):
        
        energy_array = point_data.points
        energy_label = point_data.points_label
        energy_units = point_data.points_units
        
        data_array = point_data.to_array()
        data_label = point_data.label
        data_units = point_data.units
        
        scalars_array = scalars_data.to_array()

        if scalars_array.ndim == 1:
            n_channels = 1
        else:
            n_channels = scalars_array.shape[1]
        

        logger.debug(f"Energy shape: {energy_array.shape}")
        logger.debug(f"point_values shape: {data_array.shape}")
        logger.debug(f"Scalars shape: {scalars_array.shape}")
        logger.debug(f"n_channels: {n_channels}")
        clim = self._resolve_clim(scalars_data, clim=clim)
        cmap = self._resolve_cmap(cmap)
        norm = self._resolve_norm(clim, norm)
  
        for i_channel in range(n_channels):
        
            x_data, y_data = self.orient_data(energy_array, data_array[:,i_channel])
            points = np.column_stack([x_data, y_data]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(scalars_array[:,i_channel])
            lc.set_linewidth(linewidth)
            lc.set_linestyle(linestyle)
            lc.set_alpha(alpha)

            self.ax.add_collection(lc)
            
        if show_colorbar:
            self.plot_colorbar(scalars_data, cmap=cmap, norm=norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
            
        
        self.set_dos_label(data_label, unit_label=data_units)
        self.set_energy_label(energy_label, unit_label=energy_units)
        self.finalize_axes(energy_array, data_array)
        
    def plot_scalar_fill(self,
        point_data: Property,
        scalars_data: Property,
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        clim: tuple[float | None, float | None] | None = None,
        show_colorbar: bool = True,
        **kwargs
    ):
        energy_array = point_data.points
        energy_label = point_data.points_label
        energy_units = point_data.points_units
        n_energies = energy_array.size
        
        data_array = point_data.to_array()
        data_label = point_data.label
        data_units = point_data.units

        scalars_array = scalars_data.to_array()

        n_channels = scalars_array.shape[1]
        logger.debug(f"n_channels: {n_channels}")
        
        clim = self._resolve_clim(scalars_data, clim=clim)
        norm = self._resolve_norm(clim, norm)
        cmap = self._resolve_cmap(cmap)
        
        logger.debug(f"Energy shape: {energy_array.shape}")
        logger.debug(f"point_values shape: {data_array.shape}")
        logger.debug(f"Scalars shape: {scalars_array.shape}")
        logger.debug(f"n_channels: {n_channels}")
        
        for i_channel in range(n_channels):
            channel_data_array = data_array[:,i_channel]
            channel_scalars_array = scalars_array[:,i_channel]
            for idx in range(n_energies - 1):
                segment_color = cmap(norm(channel_scalars_array[idx]))
                x_segment = energy_array[idx : idx + 2]
                y_segment = channel_data_array[idx : idx + 2]
                self.fill_between(x_segment, y_segment, color=segment_color, **kwargs)

        self.set_dos_label(data_label, unit_label=data_units)
        self.set_energy_label(energy_label, unit_label=energy_units)
        
        if show_colorbar:
            self.plot_colorbar(scalars_data, cmap=cmap, norm=norm, clim=clim, **keep_func_kwargs(kwargs, self.colorbar))
            
        self.finalize_axes(energy_array, data_array)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_clim(self, scalars_data: Property, clim: tuple[float | None, float | None] | None = None) -> tuple[float, float]:
        if clim is not None:
            return clim
        
        scalars_array = scalars_data.to_array()
        scalars_lim = scalars_data.data_lim
        
        if scalars_lim is not None:
            return scalars_lim

        finite = np.isfinite(scalars_array)
        vmin = float(np.nanmin(scalars_array[finite]))
        vmax = float(np.nanmax(scalars_array[finite]))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0

        return vmin, vmax
    
    def _resolve_norm(self, clim: tuple[float | None, float | None] | None,
                      norm: mcolors.Normalize | str | None = None, 
                      clip: bool = True) -> mcolors.Normalize:
        vmin, vmax = clim
        if norm is None:
            print(f"vmin: {vmin}, vmax: {vmax}")
            norm = mcolors.Normalize(vmin, vmax, clip=clip)
        elif isinstance(norm, str):
            norm = plt.get_norm(norm)(vmin, vmax)
        elif isinstance(norm, mcolors.Normalize):
            return norm
        else:
            raise ValueError(f"Invalid norm: {norm}")
        return norm
    
    def _resolve_cmap(self, cmap: str | mcolors.Colormap | None = None) -> mcolors.Colormap:
        if cmap is None:
            cmap = plt.get_cmap(cmap)
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        elif isinstance(cmap, mcolors.Colormap):
            cmap = cmap
        return cmap

    def plot_colorbar(self, scalars_data: Property, 
                    cmap = "plasma",
                    norm = None,
                    clim = None,
                    pad=0.02, 
                    shrink=0.8,
                    orientation="vertical",
                    location="right",
                    set_colorbar_label_kwargs=None,
                    set_colorbar_tick_params_kwargs=None,
                    **kwargs) -> None:
        
        scalars_label = scalars_data.label
        scalars_units = scalars_data.units

        clim = self._resolve_clim(scalars_data, clim=clim)
        norm = self._resolve_norm(clim, norm)
        cmap = self._resolve_cmap(cmap)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        kwargs.update({
            "pad":pad, 
            "shrink":shrink, 
            "orientation":orientation, 
            "location":location,
        })
        
 
        self._cb_orientation = orientation
        self._cb_location = location
        self._cb = self.fig.colorbar(sm, ax=self.ax, **kwargs)

        set_colorbar_label_kwargs = set_colorbar_label_kwargs if set_colorbar_label_kwargs is not None else {}
        if scalars_units is not None:
            scalars_label = f"{scalars_label} ({scalars_units})"
        self.set_colorbar_label(scalars_label, **set_colorbar_label_kwargs)
        
        set_colorbar_tick_params_kwargs = set_colorbar_tick_params_kwargs if set_colorbar_tick_params_kwargs is not None else {}
        self.set_colorbar_tick_params(**set_colorbar_tick_params_kwargs)

    def set_colorbar_label(self, label: str, 
                           rotation=270,
                           labelpad=12,
                           **kwargs):

        self._validate_colorbar()
        if self.colorbar_orientation == "vertical":
            self.colorbar_axes.set_ylabel(label, rotation=rotation, labelpad=labelpad, 
                                  **kwargs)
        else:
            self.colorbar_axes.set_xlabel(label, rotation=rotation, labelpad=labelpad, 
                              **kwargs)
        
    def set_colorbar_tick_params(self, **kwargs):
        self._validate_colorbar()
        if self.colorbar_orientation == "vertical":
            self.colorbar_axes.tick_params(axis="y", **kwargs)
        else:
            self.colorbar_axes.tick_params(axis="x", **kwargs)
        
    def set_colorbar_ticklabels(self, n_ticks: int = 5, clim: tuple[float, float] = None, labels: Sequence[str] = None, **kwargs):
        
        self._validate_colorbar()
        if (clim is None and labels is None) or (clim is not None and labels is not None):
            raise ValueError("Either clim or labels must be provided")
        elif clim is not None and labels is None:
            labels = [f"{x:.2f}" for x in np.linspace(clim[0], clim[1], n_ticks)]
        
        if self.colorbar_orientation == "vertical":
            self.colorbar_axes.set_yticklabels(labels, **kwargs)
        else:
            self.colorbar_axes.set_xticklabels(labels, **kwargs)
    
    def set_colorbar_ticks(self, 
                           ticks: Sequence[float] | ticker.Locator | None = None,
                           labels: Sequence[str] = None,
                           n_ticks: int = 5,
                           **kwargs):
        
    
        self._validate_ticks(ticks, labels)
        
        if self.colorbar_orientation == "vertical":
            self.colorbar_axes.set_yticks(ticks, labels, **kwargs)
        else:
            self.colorbar_axes.set_xticks(ticks, labels, **kwargs)
            
    def get_colorbar_ticks(self):
        if self.colorbar_orientation == "vertical":
            return self.colorbar_axes.get_yticks()
        else:
            return self.colorbar_axes.get_xticks()
        
    def get_colorbar_lim(self):
        if self.colorbar_orientation == "vertical":
            return self.colorbar_axes.get_ylim()
        else:
            return self.colorbar_axes.get_xlim()
        
    def get_colorbar_ticklabels(self):
        if self.colorbar_orientation == "vertical":
            return self.colorbar_axes.get_yticklabels()
        else:
            return self.colorbar_axes.get_xticklabels()
            
        
    def _validate_ticks(self, ticks: Sequence[float] | ticker.Locator, labels: Sequence[str]):
        
        if isinstance(ticks, ticker.Locator):
            ticks = ticks.get_ticks()
        if isinstance(labels, ticker.Locator):
            labels = labels.get_ticklabels()
        if len(ticks) != len(labels):
            raise ValueError(f"Ticks and labels must have the same length: {len(ticks)} != {len(labels)}")
        
    def _validate_colorbar(self):
        if not hasattr(self, "colorbar"):
            raise ValueError("There is no colorbar for this plotter. call colorbar() or plot() with show_colorbar=True")

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

    def set_dos_label(self, label: str = "DOS", unit_label: str = None):
        if unit_label is not None:
            label = f"{label} ({unit_label})"
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
    
    def set_energy_label(self, label: str = "Energy", unit_label: str = None):
        if unit_label is not None:
            label = f"{label} ({unit_label})"
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

        self.set_energy_lim(data=energies)
        self.set_energy_tick_params()
        
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
    def set_title(self, title: str | None, **kwargs) -> None:
        if title is not None:
            self.ax.set_title(title, **kwargs)
    
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
