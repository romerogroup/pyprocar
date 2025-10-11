"""Shared utilities for density of states plotting backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Callable

from pyvista._plot import plot
from pyprocar.core.property_store import Property
from enum import Enum
from dataclasses import dataclass, field
import re
from collections import Counter
from enum import Enum


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap

from pyprocar.utils.func_utils import keep_func_kwargs, expand_grouped_params, keep_func_kwargs_and_args


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

class ShowColorbar(Enum):
    SINGLE = "single"
    PER_CHANNEL = "per_channel"
    NONE = "none"
    
    @classmethod
    def from_string(cls, input: str|ShowColorbar|bool|None) -> "ShowColorbar":
        if isinstance(input, ShowColorbar):
            return input
        elif input is None:
            return cls.NONE
        elif isinstance(input, bool):
            return cls.SINGLE if input else cls.NONE
        
        string = input.lower()
        if string == "single":
            return cls.SINGLE
        elif string == "per_channel":
            return cls.PER_CHANNEL
        elif string == "none":
            return cls.NONE
        else:
            err_msg = f"Invalid colorbar mode: {string}. Valid modes are:\n"
            err_msg += "\n".join([f"- {mode}" for mode in cls.list_modes()])
            raise ValueError(err_msg)
        
    @classmethod
    def list_modes(cls) -> list[str]:
        """List all available colorbar modes."""
        return [mode.value for mode in cls]

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
    def from_string(cls, string: str | "AxesOrientation") -> "AxesOrientation":
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
    def from_string(cls, string: str) -> "Axis":
        if string == "x":
            return cls.X
        elif string == "y":
            return cls.Y
        elif string == "both":
            return cls.BOTH
        else:
            raise ValueError(f"Invalid axis: {string}")
        
class ChannelMode(Enum):
    FLIP = "flip"
    NORMAL = "normal"
    
    @classmethod
    def from_string(cls, string: str | "ChannelMode" | None) -> "ChannelMode":
        if isinstance(string, ChannelMode):
            return string
        elif string is None:
            return cls.NORMAL
        
        string = string.lower()
        if string == "flip":
            return cls.FLIP
        elif string == "normal":
            return cls.NORMAL
        else:
            err_msg = f"Invalid channel mode: {string}. Valid modes are:\n"
            err_msg += "\n".join([f"- {mode}" for mode in cls.list_modes()])
            raise ValueError(err_msg)
        
    @classmethod
    def list_modes(cls) -> list[str]:
        return [mode.value for mode in cls]
    

@dataclass
class DOSPlotter:
    """Lightweight wrapper around a matplotlib axis for DOS plots."""

    orientation: str = AxesOrientation.HORIZONTAL
    figsize: tuple[int, int] = (6, 4)
    dpi: int = 100
    ax: plt.Axes | None = None
    dos_lim: tuple[float, float] = None
    energy_lim: tuple[float, float] = None

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
        **kwargs
    ):
        channel_mode = ChannelMode.from_string(channel_mode)
        n_channels = scalars_data.n_channels if scalars_data else point_data.n_channels
        x_data , y_data = self.orient_data(point_data.points, point_data.to_array())
        plot_kwargs = dict(x = [], y = [], 
                           scalars = [], scalars_label = [], scalars_lim = [], 
                           vectors = [], vectors_label = [], vectors_lim = [])

        data_lim = (0, 0)
        for i_channel in range(n_channels):
            y_channel_data = y_data[:,i_channel]
            if channel_mode == ChannelMode.FLIP and i_channel != 0:
                y_channel_data = y_channel_data * -1.0
            data_min = y_channel_data.min()
            data_max = y_channel_data.max()
            data_lim = (min(data_lim[0], data_min), max(data_lim[1], data_max))
            plot_kwargs["y"].append(y_channel_data)
            plot_kwargs["x"].append(x_data)
            
            if scalars_data:
                plot_kwargs["scalars"].append(scalars_data.to_array()[:,i_channel])
                plot_kwargs["scalars_label"].append(scalars_data.label)
                plot_kwargs["scalars_lim"].append(scalars_data.rounded_data_lim[i_channel])
            if vectors_data:
                plot_kwargs["vectors"].append(vectors_data.to_array()[:,i_channel])
                plot_kwargs["vectors_label"].append(vectors_data.label[i_channel])
                plot_kwargs["vectors_lim"].append(vectors_data.rounded_data_lim[i_channel])

        # plot_kwargs = keep_func_kwargs_and_args(plot_kwargs, self.plot_scalar_line)
        additional_kwargs = keep_func_kwargs(kwargs, self.plot_scalar_line)
        print(additional_kwargs)
        # for additional_kwargs_key, additional_kwargs_value in additional_kwargs.items():
        #     plot_kwargs.setdefault(additional_kwargs_key, additional_kwargs_value)
        # additional_kwargs.update(plot_kwargs)
        # for additional_kwargs_key, additional_kwargs_value in additional_kwargs.items():
        #     try:
        #         print(additional_kwargs_key, len(additional_kwargs_value))
        #     except:
        #         pass
        # print(additional_kwargs['alpha'])

        scalars_mode = ScalarsMode.from_string(scalars_mode)
        if scalars_data and scalars_mode == ScalarsMode.LINE:
            logger.info(f"Plotting scalar line for {point_data.label}")
            # print(additional_kwargs.get("alpha"))
            x = plot_kwargs.pop("x")
            y = plot_kwargs.pop("y")
            scalars = plot_kwargs.pop("scalars")
            scalars_label = plot_kwargs.pop("scalars_label")
            scalars_lim = plot_kwargs.pop("scalars_lim")
            additional_kwargs.update(plot_kwargs)
            print(additional_kwargs)
            self.plot_scalar_line(x = x, 
                                  y = y, 
                                  scalars = scalars, 
                                  scalars_label = scalars_label, 
                                  scalars_lim = scalars_lim, 
                                  **plot_kwargs)
            # self.plot_scalar_line(point_data, scalars_data,**kwargs)
        elif scalars_data and scalars_mode == ScalarsMode.FILL:
            logger.info(f"Plotting scalar fill for {point_data.label}")
            self.plot_scalar_fill(point_data, scalars_data,**kwargs)
        elif vectors_data:
            logger.info(f"Plotting vectors for {point_data.label}")
            self.plot_vectors(point_data, vectors_data, **kwargs)
        else:
            logger.info(f"Plotting line for {point_data.label}")
            self.plot_line(point_data, **kwargs)
            
        
            
        self.set_energy_label(point_data.points_label, unit_label=point_data.points_units)
        self.set_energy_lim(point_data=point_data)
        self.set_energy_tick_params()
        
        self.set_dos_label(point_data.label, unit_label=point_data.units)
        self.set_dos_lim(lim=data_lim)
        self.set_dos_tick_params()
    
    @expand_grouped_params(use_all=True, default_mode="explode")
    def plot_scalar_line(self,
        x: np.ndarray,
        y: np.ndarray,
        scalars: np.ndarray,
        scalars_label: str | None = None,
        scalars_lim: tuple[float | None, float | None] | None = None,
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        linewidth: float = 1.5,
        linestyle: str = "-",
        alpha: float = 1.0,
        **kwargs
    ):

        # x_data, y_data = self.orient_data(energy_array, data_array)
 
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)


        plot_kwargs = keep_func_kwargs(kwargs, LineCollection)
        plot_kwargs.setdefault("clim", scalars_lim)
        plot_kwargs.setdefault("cmap", cmap)
        plot_kwargs.setdefault("norm", norm)
        plot_kwargs.setdefault("linewidth", linewidth)
        plot_kwargs.setdefault("linestyle", linestyle)
        plot_kwargs.setdefault("alpha", alpha)
        # print(plot_kwargs.get("alpha"))
        plot_kwargs.setdefault("array", scalars)
        plot_kwargs.setdefault("label", scalars_label)
        print(plot_kwargs.get("alpha"))
        
        lc = LineCollection(segments, **plot_kwargs)
        self.ax.add_collection(lc)

        # if show_colorbar == ShowColorbar.SINGLE:
        #     scalars_label = scalars_data.label
        #     scalars_unit = scalars_data.units
        #     if scalars_unit is not None:
        #         scalars_label = f"{scalars_label} ({scalars_unit})"
        #     self.plot_colorbar(label=scalars_label, cmap=cmap, norm=norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
            
        # self.set_energy_label(energy_label, unit_label=energy_units)
        # self.set_energy_lim(point_data=point_data)
        # self.set_energy_tick_params()
        
        # self.set_dos_label(data_label, unit_label=data_units)
        # self.set_dos_lim(point_data=point_data)
        # self.set_dos_tick_params()
        
        # if show_footnote:
        #     self.set_footnote(scalars_data.metadata.get("footnote"))
        
    
    
    @expand_grouped_params("linewidth", "linestyle", "alpha")
    def plot_line(
        self,
        point_data: Property,
        plot_kwargs: dict | None = None,
        channel_mode: ChannelMode | str | None = ChannelMode.NORMAL,
        **kwargs
    ):
        channel_mode = ChannelMode.from_string(channel_mode)
        
        energy_array = point_data.points
        energy_label = point_data.points_label
        energy_units = point_data.points_units
        
        data_array = point_data.to_array()
        data_label = point_data.label
        data_units = point_data.units

        metadata_labels = point_data.metadata.get("label")

        if data_array.ndim == 1:
            n_channels = 1
        else:
            n_channels = data_array.shape[1]
            
        for i_channel in range(n_channels):
            x_data, y_data = self.orient_data(energy_array, data_array[:, i_channel])
            
            if channel_mode == ChannelMode.FLIP and i_channel != 0:
                y_data *= -1.0
                
            if not plot_kwargs:
                plot_kwargs = [keep_func_kwargs(kwargs, self.ax.plot)]
                
            channel_plot_kwargs = plot_kwargs[i_channel] if len(plot_kwargs) > 1 else plot_kwargs[0]
            
            channel_label = channel_plot_kwargs.get("label", None)
            if channel_label is None:
                channel_label = metadata_labels[i_channel]
            self.ax.plot(x_data, y_data, label=channel_label, **channel_plot_kwargs)
  
        
        self.set_energy_label(energy_label, unit_label=energy_units)
        self.set_energy_lim(point_data=point_data)
        self.set_energy_tick_params()
        
        self.set_dos_label(data_label, unit_label=data_units)
        self.set_dos_lim(point_data=point_data)
        self.set_dos_tick_params()
        
    # @expand_grouped_params("linewidth", "linestyle", "alpha", "line_collection_kwargs")
    # def plot_scalar_line(self,
    #     point_data: Property,
    #     scalars_data: Property,
    #     cmap: str | mcolors.Colormap = "plasma",
    #     norm: mcolors.Normalize | str | None = None,
    #     clim: tuple[float | None, float | None] | None = None,
    #     linewidth: float = 1.5,
    #     linestyle: str = "-",
    #     alpha: float = 1.0,
    #     show_colorbar: ShowColorbar | str | bool |None = ShowColorbar.NONE,
    #     show_footnote: bool = True,
    #     line_collection_kwargs: dict | None = None,
    #     channel_mode: ChannelMode | str | None = ChannelMode.NORMAL,
    #     **kwargs
    # ):
    #     show_colorbar = ShowColorbar.from_string(show_colorbar)
    #     channel_mode = ChannelMode.from_string(channel_mode)

    #     energy_array = point_data.points
    #     energy_label = point_data.points_label
    #     energy_units = point_data.points_units
        
    #     data_array = point_data.to_array()
    #     data_label = point_data.label
    #     data_units = point_data.units
        
    #     scalars_array = scalars_data.to_array()

    #     if scalars_array.ndim == 1:
    #         n_channels = 1
    #     else:
    #         n_channels = scalars_array.shape[1]
        
    #     cmaps = self._resolve_cmap(scalars_data=scalars_data, cmap=cmap)
    #     norms = self._resolve_norm(scalars_data=scalars_data, norm=norm, clim=clim)

    #     for i_channel in range(n_channels):
    #         x_data, y_data = self.orient_data(energy_array, data_array[:,i_channel])
    #         if channel_mode == ChannelMode.FLIP and i_channel != 0:
    #             y_data *= -1.0
    #         points = np.column_stack([x_data, y_data]).reshape(-1, 1, 2)
    #         segments = np.concatenate([points[:-1], points[1:]], axis=1)

    #         if not line_collection_kwargs:
    #             line_collection_kwargs = [keep_func_kwargs(kwargs, LineCollection)]
            
  
    #         cmap = cmaps[i_channel]
    #         norm = norms[i_channel]
            
    #         channel_linestyle = linestyle[i_channel] if len(linestyle) > 1 else linestyle[0]
    #         channel_alpha = alpha[i_channel] if len(alpha) > 1 else alpha[0]
    #         channel_linewidth = linewidth[i_channel] if len(linewidth) > 1 else linewidth[0]
    #         channel_line_collection_kwargs = line_collection_kwargs[i_channel] if len(line_collection_kwargs) > 1 else line_collection_kwargs[0]

            
    #         lc = LineCollection(segments, cmap=cmap, norm=norm, **channel_line_collection_kwargs)
    #         lc.set_array(scalars_array[:,i_channel])
    #         lc.set_linewidth(channel_linewidth)
    #         lc.set_linestyle(channel_linestyle)
    #         lc.set_alpha(channel_alpha)

    #         self.ax.add_collection(lc)
            
    #         if show_colorbar == ShowColorbar.PER_CHANNEL:
    #             self.plot_colorbar(label=scalars_data.metadata.get("label")[i_channel], cmap=cmap, norm=norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
            
    #     if show_colorbar == ShowColorbar.SINGLE:
    #         scalars_label = scalars_data.label
    #         scalars_unit = scalars_data.units
    #         if scalars_unit is not None:
    #             scalars_label = f"{scalars_label} ({scalars_unit})"
    #         self.plot_colorbar(label=scalars_label, cmap=cmap, norm=norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
            
    #     self.set_energy_label(energy_label, unit_label=energy_units)
    #     self.set_energy_lim(point_data=point_data)
    #     self.set_energy_tick_params()
        
    #     self.set_dos_label(data_label, unit_label=data_units)
    #     self.set_dos_lim(point_data=point_data)
    #     self.set_dos_tick_params()
        
    #     if show_footnote:
    #         self.set_footnote(scalars_data.metadata.get("footnote"))
            
            
    
    @expand_grouped_params("fill_between_kwargs")
    def plot_scalar_fill(self,
        point_data: Property,
        scalars_data: Property,
        cmap: str | mcolors.Colormap = "plasma",
        norm: mcolors.Normalize | str | None = None,
        clim: tuple[float | None, float | None] | None = None,
        show_colorbar: ShowColorbar | str | bool = ShowColorbar.SINGLE,
        channel_mode: ChannelMode | str | None = None,
        fill_between_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        **kwargs
    ):
        channel_mode = ChannelMode.from_string(channel_mode)
        show_colorbar = ShowColorbar.from_string(show_colorbar)
        
        energy_array = point_data.points
        energy_label = point_data.points_label
        energy_units = point_data.points_units
        n_energies = energy_array.size
        
        data_array = point_data.to_array()
        data_label = point_data.label
        data_units = point_data.units

        scalars_array = scalars_data.to_array()

        n_channels = scalars_array.shape[1]

        norms = self._resolve_norm(scalars_data, norm=norm, clim=clim)
        cmaps = self._resolve_cmap(scalars_data, cmap=cmap)
        
        for i_channel in range(n_channels):
            channel_data_array = data_array[:,i_channel]
            channel_scalars_array = scalars_array[:,i_channel]
            if channel_mode == ChannelMode.FLIP and i_channel != 0:
                channel_data_array *= -1.0
            channel_cmap = cmaps[i_channel]
            channel_norm = norms[i_channel]
            if not fill_between_kwargs:
                fill_between_kwargs = [keep_func_kwargs(kwargs, self.fill_between)]
                
            channel_fill_between_kwargs = fill_between_kwargs[i_channel] if len(fill_between_kwargs) > i_channel else fill_between_kwargs[0]
                
            for idx in range(n_energies - 1):
                segment_color = channel_cmap(channel_norm(channel_scalars_array[idx]))
                x_segment = energy_array[idx : idx + 2]
                y_segment = channel_data_array[idx : idx + 2]
                
                self.fill_between(x_segment, y_segment, color=segment_color, **channel_fill_between_kwargs)

            if show_colorbar == ShowColorbar.PER_CHANNEL:
                self.plot_colorbar(scalars_data.metadata.get("label")[i_channel], cmap=channel_cmap, norm=channel_norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
        
        if show_colorbar == ShowColorbar.SINGLE:
            scalars_label = scalars_data.label
            scalars_unit = scalars_data.units
            if scalars_unit is not None:
                scalars_label = f"{scalars_label} ({scalars_unit})"
            self.plot_colorbar(label=scalars_label, cmap=cmap, norm=norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
            
        self.set_energy_label(energy_label, unit_label=energy_units)
        self.set_energy_lim(point_data=point_data)
        self.set_energy_tick_params()
        
        self.set_dos_label(data_label, unit_label=data_units)
        self.set_dos_lim(point_data=point_data)
        self.set_dos_tick_params()
        
        self.draw_baseline(value=0.0)
        
    @expand_grouped_params("quiver_kwargs")
    def plot_vectors(self,
        point_data: Property,
        vectors_data: Property,
        skip:int=1,
        angles:str='uv',
        scale=None,
        scale_units:str='inches',
        units:str='inches',
        color=None,
        clim: tuple[float | None, float | None] | None = None,
        norm: mcolors.Normalize | str | None = None,
        cmap: str | mcolors.Colormap = "plasma",
        channel_mode: ChannelMode | str | None = None,
        show_colorbar: ShowColorbar | str | bool | None = True,
        quiver_kwargs: dict | None = None,
        **kwargs
    ):
        channel_mode = ChannelMode.from_string(channel_mode)
        show_colorbar = ShowColorbar.from_string(show_colorbar)
        
        energy_array = point_data.points
        energy_label = point_data.points_label
        energy_units = point_data.points_units
        n_energies = energy_array.size
        
        data_array = point_data.to_array()
        data_label = point_data.label
        data_units = point_data.units
        
        vectors_array = vectors_data.to_array()

        n_channels = vectors_array.shape[1]
        logger.debug(f"n_channels: {n_channels}")
        
        norms = self._resolve_norm(vectors_data, norm = norm, clim = clim)
        cmaps = self._resolve_cmap(vectors_data, cmap = cmap)
        
        
        # Plot quivers
        for i_channel in range(n_channels):
            u = vectors_array[...,i_channel]                # Arrow y-component
            v = np.ones_like(vectors_array[...,i_channel])  # Arrow x-component
            vector_norms = vectors_array[...,i_channel]
            data_channel = data_array[...,i_channel]
            
            if channel_mode == ChannelMode.FLIP and i_channel != 0:
                data_channel *= -1.0
                
            channel_cmap = cmaps[i_channel] if len(cmaps) > 1 else cmaps[0]
            channel_norm = norms[i_channel] if len(norms) > 1 else norms[0]
            
            if not quiver_kwargs:
                quiver_kwargs = [keep_func_kwargs(kwargs, self.ax.quiver)]
                
            channel_quiver_kwargs = quiver_kwargs[i_channel] if len(quiver_kwargs) > 1 else quiver_kwargs[0]
            
            quiver_args = []
            quiver_args.append(energy_array[::skip])
            quiver_args.append(data_channel[::skip])
            quiver_args.append(u[::skip])
            quiver_args.append(v[::skip])
            if color is None:
                quiver_args.append(vector_norms[::skip])
                
            qv = self.ax.quiver(
                *quiver_args,
                angles=angles,
                scale=scale,
                scale_units=scale_units,
                units = units,
                color=color,
                cmap=channel_cmap,
                norm=channel_norm,
                **channel_quiver_kwargs)
            
            if show_colorbar == ShowColorbar.PER_CHANNEL:
                self.plot_colorbar(vectors_data.metadata.get("label")[i_channel], cmap=channel_cmap, norm=channel_norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
        
        
        if show_colorbar == ShowColorbar.SINGLE:
            scalars_label = vectors_data.label
            scalars_unit = vectors_data.units
            if scalars_unit is not None:
                scalars_label = f"{scalars_label} ({scalars_unit})"
            self.plot_colorbar(label=scalars_label, cmap=cmap, norm=norm, **keep_func_kwargs(kwargs, self.plot_colorbar))
            
            
        self.set_energy_label(energy_label, unit_label=energy_units)
        self.set_energy_lim(point_data=point_data)
        self.set_energy_tick_params()
        
        self.set_dos_label(data_label, unit_label=data_units)
        self.set_dos_lim(point_data=point_data)
        self.set_dos_tick_params()
        
        self.draw_baseline(0.0)
        
        return qv
    


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_clim(self, scalars_data: Property, clim: tuple[float | None, float | None] | None = None) -> tuple[float, float]:
        if clim is not None:
            return clim
        
  
        scalars_lim = scalars_data.rounded_data_lim
        
        if scalars_lim is not None:
            return scalars_lim
        else:
            return None

    def _resolve_norm(self, 
                      scalars_data: Property,
                      clim: tuple[float | None, float | None] | None = None,
                      norm: mcolors.Normalize | str | None = None, 
                      clip: bool = True) -> mcolors.Normalize:
        
        if clim is None:
            clims = self._resolve_clim(scalars_data, clim=clim)
        n_channels = scalars_data.n_channels
        
            
        norms = []
        for i_channel in range(n_channels):
            vmin, vmax = clims[i_channel]
            if norm is None:
                tmp_norm = mcolors.Normalize(vmin, vmax, clip=clip)
            elif isinstance(norm, str):
                tmp_norm = plt.get_norm(norm)(vmin, vmax)
            elif isinstance(norm, mcolors.Normalize):
                tmp_norm = norm
            else:
                raise ValueError(f"Invalid norm: {norm}")
            norms.append(tmp_norm)
        return norms
    
    def _resolve_cmap(self, 
                      scalars_data: Property, 
                      cmap: str | mcolors.Colormap | None
                      ) -> list[mcolors.Colormap]:
        n_channels = scalars_data.n_channels
        
        if not isinstance(cmap, Iterable) or isinstance(cmap, str):
            cmap = [cmap] * n_channels

            
        cmaps = []
        for i_channel in range(n_channels):
            channel_cmap = cmap[i_channel]
            if isinstance(channel_cmap, str):
                channel_cmap = plt.get_cmap(channel_cmap)
            elif isinstance(cmap, mcolors.Colormap):
                pass
            else:
                raise TypeError(f"Invalid cmap type: {type(channel_cmap)}")

            cmaps.append(channel_cmap)
        return cmaps

    def plot_colorbar(self, 
                    label: str,
                    cmap: str | mcolors.Colormap = "plasma",
                    norm: mcolors.Normalize | str | None = None,
                    pad: float = 0.02, 
                    shrink: float = 0.8,
                    orientation: str = "vertical",
                    location: str = "right",
                    set_colorbar_label_kwargs: dict | None = None,
                    set_colorbar_tick_params_kwargs: dict | None = None,
                    **kwargs) -> None:
        
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
      
        self.set_colorbar_label(label, **set_colorbar_label_kwargs)
        
        set_colorbar_tick_params_kwargs = set_colorbar_tick_params_kwargs if set_colorbar_tick_params_kwargs is not None else {}
        self.set_colorbar_tick_params(**set_colorbar_tick_params_kwargs)

    def set_colorbar_label(self, label: str, 
                           rotation=270,
                           labelpad=12,
                           **kwargs):

        self._validate_colorbar()
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            self.colorbar_axes.set_ylabel(label, rotation=rotation, labelpad=labelpad, 
                                  **kwargs)
        else:
            self.colorbar_axes.set_xlabel(label, rotation=rotation, labelpad=labelpad, 
                              **kwargs)
        
    def set_colorbar_tick_params(self, **kwargs):
        self._validate_colorbar()
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            self.colorbar_axes.tick_params(axis="y", **kwargs)
        else:
            self.colorbar_axes.tick_params(axis="x", **kwargs)
        
    def set_colorbar_ticklabels(self, n_ticks: int = 5, clim: tuple[float, float] = None, labels: Sequence[str] = None, **kwargs):
        
        self._validate_colorbar()
        if (clim is None and labels is None) or (clim is not None and labels is not None):
            raise ValueError("Either clim or labels must be provided")
        elif clim is not None and labels is None:
            labels = [f"{x:.2f}" for x in np.linspace(clim[0], clim[1], n_ticks)]
        
        if self.colorbar_orientation is AxesOrientation.VERTICAL:
            self.colorbar_axes.set_yticklabels(labels, **kwargs)
        else:
            self.colorbar_axes.set_xticklabels(labels, **kwargs)
    
    def set_colorbar_ticks(self, 
                           ticks: Sequence[float] | ticker.Locator | None = None,
                           labels: Sequence[str] = None,
                           n_ticks: int = 5,
                           **kwargs):
        
    
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
            raise ValueError(f"Ticks and labels must have the same length: {len(ticks)} != {len(labels)}")
        
    def _validate_colorbar(self):
        if not hasattr(self, "colorbar"):
            raise ValueError("There is no colorbar for this plotter. call colorbar() or plot() with show_colorbar=True")

    # ------------------------------------------------------------------
    # Orientation utilities
    # ------------------------------------------------------------------
    
    def orient_data(self, energies: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        energies = np.asarray(list(energies), dtype=np.float64)
        values = np.asarray(list(values), dtype=np.float64)

        if self.orientation is AxesOrientation.HORIZONTAL:
            return self.ax.fill_between(energies, values, baseline, **kwargs)
        return self.ax.fill_betweenx(energies, baseline, values, **kwargs)

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
    def draw_baseline(self, value: float, 
                      color="black", 
                      linewidth=0.8, 
                      linestyle="--", 
                      **kwargs) -> None:
        
        all_kwargs = dict(color=color, linewidth=linewidth, linestyle=linestyle, **kwargs)
        if self.orientation is AxesOrientation.HORIZONTAL:
            self.ax.axhline(value, **all_kwargs)
        else:
            self.ax.axvline(value, **all_kwargs)

    def draw_fermi(self, value: float,
                   color="tab:red", 
                   linewidth=1.0, 
                   linestyle="--", 
                   **kwargs) -> None:
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

    def set_footnote(self, 
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
                     **kwargs) -> None:
        
        footnote = f"footnote: {footnote}"
        if footnote is not None:
            self.ax.annotate(
            footnote,
            xy=xy, xycoords=xycoords,
            xytext=xytext, textcoords=textcoords,
            ha=ha, va=va, fontsize=fontsize, color=color,
            annotation_clip=annotation_clip,
            **kwargs)

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


    def _resolve_channel_param(self, param: Iterable[Any] | None) -> Sequence[Any]:
        if len(param) > 1:
            return param
        return param[0]