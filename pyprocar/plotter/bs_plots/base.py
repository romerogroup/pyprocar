__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as mpcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from pyprocar.core import KPath

logger = logging.getLogger(__name__)

def get_class_attributes(cls):
    class_attributes = {}
    for name, value in cls.__dict__.items():
        if not callable(value) and not name.startswith('__'):
            class_attributes[name] = value
    return class_attributes

class BasePlotter(ABC):

    def __init__(self, figsize=(8, 6), dpi=100, ax=None, **kwargs):
        self.figsize = figsize
        self.dpi = dpi

        self.ax = ax
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.fig = plt.gcf()

        self.data_store = {}
        self.values_dict = {}
        self._legend_handles = []
        
        self.x = None
        self.update_instance_params(**kwargs)
        
    @abstractmethod
    def _plot(self, kpath: KPath, bands: np.ndarray,  **kwargs):
        pass
    

    @property
    def class_plot_params(self):
        class_attrs = get_class_attributes(self.__class__)
        tmp_attrs = class_attrs.copy()
        for attr_name, value in tmp_attrs.items():
            if isinstance(value, property):
                class_attrs.pop(attr_name)
            elif attr_name == "_abc_impl":
                class_attrs.pop(attr_name)
        return class_attrs
    
    @property
    def instance_plot_params(self):
        class_attrs = get_class_attributes(self.__class__)
        
        instance_attrs = {}
        for attr_name, value in class_attrs.items():
            instance_attrs[attr_name] = getattr(self, attr_name, None)

        return instance_attrs
    
    @property
    def plot_params(self):
        return self.instance_plot_params
    
    def update_instance_params(self, **kwargs):
        plot_params = self.instance_plot_params
        for plot_param_name, plot_param_value in kwargs.items():
            if plot_param_name in plot_params:
                setattr(self, plot_param_name, plot_param_value)
        return self.instance_plot_params
        
    
    def plot(self, kpath: KPath, bands: np.ndarray, **kwargs):
        self.update_instance_params(**kwargs)
        self._plot(kpath, bands, **kwargs)
        
    def set_xlim(self, xlim:List[float] = None, **kwargs):
        if xlim is None:
            xlim = (self.x[0], self.x[-1])
            
        self.ax.set_xlim(xlim, **kwargs)
        
    def set_ylim(self, ylim:List[float] = None, **kwargs):
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

    def set_xticks(self, tick_positions: List[int] = None, tick_names: List[str] = None, color: str = "black"):
        """Set high-symmetry tick marks and labels using the current k-path.

        Parameters
        ----
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
        if tick_positions is None and hasattr(self, "kpath") and self.kpath is not None:
            tick_positions = self.kpath.tick_positions
        if tick_names is None and hasattr(self, "kpath") and self.kpath is not None:
            tick_names = self.kpath.tick_names

        if tick_positions is not None:
            for ipos in tick_positions:
                if 0 <= ipos < len(self.x):
                    self.ax.axvline(self.x[ipos], color=color)
            self.ax.set_xticks(self.x[tick_positions])
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)

    def set_yticks(self, major: float = None, minor: float = None, interval: List[float] = None):
        """Set y-axis tick locators using heuristics if not provided.

        Parameters
        ----
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

    def set_xlabel(self, label: str = "K vector", **kwargs):
        """Set x-axis label.

        Parameters
        ----
        label : str, optional
            Axis label.
        """
        self.ax.set_xlabel(label, **kwargs)

    def set_ylabel(self, label: str = r"E (eV)", **kwargs):
        """Set y-axis label.

        Parameters
        ----
        label : str, optional
            Axis label.
        """
        self.ax.set_ylabel(label, **kwargs)

    def set_title(self, title: str = "Band Structure", **kwargs):
        """Set plot title."""
        self.ax.set_title(title, **kwargs)

    def set_colorbar_title(self, title: str = "Atomic Orbital Projections", **kwargs):
        """Set colorbar title if a colorbar exists."""
        if self.cb is not None:
            self.cb.ax.tick_params(labelsize=kwargs.pop("labelsize", None))
            self.cb.set_label(title, **kwargs)

    def draw_fermi(self, fermi_level: float = 0.0, color: str = "k", linestyle: str = "--", linewidth: float = 1.0):
        """Draw a horizontal Fermi level line."""
        self.ax.axhline(y=fermi_level, color=color, linestyle=linestyle, linewidth=linewidth)

    def grid(self, enabled: bool = True, which: str = "both", color: str = "#cccccc", linestyle: str = ":", linewidth: float = 0.8):
        """Configure grid display."""
        if enabled:
            self.ax.grid(enabled, which=which, color=color, linestyle=linestyle, linewidth=linewidth)

    def legend(self, labels: List[str] = None, **kwargs):
        """Show legend; uses stored handles when available.

        Parameters
        ----
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

    def save(self, filename: str = "bands.pdf", dpi: Optional[int] = None, bbox_inches: str = "tight"):
        """Save the current figure to disk."""
        plt.savefig(filename, dpi=(dpi or self.dpi), bbox_inches=bbox_inches)
        plt.clf()

    def export_data(self, filename: str):
        """Export recorded plot arrays to CSV/TXT/JSON/DAT.

        Parameters
        ----
        filename : str
            Output path; extension defines format.
        """
        possible_file_types = ["csv", "txt", "json", "dat"]
        file_type = filename.split(".")[-1]
        if file_type not in possible_file_types:
            raise ValueError(f"The file type must be {possible_file_types}")
        if not self.values_dict:
            raise ValueError("No values recorded. Plot first before exporting.")

        values = {}
        for key, value in self.values_dict.items():
            if value is None:
                continue
            arr = np.atleast_1d(value)
            if arr.size > 0:
                values[key] = arr

        column_names = list(values.keys())
        sorted_columns = []
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
            serializable = {k: np.asarray(v).tolist() for k, v in values.items()}
            with open(filename, "w") as outfile:
                json.dump(serializable, outfile)

    # ---- helpers ----
    def _record_kpath_exports(self, kpath: KPath):
        self.values_dict["kpath_values"] = self.x
        tick_names = []
        if kpath is not None:
            for i, _x in enumerate(self.x):
                name = ""
                for i_tick, pos in enumerate(kpath.tick_positions):
                    if i == pos:
                        name = kpath.tick_names[i_tick]
                        break
                tick_names.append(name)
        self.values_dict["kpath_tick_names"] = tick_names
        
    def show(self):
        plt.show()    
        
    def _validate_data(self, 
                               bands: np.ndarray | None, 
                               scalars: np.ndarray | None = None,
                               vectors: np.ndarray | None = None):
        if bands.ndim == 2:
            bands = bands[..., np.newaxis]
        n_spin_channels = bands.shape[-1]
 
        if scalars is not None and scalars.ndim == 2:
            scalars = scalars[..., np.newaxis]
        if scalars is not None and scalars.ndim != bands.ndim:
            error_message = "scalars must have the same number of dimensions as bands"
            error_message += f"bands has {bands.ndim} dimensions, scalars has {scalars.ndim} dimensions"
            error_message += f"Use a built in method in ElectronicBandStructurePath to get the scalars"
            raise ValueError(error_message)
        elif scalars is not None and scalars.ndim == bands.ndim and scalars.shape[-1] != n_spin_channels:
            error_message = "scalars must have the same number of spin channels as bands."
            error_message += f"bands has {n_spin_channels} spin channels, scalars has {scalars.shape[-1]} spin channels\n"
            error_message += f"This error is likely due to a non-colinear calculation where the scalars can have spin components\n"
            raise ValueError(error_message)
        
        if vectors is not None and vectors.ndim == 2:
            vectors = vectors[...,np.newaxis]
        if vectors is not None and vectors.ndim != bands.ndim:
            error_message = "vectors must have the same number of dimensions as bands"
            error_message += f"bands has {bands.ndim} dimensions, vectors has {vectors.ndim} dimensions"
            error_message += f"Use a built in method in ElectronicBandStructurePath to get the scalars"
            raise ValueError(error_message)
        elif vectors is not None and vectors.ndim == bands.ndim and vectors.shape[-1] != n_spin_channels:
            error_message = "vectors must have the same number of spin channels as bands."
            error_message += f"bands has {n_spin_channels} spin channels, vectors has {vectors.shape[1]} spin channels\n"
            error_message += f"This error is likely due to a non-colinear calculation where the vectors can have spin components\n"
            raise ValueError(error_message)
        
        return bands, scalars, vectors
    
    # def colorbar(self, **kwargs):
    #     if not hasattr(self, "scalar_mappable"):
    #         raise ValueError("No scalar mappable to plot colorbar for")
    #     if self.scalar_mappable is None:
    #         raise ValueError("Scalar mappable is None, plot the data first")
    #     self.cb = self.fig.colorbar(self.scalar_mappable, ax=self.ax, **kwargs)

    #     return self.cb
    
    # ---- color mapping resolver ----
    def _resolve_colormap(self,
                          data: np.ndarray | None,
                          cmap: str | mpcolors.Colormap = "plasma",
                          norm: str | mpcolors.Normalize | type | None = "auto",
                          clim: Tuple[float | None, float | None] | None = None,
                          ):
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
       
        vmin = None
        vmax = None
        if clim is not None:
            vmin, vmax = clim
        # Compute from data if needed
        if data is not None:
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
            int_pad = 0.0    # extra pad after floor/ceil

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

            if infer_vmin:
                vmin = _snap_min(vmin)
            if infer_vmax:
                vmax = _snap_max(vmax)

            # Ensure vmin < vmax; if equal after snapping, expand slightly
            if vmin is not None and vmax is not None and vmin >= vmax:
                eps = 1e-8
                if infer_vmin:
                    vmin = vmin - eps
                else:
                    vmax = vmax + eps

        # Resolve Normalize
        if isinstance(norm, mpcolors.Normalize):
            norm_obj = norm
        elif isinstance(norm, type) and issubclass(norm, mpcolors.Normalize):
            norm_obj = norm(vmin=vmin, vmax=vmax)
        elif norm in ("auto", None):
            norm_obj = mpcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm_obj = mpcolors.Normalize(vmin=vmin, vmax=vmax)

        # Resolve cmap
        cmap_obj = cmap
        # Build a ScalarMappable for colorbar convenience
        scalar_mappable = cm.ScalarMappable(norm=norm_obj, cmap=cmap_obj)
        if data is not None:
            scalar_mappable.set_array(np.asarray(data).ravel())
            
        return norm_obj, cmap_obj, scalar_mappable
    
    
    
    
        