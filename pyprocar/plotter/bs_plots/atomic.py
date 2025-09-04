__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import json
import logging
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
from pyprocar.plotter.bs_plots.base import BasePlotter

logger = logging.getLogger(__name__)

def get_class_attributes(cls):
    class_attributes = {}
    for name, value in cls.__dict__.items():
        if not callable(value) and not name.startswith('__'):
            class_attributes[name] = value
    return class_attributes


class AtomicLevelsPlot(BasePlotter):
    elimit: Tuple[float, float] = None
    labels_prefix: str = "s"
    cmap: str = "plasma"
    norm: Union[mpcolors.Normalize, type] = None
    clim: Tuple[float, float] = (None, None)
    show_colorbar: bool | None = True
    colorbar_kwargs: dict | None = None
    linewidth: float = None
    line_collection_kwargs: dict | None = None
    show_text: bool = True
    
    # Resolve colormap
    def _plot(self, kpath: KPath, bands: np.ndarray, scalars: np.ndarray = None, **kwargs):
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
            cmap=self.cmap,
            norm=self.norm,
            clim=self.clim,
        )

        # Remove x ticks for atomic levels and compute text bbox in data units
        self.ax.xaxis.set_major_locator(plt.NullLocator())

        
        # Determine a representative text bbox in data coordinates
        n_bands = bands.shape[1]
        sample_text = f"{self.labels_prefix}-0 : b-{n_bands}"
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
        last_lc = None
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
                lc = LineCollection(segments, **self.line_collection_kwargs)
                if scalars is not None:
                    level_scalar = float(np.asarray(scalars[0, iband, ispin]))
                    lc.set_array(np.array([level_scalar]))
                self.ax.add_collection(lc)
                last_lc = lc

                if self.show_text:
                    # if vertical overlap, toggle lateral shift
                    if last_y is not None and y < (last_y + h):
                        shift_state = 1 - shift_state
                    else:
                        shift_state = 0
                    x_pos = x_base + (2.0 * w if shift_state == 1 else 0.0)
                    # Clamp inside current xlim
                    xmin, xmax = self.ax.get_xlim()
                    x_pos = min(max(x_pos, xmin + 0.05 * w), xmax - 0.05 * w)
                    self.ax.text(x_pos, y, f"{self.labels_prefix}-{ispin} : b-{iband+1}")
                    last_y = y
        
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

    

    
# if __name__ == "__main__":
#     scatter = Scatter(clim=(0, 10))
#     print(scatter.class_plot_params)
#     print(scatter.instance_plot_params)