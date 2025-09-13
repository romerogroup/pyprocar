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


class OverlayPlot(BasePlotter):
    colors: Optional[List[str]] = None
    norm: str | mpcolors.Normalize | type | None = "auto"
    clim: Tuple[float | None, float | None] | None = None
    cmap: str | mpcolors.Colormap = "plasma"
    cmap_list: Optional[List[str]] = None
    fill_alpha: float = 0.3
    linewidth: float = 1.0
    fill_between_kwargs: dict | None = {}
    show_colorbar: bool | None = None
    colorbar_kwargs: dict | None = {}
    
    # Resolve colormap
    def _plot(self, kpath: KPath, bands: np.ndarray, weights: List[np.ndarray], labels: Optional[List[str]] = None, **kwargs):
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)

        # Validate data
        bands, _, _ = self._validate_data(bands=bands)
        for i, weight in enumerate(weights):
            _, weight, _ = self._validate_data(bands=bands, scalars=weight)
            weights[i] = weight

        # Default colormap names per overlay, fallback if explicit colors not given
        if self.cmap_list is None:
            self.cmap_list = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
        if self.colors is None:
            # Sample mid tone from each colormap to derive RGBA colors
            colors = []
            for i in range(len(weights)):
                cmap_name = self.cmap_list[i % len(self.cmap_list)]
                cmap = plt.get_cmap(cmap_name)
                colors.append(cmap(0.7))
                
            cmaps=[]
            norms=[]
            scalar_mappables=[]
            for i in range(len(weights)):
                resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
                    data=weights[i],
                    cmap=self.cmap_list[i % len(self.cmap_list)],
                    norm=self.norm,
                    clim=self.clim,
                )
                cmaps.append(resolved_cmap)
                norms.append(resolved_norm)
                scalar_mappables.append(scalar_mappable)
                

        # Iterate over weights and fill between
        legend_handles = []
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
                    
                    if colors is None:
                        cmap = cmaps[widx]
                        norm = norms[widx]
                    else:
                        cmap = None
                        norm = None
                    
                    self.ax.fill_between(
                        self.x,
                        y - width_arr / 2.0,
                        y + width_arr / 2.0,
                        color=color,
                        cmap=cmap,
                        norm=norm,
                        **self.fill_between_kwargs,
                    )
            legend_label = labels[widx] if labels and widx < len(labels) else f"overlay-{widx+1}"
            legend_handles.append(mpatches.Patch(color=color, label=legend_label, alpha=self.fill_alpha))
        
        self._legend_handles = legend_handles
        self.legend()

    

    
# if __name__ == "__main__":
#     scatter = Scatter(clim=(0, 10))
#     print(scatter.class_plot_params)
#     print(scatter.instance_plot_params)