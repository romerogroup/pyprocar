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
from pyprocar.utils.inspect_utils import keep_func_kwargs

logger = logging.getLogger(__name__)

def get_class_attributes(cls):
    class_attributes = {}
    for name, value in cls.__dict__.items():
        if not callable(value) and not name.startswith('__'):
            class_attributes[name] = value
    return class_attributes


class ScatterPlot(BasePlotter):
    cmap: str | mpcolors.Colormap = "plasma"
    norm: Optional[str | mpcolors.Normalize | type] = "auto"
    clim: Optional[Tuple[Optional[float], Optional[float]]] = None
    show_colorbar: bool | None = True
    scatter_kwargs: dict = {}
    colorbar_title: str | None = "Scalars"
    colorbar_kwargs: dict = {}
    
    # Resolve colormap
    def _plot(self, kpath: KPath, bands: np.ndarray, scalars: np.ndarray = None, **kwargs):
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        # Validate data
        bands, scalars, _ = self._validate_data(bands, scalars)
        
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=self.cmap,
            norm=self.norm,
            clim=self.clim,
        )
        self.scatter_kwargs["cmap"] = resolved_cmap
        self.scatter_kwargs["norm"] = resolved_norm
        self.scatter_kwargs["clim"] = self.clim if "clim" not in self.scatter_kwargs else self.scatter_kwargs["clim"]
    
        created_collections: dict[tuple[int, int], PathCollection] = {}
        
        # Plot scatter
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin in range(n_spin_channels):
            data=None
            if scalars is not None:
                data = scalars[..., ispin]
                
            for iband in range(n_bands):
                y = bands[:, iband, ispin]
                c_vals = None if data is None else data[:, iband]
                coll = self.ax.scatter(self.x, y, c=c_vals, **self.scatter_kwargs)
                created_collections[(iband, ispin)] = coll

        # Add colorbar if requested
        if scalars is not None and self.show_colorbar:
            self.colorbar_kwargs["label"] = self.colorbar_title
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **self.colorbar_kwargs)
                    
        return created_collections
    
    

    
# if __name__ == "__main__":
#     scatter = Scatter(clim=(0, 10))
#     print(scatter.class_plot_params)
#     print(scatter.instance_plot_params)