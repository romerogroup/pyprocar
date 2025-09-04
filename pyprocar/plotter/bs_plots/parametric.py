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


class ParametricPlot(BasePlotter):
    cmap: str | mpcolors.Colormap = "plasma"
    norm: Optional[str | mpcolors.Normalize | type] = "auto"
    clim: Optional[Tuple[Optional[float], Optional[float]]] = None
    linewidth:float = 2.0
    collection_kwargs: dict | None = None
    show_colorbar: bool | None = True
    colorbar_kwargs: dict | None = None
    
    # Resolve colormap
    def _plot(self, kpath: KPath, bands: np.ndarray, scalars: np.ndarray = None, **kwargs):
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        # Validate data
        bands, scalars, _ = self._validate_data(bands, scalars)
        
        resolved_norm, resolved_cmap, self.scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=self.cmap,
            norm=self.norm,
            clim=self.clim,
        )
    
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        # Validate data
        bands, scalars, _ = self._validate_data(bands=bands, scalars=scalars)
        
        # Resolve colormap
        resolved_norm, resolved_cmap, self.scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=self.cmap,
            norm=self.norm,
            clim=self.clim,
        )
        
        # Prepare data
        width_weights = np.ones_like(bands)
        mbands = np.ma.masked_array(bands, False)
        # if width_weights is not None:
        #     logger.info(f"___Applying width mask___")
        #     mbands = np.ma.masked_array(
        #         bands,
        #         np.abs(width_weights) < width_mask,
        #     )
        # if color_mask is not None:
        #     logger.info(f"___Applying color mask___")
        #     mbands = np.ma.masked_array(
        #         self.ebs.bands,
        #         np.abs(color_weights) < color_mask,
        #     )
        
        # Plot parametric bands
        last_lc = None
        created_collections: dict[tuple[int, int], LineCollection] = {}
        
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin_channel in range(n_spin_channels):
            for iband in range(n_bands):
                points = np.array([self.x, mbands[:, iband, ispin_channel]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments,  **self.collection_kwargs)
                
                # Handle colors
                if scalars is not None:
                    lc.set_array(scalars[:, iband, ispin_channel])
                    lc.set_cmap(resolved_cmap)
                    lc.set_norm(resolved_norm)
                    
                lc.set_linewidth(width_weights[:, iband, ispin_channel] * self.linewidth)
                self.ax.add_collection(lc)
                last_lc = lc
                created_collections[(iband, ispin_channel)] = lc
                
        # Set default plot parameters
        self.set_xlim()
        ymin = float(bands.min())
        ymax = float(bands.max())
        elimit = (ymin, ymax)
        self.set_ylim(elimit)
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()
        
        # Record exportable data
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                bkey = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[bkey] = bands[:, iband, ispin]
                if scalars is not None:
                    pkey = f"projections__parametric__band-{iband}_spinChannel-{ispin}"
                    self.values_dict[pkey] = scalars[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_collections
    
    

    
# if __name__ == "__main__":
#     scatter = Scatter(clim=(0, 10))
#     print(scatter.class_plot_params)
#     print(scatter.instance_plot_params)