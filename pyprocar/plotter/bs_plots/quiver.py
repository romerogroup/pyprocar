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
from pyprocar.core.property_store import Property
from pyprocar.plotter.bs_plots.base import BasePlotter

logger = logging.getLogger(__name__)

def get_class_attributes(cls):
    class_attributes = {}
    for name, value in cls.__dict__.items():
        if not callable(value) and not name.startswith('__'):
            class_attributes[name] = value
    return class_attributes


class QuiverPlot(BasePlotter):
    skip:int=1
    angles:str='uv'
    scale=None
    scale_units:str='inches'
    units:str='inches'
    color=None
    cmap: str | mpcolors.Colormap = "plasma"
    norm: str | mpcolors.Normalize | type | None = "auto"
    clim:Tuple[float | None, float | None] | None = None
    show_colorbar: bool | None = True
    quiver_kwargs: dict = {}
    colorbar_title: str | None = "Vectors"
    colorbar_kwargs: dict = {}
    
    
    # Resolve colormap
    def _plot(self, kpath: KPath, bands: np.ndarray, vectors: np.ndarray | Property = None, **kwargs):

        # Validate data
        bands, _, vectors = self._validate_data(bands=bands, vectors=vectors)
        
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        
        
        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=vectors,
            cmap=self.cmap,
            norm=self.norm,
            clim=self.clim,
        )
        self.quiver_kwargs["cmap"] = resolved_cmap
        self.quiver_kwargs["norm"] = resolved_norm
        self.quiver_kwargs["clim"] = self.clim if "clim" not in self.quiver_kwargs else self.quiver_kwargs["clim"]
    
        # Plot quivers
        created_quivers: dict[tuple[int, int], object] = {}
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin_channel in range(n_spin_channels):
            u = vectors[...,ispin_channel]                # Arrow y-component
            v = np.ones_like(vectors[...,ispin_channel])  # Arrow x-component
            vector_norms = vectors[...,ispin_channel]
            current_bands = bands[...,ispin_channel]
        
            for iband in range(n_bands):
                band_u = u[...,iband]
                band_v = v[...,iband]
                band_current_bands = current_bands[...,iband]
    
                quiver_args = []
                quiver_args.append(self.x[::self.skip])
                quiver_args.append(band_current_bands[::self.skip])
                quiver_args.append(band_u[::self.skip])
                quiver_args.append(band_v[::self.skip])
                
                if self.color is None:
                    quiver_args.append(vector_norms[...,iband])
                    
                qv = self.ax.quiver(
                    *quiver_args,
                    angles=self.angles,
                    scale=self.scale,
                    scale_units=self.scale_units,
                    units = self.units,
                    color=self.color,
                    **self.quiver_kwargs)
                created_quivers[(iband, ispin_channel)] = qv
                
        if vectors is not None and self.show_colorbar:
            self.colorbar_kwargs["label"] = self.colorbar_title
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **self.colorbar_kwargs)
                
        return created_quivers
    
    

    
# if __name__ == "__main__":
#     scatter = Scatter(clim=(0, 10))
#     print(scatter.class_plot_params)
#     print(scatter.instance_plot_params)