from typing import List, Optional, Tuple

import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np

from pyprocar.core import KPath
from pyprocar.plotter.bs_plots.atomic import AtomicLevelsPlot
from pyprocar.plotter.bs_plots.base import BasePlotter
from pyprocar.plotter.bs_plots.overlay import OverlayPlot
from pyprocar.plotter.bs_plots.parametric import ParametricPlot
from pyprocar.plotter.bs_plots.quiver import QuiverPlot
from pyprocar.plotter.bs_plots.scatter import ScatterPlot


class BandsPlotter(BasePlotter):
    cmap: str | mpcolors.Colormap = "plasma"
    norm: Optional[str | mpcolors.Normalize | type] = "auto"
    clim: Optional[Tuple[Optional[float], Optional[float]]] = None
    
    
    def __init__(self, figsize=(8, 6), dpi=100, ax=None, **kwargs):
        super().__init__(figsize=figsize, dpi=dpi, ax=ax, **kwargs)
        self._plotters = []
    
    def _plot(self, kpath: KPath, bands: np.ndarray, **kwargs):
        pass
    
    @property
    def plotters(self):
        return self._plotters
    
    def _add_plotter(self, plotter: BasePlotter, *args, **kwargs):
        self._plotters.append(plotter)
        return plotter.plot(*args, **kwargs)
        
    def overlay(self, kpath: KPath, bands: np.ndarray, weights: List[np.ndarray], **kwargs):
        p = OverlayPlot(ax=self.ax, **self.instance_plot_params)
        return self._add_plotter(p, kpath, bands, weights=weights, **kwargs)
        
    def scatter(self, kpath: KPath, bands: np.ndarray, scalars: np.ndarray = None, **kwargs):
        p = ScatterPlot(ax=self.ax, **self.instance_plot_params)
        return self._add_plotter(p, kpath, bands, scalars=scalars, **kwargs)
    
    def parametric(self, kpath: KPath, bands: np.ndarray, scalars: np.ndarray = None, **kwargs):
        p = ParametricPlot(ax=self.ax, **self.instance_plot_params)
        return self._add_plotter(p, kpath, bands, scalars=scalars, **kwargs)
    
    def quiver(self, kpath: KPath, bands: np.ndarray, vectors: np.ndarray = None, **kwargs):
        p = QuiverPlot(ax=self.ax, **self.instance_plot_params)
        return self._add_plotter(p, kpath, bands, vectors=vectors, **kwargs)

    def atomic(self, bands: np.ndarray, **kwargs):
        p = AtomicLevelsPlot(ax=self.ax, **self.instance_plot_params)
        return self._add_plotter(p, bands, **kwargs)
        
        
        