__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator

from pyprocar.core import ElectronicBandStructure, KPath
from pyprocar.core.ebs import ElectronicBandStructurePath

logger = logging.getLogger(__name__)

# Base styling that all modes inherit
@dataclass  
class BasePlotStyle:
    """Base styling configuration that all plot modes inherit"""
    color: Union[str, List[str], Dict[int, str]] = None
    linestyle: Union[str, List[str], Dict[int, str]] = None  
    linewidth: Union[float, List[float], Dict[int, float]] = None
    alpha: Union[float, List[float], Dict[int, float]] = None
    label: Union[str, List[str], Dict[int, str]] = None
    extra_kwargs: Dict[str, Union[Any, List[Any], Dict[int, Any]]] = field(default_factory=dict)

# Mode-specific configurations
@dataclass
class PlainBandStyle(BasePlotStyle):
    """Configuration for plain band plotting"""
    pass

@dataclass
class ScatterPlotStyle(BasePlotStyle):
    """Configuration for scatter plot mode"""
    marker: Union[str, List[str], Dict[int, str]] = None
    markersize: Union[float, List[float], Dict[int, float]] = None
    markeredgecolor: Union[str, List[str], Dict[int, str]] = None
    markerfacecolor: Union[str, List[str], Dict[int, str]] = None

@dataclass
class ParametricPlotStyle(BasePlotStyle):
    """Configuration for parametric plot mode"""
    cmap: Union[str, List[str], Dict[int, str]] = None
    vmin: Union[float, List[float], Dict[int, float]] = None
    vmax: Union[float, List[float], Dict[int, float]] = None

@dataclass
class OverlayPlotStyle(BasePlotStyle):
    """Configuration for overlay plot mode"""
    fill_alpha: Union[float, List[float], Dict[int, float]] = None
    edge_alpha: Union[float, List[float], Dict[int, float]] = None

# Strategy pattern with local defaults
class PlotModeStrategy(ABC):
    """Abstract base class for plot mode strategies"""
    
    @abstractmethod
    def get_default_style(self, n_spin_channels: int) -> BasePlotStyle:
        """Get default styling for this mode based on number of spin channels"""
        pass
    
    @abstractmethod
    def plot(self, plotter: 'BandStructurePlotter', ebs_name: str, 
             ebs: 'ElectronicBandStructurePath', style: BasePlotStyle, **kwargs) -> None:
        """Execute the plotting strategy"""
        pass

class PlainBandStrategy(PlotModeStrategy):
    def get_default_style(self, n_spin_channels: int) -> PlainBandStyle:
        """Get default styling for plain bands"""
        if n_spin_channels == 1:
            return PlainBandStyle(
                color=["#1f77b4"],           # Blue
                linestyle=["-"],             # Solid
                linewidth=[1.5],
                alpha=[1.0]
            )
        elif n_spin_channels == 2:
            return PlainBandStyle(
                color=["#1f77b4", "#ff7f0e"],    # Blue, Orange  
                linestyle=["-", "--"],            # Solid, Dashed
                linewidth=[1.5, 1.5],
                alpha=[1.0, 0.8]                 # Spin-down slightly transparent
            )
        else:
            # Fallback for unexpected cases
            return PlainBandStyle(
                color=["#1f77b4"] * n_spin_channels,
                linestyle=["-"] * n_spin_channels,
                linewidth=[1.5] * n_spin_channels,
                alpha=[1.0] * n_spin_channels
            )
    
    def plot(self, plotter, data, style: PlainBandStyle, **kwargs):
        
        # Merge user style with defaults
        default_style = self.get_default_style(data.n_spin_channels)
        final_style = plotter._merge_styles(style, default_style)
        
        # Prepare per-spin plot kwargs
        spin_kwargs = plotter._prepare_plot_kwargs(final_style, ebs.n_spin_channels, **kwargs)
        
        for ispin in ebs.spin_channels:
            plot_kwargs = spin_kwargs[ispin].copy()
            if "label" not in plot_kwargs or plot_kwargs["label"] is None:
                plot_kwargs["label"] = plotter._generate_label(ebs_name, ebs, ispin)
            
            plotter.ax.plot(
                bands_data["distances"], 
                bands_data["energies"][..., ispin], 
                **plot_kwargs
            )

class ScatterStrategy(PlotModeStrategy):
    def get_default_style(self, n_spin_channels: int) -> ScatterPlotStyle:
        """Get default styling for scatter plots"""
        if n_spin_channels == 1:
            return ScatterPlotStyle(
                color=["#1f77b4"],
                marker=["o"],
                markersize=[20],
                alpha=[0.7],
                markeredgecolor=["black"],
                linewidth=[0.5]  # For marker edges
            )
        elif n_spin_channels == 2:
            return ScatterPlotStyle(
                color=["#1f77b4", "#ff7f0e"],
                marker=["o", "s"],               # Circle, Square
                markersize=[20, 18],
                alpha=[0.8, 0.6],               # Spin-down more transparent
                markeredgecolor=["darkblue", "darkorange"],
                linewidth=[0.5, 0.5]
            )
        else:
            # Fallback for unexpected cases
            default_markers = ["o", "s", "^", "D"]  # Circle, square, triangle, diamond
            return ScatterPlotStyle(
                color=["#1f77b4"] * n_spin_channels,
                marker=[default_markers[i % len(default_markers)] for i in range(n_spin_channels)],
                markersize=[20] * n_spin_channels,
                alpha=[0.7] * n_spin_channels
            )
    
    def plot(self, plotter, ebs_name, ebs, style: ScatterPlotStyle, 
             atoms: List[int] = None, orbitals: List[int] = None,
             width_mask: np.ndarray = None, color_mask: np.ndarray = None,
             width_weights: np.ndarray = None, color_weights: np.ndarray = None,
             **kwargs):
        
        bands_data = plotter.get_projected_bands_data(ebs, atoms=atoms, orbitals=orbitals)
        
        # Apply masks if provided
        if width_mask is not None or color_mask is not None:
            # Handle masking logic (from old ebs_plot.py)
            pass
        
        # Merge user style with defaults
        default_style = self.get_default_style(ebs.n_spin_channels)
        final_style = plotter._merge_styles(style, default_style)
        
        spin_kwargs = plotter._prepare_plot_kwargs(final_style, ebs.n_spin_channels, **kwargs)
        
        for ispin in ebs.spin_channels:
            plot_kwargs = spin_kwargs[ispin].copy()
            
            # Handle weight-based sizing and coloring
            sizes = width_weights[..., ispin] if width_weights is not None else plot_kwargs.get('markersize', 20)
            colors = color_weights[..., ispin] if color_weights is not None else plot_kwargs.get('color')
            
            # Remove conflicting keys for scatter
            scatter_kwargs = {k: v for k, v in plot_kwargs.items() 
                            if k not in ['color', 'markersize']}
            
            plotter.ax.scatter(
                bands_data["distances"],
                bands_data["energies"][..., ispin],
                s=sizes,
                c=colors,
                **scatter_kwargs
            )

class ParametricStrategy(PlotModeStrategy):
    def get_default_style(self, n_spin_channels: int) -> ParametricPlotStyle:
        """Get default styling for parametric plots"""
        if n_spin_channels == 1:
            return ParametricPlotStyle(
                cmap=["viridis"],
                linewidth=[2.0],
                alpha=[0.8]
            )
        elif n_spin_channels == 2:
            return ParametricPlotStyle(
                cmap=["viridis", "plasma"],     # Different colormaps per spin
                linewidth=[2.0, 1.8],
                alpha=[0.9, 0.7]
            )
        else:
            default_cmaps = ["viridis", "plasma", "inferno", "magma"]
            return ParametricPlotStyle(
                cmap=[default_cmaps[i % len(default_cmaps)] for i in range(n_spin_channels)],
                linewidth=[2.0] * n_spin_channels,
                alpha=[0.8] * n_spin_channels
            )
    
    def plot(self, plotter, ebs_name, ebs, style: ParametricPlotStyle,
             atoms: List[int] = None, orbitals: List[int] = None,
             width_weights: np.ndarray = None, color_weights: np.ndarray = None,
             elimit: List[float] = None, **kwargs):
        
        from matplotlib.collections import LineCollection
        
        bands_data = plotter.get_projected_bands_data(ebs, atoms=atoms, orbitals=orbitals)
        
        # Merge user style with defaults
        default_style = self.get_default_style(ebs.n_spin_channels)
        final_style = plotter._merge_styles(style, default_style)
        
        spin_kwargs = plotter._prepare_plot_kwargs(final_style, ebs.n_spin_channels, **kwargs)
        
        for ispin in ebs.spin_channels:
            plot_kwargs = spin_kwargs[ispin].copy()
            
            # Create line segments for LineCollection
            points = np.array([bands_data["distances"], bands_data["energies"][..., ispin]]).T
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
            
            # Color data
            colors = color_weights[..., ispin] if color_weights is not None else None
            
            lc = LineCollection(segments, 
                              cmap=plot_kwargs.get('cmap', 'viridis'),
                              linewidths=plot_kwargs.get('linewidth', 2.0),
                              alpha=plot_kwargs.get('alpha', 0.8))
            
            if colors is not None:
                lc.set_array(colors)
                
            plotter.ax.add_collection(lc)

class OverlayStrategy(PlotModeStrategy):
    def get_default_style(self, n_spin_channels: int) -> OverlayPlotStyle:
        """Get default styling for overlay plots"""
        if n_spin_channels == 1:
            return OverlayPlotStyle(
                color=["#1f77b4"],
                fill_alpha=[0.3],
                edge_alpha=[0.8],
                linewidth=[1.0]
            )
        elif n_spin_channels == 2:
            return OverlayPlotStyle(
                color=["#1f77b4", "#ff7f0e"],
                fill_alpha=[0.3, 0.2],
                edge_alpha=[0.8, 0.6],
                linewidth=[1.0, 1.0]
            )
        else:
            return OverlayPlotStyle(
                color=["#1f77b4"] * n_spin_channels,
                fill_alpha=[0.3] * n_spin_channels,
                edge_alpha=[0.8] * n_spin_channels,
                linewidth=[1.0] * n_spin_channels
            )
    
    def plot(self, plotter, ebs_name, ebs, style: OverlayPlotStyle,
             weights: np.ndarray = None, **kwargs):
        
        bands_data = plotter.get_projected_bands_data(ebs)
        
        # Merge user style with defaults
        default_style = self.get_default_style(ebs.n_spin_channels)
        final_style = plotter._merge_styles(style, default_style)
        
        spin_kwargs = plotter._prepare_plot_kwargs(final_style, ebs.n_spin_channels, **kwargs)
        
        for ispin in ebs.spin_channels:
            plot_kwargs = spin_kwargs[ispin].copy()
            
            # Fill between logic for overlay
            if weights is not None:
                width = weights[..., ispin]
                plotter.ax.fill_between(
                    bands_data["distances"],
                    bands_data["energies"][..., ispin] - width/2,
                    bands_data["energies"][..., ispin] + width/2,
                    alpha=plot_kwargs.get('fill_alpha', 0.3),
                    color=plot_kwargs.get('color'),
                    **{k: v for k, v in plot_kwargs.items() 
                       if k not in ['fill_alpha', 'color']}
                )

# Simplified main plotter class
class BandStructurePlotter:
    
    def __init__(self, figsize=(8, 6), dpi=100, ax=None):
        self.figsize = figsize
        self.dpi = dpi

        self.ax = ax
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)

        self.data_store = {}

        # Strategy registry - defaults handled by strategies themselves
        self.strategies = {
            'plain': PlainBandStrategy(),
            'scatter': ScatterStrategy(), 
            'parametric': ParametricStrategy(),
            'overlay': OverlayStrategy()
        }
        
        self.x = None
        
    def plot(self, 
             kpath: KPath, 
             bands: np.ndarray,  **kwargs):
        self.x = kpath.get_distances(as_segments=False)
        
        logger.info(f"kpath shape: {x.shape}")
        logger.info(f"bands shape: {bands.shape}")

        n_spin_channels = bands.shape[-1]
        for ispin in range(n_spin_channels):
       
            self.ax.plot(self.x, bands[..., ispin], linestyle="--", **kwargs)
            
        
        
    def plot_scatter(self, 
             kpath: KPath, 
             bands: np.ndarray, 
             scalars: np.ndarray = None, **kwargs):
        self.x = kpath.get_distances(as_segments=False)
        
        logger.info(f"kpath shape: {self.x.shape}")
        logger.info(f"bands shape: {bands.shape}")
        logger.info(f"scalars shape: {scalars.shape}")
        

        n_spin_channels = bands.shape[-1]

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

        for ispin in range(n_spin_channels):
            data=None
            if scalars is not None:
                data = scalars[..., ispin]
                
            self.ax.scatter(self.x, bands[..., ispin], c=data, linestyle="--", **kwargs)
            
            
    def plot_parametric(self, 
             kpath: KPath, 
             bands: np.ndarray, 
             scalars: np.ndarray = None, 
             cmap: str = "plasma",
             norm: mpcolors.Normalize = None,
             clim:tuple = (None, None),
             linewidth:float = 2.0,
             **kwargs):
        
        if not hasattr(self, "norm") or not hasattr(self, "clim") or not hasattr(self, "cmap"):
            self.set_scalar_mappable(norm=norm, clim=clim, cmap=cmap)
            
        self.x = kpath.get_distances(as_segments=False)
        
        logger.info(f"kpath shape: {self.x.shape}")
        logger.info(f"bands shape: {bands.shape}")
        logger.info(f"scalars shape: {scalars.shape}")
        

        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        
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
        
        # if width_weights is None:
        #     width_weights = np.ones_like(bands)
        #     linewidth = 2.0
        
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
        
        for ispin_channel in range(n_spin_channels):
            for iband in range(n_bands):
                points = np.array([self.x, mbands[:, iband, ispin_channel]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments,  **kwargs)
                
                # Handle colors
                if len(scalars) > 0:
                    lc.set_array(scalars[:, iband, ispin_channel])
                    lc.set_cmap(self.cmap)
                    lc.set_norm(self.norm)
                    
                lc.set_linewidth(width_weights[:, iband, ispin_channel] * linewidth)
                handle = self.ax.add_collection(lc)

            
    def plot_quiver(self, 
            kpath: KPath, 
            bands: np.ndarray, 
            vectors: np.ndarray = None,
            skip:int=1,
            angles:str='uv',
            scale=None,
            scale_units:str='inches',
            units:str='inches',
            color=None,
            **kwargs):
        self.x = kpath.get_distances(as_segments=False)
        
        logger.info(f"kpath shape: {self.x.shape}")
        logger.info(f"bands shape: {bands.shape}")
        logger.info(f"vectors shape: {vectors.shape}")
        if bands.ndim == 2:
            bands = bands[...,np.newaxis]
        if vectors.ndim == 2:
            vectors = vectors[...,np.newaxis]

        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        
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
        
        
        
        for ispin_channel in range(n_spin_channels):
            u = vectors[...,ispin_channel]                # Arrow y-component
            v = np.ones_like(vectors[...,ispin_channel])  # Arrow x-component
            current_bands = bands[...,ispin_channel]
        
            for iband in range(n_bands):
                band_u = u[...,iband]
                band_v = v[...,iband]
                band_current_bands = current_bands[...,iband]
    
                quiver_args = []
                quiver_args.append(self.x[::skip])
                quiver_args.append(band_current_bands[::skip])
                quiver_args.append(band_u[::skip])
                quiver_args.append(band_v[::skip])
                if color is None:
                    quiver_args.append(band_u)
                    
                self.ax.quiver(
                    *quiver_args,
                    angles=angles,
                    scale=scale,
                    scale_units=scale_units,
                    units = units,
                    color=color,
                    **kwargs)
                
    def set_scalar_mappable(self, 
                            norm:mpcolors.Normalize = None, 
                            clim:tuple = (None, None), 
                            cmap:str = "plasma"):
        
        vmin = clim[0]
        vmax = clim[1]
        if vmin is None:
            vmin=-0.5
        if vmax is None:
            vmax=0.5
        if norm is None:
            norm = mpcolors.Normalize
            
        norm = norm(vmin, vmax)
        self.norm = norm
        self.clim = clim
        self.cmap = cmap

        self.cm = cm.ScalarMappable(norm=norm, cmap=cmap)
            
    def set_xlim(self, xlim:List[float] = None, **kwargs):
        if xlim is None:
            xlim = (self.x[0], self.x[-1])
            
        self.ax.set_xlim(xlim, **kwargs)
        
    def set_ylim(self, ylim:List[float] = None, **kwargs):
        if ylim is None:
            ylim = (self.flefffy.min(), self.y.max())
            
        self.ax.set_ylim(ylim, **kwargs)
        
    def show(self):
        plt.show()
    
    # def add_ebs(self, ebs: ElectronicBandStructurePath, name=None):
    #     """A method to add an ElectronicBandStructure object to the plot"""
    #     n_ebs = len(self.ebs_store)
    #     if name is None:
    #         name = f"ebs-{n_ebs}"
            
    #     if self.zero_fermi:
    #         ebs.shift_bands(-1*ebs.fermi, inplace=True)
            
    #     self.ebs_store[name] = ebs
    #     self.kpath = ebs.kpath
    #     self.x = self.kpath.get_distances(as_segments=False)
    
    # def plot(self, mode: str = 'plain', style: BasePlotStyle = None, **kwargs):
    #     """
    #     Universal plotting interface that delegates to appropriate strategy.
        
    #     Parameters
    #     ----------
    #     mode : str
    #         Plot mode: 'plain', 'scatter', 'parametric', 'overlay'
    #     style : mode-specific style object
    #         Styling configuration appropriate for the mode
    #     **kwargs : dict
    #         Mode-specific parameters and matplotlib parameters
    #     """
    #     if mode not in self.strategies:
    #         raise ValueError(f"Unknown plot mode: {mode}. Available: {list(self.strategies.keys())}")
        
    #     strategy = self.strategies[mode]
        
    #     for ebs_name, ebs in self.ebs_store.items():
    #         # If no style provided, strategy will use its defaults
    #         if style is None:
    #             style = strategy.get_default_style(ebs.n_spin_channels)
            
    #         strategy.plot(self, ebs_name, ebs, style, **kwargs)
    
    # # Convenience methods with mode-specific signatures
    # def plot_plain_bands(self, style: PlainBandStyle = None, **kwargs):
    #     """Plot plain bands"""
    #     self.plot('plain', style, **kwargs)
    
    # def plot_scatter(self, style: ScatterPlotStyle = None, 
    #                  atoms: List[int] = None, orbitals: List[int] = None,
    #                  width_mask: np.ndarray = None, color_mask: np.ndarray = None,
    #                  width_weights: np.ndarray = None, color_weights: np.ndarray = None,
    #                  **kwargs):
    #     """Plot scatter with projections"""
    #     self.plot('scatter', style, atoms=atoms, orbitals=orbitals,
    #              width_mask=width_mask, color_mask=color_mask,
    #              width_weights=width_weights, color_weights=color_weights, **kwargs)
    
    # def plot_parametric(self, style: ParametricPlotStyle = None,
    #                    atoms: List[int] = None, orbitals: List[int] = None,
    #                    width_weights: np.ndarray = None, color_weights: np.ndarray = None,
    #                    elimit: List[float] = None, **kwargs):
    #     """Plot parametric with projections"""
    #     self.plot('parametric', style, atoms=atoms, orbitals=orbitals,
    #              width_weights=width_weights, color_weights=color_weights,
    #              elimit=elimit, **kwargs)
    
    # def plot_overlay(self, style: OverlayPlotStyle = None,
    #                 weights: np.ndarray = None, **kwargs):
    #     """Plot overlay"""
    #     self.plot('overlay', style, weights=weights, **kwargs)
    
    # def register_plot_mode(self, name: str, strategy: PlotModeStrategy):
    #     """Register a new plot mode - strategy includes its own defaults"""
    #     self.strategies[name] = strategy
