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

logger = logging.getLogger(__name__)

# Removed legacy style/strategy scaffolding in favor of simpler API.

# Simplified main plotter class

@dataclass
class PlainBandStyle:
    color: str = "black"
    linestyle: str = "-"
    linewidth: float = 1.0 
    alpha: float = 1.0
    label: str = None
    extra_kwargs: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, style_dict: dict):
        return cls(**style_dict)
    
    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}
    
    


class BandStructurePlotter:
    """Visualizer for band-structure arrays from a model layer.

    Parameters
    ----
    figsize : tuple of float, optional
        Figure size (width, height). Default is (8, 6).
    dpi : int, optional
        Dots-per-inch for the figure. Default is 100.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure/axes are created.
    """

    def __init__(self, figsize=(8, 6), dpi=100, ax=None):
        self.figsize = figsize
        self.dpi = dpi

        self.ax = ax
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            self.fig = plt.gcf()

        self.data_store = {}
        self.values_dict = {}
        self.cb = None
        self._legend_handles = []
        
        self.x = None
        
    def plot(self, 
             kpath: KPath, 
             bands: np.ndarray,
             line_kwargs: dict | None = None,
             **kwargs):
        """Plot plain band structure lines.

        Parameters
        ----
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        line_kwargs : dict, optional
            Keyword arguments forwarded to `Axes.plot` for line styling.
        **kwargs
            Additional matplotlib line keywords (fallback, merged under
            the hood; explicit `line_kwargs` takes precedence).
        """
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        bands, _ , _ = self._validate_data(bands=bands)
        
        # Merge kwargs: generic kwargs as fallback, line_kwargs override
        merged_line_kwargs = {}
 
        if line_kwargs:
            merged_line_kwargs.update(kwargs)
        # Preserve previous default linestyle if user didn't supply one
        if "linestyle" not in merged_line_kwargs:
            merged_line_kwargs["linestyle"] = "-"
            
            
        created_lines: dict[tuple[int, int], Line2D] = {}
        n_bands = bands.shape[1]
        n_spin_channels = bands.shape[-1]
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                ret = self.ax.plot(self.x, bands[:, iband, ispin], **merged_line_kwargs)
                created_lines[(iband, ispin)] = ret[0]

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
            for iband in range(bands.shape[1]):
                key = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[key] = bands[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_lines
            
    def plot_scatter(self, 
             kpath: KPath, 
             bands: np.ndarray, 
             scalars: np.ndarray = None,
             scatter_kwargs: dict | None = None,
             cmap: str | mpcolors.Colormap = "plasma",
             norm: str | mpcolors.Normalize | type | None = "auto",
             clim: Tuple[float | None, float | None] | None = None,
             show_colorbar: bool | None = True,
             colorbar_kwargs: dict | None = None,
             **kwargs):
        """Plot band energies as scatter, optionally colored by `scalars`.

        Parameters
        ----
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        scalars : ndarray, optional
            Colormap scalars with the same shape as `bands`.
        scatter_kwargs : dict, optional
            Keyword arguments forwarded to `Axes.scatter`.
        **kwargs
            Additional matplotlib scatter keywords (fallback, merged under
            the hood; explicit `scatter_kwargs` takes precedence).
        """
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        # Validate data
        bands, scalars, _ = self._validate_data(bands, scalars)
        
        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=cmap,
            norm=norm,
        )
        
        # Merge kwargs for scatter
        merged_scatter_kwargs = {}
        if kwargs:
            merged_scatter_kwargs.update(kwargs)
        if scatter_kwargs:
            merged_scatter_kwargs.update(scatter_kwargs)
            
        merged_scatter_kwargs["norm"] = resolved_norm
        merged_scatter_kwargs["cmap"] = resolved_cmap

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
                coll = self.ax.scatter(self.x, y, c=c_vals, **merged_scatter_kwargs)
                created_collections[(iband, ispin)] = coll

        # Add colorbar if requested
        if scalars is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)
            
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
            for iband in range(bands.shape[1]):
                bkey = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[bkey] = bands[:, iband, ispin]
                if scalars is not None:
                    pkey = f"projections__scatter__band-{iband}_spinChannel-{ispin}"
                    self.values_dict[pkey] = scalars[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_collections
            
            
    def plot_parametric(self, 
             kpath: KPath, 
             bands: np.ndarray, 
             scalars: np.ndarray = None, 
             cmap: str | mpcolors.Colormap = "plasma",
             norm: str | mpcolors.Normalize | type | None = "auto",
             clim:Tuple[float | None, float | None] | None = None,
             linewidth:float = 2.0,
             collection_kwargs: dict | None = None,
             show_colorbar: bool | None = True,
             colorbar_kwargs: dict | None = None,
             **kwargs):
        """Plot parametric bands colored by `scalars` via LineCollection.

        Parameters
        ----
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        scalars : ndarray, optional
            Colormap scalars with the same shape as `bands`.
        cmap : str, optional
            Colormap name. Default is "plasma".
        norm : matplotlib.colors.Normalize, optional
            Normalization instance or class. If class, built with `clim`.
        clim : tuple, optional
            (vmin, vmax). Defaults to (None, None).
        linewidth : float, optional
            Base linewidth for segments.
        collection_kwargs : dict, optional
            Keyword arguments forwarded to `LineCollection` construction.
        colorbar_kwargs : dict, optional
            Keyword arguments forwarded to `Figure.colorbar` when a
            colorbar is added.
        **kwargs
            Additional LineCollection kwargs (fallback, merged under the
            hood; explicit `collection_kwargs` takes precedence).
        """
        
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        # Validate data
        bands, scalars, _ = self._validate_data(bands=bands, scalars=scalars)
        
        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=scalars,
            cmap=cmap,
            norm=norm,
            clim=clim,
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
        
        # Merge kwargs for LineCollection
        merged_collection_kwargs = {}
        if kwargs:
            merged_collection_kwargs.update(kwargs)
        if collection_kwargs:
            merged_collection_kwargs.update(collection_kwargs)
            
        # Plot parametric bands
        last_lc = None
        created_collections: dict[tuple[int, int], LineCollection] = {}
        
        n_spin_channels = bands.shape[-1]
        n_bands = bands.shape[1]
        for ispin_channel in range(n_spin_channels):
            for iband in range(n_bands):
                points = np.array([self.x, mbands[:, iband, ispin_channel]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments,  **merged_collection_kwargs)
                
                # Handle colors
                if scalars is not None:
                    lc.set_array(scalars[:, iband, ispin_channel])
                    lc.set_cmap(resolved_cmap)
                    lc.set_norm(resolved_norm)
                    
                lc.set_linewidth(width_weights[:, iband, ispin_channel] * linewidth)
                self.ax.add_collection(lc)
                last_lc = lc
                created_collections[(iband, ispin_channel)] = lc
                
        # Add colorbar if requested
        if scalars is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)
        
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
            cmap: str | mpcolors.Colormap = "plasma",
            norm: str | mpcolors.Normalize | type | None = "auto",
            clim:Tuple[float | None, float | None] | None = None,
            quiver_kwargs: dict | None = None,
            show_colorbar: bool | None = True,
            colorbar_kwargs: dict | None = None,
            **kwargs):
        """Plot vector-valued data (e.g., velocities) as arrows along bands.

        Parameters
        ----
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        vectors : ndarray
            Vector magnitudes aligned with `bands` (same shape). These values
            are drawn vertically; arrows are oriented horizontally with unit x.
        skip : int, optional
            Plot every `skip`-th point to reduce clutter. Default is 1.
        angles : str, optional
            Quiver angles mode. Default is 'uv'.
        scale, scale_units, units, color : optional
            Passed through to `Axes.quiver`.
        **kwargs
            Additional `Axes.quiver` kwargs.
        """
        if vectors is None:
            raise ValueError("vectors must be provided for plot_quiver")

        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)
        
        # Validate data
        bands, _, vectors = self._validate_data(bands=bands, vectors=vectors)
        
        # Resolve colormap
        resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
            data=vectors,
            cmap=cmap,
            norm=norm,
            clim=clim,
        )
        
        # Merge kwargs and quiver_kwargs
        merged_collection_kwargs = {}
        if kwargs:
            merged_collection_kwargs.update(kwargs)
        if quiver_kwargs:
            merged_collection_kwargs.update(quiver_kwargs)
            
        merged_collection_kwargs["norm"] = resolved_norm
        merged_collection_kwargs["cmap"] = resolved_cmap
        
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
                quiver_args.append(self.x[::skip])
                quiver_args.append(band_current_bands[::skip])
                quiver_args.append(band_u[::skip])
                quiver_args.append(band_v[::skip])
                if color is None:
                    quiver_args.append(vector_norms[...,iband])
                    
                qv = self.ax.quiver(
                    *quiver_args,
                    angles=angles,
                    scale=scale,
                    scale_units=scale_units,
                    units = units,
                    color=color,
                    **merged_collection_kwargs)
                created_quivers[(iband, ispin_channel)] = qv
                
        # Add colorbar if requested
        if vectors is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)

        # Record exportable bands
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                key = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[key] = bands[:, iband, ispin]
        self._record_kpath_exports(kpath)
        return created_quivers

    def plot_atomic_levels(
        self,
        bands: np.ndarray,
        elimit: Tuple[float, float] = None,
        labels_prefix: str = "s",
        scalars: np.ndarray = None,
        cmap: str = "plasma",
        norm: Union[mpcolors.Normalize, type] = None,
        clim: Tuple[float, float] = (None, None),
        show_colorbar: bool | None = True,
        colorbar_kwargs: dict | None = None,
        linewidth: float = None,
        line_collection_kwargs: dict | None = None,
        show_text: bool = True,
    ) -> None:
        """Plot atomic-like energy levels for a single k-point input.

        This duplicates the single k-point to draw short horizontal segments
        and annotates them with band indices, attempting to avoid label overlap.
        When `scalars` is provided, segments are colored similarly to
        `plot_parametric` using a LineCollection and a colorbar is added.

        Parameters
        ----
        bands : ndarray
            Energies with shape (1, n_bands, n_spins) or (1, n_bands).
        elimit : tuple of float, optional
            y-limits (ymin, ymax). If None, inferred from data.
        labels_prefix : str, optional
            Prefix for the spin label in text. Default is 's'.
        scalars : ndarray, optional
            Scalar values compatible with bands shape (1, n_bands, n_spins) or
            broadcastable; used to color each level segment.
        cmap : str, optional
            Colormap name. Default 'plasma'.
        norm : Normalize or type, optional
            Normalization instance or class; if class, constructed with `clim`.
        clim : tuple of float, optional
            (vmin, vmax) for normalization. Defaults to (None, None).
        linewidth : float, optional
            Line width for colored segments. Default 2.0.
        show_text : bool, optional
            Whether to draw text labels near levels. Default True.
        """

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
            cmap=cmap,
            norm=norm,
            clim=clim,
        )
        
        # Merge kwargs and line_collection_kwargs
        merged_collection_kwargs = {}
        if line_collection_kwargs:
            merged_collection_kwargs.update(line_collection_kwargs)
        merged_collection_kwargs["norm"] = resolved_norm
        merged_collection_kwargs["cmap"] = resolved_cmap
        
        # Remove x ticks for atomic levels and compute text bbox in data units
        self.ax.xaxis.set_major_locator(plt.NullLocator())

        
        # Determine a representative text bbox in data coordinates
        n_bands = bands.shape[1]
        sample_text = f"{labels_prefix}-0 : b-{n_bands}"
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
                lc = LineCollection(segments, **merged_collection_kwargs)
                if scalars is not None:
                    level_scalar = float(np.asarray(scalars[0, iband, ispin]))
                    lc.set_array(np.array([level_scalar]))
                self.ax.add_collection(lc)
                last_lc = lc

                if show_text:
                    # if vertical overlap, toggle lateral shift
                    if last_y is not None and y < (last_y + h):
                        shift_state = 1 - shift_state
                    else:
                        shift_state = 0
                    x_pos = x_base + (2.0 * w if shift_state == 1 else 0.0)
                    # Clamp inside current xlim
                    xmin, xmax = self.ax.get_xlim()
                    x_pos = min(max(x_pos, xmin + 0.05 * w), xmax - 0.05 * w)
                    self.ax.text(x_pos, y, f"{labels_prefix}-{ispin} : b-{iband+1}")
                    last_y = y
        
        # Add colorbar if requested
        if scalars is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)
        
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

    def plot_overlay(
        self,
        kpath: KPath,
        bands: np.ndarray,
        weights: List[np.ndarray],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        norm: str | mpcolors.Normalize | type | None = "auto",
        clim: Tuple[float | None, float | None] | None = None,
        cmap: str | mpcolors.Colormap = "plasma",
        cmap_list: Optional[List[str]] = None,
        fill_alpha: float = 0.3,
        linewidth: float = 1.0,
        fill_between_kwargs: dict | None = None,
        show_colorbar: bool | None = None,
        colorbar_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """Plot overlays by filling band envelopes using provided weights.

        Parameters
        ----
        kpath : KPath
            K-path defining cumulative distances along x.
        bands : ndarray
            Energies with shape (n_k, n_bands, n_spins) or (n_k, n_bands).
        weights : list of ndarray
            Each weight array must match `bands` shape; it defines a thickness
            around each band: [E - w/2, E + w/2] is filled.
        colors : list of str, optional
            Colors per weight set. If None, matplotlib cycles colors.
        fill_alpha : float, optional
            Alpha for filled region. Default is 0.3.
        edge_alpha : float, optional
            Alpha for the edges. Default is 0.8.
        linewidth : float, optional
            Edge line width. Default is 1.0.
        **kwargs
            Additional `Axes.fill_between` kwargs.
        """
        self.kpath = kpath
        self.x = kpath.get_distances(as_segments=False)

        # Validate data
        bands, _, _ = self._validate_data(bands=bands)
        for i, weight in enumerate(weights):
            _, weight, _ = self._validate_data(bands=bands, scalars=weight)
            weights[i] = weight

        # Default colormap names per overlay, fallback if explicit colors not given
        if cmap_list is None:
            cmap_list = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
        if colors is None:
            # Sample mid tone from each colormap to derive RGBA colors
            colors = []
            for i in range(len(weights)):
                cmap_name = cmap_list[i % len(cmap_list)]
                cmap = plt.get_cmap(cmap_name)
                colors.append(cmap(0.7))
                
            cmaps=[]
            norms=[]
            scalar_mappables=[]
            for i in range(len(weights)):
                resolved_norm, resolved_cmap, scalar_mappable = self._resolve_colormap(
                    data=weights[i],
                    cmap=cmap_list[i % len(cmap_list)],
                    norm=norm,
                    clim=clim,
                )
                cmaps.append(resolved_cmap)
                norms.append(resolved_norm)
                scalar_mappables.append(scalar_mappable)
                
                
        # Merge kwargs and fill_between_kwargs
        merged_collection_kwargs = {}
        if kwargs:
            merged_collection_kwargs.update(kwargs)
        if fill_between_kwargs:
            merged_collection_kwargs.update(fill_between_kwargs)
            
        merged_collection_kwargs["alpha"] = fill_alpha
        merged_collection_kwargs["linewidth"] = linewidth


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
                        **merged_collection_kwargs,
                    )
            legend_label = labels[widx] if labels and widx < len(labels) else f"overlay-{widx+1}"
            legend_handles.append(mpatches.Patch(color=color, label=legend_label, alpha=fill_alpha))
            
        # Add colorbar if requested
        if colors is not None and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            self.cb = self.fig.colorbar(scalar_mappable, ax=self.ax, **colorbar_kwargs)

        # Export
        for ispin in range(n_spin_channels):
            for iband in range(n_bands):
                key = f"bands__band-{iband}_spinChannel-{ispin}"
                self.values_dict[key] = bands[:, iband, ispin]
        self._record_kpath_exports(kpath)
        self._legend_handles = legend_handles
        
        self.set_xlim()
        self.set_ylim()
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()
        self.legend()
                        
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
