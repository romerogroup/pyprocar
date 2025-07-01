__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import json
import logging
from typing import Any, Dict, List

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from pyvista import ColorLike

from pyprocar.core import BandStructure2D, ElectronicBandStructurePath

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

class BandStructurePlotter:
    """
    A plotter class for band structure using matplotlib.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size (width, height) in inches, by default (8, 6)
    dpi : int, optional
        Figure resolution in dots per inch, by default 100
    """

    def __init__(self, figsize=(8, 6), dpi=100, ax=None):
        self.figsize = figsize
        self.dpi = dpi

        self.ax = ax
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)

        self.ebs_store = {}
        self.kpath=None

        self.set_plot_settings()
        
        
    def _initiate_plot_args(self):
        """Helper method to initialize the plot options"""
        self.set_xticks()
        self.set_yticks()
        self.set_xlabel()
        self.set_ylabel()
        self.set_xlim()
        self.set_ylim()
        
    def add_ebs(self, ebs: ElectronicBandStructurePath, name=None):
        """A method to add an ElectronicBandStructure object to the plot"""
        n_ebs = len(self.ebs_store)
        if name is None:
            name = f"ebs-{n_ebs}"
        self.ebs_store[name] = ebs
        self.kpath = ebs.kpath
        self.x = self.kpath.get_distances(as_segments=False)
        
        
        
    def draw_fermi(
        self,
        level: float = 0,
        color: str = "grey",
        linestyle: str = "--",
        linewidth: float = 1.0,
        **kwargs
        ):
        """A method to draw the fermi line

        Parameters
        ----------
        fermi_level : str, optional
            The energy level to draw the line
        """
        self.ax.axhline(
            y=level,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs
        )
        
    def is_ebs_store_empty(self):
        """A method to check if the ebs store is empty"""
        return len(self.ebs_store) == 0
        
        
    def set_xticks(
        self,
        tick_positions: List[int] = None,
        tick_names: List[str] = None,
        color: str = "black",
    ):
        """A method to set the x ticks

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None
        color : str, optional
            A color for the ticks, by default "black"
        """
        if self.kpath is not None and self.tick_positions is None:
            self.tick_positions = self.kpath.tick_positions
        if self.kpath is not None and self.tick_names is None:
            self.tick_names = self.kpath.tick_names
            
        if tick_positions is not None:
            for ipos in tick_positions:
                self.ax.axvline(self.x[ipos], color=color)
            self.ax.set_xticks(self.x[tick_positions])

        if tick_names is not None:
            logger.debug(f"tick_names: {tick_names}")
            self.ax.set_xticklabels(tick_names)

        
    def set_tick_params(self, **kwargs):
        """A method to set the tick parameters"""
        self.ax.tick_params(**kwargs)

    def set_yticks(
        self, 
        interval: List[float] = None,
        major: float = None, 
        minor: float = None, 
        minor_y_tick_params: Dict[str, Any] = None,
        major_y_tick_params: Dict[str, Any] = None,
        major_locator = None,
        minor_locator = None,
    ):
        """A method to set the y ticks

        Parameters
        ----------
        major : float, optional
            A float to set the major tick locator, by default None
        minor : float, optional
            A float to set the the minor tick Locator, by default None
        interval : List[float], optional
            The interval of the ticks, by default None
        """
        interval = abs(interval[1] - interval[0])
        if interval < 30 and interval >= 20:
            major = 5
            minor = 1
        elif interval < 20 and interval >= 10:
            major = 4
            minor = 0.5
        elif interval < 10 and interval >= 5:
            major = 2
            minor = 0.2
        elif interval < 5 and interval >= 3:
            major = 1
            minor = 0.1
        elif interval < 3 and interval >= 1:
            major = 0.5
            minor = 0.1
        else:
            pass

        if major is not None:
            self.ax.yaxis.set_major_locator(MultipleLocator(major))
        if minor is not None:
            self.ax.yaxis.set_minor_locator(MultipleLocator(minor))

        self.ax.tick_params(**major_y_tick_params)
        self.ax.tick_params(**minor_y_tick_params)

    def set_xlim(
        self, interval: List[float] = None, ktick_interval: List[float] = None
    ):
        """A method to set the x limit

        Parameters
        ----------
        interval : List[float], optional
            A list containing the begining and the end of the interval, by default None
        """
        if interval is None:
            interval = (self.x[0], self.x[-1])
        if ktick_interval:
            ktick_start = ktick_interval[0]
            ktick_end = ktick_interval[1]
            interval = (self.x[ktick_start], self.x[ktick_end])

        self.ax.set_xlim(interval)

    def set_ylim(self, interval: List[float] = None):
        """A method to set the y limit

        Parameters
        ----------
        interval : List[float], optional
            A list containing the begining and the end of the interval, by default None
        """
        if interval is None:
            interval = (
                self.ebs.bands.min() - abs(self.ebs.bands.min()) * 0.1,
                self.ebs.bands.max() * 1.1,
            )
        self.ax.set_ylim(interval)

    def set_xlabel(self, label: str = None):
        """A method to set the x label

        Parameters
        ----------
        label : str, optional
            String fo the x label name, by default "K vector"
        """
        if label is None:
            label = self.config.x_label
            self.ax.set_xlabel(label, **self.config.x_label_params)

    def set_ylabel(self, label: str = None, 
                   shifted_by_fermi: bool = True,
                   **kwargs
                   ):
        """A method to set the y label

        Parameters
        ----------
        label : str, optional
            String fo the y label name, by default r"E - E$ (eV)"
        """
        if label is None and shifted_by_fermi:
            label = r"E - E$_F$ (eV)"
        elif label is None and not shifted_by_fermi:
            label = "E (eV)"

        self.ax.set_ylabel(label, **kwargs)

    def set_title(self, label: str, **kwargs):
        """A method to set the title

        Parameters
        ----------
        label : str, optional
            String for the title, by default "Band Structure"
        """
        self.ax.set_title(label=label, **kwargs)





class BandStructure2DPlotter(pv.Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._meshes = []
        
    def add_brillouin_zone(
        self,
        brillouin_zone: pv.PolyData = None,
        style: str = "wireframe",
        line_width: float = 2.0,
        color: ColorLike = "black",
        opacity: float = 1.0,
    ):
        self.add_mesh(
            brillouin_zone,
            style=style,
            line_width=line_width,
            color=color,
            opacity=opacity,
        )
        
    def add_surface(
        self,
        bs2d: BandStructure2D,
        normalize: bool = False,
        add_texture_args: dict = None,
        add_active_vectors: bool = False,
        show_scalar_bar: bool = True,
        add_mesh_args: dict = None,
        **kwargs,
    ):
        logger.info(f"____Adding Surface to Plotter____")

        if add_texture_args is None:
            add_texture_args = {}
        add_texture_args["name"] = add_texture_args.get("name", "vectors")

        if add_mesh_args is None:
            add_mesh_args = {}
            
        
        self.add_mesh(bs2d, **add_mesh_args)

        # if show_scalar_bar:
        #     active_scalar_name = fermi_surface.active_scalars_name
        #     if "norm" in active_scalar_name:
        #         active_scalar_name = active_scalar_name.replace("-norm", "")
        #     add_mesh_args["show_scalar_bar"] = add_mesh_args.get(
        #         "show_scalar_bar", True
        #     )
        #     add_mesh_args["scalar_bar_args"] = add_mesh_args.get("scalar_bar_args", {})
        #     add_mesh_args["scalar_bar_args"]["title"] = add_mesh_args.get(
        #         "scalar_bar_args", {}
        #     ).get("title", active_scalar_name)

        # add_mesh_args["cmap"] = add_mesh_args.get("cmap", "plasma")
        # add_mesh_args["clim"] = add_mesh_args.get("clim", None)
        # add_mesh_args["name"] = add_mesh_args.get("name", "surface")
        # add_mesh_args.update(kwargs)

        # clim = add_mesh_args.get("clim", None)
        # cmap = add_mesh_args.get("cmap", "plasma")

        # if normalize:
        #     scalars = normalize_to_range(fermi_surface.active_scalars, clim=clim)
        #     add_mesh_args["scalars"] = scalars
        # add_mesh_args["scalars"] = add_mesh_args.get("scalars", None)

        # self.add_mesh(fermi_surface, **add_mesh_args)

        # if add_active_vectors:
        #     # aligning the texture colors with the surface colors
        #     add_texture_args["cmap"] = add_texture_args.get("cmap", cmap)
        #     add_texture_args["clim"] = add_texture_args.get("clim", clim)

        #     self.add_texture(fermi_surface, **add_texture_args)

    def add_bounds(self, bs2d: BandStructure2D, ulim=None, vlim=None, elim=None, elim_offset=1, **kwargs):
        current_bounds = bs2d.bounds
        if ulim is None:
            ulim = (current_bounds[0], current_bounds[1])
        if vlim is None:
            vlim = (current_bounds[2], current_bounds[3])
        if elim is None:
            elim = (bs2d.ebs.efermi - elim_offset, bs2d.ebs.efermi + elim_offset)
        
        bounds = (*ulim, *vlim, *elim)
            
        self.show_bounds(bs2d, bounds=bounds, **kwargs)
        