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

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyprocar.core import KPath
from pyprocar.core.property_store import Property

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
        self.set_default_plot_parameters(bands)
        
    def set_default_plot_parameters(self, bands: np.ndarray):
        self.set_xlim()
        ymin = float(bands.min())
        ymax = float(bands.max())
        elimit = (ymin, ymax)
        self.set_ylim(elimit)
        self.set_yticks()
        self.set_xticks()
        self.set_xlabel()
        self.set_ylabel()
        
    def _setup_colorbar(self, dos_projected, dos_total_projected):

        vmin, vmax = self._get_color_limits(dos_projected, dos_total_projected)
        cmap = mpl.cm.get_cmap(self.config.cmap)

        if self.config.plot_bar:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cb = self.fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax
            )
            cb.ax.tick_params(labelsize=self.config.colorbar_tick_labelsize)
            cb.set_label(
                self.config.colorbar_title,
                size=self.config.colorbar_title_size,
                rotation=270,
                labelpad=self.config.colorbar_title_padding,
            )
        
    def _get_color_limits(self, dos_projected, dos_total_projected):
        if self.config.clim:
            self.clim = self.config.clim
        else:
            self.clim = [0, 0]
            self.clim[0] = dos_projected.min() / dos_total_projected.max()
            self.clim[1] = dos_projected.max() / dos_total_projected.max()
        return self.clim
        
    def _set_plot_limits(self, spin_channels):
        total_max = 0
        for ispin in range(len(spin_channels)):
            tmp_max = self.dos.total[ispin].max()
            if tmp_max > total_max:
                total_max = tmp_max

        if self.orientation == "horizontal":
            x_label = self.config.x_label
            y_label = self.config.y_label
            xlim = [self.dos.energies.min(), self.dos.energies.max()]
            ylim = (
                [-self.dos.total.max(), total_max]
                if len(spin_channels) == 2
                else [0, total_max]
            )
        elif self.orientation == "vertical":
            x_label = self.config.y_label
            y_label = self.config.x_label
            xlim = (
                [-self.dos.total.max(), total_max]
                if len(spin_channels) == 2
                else [0, total_max]
            )
            ylim = [self.dos.energies.min(), self.dos.energies.max()]

        self.set_xlabel(x_label)
        self.set_ylabel(y_label)
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        
        
    def set_xticks(
        self, tick_positions: List[int] = None, tick_names: List[str] = None
    ):
        """A method to set the xticks of the plot

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None

        """

        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)
        if self.config.major_x_tick_params:
            self.ax.tick_params(**self.config.major_x_tick_params)
        if self.config.minor_x_tick_params:
            self.ax.tick_params(**self.config.minor_x_tick_params)
        return None

    def set_yticks(
        self, tick_positions: List[int] = None, tick_names: List[str] = None
    ):
        """A method to set the yticks of the plot

        Parameters
        ----------
        tick_positions : List[int], optional
            A list of tick positions, by default None
        tick_names : List[str], optional
            A list of tick names, by default None

        """
        if tick_positions is not None:
            self.ax.set_xticks(tick_positions)
        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)

        if self.config.major_y_tick_params:
            self.ax.tick_params(**self.config.major_y_tick_params)
        if self.config.minor_y_tick_params:
            self.ax.tick_params(**self.config.minor_y_tick_params)
        return None

    def set_xlim(self, interval: List[int] = None):
        """A method to set the xlim of the plot

        Parameters
        ----------
        interval : List[int], optional
            The x interval, by default None
        """
        if interval is not None:
            self.ax.set_xlim(interval)
        return None

    def set_ylim(self, interval: List[int] = None):
        """A method to set the ylim of the plot

        Parameters
        ----------
        interval : List[int], optional
            The y interval, by default None
        """
        if interval is not None:
            self.ax.set_ylim(interval)

        return None

    def set_xlabel(self, label: str):
        """A method to set the x label

        Parameters
        ----------
        label : str
            The x label name

        Returns
        -------
        None
            None
        """
        if self.config.x_label:
            self.ax.set_xlabel(self.config.x_label, **self.config.x_label_params)
        else:
            self.ax.set_xlabel(label, **self.config.x_label_params)
        return None

    def set_ylabel(self, label: str):
        """A method to set the y label

        Parameters
        ----------
        label : str
            The y label name

        Returns
        -------
        None
            None
        """
        if self.config.y_label:
            self.ax.set_ylabel(self.config.y_label, **self.config.y_label_params)
        else:
            self.ax.set_ylabel(label, **self.config.y_label_params)

    def legend(self, labels: List[str] = None):
        """A method to include the legend

        Parameters
        ----------
        label : str
            The labels for the legend

        Returns
        -------
        None
            None
        """
        if labels == None:
            labels = self.labels
        if self.config.legend and len(labels) != 0:
            if len(self.handles) != len(labels):
                raise ValueError(
                    f"The number of labels and handles should be the same, currently there are {len(self.handles)} handles and {len(labels)} labels"
                )
            self.ax.legend(self.handles, labels, **self.config.legend_params)
        return None
    
    def set_title(self, title: str = ""):
        """A method to set the title of the plot
        """
        if self.config.title:
            title = self.config.title
        self.ax.set_title(title, **self.config.title_params)
        return None

    def draw_fermi(self, value, orientation: str = "horizontal"):
        """A method to draw the fermi surface

        Parameters
        ----------
        orientation : str, optional
            Boolean to plot vertical or horizontal, by default 'horizontal'
        color : str, optional
            A color , by default "blue"
        linestyle : str, optional
            THe line style, by default "dotted"
        linewidth : float, optional
            The linewidth, by default 1

        Returns
        -------
        None
            None
        """
        if orientation == "horizontal":
            self.ax.axvline(
                x=value,
                color=self.config.fermi_color,
                linestyle=self.config.fermi_linestyle,
                linewidth=self.config.fermi_linewidth,
            )
        elif orientation == "vertical":
            self.ax.axhline(
                y=value,
                color=self.config.fermi_color,
                linestyle=self.config.fermi_linestyle,
                linewidth=self.config.fermi_linewidth,
            )
        return None

    def draw_baseline(self, value, orientation: str = "horizontal"):
        """A method to draw the baseline

        Parameters
        ----------
        value : float
            The value of the baseline
        """
        if orientation == "horizontal":
            self.ax.axhline(y=value, **self.config.baseline_params)
        elif orientation == "vertical":
            self.ax.axvline(x=value, **self.config.baseline_params)
        return None

    def grid(self):
        """A method to include a grid on the plot.

        Returns
        -------
        None
            None
        """
        if self.config.grid:
            self.ax.grid(
                self.config.grid,
                which=self.config.grid_which,
                color=self.config.grid_color,
                linestyle=self.config.grid_linestyle,
                linewidth=self.config.grid_linewidth,
            )
        return None

    def show(self):
        """A method to show the plot

        Returns
        -------
        None
            None
        """
        plt.show()
        return None

    def save(self, filename: str = "dos.pdf"):
        """A method to save the plot

        Parameters
        ----------
        filename : str, optional
            The filename, by default 'dos.pdf'

        Returns
        -------
        None
            None
        """

        plt.savefig(filename, dpi=self.config.dpi, bbox_inches="tight")
        plt.clf()
        return None

    def update_config(self, config_dict):
        for key, value in config_dict.items():
            self.config[key]["value"] = value

    def export_data(self, filename):
        """
        This method will export the data to a csv file

        Parameters
        ----------
        filename : str
            The file name to export the data to

        Returns
        -------
        None
            None
        """
        possible_file_types = ["csv", "txt", "json", "dat"]
        file_type = filename.split(".")[-1]
        if file_type not in possible_file_types:
            raise ValueError(f"The file type must be {possible_file_types}")
        if self.values_dict is None:
            raise ValueError("The data has not been plotted yet")

        column_names = list(self.values_dict.keys())
        sorted_column_names = [None] * len(column_names)
        index = 0
        for column_name in column_names:
            if "energies" in column_name.split("_")[0]:
                sorted_column_names[index] = column_name
                index += 1

        for column_name in column_names:
            if "dosTotalSpin" in column_name.split("_")[0]:
                sorted_column_names[index] = column_name
                index += 1
        for ispin in range(2):
            for column_name in column_names:

                if "spinChannel-0" in column_name.split("_")[0] and ispin == 0:
                    sorted_column_names[index] = column_name
                    index += 1
                if "spinChannel-1" in column_name.split("_")[0] and ispin == 1:
                    sorted_column_names[index] = column_name
                    index += 1

        column_names.sort()
        if file_type == "csv":
            df = pd.DataFrame(self.values_dict)
            df.to_csv(filename, columns=sorted_column_names, index=False)
        elif file_type == "txt":
            df = pd.DataFrame(self.values_dict)
            df.to_csv(filename, columns=sorted_column_names, sep="\t", index=False)
        elif file_type == "json":
            with open(filename, "w") as outfile:
                for key, value in self.values_dict.items():
                    self.values_dict[key] = value.tolist()
                json.dump(self.values_dict, outfile)
        elif file_type == "dat":
            df = pd.DataFrame(self.values_dict)
            df.to_csv(filename, columns=sorted_column_names, sep=" ", index=False)
