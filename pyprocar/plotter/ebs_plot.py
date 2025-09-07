__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import json
import logging
import os
from typing import List

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator

from pyprocar.core import ElectronicBandStructure, KPath

logger = logging.getLogger(__name__)


class EBSPlot:
    """
    A class to plot an electronic band structure.

    Parameters
    ----------
    ebs : ElectronicBandStructure
        An electronic band structure object pyprocar.core.ElectronicBandStructure.
    kpath : KPath, optional
        A kpath object pyprocar.core.KPath. The default is None.
    ax : mpl.axes.Axes, optional
        A matplotlib Axes object. If provided the plot will be located at that ax.
        The default is None.
    spin : List[int], optional
        A list of the spins
        The default is None.

    Returns
    -------
    None.

    """

    def __init__(
        self,
        ebs: ElectronicBandStructure,
        kpath: KPath = None,
        ax: mpl.axes.Axes = None,
        spins: List[int] = None,
        kdirect: bool = True,
        config=None,
    ):
        self.config = config

        self.ebs = ebs
        self.kpath = kpath
        self.spins = spins
        self.kdirect = kdirect

        self.values_dict = {}

        if self.spins is None:
            self.spins = range(self.ebs.nspins)
        self.nspins = len(self.spins)
        if self.ebs.is_non_collinear:
            self.spins = [0]
        self.handles = []

        figsize = tuple(self.config.figure_size)
        if ax is None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax

        logger.info("___Initializing EBSPlot___")

        logger.debug("Kpath: %s", self.kpath)
        logger.debug("Non-collinear: %s", self.ebs.is_non_collinear)
        logger.debug("Spins: %s", self.spins)
        logger.debug("Kdirect: %s", self.kdirect)
        logger.debug("Config: %s", self.config)
        logger.debug("EBS: %s", self.ebs)

        # need to initiate kpath if kpath is not defined.
        self.x = self._get_x()

        self._initiate_plot_args()

        return None

    def _initiate_plot_args(self):
        """Helper method to initialize the plot options"""
        self.set_xticks()
        self.set_yticks()
        self.set_xlabel()
        self.set_ylabel()
        self.set_xlim()
        self.set_ylim()

    def _get_x(self):
        """
        Provides the x axis data of the plots

        Returns
        -------
        np.ndarray
            x-axis data.

        """
        logger.info("___Getting x values___")
        self.values_dict[f"k_current"] = []
        pos = 0
        if self.kpath is not None and self.kpath.nsegments == len(self.kpath.ngrids):
            logger.info(
                "Kpath exists and nsegments == ngrids. Creating path from kpath"
            )
            logger.debug(f"ngrids: {self.kpath.ngrids}")
            logger.debug(f"nsegments: {self.kpath.nsegments}")
            
            k_current = None
            for isegment in range(self.kpath.nsegments):
                kstart, kend = self.kpath.special_kpoints[isegment]
                if self.kdirect is False:
                    kstart = np.dot(self.ebs.reciprocal_lattice, kstart)
                    kend = np.dot(self.ebs.reciprocal_lattice, kend)

                distance = np.linalg.norm(kend - kstart)
                if isegment == 0:
                    x = np.linspace(pos, pos + distance, self.kpath.ngrids[isegment])
                    k_current = kstart
                    self.values_dict[f"k_current"].append(k_current.tolist())
                else:
                    x = np.append(
                        x,
                        np.linspace(pos, pos + distance, self.kpath.ngrids[isegment]),
                        axis=0,
                    )

                k_unit_dir = (kend - kstart) / distance
                for i in range(x.shape[0]):
                    k_current += k_unit_dir * distance
                    self.values_dict[f"k_current"].append(k_current.tolist())

                pos += distance
        else:
            logger.info(
                "Kpath does not exist or nsegments != ngrids. Creating x from kpoints"
            )

            x = np.arange(0, self.ebs.kpoints.shape[0])

        return np.array(x).reshape(
            -1,
        )

    def plot_bands(self):
        """
        Plot the plain band structure.

        Returns
        -------
        None.

        """
        values_dict = {}

        for ispin in self.spins:
            if len(self.spins) == 1:
                color = self.config.color
            else:
                color = self.config.spin_colors[ispin]

            for iband in range(self.ebs.nbands):
                handle = self.ax.plot(
                    self.x,
                    self.ebs.bands[:, iband, ispin],
                    color=color,
                    alpha=self.config.opacity[ispin],
                    linestyle=self.config.linestyle[ispin],
                    label=self.config.label[ispin],
                    linewidth=self.config.linewidth[ispin],
                )
                self.handles.append(handle)

                band_name = f"band-{iband}_spinChannel-{str(ispin)}"
                values_dict[f"bands_{band_name}"] = self.ebs.bands[:, iband, ispin]

        values_dict["kpath_values"] = self.x
        values_dict["kpath_tick_names"] = []
        for i, x in enumerate(self.x):
            tick_name = ""
            if self.kpath is not None:
                for i_tick, tick_position in enumerate(self.kpath.tick_positions):
                    if i == tick_position:
                        tick_name = self.kpath.tick_names[i_tick]
                values_dict["kpath_tick_names"].append(tick_name)
        self.values_dict = values_dict

    def plot_scatter(
        self,
        width_mask: np.ndarray = None,
        color_mask: np.ndarray = None,
        spins: List[int] = None,
        width_weights: np.ndarray = None,
        color_weights: np.ndarray = None,
        labels=None,
    ):
        """A method to plot a scatter plot

        Parameters
        ----------
        width_mask : np.ndarray, optional
            The width mask, by default None
        color_mask : np.ndarray, optional
            The color mask, by default None
        spins : List[int], optional
            A list of spins, by default None
        width_weights : np.ndarray, optional
            The width weight of each point, by default None
        color_weights : np.ndarray, optional
            The color weights at each point, by default None
        """
        values_dict = {}

        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]

        if width_weights is None:
            width_weights = np.ones_like(self.ebs.bands)
            markersize = self.config.markersize
        else:
            markersize = [l * 30 for l in self.config.markersize]

        if width_mask is not None or color_mask is not None:
            if width_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(width_weights) < width_mask
                )
            if color_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(color_weights) < color_mask
                )
        else:
            # Faking a mask, all elemtnet are included
            mbands = np.ma.masked_array(self.ebs.bands, False)

        if color_weights is not None:
            vmin = self.config.clim[0]
            vmax = self.config.clim[1]
            if vmin is None:
                # only the actual spin values are to be used (i.e. we
                # are plotting the density, then negative values from
                # spin projections are nonsense )
                vmin = color_weights[:, :, spins].min()
            if vmax is None:
                vmax = color_weights[:, :, spins].max()

        for ispin in spins:
            for iband in range(self.ebs.nbands):
                if len(self.spins) == 1:
                    color = self.config.color
                else:
                    color = self.config.spin_colors[ispin]
                if color_weights is None:
                    sc = self.ax.scatter(
                        self.x,
                        mbands[:, iband, ispin],
                        c=color,
                        s=width_weights[:, iband, ispin] * markersize[ispin],
                        # edgecolors="none",
                        linewidths=self.config.linewidth[ispin],
                        cmap=self.config.cmap,
                        vmin=vmin,
                        vmax=vmax,
                        marker=self.config.marker[ispin],
                        alpha=self.config.opacity[ispin],
                    )
                else:
                    sc = self.ax.scatter(
                        self.x,
                        mbands[:, iband, ispin],
                        c=color_weights[:, iband, ispin],
                        s=width_weights[:, iband, ispin] * markersize[ispin],
                        # edgecolors="none",
                        linewidths=self.config.linewidth[ispin],
                        cmap=self.config.cmap,
                        vmin=vmin,
                        vmax=vmax,
                        marker=self.config.marker[ispin],
                        alpha=self.config.opacity[ispin],
                    )

                band_name = f"band-{iband}_spinChannel-{str(ispin)}"
                values_dict[f"bands__{band_name}"] = self.ebs.bands[:, iband, ispin]
                projection_name = labels[0]
                if color_weights is not None:
                    values_dict[f"projections__{projection_name}__{band_name}"] = (
                        color_weights[:, iband, ispin]
                    )

        if self.config.plot_color_bar and color_weights is not None:
            self.cb = self.fig.colorbar(sc, ax=self.ax)

        values_dict["kpath_values"] = self.x
        tick_names = []
        for i, x in enumerate(self.x):
            tick_name = ""
            if self.kpath is not None:
                for i_tick, tick_position in enumerate(self.kpath.tick_positions):
                    if i == tick_position:
                        tick_name = self.kpath.tick_names[i_tick]
                tick_names.append(tick_name)
        values_dict["kpath_tick_names"] = tick_names

        self.values_dict = values_dict

    def plot_parameteric(
        self,
        spins: List[int] = None,
        width_mask: np.ndarray = None,
        color_mask: np.ndarray = None,
        width_weights: np.ndarray = None,
        color_weights: np.ndarray = None,
        elimit: List[float] = None,
        labels=None,
    ):
        """A method to plot a scatter plot

        Parameters
        ----------
        spins : List[int], optional
            A list of spins, by default None
        width_mask : np.ndarray, optional
            The width mask, by default None
        color_mask : np.ndarray, optional
            The color mask, by default None
        width_weights : np.ndarray, optional
            The width weight of each point, by default None
        color_weights : np.ndarray, optional
            The color weights at each point, by default None
        elimit : List[float], optional
            Energy range to plot. Only useful if the band index is written
        """
        values_dict = {}
        if labels is None:
            labels = [""]

        # if there is only a single k-point the method for atomic
        # levels will be called to fake another kpoint and then
        # exit. `plot_atomic_levels` will invoke this method again to
        # get the actual plot
        if len(self.ebs.kpoints) == 1:
            self.plot_atomic_levels(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins,
                elimit=elimit,
            )
            return

        if width_weights is None:
            width_weights = np.ones_like(self.ebs.bands)
            linewidth = self.config.linewidth
        else:
            linewidth = [l * 5 for l in self.config.linewidth]

        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]

        if width_mask is not None or color_mask is not None:
            if width_mask is not None:
                logger.info(f"___Applying width mask___")
                mbands = np.ma.masked_array(
                    self.ebs.bands,
                    np.abs(width_weights) < width_mask,
                )
            if color_mask is not None:
                logger.info(f"___Applying color mask___")
                mbands = np.ma.masked_array(
                    self.ebs.bands,
                    np.abs(color_weights) < color_mask,
                )
        else:
            # Faking a mask, all elemtnet are included
            logger.info(f"___No mask applied___")
            mbands = np.ma.masked_array(self.ebs.bands, False)
        if color_weights is not None:
            vmin = self.config.clim[0]
            vmax = self.config.clim[1]
            if vmin is None:
                vmin = color_weights[:, :, spins].min()
            if vmax is None:
                vmax = color_weights[:, :, spins].max()
            norm = mpl.colors.Normalize(vmin, vmax)

        for ispin in spins:
            for iband in range(self.ebs.nbands):
                if len(self.spins) == 1:
                    color = self.config.color
                else:
                    color = self.config.spin_colors[ispin]
                points = np.array([self.x, mbands[:, iband, ispin]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # this is to delete the segments on the high sym points
                x = self.x
                # segments = np.delete(
                #     segments, np.where(x[1:] == x[:-1])[0], axis=0)
                if color_weights is None:
                    lc = LineCollection(
                        segments, colors=color, linestyle=self.config.linestyle[ispin]
                    )
                else:
                    lc = LineCollection(
                        segments, cmap=plt.get_cmap(self.config.cmap), norm=norm
                    )
                    lc.set_array(color_weights[:, iband, ispin])
                lc.set_linewidth(width_weights[:, iband, ispin] * linewidth[ispin])
                lc.set_linestyle(self.config.linestyle[ispin])
                handle = self.ax.add_collection(lc)

                band_name = f"band-{iband}_spinChannel-{str(ispin)}"
                projection_name = labels[0]
                values_dict[f"bands__{band_name}"] = self.ebs.bands[:, iband, ispin]
                if color_weights is not None:
                    values_dict[f"projections__{projection_name}__{band_name}"] = (
                        color_weights[:, iband, ispin]
                    )
            # if color_weights is not None:
            #     handle.set_color(color_map[iweight][:-1].lower())
            handle.set_linewidth(linewidth)

            # mpatches.Patch(color='red', label='The red data')
            self.handles.append(mpatches.Patch(color="red", label="The red data"))

        if self.config.plot_color_bar and color_weights is not None:
            self.cb = self.fig.colorbar(lc, ax=self.ax)

        values_dict["kpath_values"] = self.x
        tick_names = []
        for i, x in enumerate(self.x):
            tick_name = ""
            if self.kpath is not None:
                for i_tick, tick_position in enumerate(self.kpath.tick_positions):
                    if i == tick_position:
                        tick_name = self.kpath.tick_names[i_tick]
                tick_names.append(tick_name)
        values_dict["kpath_tick_names"] = tick_names
        self.values_dict = values_dict

    def plot_parameteric_overlay(
        self,
        spins: List[int] = None,
        weights: np.ndarray = None,
        labels: str = None,
    ):
        """A method to plot the parametric overlay

        Parameters
        ----------
        spins : List[int], optional
            A list of spins, by default None
        weights : np.ndarray, optional
            The weights of each point, by default None
        """
        values_dict = {}
        if labels is None:
            labels = [""]

        linewidth = [l * 7 for l in self.config.linewidth]
        if type(self.config.cmap) is str:
            color_map = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
        else:
            color_map = self.config.cmap
        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]

        for iweight, weight in enumerate(weights):
            cmap = plt.get_cmap(color_map[iweight])
            vmin = self.config.clim[0]
            vmax = self.config.clim[1]
            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 1
            norm = mpl.colors.Normalize(vmin, vmax)
            for ispin in spins:
                # plotting
                for iband in range(self.ebs.nbands):
                    points = np.array(
                        [self.x, self.ebs.bands[:, iband, ispin]]
                    ).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    # this is to delete the segments on the high sym points
                    x = self.x
                    segments = np.delete(segments, np.where(x[1:] == x[:-1])[0], axis=0)
                    lc = LineCollection(
                        segments,
                        cmap=cmap,
                        norm=norm,
                        alpha=self.config.opacity[ispin],
                    )
                    lc.set_array(weight[:, iband, ispin])
                    lc.set_linewidth(weight[:, iband, ispin] * linewidth[ispin])

                    self.ax.add_collection(lc)

                    band_name = f"band-{iband}_spinChannel-{str(ispin)}"
                    projection_name = labels[iweight]
                    values_dict[f"bands__{band_name}"] = self.ebs.bands[:, iband, ispin]
                    if weights is not None:
                        values_dict[f"projections__{projection_name}__{band_name}"] = (
                            weight[:, iband, ispin]
                        )

            self.handles.append(
                mpatches.Patch(
                    color=color_map[iweight][:-1].lower(), label=labels[iweight]
                )
            )

            if self.config.plot_color_bar:
                self.cb = self.fig.colorbar(lc, ax=self.ax)

        values_dict["kpath_values"] = self.x
        tick_names = []
        for i, x in enumerate(self.x):
            tick_name = ""
            if self.kpath is not None:
                for i_tick, tick_position in enumerate(self.kpath.tick_positions):
                    if i == tick_position:
                        tick_name = self.kpath.tick_names[i_tick]
                tick_names.append(tick_name)
        values_dict["kpath_tick_names"] = tick_names
        self.values_dict = values_dict

    def plot_atomic_levels(
        self,
        spins: List[int] = None,
        width_mask: np.ndarray = None,
        color_mask: np.ndarray = None,
        width_weights: np.ndarray = None,
        color_weights: np.ndarray = None,
        elimit: List[float] = None,
        labels=None,
    ):
        """A method to plot a scatter plot

        Parameters
        ----------
        spins : List[int], optional
            A list of spins, by default None
        width_mask : np.ndarray, optional
            The width mask, by default None
        color_mask : np.ndarray, optional
            The color mask, by default None
        width_weights : np.ndarray, optional
            The width weight of each point, by default None
        color_weights : np.ndarray, optional
            The color weights at each point, by default None
        elimit : List[float], optional
            The energy range to plot.
        """
        if labels is None:
            labels = [""]
        self.ebs.bands = np.vstack((self.ebs.bands, self.ebs.bands))
        self.ebs.projected = np.vstack((self.ebs.projected, self.ebs.projected))
        self.ebs.kpoints = np.vstack((self.ebs.kpoints, self.ebs.kpoints))
        self.ebs.kpoints[0][-1] += 1
        self.x = self._get_x()
        print("Atomic plot: bands.shape  :", self.ebs.bands.shape)
        print("Atomic plot: spd.shape    :", self.ebs.projected.shape)
        print("Atomic plot: kpoints.shape:", self.ebs.kpoints.shape)
        self.ax.xaxis.set_major_locator(plt.NullLocator())
        # labels on each band

        if elimit:
            emin, emax = elimit[0], elimit[1]
        else:
            emin, emax = np.min(self.ebs.bands), np.max(self.ebs.bands)
        # print('Energy range', emin, emax)

        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]
        # cointainers for the bounding boxes of the text elements
        Nspin = len(spins)
        texts = []
        for ispin in spins:
            for i in range(len(self.ebs.bands[0, :, ispin])):
                energy = self.ebs.bands[0, i, ispin]
                if energy > emin and energy < emax:
                    txt = [0, energy, f"s-{ispin} : " + "b-" + str(i + 1)]
                    texts.append(txt)
        # sorting the texts
        texts.sort(key=lambda x: x[1])

        # I need to set the energy limits
        self.set_ylim(elimit)
        self.set_xlim()

        # knowing the text size
        txt = texts[-1]
        txt = plt.text(*txt)
        bbox = txt.get_window_extent()
        bbox_data = self.ax.transData.inverted().transform_bbox(bbox)
        w, h = bbox_data.width, bbox_data.height
        txt.remove()
        # print('Width, ', w, '. Height,', h)

        shift = 0
        txt = texts[0]
        self.ax.text(*txt)
        for i in range(1, len(texts)):
            txt = texts[i]
            last = texts[i - 1]
            y, y0 = txt[1], last[1]
            # if there there is vertical overlap
            if y < y0 + h:
                # print('overlap', y, y0+h)
                # I shift it laterally (the shift can be 0)
                shift += 1
                if shift == 2:
                    shift = 0

                txt[0] = txt[0] + w * 1.5 * shift
            else:
                shift = 0

            # print(txt)
            self.ax.text(*txt)

        self.plot_parameteric(
            color_weights=color_weights,
            width_weights=width_weights,
            color_mask=color_mask,
            width_mask=width_mask,
            spins=spins,
            labels=labels,
        )

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
        logger.info("___Setting x ticks___")
        logger.debug(
            "Setting x ticks:\n\tpositions=%s\n\tnames=%s\n\tcolor=%s",
            self.kpath.tick_positions if self.kpath is not None else tick_positions,
            self.kpath.tick_names if self.kpath is not None else tick_names,
            color,
        )

        if self.kpath is not None:
            if tick_positions is None:
                tick_positions = self.kpath.tick_positions
                logger.debug("Overriding default tick_positions: %s", tick_positions)
            if tick_names is None:
                tick_names = self.kpath.tick_names
                logger.debug("Overriding default tick_names: %s", tick_names)

        if tick_positions is not None:
            for ipos in tick_positions:
                self.ax.axvline(self.x[ipos], color=color)
            self.ax.set_xticks(self.x[tick_positions])

        if tick_names is not None:
            self.ax.set_xticklabels(tick_names)

        self.ax.tick_params(**self.config.major_x_tick_params)

    def set_yticks(
        self, major: float = None, minor: float = None, interval: List[float] = None
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
        # if (major is None or minor is None):
        if interval is None:
            interval = (
                self.ebs.bands.min() - abs(self.ebs.bands.min()) * 0.1,
                self.ebs.bands.max() * 1.1,
            )

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

        if self.config.multiple_locator_y_major_value is not None:
            major = self.config.multiple_locator_y_major_value
        if self.config.multiple_locator_y_minor_value is not None:
            minor = self.config.multiple_locator_y_minor_value

        if (
            self.config.major_y_locator is not None
            or self.config.minor_y_locator is not None
        ):
            if self.config.major_y_locator is not None:
                self.ax.yaxis.set_major_locator(self.config.major_y_locator)
            if self.config.minor_y_locator is not None:
                self.ax.yaxis.set_minor_locator(self.config.minor_y_locator)
        else:
            if major is not None:
                self.ax.yaxis.set_major_locator(MultipleLocator(major))
            if minor is not None:
                self.ax.yaxis.set_minor_locator(MultipleLocator(minor))

        self.ax.tick_params(**self.config.major_y_tick_params)
        self.ax.tick_params(**self.config.minor_y_tick_params)

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

    def set_ylabel(self, label: str = r"E - E$_F$ (eV)"):
        """A method to set the y label

        Parameters
        ----------
        label : str, optional
            String fo the y label name, by default r"E - E$ (eV)"
        """
        self.ax.set_ylabel(label, **self.config.y_label_params)

    def set_title(self, title: str = "Band Structure"):
        """A method to set the title

        Parameters
        ----------
        title : str, optional
            String for the title, by default "Band Structure"
        """
        if self.config.title:
            self.ax.set_title(label=self.config.title, **self.config.title_params)

    def set_colorbar_title(self, title: str = None):
        """A method to set the title of the color bar

        Parameters
        ----------
        title : str, optional
            String for the title, by default "Atomic Orbital Projections"
        """
        if title:
            title = title
        else:
            title = self.config.colorbar_title
            
        if not hasattr(self, "cb"):
            return None

        if self.config.colorbar_tick_params:
            self.cb.ax.tick_params(**self.config.colorbar_tick_params)
        else:
            self.cb.ax.tick_params(labelsize=self.config.colorbar_tick_labelsize)

        if self.config.colorbar_label_params:
            self.cb.set_label(title, **self.config.colorbar_label_params)
        else:
            self.cb.set_label(
                title,
                size=self.config.colorbar_title_size,
                rotation=270,
                labelpad=self.config.colorbar_title_padding,
            )

    def legend(self, labels: List[str] = None):
        """A methdo to plot the legend

        Parameters
        ----------
        labels : List[str], optional
            A list of strings for the labels of each element for the legend, by default None
        """
        if labels == None:
            labels = self.config.label

        if self.config.legend:
            self.ax.legend(self.handles, labels)

    def draw_fermi(
        self,
        fermi_level: float = 0,
    ):
        """A method to draw the fermi line

        Parameters
        ----------
        fermi_level : str, optional
            The energy level to draw the line
        """
        self.ax.axhline(
            y=fermi_level,
            color=self.config.fermi_color,
            linestyle=self.config.fermi_linestyle,
            linewidth=self.config.fermi_linewidth,
        )

    def grid(self):
        """A method to plot a grid"""
        if self.config.grid:
            self.ax.grid(
                self.config.grid,
                which=self.config.grid_which,
                color=self.config.grid_color,
                linestyle=self.config.grid_linestyle,
                linewidth=self.config.grid_linewidth,
            )

    def show(self):
        """A method to show the plot"""
        plt.show()

    def save(self, filename: str = "bands.pdf"):
        """A method to save the plot

        Parameters
        ----------
        filename : str, optional
            A string for the file name, by default 'bands.pdf'
        """
        plt.savefig(filename, dpi=self.config.dpi, bbox_inches="tight")
        plt.clf()

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

        logger.info("___Exporting data to %s___", filename)

        values_dict = {}
        for key, value in self.values_dict.items():
            logger.debug(
                "Column: %s, Type: %s, Shape: %s", key, type(value), len(value)
            )

            if len(value) != 0:
                values_dict[key] = value

        column_names = list(values_dict.keys())
        sorted_column_names = [None] * len(column_names)
        index = 0
        for column_name in column_names:
            if "kpath_values" == column_name:
                sorted_column_names[index] = column_name
                index += 1
            if "kpath_tick_names" == column_name:
                sorted_column_names[index] = column_name
                index += 1
            if "k_current" == column_name:
                sorted_column_names[index] = column_name
                index += 1
        for ispin in range(2):
            for column_name in column_names:

                if "spinChannel-0" in column_name.split("_")[-1] and ispin == 0:
                    sorted_column_names[index] = column_name
                    index += 1
                if "spinChannel-1" in column_name.split("_")[-1] and ispin == 1:
                    sorted_column_names[index] = column_name
                    index += 1

        column_names.sort()
        if file_type == "csv":
            df = pd.DataFrame(values_dict)
            df.to_csv(filename, columns=sorted_column_names, index=False)
        elif file_type == "txt":
            df = pd.DataFrame(values_dict)
            df.to_csv(filename, columns=sorted_column_names, sep="\t", index=False)
        elif file_type == "json":
            with open(filename, "w") as outfile:
                for key, value in values_dict.items():
                    values_dict[key] = value.tolist()
                json.dump(values_dict, outfile)
        elif file_type == "dat":
            df = pd.DataFrame(values_dict)
            df.to_csv(filename, columns=sorted_column_names, sep=" ", index=False)
