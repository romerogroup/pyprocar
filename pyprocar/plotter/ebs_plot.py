import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
import sys
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


class EBSPlot:
    def __init__(self, ebs, kpath=None, ax=None, spins=None, colors=None, opacities=None, linestyles=None, linewidths=None, labels=None):
        """
        class to plot an electronic band structure.

        Parameters
        ----------
        ebs : object
            An electronic band structure object pyprocar.core.ElectronicBandStructure.
        kpath : object, optional
            A kpath object pyprocar.core.KPath. The default is None.
        ax : object, optional
            A matplotlib Axes object. If provided the plot will be located at that ax.
            The default is None.

        Returns
        -------
        None.

        """
        self.ebs = ebs
        self.kpath = kpath
        self.spins = spins
        self.colors = colors
        self.opacities = opacities
        self.linestyles = linestyles
        self.linewidths = linewidths
        self.labels = labels
        self.handles = []

        if ax is None:
            self.fig = plt.figure(figsize=(9, 6))
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax
        # need to initiate kpath if kpath is not defined.
        self.x = self._get_x()

        self._initiate_plot_args()

    def _initiate_plot_args(self):

        if self.spins is None:
            self.spins = range(self.ebs.nspins)
        self.nspins = len(self.spins)

        if self.colors is None:
            self.colors = np.array(["black", "red"])[:self.nspins]
        if self.opacities is None:
            self.opacities = np.array([1.0, 1.0])[:self.nspins]
        if self.linestyles is None:
            self.linestyles = np.array(["solid", "dashed"])[:self.nspins]
        if self.linewidths is None:
            self.linewidths = np.array([1, 1])[:self.nspins]
        if self.labels is None:
            self.labels = np.array([r'$\uparrow$',
                                    r'$\downarrow$'])[:self.nspins]
        # print(self.nspins)
        # print(self.colors)
        # print(self.opacities)
        # print(self.linestyles)
        # print(self.linewidths)
        # print(self.labels)

    def _get_x(self):
        """
        provides the x axis data of the plots

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        pos = 0
        for isegment in range(self.kpath.nsegments):
            kstart, kend = self.kpath.special_kpoints[isegment]
            distance = np.linalg.norm(kend - kstart)
            if isegment == 0:
                x = np.linspace(pos, pos + distance,
                                self.kpath.ngrids[isegment])
            else:
                x = np.append(
                    x,
                    np.linspace(pos, pos + distance,
                                self.kpath.ngrids[isegment]),
                    axis=0,
                )
            pos += distance
        return np.array(x).reshape(-1,)

    def plot_bands(self):
        """
        Plot the plain band structure.

        Parameters
        ----------
        spins : list, optional
            A list of the spins to be plotted. The default is None.
        color : string, optional
            Color for the bands. The default is "blue".
        opacity : float, optional
            Opacity level between 0.0 and 1.0. The default is 1.0.

        Returns
        -------
        None.

        """

        for ispin in range(self.ebs.bands.shape[2]):
            for iband in range(self.ebs.nbands):
                handle = self.ax.plot(
                    self.x, self.ebs.bands[:, iband, ispin], color=self.colors[ispin], alpha=self.opacities[
                        ispin], linestyle=self.linestyles[ispin], label=self.labels[ispin], linewidth=self.linewidths[ispin],
                )
            self.handles.append(handle[0])

    def plot_order(self):
        for ispin in range(self.ebs.bands.shape[2]):
            for iband in range(self.ebs.nbands):
                self.ax.plot(
                    self.x, self.ebs.bands[:, iband, ispin], alpha=self.opacities[
                        ispin],  linewidth=self.linewidths[ispin],
                )


    def plot_scatter(self,
                     spins=None,
                     width_mask=None,
                     color_mask=None,
                     cmap=None,
                     vmin=None,
                     vmax=None,
                     marker="o",
                     opacity=1.0,
                     width_weights=None,
                     color_weights=None,
                     plot_color_bar=True,
                     ):
        if cmap is None:
            cmap = 'viridis'

        if width_weights is None:
            width_weights = np.ones_like(self.ebs.bands)
        else:
            self.linewidths =[l*30 for l in self.linewidths]

        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]

        if width_mask is not None or color_mask is not None:
            if width_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(width_weights) < width_mask)
            if color_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(color_weights) < color_mask)
        else:
            # Faking a mask, all elemtnet are included
            mbands = np.ma.masked_array(self.ebs.bands, False)

        if color_weights is not None:

            if vmin is None:
                vmin = color_weights.min()
            if vmax is None:
                vmax = color_weights.max()
            print("normalizing to : ", (vmin, vmax))
            norm = matplotlib.colors.Normalize(vmin, vmax)

        for ispin in spins:
            for iband in range(self.ebs.nbands):
                if color_weights is None:
                    sc = self.ax.scatter(
                        self.x,
                        mbands[:, iband, ispin],
                        c=self.colors[ispin],
                        s=width_weights[:, iband, ispin].round(
                            2)*self.linewidths[ispin],
                        edgecolors="none",
                        linewidths=0,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        marker=marker,
                        alpha=opacity,
                    )
                else:
                    sc = self.ax.scatter(
                        self.x,
                        mbands[:, iband, ispin],
                        c=color_weights[:, iband, ispin].round(2),
                        s=width_weights[:, iband, ispin].round(
                            2)*self.linewidths[ispin],
                        edgecolors="none",
                        linewidths=0,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        marker=marker,
                        alpha=opacity,
                        norm=norm,
                    )
        if plot_color_bar and color_weights is not None:
            cb = self.fig.colorbar(sc, ax=self.ax)
            cb.ax.tick_params(labelsize=20)

    def plot_parameteric(
        self,
        spins=None,
        cmap=None,
        vmin=None,
        vmax=None,
        width_mask=None,
        color_mask=None,
        opacity=1.0,
        width_weights=None,
        color_weights=None,
        plot_color_bar=False,
    ):
        if cmap is None:
            cmap = 'viridis'
        if width_weights is None:
            width_weights = np.ones_like(self.ebs.bands)
        else:
            self.linewidths = [l*5 for l in self.linewidths]

        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]

        if width_mask is not None or color_mask is not None:
            if width_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(width_weights) < width_mask)
            if color_mask is not None:
                mbands = np.ma.masked_array(
                    self.ebs.bands, np.abs(color_weights) < color_mask)
        else:
            # Faking a mask, all elemtnet are included
            mbands = np.ma.masked_array(self.ebs.bands, False)

        if color_weights is not None:

            if vmin is None:
                vmin = color_weights.min()
            if vmax is None:
                vmax = color_weights.max()
            print("normalizing to : ", (vmin, vmax))
            norm = matplotlib.colors.Normalize(vmin, vmax)

        for ispin in spins:
            # plotting
            for iband in range(self.ebs.nbands):

                points = np.array(
                    [self.x, mbands[:, iband, ispin]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # this is to delete the segments on the high sym points
                x = self.x
                segments = np.delete(
                    segments, np.where(x[1:] == x[:-1])[0], axis=0)
                if color_weights is None:
                    lc = LineCollection(
                        segments, colors=self.colors[ispin])
                else:
                    lc = LineCollection(
                        segments, cmap=plt.get_cmap(cmap), norm=norm)
                    lc.set_array(color_weights[:, iband, ispin])
                lc.set_linewidth(
                    width_weights[:, iband, ispin]*self.linewidths[ispin])
                self.ax.add_collection(lc)
        if plot_color_bar and color_weights is not None:
            cb = self.fig.colorbar(lc, ax=self.ax)
            cb.ax.tick_params(labelsize=20)

    def plot_parameteric_overlay(self,
                                 spins=None,
                                 cmaps=None,
                                 vmin=None,
                                 vmax=None,
                                 weights=None,
                                 plot_color_bar=False,
                                 ):

        self.linewidths = [l*7 for l in self.linewidths]
        if cmaps is None:
            cmaps = ['Reds', "Blues", "Greens",
                     "Purples", "Oranges", "Greys"]
        if spins is None:
            spins = range(self.ebs.nspins)
        if self.ebs.is_non_collinear:
            spins = [0]
        for iweight, weight in enumerate(weights):

            if vmin is None:
                vmin = 0
            if vmax is None:
                vmax = 1
            norm = matplotlib.colors.Normalize(vmin, vmax)

            for ispin in spins:
                # plotting
                for iband in range(self.ebs.nbands):

                    points = np.array(
                        [self.x, self.ebs.bands[:, iband, ispin]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate(
                        [points[:-1], points[1:]], axis=1)
                    # this is to delete the segments on the high sym points
                    x = self.x
                    segments = np.delete(
                        segments, np.where(x[1:] == x[:-1])[0], axis=0)
                    lc = LineCollection(
                        segments, cmap=plt.get_cmap(cmaps[iweight]), norm=norm, alpha=self.opacities[0])
                    lc.set_array(weight[:, iband, ispin])
                    lc.set_linewidth(
                        weight[:, iband, ispin]*self.linewidths[ispin])
                    handle = self.ax.add_collection(lc)
            handle.set_color(cmaps[iweight][:-1].lower())
            handle.set_linewidth(self.linewidths)
            self.handles.append(handle)
            if plot_color_bar:
                cb = self.fig.colorbar(lc, ax=self.ax)
                cb.ax.tick_params(labelsize=20)

    def set_xticks(self, tick_positions=None, tick_names=None, color="black"):
        if tick_positions is None:
            tick_positions = self.kpath.tick_positions
        if tick_names is None:
            tick_names = self.kpath.tick_names
        for ipos in tick_positions:
            self.ax.axvline(self.x[ipos], color=color)
        self.ax.set_xticks(self.x[tick_positions])
        self.ax.set_xticklabels(tick_names)

    def set_yticks(self, major=None, minor=None, interval=None):
        if (major is None or minor is None):
            if interval is None:
                interval = (self.ebs.bands.min()-abs(self.ebs.bands.min())
                            * 0.1, self.ebs.bands.max()*1.1)

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
        if major is not None and minor is not None:
            self.ax.yaxis.set_major_locator(MultipleLocator(major))
            self.ax.yaxis.set_minor_locator(MultipleLocator(minor))
        self.ax.tick_params(
            which='major',
            axis="y",
            direction="inout",
            width=1,
            length=5,
            labelright=False,
            right=True,
            left=True)  # )

        self.ax.tick_params(
            which='minor',
            axis="y",
            direction="in",
            left=True,
            right=True)

    def set_xlim(self, interval=None):
        if interval is None:
            interval = (self.x[0], self.x[-1])
        self.ax.set_xlim(interval)

    def set_ylim(self, interval=None):
        if interval is None:
            interval = (self.ebs.bands.min()-abs(self.ebs.bands.min())
                        * 0.1, self.ebs.bands.max()*1.1)
        self.ax.set_ylim(interval)

    def set_xlabel(self, label="K vector"):
        self.ax.set_xlabel(label)

    def set_ylabel(self, label=r"E - E$_F$ (eV)"):
        self.ax.set_ylabel(label)

    def legend(self):
        self.ax.legend(self.handles, self.labels)

    def draw_fermi(self, color="blue", linestyle="dotted"):
        self.ax.axhline(y=0, color=color, linestyle=linestyle)
