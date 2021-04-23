import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
import sys


class EBSPlot:
    def __init__(self, ebs, kpath=None, ax=None):
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
        if ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax
        # need to initiate kpath if kpath is not defined.
        self.x = self._get_x()

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

    def plot_bands(self, spins=None, color="blue", opacity=1.0):
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
        if spins is None:
            spins = range(self.ebs.nspins)
        for ispin in spins:
            for iband in range(self.ebs.nbands):
                self.ax.plot(
                    self.x, self.ebs.bands[:, iband,
                                           ispin], color=color, alpha=opacity,
                )

    def plot_scatter(self,
                     spins=None,
                     size=50,
                     mask=None,
                     cmap="Reds",
                     vmin=0,
                     vmax=1,
                     marker="o",
                     opacity=1.0,
                     size_weights=None,
                     color_weights=None,
                     plot_bar=True,
                     ):
        """
        

        Parameters
        ----------
        spins : TYPE, optional
            DESCRIPTION. The default is None.
        size : TYPE, optional
            DESCRIPTION. The default is 50.
        mask : TYPE, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is "Reds".
        vmin : TYPE, optional
            DESCRIPTION. The default is 0.
        vmax : TYPE, optional
            DESCRIPTION. The default is 1.
        marker : TYPE, optional
            DESCRIPTION. The default is "o".
        opacity : TYPE, optional
            DESCRIPTION. The default is 1.0.
        size_weights : TYPE, optional
            DESCRIPTION. The default is None.
        color_weights : TYPE, optional
            DESCRIPTION. The default is None.
        plot_bar : TYPE, optional
            DESCRIPTION. The default is True.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if spins is None:
            spins = range(self.ebs.nspins)
        if size_weights is None:
            size_weights = np.ones(
                shape=(self.ebs.nkpoints, self.ebs.nbands, self.ebs.nspins))
        if color_weights is None:
            color_weights = np.ones(
                shape=(self.ebs.nkpoints, self.ebs.nbands, self.ebs.nspins))
            plot_bar = False
        if mask is not None:
            mbands = np.ma.masked_array(
                self.ebs.bands, np.abs(color_weights) < mask)
        else:
            mbands = self.ebs.bands
        for ispin in spins:
            for iband in range(self.ebs.nbands):
                self.ax.scatter(
                    self.x,
                    mbands[:, iband, ispin],
                    c=color_weights[:, iband, ispin].round(2),
                    s=color_weights[:, iband, ispin].round(2)*size,
                    edgecolors="none",
                    linewidths=0,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    marker=marker,
                    alpha=opacity
                )
        # if plot_bar:
        #     cb = self.fig.colorbar(lc, ax=self.ax)
        #     cb.ax.tick_params(labelsize=20)

    def plot_parameteric(
        self,
        spins=None,
        cmap="jet",
        vmin=None,
        vmax=None,
        mask=None,
        linewidth=10,
        opacity=1.0,
        width_weights=None,
        color_weights=None,
        plot_bar=False,
    ):
        if spins is None:
            spins = range(self.ebs.nspins)
        if mask is not None:
            mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
        else:
            # Faking a mask, all elemtnet are included
            mbands = np.ma.masked_array(self.ebs.bands, False)

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
                    [self.x, self.ebs.bands[:, iband, ispin]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(
                    segments, cmap=plt.get_cmap(cmap), norm=norm)
                lc.set_array(color_weights[:, iband, ispin])
                lc.set_linewidth(linewidth*width_weights[:, iband, ispin])
                self.ax.add_collection(lc)
        if plot_bar:
            cb = self.fig.colorbar(lc, ax=self.ax)
            cb.ax.tick_params(labelsize=20)


    def set_xticks(self, color="black"):
        for ipos in self.kpath.tick_positions:
            self.ax.axvline(self.x[ipos], color=color)
        self.ax.set_xticks(self.x[self.kpath.tick_positions])
        self.ax.set_xticklabels(self.kpath.tick_names)

    def set_xlim(self, interval=None):
        if interval is None:
            interval = (self.x[0], self.x[-1])
        self.ax.set_xlim(interval)

    def set_ylim(self, interval):
        if interval is None:
            interval = (self.ebs.bands.min()-abs(self.ebs.bands.min())
                        * 0.1, self.ebs.bands.max()*1.1)
        self.ax.set_ylim(interval)

    def draw_fermi(self, color="red", linestyle="--"):
        self.ax.axhline(y=0, color=color, linestyle=linestyle)
