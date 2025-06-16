import sys
import re
import logging

import numpy as np
import matplotlib.pyplot as plt

class ProcarPlot:
    """A depeciated class ot plot the band structure
    """
    def __init__(self, bands, spd, kpoints=None):
        self.bands = bands.transpose()
        self.spd = spd.transpose()
        self.kpoints = kpoints
        return

    def plotBands(
        self,
        size=0.02,
        marker="o",
        ticks=None,
        color="blue",
        discontinuities=[],
        figsize=(13, 9),
        ax=None,
    ):

        if not ax:
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
            ax = fig.add_subplot(111)
        else:
            fig = plt.gca()

        if size is not None:
            size = size / 2

        if self.kpoints is not None:
            xaxis = [0]

            #### MODIFIED FOR DISCONTINOUS BANDS ####
            if ticks:
                ticks, ticksNames = list(zip(*ticks))

                # counters for number of discontinuities
                icounter = 1
                ii = 0

                for i in range(1, len(self.kpoints) - len(discontinuities)):
                    d = self.kpoints[icounter] - self.kpoints[icounter - 1]
                    d = np.sqrt(np.dot(d, d))
                    xaxis.append(d + xaxis[-1])
                    icounter += 1
                    ii += 1
                    if ii in discontinuities:
                        icounter += 1
                        ii += 1
                        xaxis.append(xaxis[-1])
                xaxis = np.array(xaxis)

                # plotting

                for i_tick in range(len(ticks) - 1):
                    x = xaxis[ticks[i_tick] : ticks[i_tick + 1] + 1]
                    y = self.bands.transpose()[ticks[i_tick] : ticks[i_tick + 1] + 1, :]
                    ax.plot(x, y, "r-", marker=marker, markersize=size, color=color)

            #### END  OF MODIFIED DISCONTINUOUS BANDS ####

            # if ticks are not given
            else:
                for i in range(1, len(self.kpoints)):
                    d = self.kpoints[i - 1] - self.kpoints[i]
                    d = np.sqrt(np.dot(d, d))
                    xaxis.append(d + xaxis[-1])
                xaxis = np.array(xaxis)
                ax.plot(
                    xaxis,
                    self.bands.transpose(),
                    "r-",
                    marker=marker,
                    markersize=size,
                    color=color,
                )

        # if self.kpoints is None
        else:
            xaxis = np.arange(len(self.bands))
            ax.plot(
                xaxis,
                self.bands.transpose(),
                "r-",
                marker=marker,
                markersize=size,
                color=color,
            )

        ax.set_xlim(xaxis.min(), xaxis.max())

        # Handling ticks
        if ticks:
            # added for meta-GGA calculations
            if ticks[0] > 0:
                ax.set_xlim(left=xaxis[ticks[0]])
            ticks = [xaxis[x] for x in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticksNames)
            for xc in ticks:
                ax.axvline(x=xc, color="k")
        ax.axhline(color="black", linestyle="--")

        return fig, ax

    def parametricPlot(
        self,
        cmap="jet",
        vmin=None,
        vmax=None,
        mask=None,
        ticks=None,
        discontinuities=[],
        ax=None,
        figsize=(13, 9),
        plot_bar=True,
        linewidth=1,
    ):
        from matplotlib.collections import LineCollection
        import matplotlib

        # fig = plt.figure() # use plt.gca() since it won't create a new figure for band comparison
        if ax is None:
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        bsize, ksize = self.bands.shape

        # print self.bands
        if mask is not None:
            mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
        else:
            # Faking a mask, all elemtnet are included
            mbands = np.ma.masked_array(self.bands, False)

        if vmin is None:
            vmin = self.spd.min()
        if vmax is None:
            vmax = self.spd.max()
        print("normalizing to : ", (vmin, vmax))
        norm = matplotlib.colors.Normalize(vmin, vmax)

        if self.kpoints is not None:
            xaxis = [0]

            #### MODIFIED FOR DISCONTINOUS BANDS ####
            if ticks:
                ticks, ticksNames = list(zip(*ticks))

                # counters for number of discontinuities
                icounter = 1
                ii = 0

                for i in range(1, len(self.kpoints) - len(discontinuities)):
                    d = self.kpoints[icounter] - self.kpoints[icounter - 1]
                    d = np.sqrt(np.dot(d, d))
                    xaxis.append(d + xaxis[-1])
                    icounter += 1
                    ii += 1
                    if ii in discontinuities:
                        icounter += 1
                        ii += 1
                        xaxis.append(xaxis[-1])
                xaxis = np.array(xaxis)

                # plotting
                for y, z in zip(mbands, self.spd):
                    points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm)
                    lc.set_array(z)
                    lc.set_linewidth(linewidth)
                    ax.add_collection(lc)
                if plot_bar:
                    cb = fig.colorbar(lc, ax=ax)
                    cb.ax.tick_params(labelsize=20)
                ax.set_xlim(xaxis.min(), xaxis.max())
                ax.set_ylim(mbands.min(), mbands.max())

            #### END  OF MODIFIED DISCONTINUOUS BANDS ####

            # if ticks are not given
            else:
                for i in range(1, len(self.kpoints)):
                    d = self.kpoints[i - 1] - self.kpoints[i]
                    d = np.sqrt(np.dot(d, d))
                    xaxis.append(d + xaxis[-1])
                xaxis = np.array(xaxis)

                # plotting
                for y, z in zip(mbands, self.spd):
                    points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm)
                    lc.set_array(z)
                    lc.set_linewidth(linewidth)
                    ax.add_collection(lc)
                if plot_bar:
                    cb = fig.colorbar(lc, ax=ax)
                    cb.ax.tick_params(labelsize=20)
                ax.set_xlim(xaxis.min(), xaxis.max())
                ax.set_ylim(mbands.min(), mbands.max())

        # if self.kpoints is None
        else:
            xaxis = np.arange(ksize)
            for y, z in zip(mbands, self.spd):
                points = np.array([xaxis, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm)
                lc.set_array(z)
                lc.set_linewidth(linewidth)
                ax.add_collection(lc)
            if plot_bar:
                cb = fig.colorbar(lc, ax=ax)
                cb.ax.tick_params(labelsize=20)
            ax.set_xlim(xaxis.min(), xaxis.max())
            ax.set_ylim(mbands.min(), mbands.max())

        # handling ticks
        if ticks:
            # added for meta-GGA calculations
            if ticks[0] > 0:
                ax.set_xlim(left=xaxis[ticks[0]])
            ticks = [xaxis[x] for x in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticksNames)
            for xc in ticks:
                ax.axvline(x=xc, color="black")

        ax.axhline(color="black", linestyle="--")

        return fig, ax

    def scatterPlot(
        self,
        size=50,
        mask=None,
        cmap="hot_r",
        vmax=None,
        vmin=None,
        marker="o",
        ticks=None,
        discontinuities=[],
        ax=None,
    ):
        bsize, ksize = self.bands.shape
        # plotting
        if not ax:
            fig = plt.figure(figsize=(13, 9))
            fig.tight_layout()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        if self.kpoints is not None:
            xaxis = [0]

            #### MODIFIED FOR DISCONTINOUS BANDS ####
            if ticks:
                ticks, ticksNames = list(zip(*ticks))

                # counters for number of discontinuities
                icounter = 1
                ii = 0

                for i in range(1, len(self.kpoints) - len(discontinuities)):
                    d = self.kpoints[icounter] - self.kpoints[icounter - 1]
                    d = np.sqrt(np.dot(d, d))
                    xaxis.append(d + xaxis[-1])
                    icounter += 1
                    ii += 1
                    if ii in discontinuities:
                        icounter += 1
                        ii += 1
                        xaxis.append(xaxis[-1])
                xaxis = np.array(xaxis)

                xaxis.shape = (1, ksize)
                xaxis = xaxis.repeat(bsize, axis=0)
                if mask is not None:
                    mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
                else:
                    mbands = self.bands

                scatter = ax.scatter(
                    xaxis,
                    mbands,
                    c=self.spd,
                    s=size,
                    linewidths=0,
                    cmap=cmap,
                    vmax=vmax,
                    vmin=vmin,
                    marker=marker,
                    edgecolors="none",
                )
                fig.colorbar(scatter)
                ax.set_xlim(xaxis.min(), xaxis.max())

            #### END  OF MODIFIED DISCONTINUOUS BANDS ####

            # if ticks are not given
            else:
                for i in range(1, len(self.kpoints)):
                    d = self.kpoints[i - 1] - self.kpoints[i]
                    d = np.sqrt(np.dot(d, d))
                    xaxis.append(d + xaxis[-1])
                xaxis = np.array(xaxis)

                xaxis.shape = (1, ksize)
                xaxis = xaxis.repeat(bsize, axis=0)
                if mask is not None:
                    mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
                else:
                    mbands = self.bands

                scatter = ax.scatter(
                    xaxis,
                    mbands,
                    c=self.spd,
                    s=size,
                    linewidths=0,
                    cmap=cmap,
                    vmax=vmax,
                    vmin=vmin,
                    marker=marker,
                    edgecolors="none",
                )

                plt.colorbar(scatter)
                ax.set_xlim(xaxis.min(), xaxis.max())

        # if kpoints is None
        else:
            xaxis = np.arange(ksize)
            xaxis.shape = (1, ksize)
            xaxis = xaxis.repeat(bsize, axis=0)
            if mask is not None:
                mbands = np.ma.masked_array(self.bands, np.abs(self.spd) < mask)
            else:
                mbands = self.bands

            ax.scatter(
                xaxis,
                mbands,
                c=self.spd,
                s=size,
                linewidths=0,
                cmap=cmap,
                vmax=vmax,
                vmin=vmin,
                marker=marker,
                edgecolors="none",
            )

            plt.colorbar(ax=ax)
            ax.set_xlim(xaxis.min(), xaxis.max())

        # handling ticks
        if ticks:
            # added for meta-GGA calculations
            if ticks[0] > 0:
                ax.set_xlim(left=xaxis[0, ticks[0]])
            ticks = [xaxis[0, x] for x in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticksNames)

        ax.axhline(color="black", linestyle="--")

        return fig, ax

    def atomicPlot(self, cmap="hot_r", vmin=None, vmax=None, ax=None):
        """
    Just a handler to parametricPlot. Useful to plot energy levels.

    It adds a fake k-point. Shouldn't be invoked with more than one
    k-point
    ax not implemented here, not need
    """

        print("Atomic plot: bands.shape  :", self.bands.shape)
        print("Atomic plot: spd.shape    :", self.spd.shape)
        print("Atomic plot: kpoints.shape:", self.kpoints.shape)

        self.bands = np.hstack((self.bands, self.bands))
        self.spd = np.hstack((self.spd, self.spd))
        self.kpoints = np.vstack((self.kpoints, self.kpoints))
        self.kpoints[0][-1] += 1
        print("Atomic plot: bands.shape  :", self.bands.shape)
        print("Atomic plot: spd.shape    :", self.spd.shape)
        print("Atomic plot: kpoints.shape:", self.kpoints.shape)
        
        print("Foooooooooooooooooo", self.kpoints)

        fig, ax1 = self.parametricPlot(cmap, vmin, vmax, ax=ax)

        #        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        ax1.xaxis.set_major_locator(plt.NullLocator())
        # labels on each band
        for i in range(len(self.bands[:, 0])):
            # print i, self.bands[i]
            ax1.text(0, self.bands[i, 0], str(i + 1))
            bbox = txt.get_window_extent()
            print('bbox', bbox)
        return fig, ax1
