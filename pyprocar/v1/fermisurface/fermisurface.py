import numpy as np
import re
import logging
import matplotlib.pyplot as plt
import sys


class FermiSurface:
    def __init__(self, kpoints, bands, spd, loglevel=logging.WARNING):
        """FermiSurface: Class to build and to plot a 2D fermi surface.  It
    finds the relevant bands (crossig the Fermi level) and interpolate
    them

    args:
      kpoints: Numpy array with kpoints Nx3.
      bands: the bands with Fermi energy already substracted!, numpy array,
             Nkpoints x Nbands.
      spd: character (atomic, orbital) of each bands at each Kpoint, numpy
           array Nkpoints x Nbands.
      loglevel(=logging.WARNING): the verbosity level.
    """
        # Since some time ago Kpoints are in cartesian coords (ready to use)
        self.kpoints = kpoints
        self.bands = bands
        self.spd = spd
        self.useful = None  # List of useful bands (filled in findEnergy)
        self.energy = None

        self.log = logging.getLogger("FermiSurface")
        self.log.setLevel(loglevel)
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(
            logging.Formatter("%(name)s::%(levelname)s: " "%(message)s")
        )
        self.ch.setLevel(logging.DEBUG)
        self.log.addHandler(self.ch)

        self.log.debug("FermiSurface.init: ...")
        self.log.info("Kpoints.shape : " + str(self.kpoints.shape))
        self.log.info("bands.shape   : " + str(self.bands.shape))
        self.log.info("spd.shape     : " + str(self.spd.shape))
        self.log.debug("FermiSurface.init: ...Done")
        return

    def FindEnergy(self, energy):
        self.log.debug("FindEnergy: ...")
        self.energy = energy
        self.log.info("Energy   : " + str(energy))
        bands = self.bands.transpose()
        # searching for bands crossing the desired energy
        self.useful = np.where(
            np.logical_and(bands.min(axis=1) < energy, bands.max(axis=1) > energy)
        )
        self.log.info("set of useful bands    : " + str(self.useful))
        bands = bands[self.useful]
        self.log.debug("new bands.shape : " + str(bands.shape))
        if len(bands) == 0:
            self.log.error("No bands found in that range. Check your data. Returning")
            raise RuntimeError("No bands to plot")
        self.log.debug("FindEnergy: ...Done")
        return

    def Plot(self, interpolation=200, mask=None):
        """Only 2D layer geometry along z"""
        self.log.debug("Plot: ...")
        from scipy.interpolate import griddata

        if self.useful is None:
            raise RuntimeError("self.FindEnergy() must be called before Plotting")

        # selecting components of K-points
        x, y = self.kpoints[:, 0], self.kpoints[:, 1]
        self.log.debug("k_x[:10], k_y[:10] values" + str([x[:10], y[:10]]))

        bands = self.bands.transpose()[self.useful]

        # and new, interpolated component
        xmax, xmin = x.max(), x.min()
        ymax, ymin = y.max(), y.min()
        self.log.debug("xlim = " + str([xmin, xmax]) + "  ylim = " + str([ymin, ymax]))
        xnew, ynew = np.mgrid[
            xmin : xmax : interpolation * 1j, ymin : ymax : interpolation * 1j
        ]

        # interpolation
        bnew = []
        for band in bands:
            self.log.debug("Interpolating ...")
            bnew.append(griddata((x, y), band, (xnew, ynew), method="cubic"))

        plots = [
            plt.contour(
                xnew,
                ynew,
                z,
                [self.energy],
                linewidths=0.5,
                colors="k",
                linestyles="solid",
            )
            for z in bnew
        ]
        plt.axis("equal")
        self.log.debug("Plot: ...Done")
        return plots

    def st(self, sx, sy, sz, spin=None, noarrow=False, interpolation=300):
        """Only 2D layer geometry along z. It is like a enhanced version
    of 'plot' method.

    sx, sy, sz are spin projected Nkpoints x Nbands numpy arrays. They
    also are (already) projected by orbital and atom (from other
    class)

    """
        self.log.debug("st: ...")
        from scipy.interpolate import griddata

        if self.useful is None:
            raise RuntimeError("self.FindEnergy() must be called before Plotting")

        # selecting components of K-points
        x, y = self.kpoints[:, 0], self.kpoints[:, 1]

        bands = self.bands.transpose()[self.useful]

        sx = sx.transpose()[self.useful]
        sy = sy.transpose()[self.useful]
        sz = sz.transpose()[self.useful]

        # and new, interpolated component
        xmax, xmin = x.max(), x.min()
        ymax, ymin = y.max(), y.min()
        self.log.debug("xlim = " + str([xmin, xmax]) + "  ylim = " + str([ymin, ymax]))
        xnew, ynew = np.mgrid[
            xmin : xmax : interpolation * 1j, ymin : ymax : interpolation * 1j
        ]

        # interpolation
        bnew = []
        for band in bands:
            self.log.debug("Interpolating ...")
            bnew.append(griddata((x, y), band, (xnew, ynew), method="cubic"))

        linewidths = 0.7
        if noarrow:
            # print "second no arrow\n"
            linewidths = 0.2
        cont = [
            plt.contour(
                xnew,
                ynew,
                z,
                [self.energy],
                linewidths=linewidths,
                colors="k",
                linestyles="solid",
            )
            for z in bnew
        ]
        plt.axis("equal")

        for (contour, spinX, spinY, spinZ) in zip(cont, sx, sy, sz):
            # The previous interp. yields the level curves, nothing more is
            # useful from there
            paths = contour.collections[0].get_paths()
            verts = [xx.vertices for xx in paths]
            points = np.concatenate(verts)
            self.log.debug("Fermi surf. points.shape: " + str(points.shape))

            newSx = griddata((x, y), spinX, (points[:, 0], points[:, 1]))
            newSy = griddata((x, y), spinY, (points[:, 0], points[:, 1]))
            newSz = griddata((x, y), spinZ, (points[:, 0], points[:, 1]))

            self.log.info("newSx.shape: " + str(newSx.shape))

            import matplotlib.colors as colors

            if noarrow is False:
                plt.quiver(
                    points[::6, 0],
                    points[::6, 1],
                    newSx[::6],
                    newSy[::6],
                    newSz[::6],
                    scale_units="xy",
                    angles="xy",
                    norm=colors.Normalize(-0.5, 0.5),
                )
            else:
                # a dictionary to select the right spin component
                spinDict = {1: newSx[::6], 2: newSy[::6], 3: newSz[::6]}
                plt.scatter(
                    points[::6, 0],
                    points[::6, 1],
                    c=spinDict[spin],
                    s=50,
                    edgecolor="none",
                    alpha=1.0,
                    marker=".",
                    cmap="seismic",
                    norm=colors.Normalize(-0.5, 0.5),
                )
        plt.colorbar()
        plt.axis("equal")
        font = {"size": 16}
        plt.rc("font", **font)

        self.log.debug("st: ...Done")
        return
