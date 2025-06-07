__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import logging
import os
import re
import sys
from typing import List

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from matplotlib import colors as mpcolors
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
from skimage import measure

from pyprocar.utils import ROOT, ConfigManager

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


def validate_band_indices(band_indices):
    """Validate the band indices"""
    if band_indices is None:
        return None
    elif all(isinstance(x, list) for x in band_indices):
        return band_indices
    elif all(isinstance(x, int) for x in band_indices):
        return [band_indices]
    else:
        raise ValueError(
            "Invalid band indices. Band indices must be a list of lists of integers or a list of integers. This represents selecting the bands for each spin.\n"
            "Example: \n [[0,1], [2,3]] means that the first band is selected for the first spin and the second band is selected for the second spin."
        )


class FermiSurface:
    """This object is used to help plot the 2d fermi surface

    Parameters
    ----------
    kpoints : np.ndarray
        Numpy array with kpoints Nx3.
    bands :  np.ndarray
        Numpy array with the bands. The Fermi energy is already substracted.
        (n_kpoints, n_bands)
    spd : np.ndarray
        Numpy array with the projections. Expected size (n_kppints,n_bands,n_spins,n_orbitals,n_atoms)
    cmap : str
        The cmap to use. default = 'jet
    band_indices : List[List]
        A list of list that contains band indices for a given spin
    band_colors : List[List]
        A list of list that contains colors for the band index
        corresponding the band_indices for a given spin
    loglevel : _type_, optional
        The verbosity level., by default logging.WARNING
    """

    def __init__(
        self,
        kpoints,
        bands,
        spd,
        band_indices: List[List] = None,
        band_colors: List[List] = None,
        **kwargs,
    ):

        # Since some time ago Kpoints are in cartesian coords (ready to use)
        self.kpoints = kpoints
        self.bands = bands
        self.spd = spd
        self.band_indices = band_indices
        self.band_colors = band_colors

        self.useful = None  # List of useful bands (filled in findEnergy)
        self.energy = None

        logger.debug("FermiSurface.init: ...")
        logger.info("Kpoints.shape : " + str(self.kpoints.shape))
        logger.info("bands.shape   : " + str(self.bands.shape))
        logger.info("spd.shape     : " + str(self.spd.shape))
        logger.debug("FermiSurface.init: ...Done")

        self.band_indices = validate_band_indices(band_indices)

        config_manager = ConfigManager(
            os.path.join(ROOT, "pyprocar", "cfg", "fermi_surface_2d.yml")
        )

        config_manager.update_config(kwargs)
        self.config = config_manager.get_config()
        return None

    def find_energy(self, energy):
        """A method to find bands which are near a given energy

        Parameters
        ----------
        energy : float
            The energy to search for bands around.

        Returns
        -------
        None
            None

        Raises
        ------
        RuntimeError
            If no bands are found, raise an error.
        """
        self.energy = energy
        logger.info("Energy   : " + str(energy))

        # searching for bands crossing the desired energy
        bands_to_plot = False
        self.useful_bands_by_spins = []
        for i_spin in range(self.bands.shape[2]):
            bands = self.bands[:, :, i_spin]

            indices = np.where(
                np.logical_and(bands.min(axis=0) < energy, bands.max(axis=0) > energy)
            )[0]
            self.useful_bands_by_spins.append(indices)

            if len(indices) != 0:
                bands_to_plot = True
                print(
                    f"Band indices near iso-surface: (bands.shape={bands.shape}) spin-{i_spin} | bands-{indices}"
                )
        if not bands_to_plot:
            user_logger.error(
                f"Could not find any bands crossing the energy ({energy} eV) relative to the fermi energy.\n"
                "Please check the energy and the bands:\n\n"
                "1. Try shifting the energy to find crossings of the bands and the energy.\n"
                "2. Check the density of states to see where the bands are in terms of energy.\n"
                "3. Check the bands to see if they are crossing the energy."
            )
            raise RuntimeError("No bands to plot")
        return None

    def plot(self, mode: str, interpolation=500):
        """This method plots the 2d fermi surface along the z axis

        Only 2D layer geometry along z

        Parameters
        ----------
        mode : str, optional
            This parameters sets the mode,
        interpolation : int, optional
            The interpolation level, by default 500

        Returns
        -------
        List[plots]
            Returns a list of matplotlib.pyplot instances

        Raises
        ------
        RuntimeError
            Raise error if find energy was not called before plotting.
        """
        logger.debug("Plot: ...")
        from scipy.interpolate import griddata

        if self.useful_bands_by_spins is None:
            raise RuntimeError("self.find_energy() must be called before Plotting")

        # selecting components of K-points
        x, y = self.kpoints[:, 0], self.kpoints[:, 1]
        logger.debug("k_x[:10], k_y[:10] values" + str([x[:10], y[:10]]))

        # and new, interpolated component
        xmax, xmin = x.max(), x.min()
        ymax, ymin = y.max(), y.min()
        logger.debug("xlim = " + str([xmin, xmax]) + "  ylim = " + str([ymin, ymax]))
        xnew, ynew = np.mgrid[
            xmin : xmax : interpolation * 1j, ymin : ymax : interpolation * 1j
        ]

        unique_x = xnew[:, 0]
        unique_y = ynew[0, :]

        # interpolation
        n_spins = self.bands.shape[2]
        for i_spin in range(n_spins):
            # transpose so bands axis is first
            if self.band_indices is None:
                bands = self.bands[
                    :, self.useful_bands_by_spins[i_spin], i_spin
                ].transpose()
                spd = self.spd[
                    :, self.useful_bands_by_spins[i_spin], i_spin
                ].transpose()
                band_labels = np.unique(self.useful_bands_by_spins[i_spin])
            else:
                bands = self.bands[:, self.band_indices[i_spin], i_spin].transpose()
                spd = self.spd[:, self.band_indices[i_spin], i_spin].transpose()
                band_labels = np.unique(self.band_indices[i_spin])

            if spd.shape[0] == 0:
                continue

            # Normalizing
            vmin = self.config["clim"]["value"][0]
            vmax = self.config["clim"]["value"][1]
            if vmin is None:
                vmin = spd.min()
            if vmax is None:
                vmax = spd.max()
            norm = mpcolors.Normalize(vmin, vmax)

            # Interpolating band energies on to new grid
            bnew = []
            logger.debug("Interpolating ...")
            for i_band, band in enumerate(bands):

                bnew.append(griddata((x, y), band, (xnew, ynew), method="cubic"))
            bnew = np.array(bnew)

            # Generates colors per band
            n_bands = bands.shape[0]
            cmap = cm.get_cmap(self.config["cmap"]["value"])
            if i_spin == 1:
                factor = 0.25
            else:
                factor = 0
            solid_color_surface = np.arange(n_bands) / n_bands + factor
            band_colors = np.array(
                [cmap(norm(x)) for x in solid_color_surface[:]]
            ).reshape(-1, 4)
            plots = []

            for i_band, band_energies in enumerate(bnew):
                contours = measure.find_contours(band_energies, self.energy)
                for i_contour, contour in enumerate(contours):
                    # measure.find contours returns a list of coordinates indcies of the mesh.
                    # However, due to the algorithm they take values that are in between mesh points.
                    # We need to interpolate the values to the original kmesh
                    x_vals = contour[:, 0]
                    y_vals = contour[:, 1]
                    x_interp = np.interp(
                        x_vals, np.arange(0, unique_x.shape[0]), unique_x
                    )
                    y_interp = np.interp(
                        y_vals, np.arange(0, unique_y.shape[0]), unique_y
                    )
                    points = np.array([[x_interp, y_interp]])
                    points = np.moveaxis(points, -1, 0)

                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    if mode == "plain":
                        lc = LineCollection(
                            segments,
                            colors=self.config["color"]["value"][i_spin],
                            linestyle=self.config["linestyle"]["value"][i_spin],
                        )
                        lc.set_linestyle(self.config["linestyle"]["value"][i_spin])
                    if mode == "plain_bands":
                        if self.band_colors is None:
                            c = band_colors[i_band]
                        else:
                            c = self.band_colors[i_spin][i_band]
                        lc = LineCollection(segments, colors=c)
                        # lc.set_array(c)
                        if i_contour == 0:
                            if n_spins == 2:
                                label = f"Band {band_labels[i_band]}- Spin - {i_spin}"
                            else:
                                label = f"Band {band_labels[i_band]}"
                            lc.set_label(label)
                    if mode == "parametric":
                        c = griddata(
                            (x, y), spd[i_band, :], (x_interp, y_interp), method="cubic"
                        )
                        lc = LineCollection(
                            segments,
                            cmap=plt.get_cmap(self.config["cmap"]["value"]),
                            norm=norm,
                        )
                        lc.set_array(c)

                    plt.gca().add_collection(lc)

        if mode == "parametric":
            cbar = plt.colorbar(lc, ax=plt.gca())
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_ylabel("Atomic Orbital Projections", labelpad=10, rotation=270)

        plt.axis("equal")
        logger.debug("Plot: ...Done")
        # return plots

    def spin_texture(self, sx, sy, sz, spin=None, interpolation=300):
        """This method plots spin texture of the 2d fermi surface

        Only 2D layer geometry along z. It is like a enhanced version of 'plot' method.

        sx, sy, sz are spin projected Nkpoints x Nbands numpy arrays. They
        also are (already) projected by orbital and atom (from other
        class)

        Parameters
        ----------
        sx : np.ndarray
            Spin projected array for the x component. size (n_kpoints,n_bands)
        sy : np.ndarray
            Spin projected array for the y component. size (n_kpoints,n_bands)
        sz : np.ndarray
            Spin projected array for the z component. size (n_kpoints,n_bands)
        spin : List or array-like, optional
            List of marker colors for the arrows, by default None

        interpolation : int, optional
            The interpolation level, by default 300

        Raises
        ------
        RuntimeError
            Raise error if find energy was not called before plotting.
        """

        logger.debug("spin_texture: ...")

        if self.useful_bands_by_spins is None:
            raise RuntimeError("self.find_energy() must be called before plotting")

        # selecting components of K-points
        x, y = self.kpoints[:, 0], self.kpoints[:, 1]

        if self.band_indices is None:
            bands = self.bands[:, self.useful_bands_by_spins[0], 0].transpose()
            band_labels = np.unique(self.useful_bands_by_spins[0])
            sx = sx[:, self.useful_bands_by_spins[0]].transpose()
            sy = sy[:, self.useful_bands_by_spins[0]].transpose()
            sz = sz[:, self.useful_bands_by_spins[0]].transpose()
        else:
            bands = self.bands[:, self.band_indices[0], 0].transpose()

            band_labels = np.unique(self.band_indices[0])
            sx = sx[:, self.band_indices[0]].transpose()
            sy = sy[:, self.band_indices[0]].transpose()
            sz = sz[:, self.band_indices[0]].transpose()

        # and new, interpolated component
        xmax, xmin = x.max(), x.min()
        ymax, ymin = y.max(), y.min()

        logger.debug("xlim = " + str([xmin, xmax]) + "  ylim = " + str([ymin, ymax]))

        xnew, ynew = np.mgrid[
            xmin : xmax : interpolation * 1j, ymin : ymax : interpolation * 1j
        ]

        # interpolation
        bnew = []
        for band in bands:
            logger.debug("Interpolating ...")
            interp_bands = griddata((x, y), band, (xnew, ynew), method="cubic")
            bnew.append(interp_bands)

        # Normalizing
        vmin = self.config["clim"]["value"][0]
        vmax = self.config["clim"]["value"][1]
        if vmin is None:
            vmin = -0.5
        if vmax is None:
            vmax = 0.5
        norm = mpcolors.Normalize(vmin, vmax)

        cont = [
            plt.contour(
                xnew,
                ynew,
                z,
                [self.energy],
                linewidths=self.config["linewidth"]["value"],
                colors="k",
                linestyles="solid",
                alpha=self.config["contour_alpha"]["value"],
            )
            for z in bnew
        ]

        if len(cont) == 0:
            raise RuntimeError("Could not find any contours at this energy")

        x_limits = [0, 0]
        y_limits = [0, 0]
        for i_band, (contour, spinX, spinY, spinZ) in enumerate(zip(cont, sx, sy, sz)):
            # The previous interp. yields the level curves, nothing more is
            # useful from there
            paths = contour.get_paths()
            if paths:
                verts = [path.vertices for path in paths]
                points = np.concatenate(verts)
                x_limits = [
                    min(x_limits[0], points[:, 0].min()),
                    max(x_limits[1], points[:, 0].max()),
                ]
                y_limits = [
                    min(y_limits[0], points[:, 1].min()),
                    max(y_limits[1], points[:, 1].max()),
                ]

                logger.debug("Fermi surf. points.shape: " + str(points.shape))
                newSx = griddata((x, y), spinX, (points[:, 0], points[:, 1]))
                newSy = griddata((x, y), spinY, (points[:, 0], points[:, 1]))
                newSz = griddata((x, y), spinZ, (points[:, 0], points[:, 1]))
                logger.info("newSx.shape: " + str(newSx.shape))
                if self.config["arrow_size"]["value"] is not None:
                    # This is so the density scales the way you think. increasing number means increasing density.
                    # The number in the numerator is so it scales reasonable with 0-20
                    scale = 10 / self.config["arrow_size"]["value"]
                    scale_units = "xy"
                    angles = "xy"
                else:
                    scale = None
                    scale_units = "xy"
                    angles = "xy"

                # This is so the density scales the way you think. increasing number means increasing density.
                # The number in the numerator is so it scales reasonable with 0-20
                arrow_density = 50 // self.config["arrow_density"]["value"]
                if self.config["spin_projection"]["value"] == "z":
                    color = newSz[::arrow_density]
                elif self.config["spin_projection"]["value"] == "y":
                    color = newSy[::arrow_density]
                elif self.config["spin_projection"]["value"] == "x":
                    color = newSx[::arrow_density]
                elif self.config["spin_projection"]["value"] == "x^2":
                    color = newSx[::arrow_density] ** 2
                elif self.config["spin_projection"]["value"] == "y^2":
                    color = newSy[::arrow_density] ** 2
                elif self.config["spin_projection"]["value"] == "z^2":
                    color = newSz[::arrow_density] ** 2

                if self.config["no_arrow"]["value"]:
                    # a dictionary to select the right spin component
                    # spinDict = {0: newSx[::self.config['arrow_density']['value']],
                    #             1: newSy[::self.config['arrow_density']['value']],
                    #             2: newSz[::self.config['arrow_density']['value']]}

                    plt.scatter(
                        points[::arrow_density, 0],
                        points[::arrow_density, 1],
                        c=color,
                        # spinDict[spin],
                        s=50,
                        edgecolor="none",
                        alpha=1.0,
                        marker=self.config["marker"]["value"],
                        cmap=self.config["cmap"]["value"],
                        norm=norm,
                    )

                else:
                    if (
                        self.config["arrow_color"]["value"] is not None
                        or self.band_colors is not None
                    ):
                        if self.band_colors is not None:
                            c = self.band_colors[0][i_band]
                        if self.config["arrow_color"]["value"] is not None:
                            c = self.config["arrow_color"]["value"]
                        plt.quiver(
                            points[::arrow_density, 0],  # Arrow position x-component
                            points[::arrow_density, 1],  # Arrow position y-component
                            newSx[::arrow_density],  # Arrow direction x-component
                            newSy[::arrow_density],  # Arrow direction y-component
                            scale=scale,
                            scale_units=scale_units,
                            angles=angles,
                            color=c,
                        )
                    else:

                        plt.quiver(
                            points[::arrow_density, 0],  # Arrow position x-component
                            points[::arrow_density, 1],  # Arrow position y-component
                            newSx[::arrow_density],  # Arrow direction x-component
                            newSy[::arrow_density],  # Arrow direction y-component
                            color,  # Color for each arrow
                            scale=scale,
                            scale_units=scale_units,
                            angles=angles,
                            cmap=self.config["cmap"]["value"],
                            norm=norm,
                        )

        if self.config["plot_color_bar"]["value"]:
            cbar = plt.colorbar()
            if len(self.config["spin_projection"]["value"].split("^")) == 2:
                tmp = self.config["spin_projection"]["value"].split("^")
                label = f"S$_{tmp[0]}^{tmp[1]}$ projection"
            else:
                tmp = self.config["spin_projection"]["value"].split("^")
                label = f"S$_{tmp[0]}$ projection"
            cbar.ax.set_ylabel(label, rotation=270)

        xlimits = (
            x_limits[0] - abs(x_limits[0]) * 0.1,
            x_limits[1] + abs(x_limits[1]) * 0.1,
        )
        ylimits = (
            y_limits[0] - abs(y_limits[0]) * 0.1,
            y_limits[1] + abs(y_limits[1]) * 0.1,
        )
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        font = {"size": 16}
        plt.rc("font", **font)

        logger.debug("st: ...Done")
        return None

    def add_axes_labels(self):
        """
        Method to add labels to matplotlib plot
        """
        if self.config["add_axes_labels"]["value"]:
            plt.ylabel(self.config["y_label"]["value"])
            plt.xlabel(self.config["x_label"]["value"])

    def add_legend(self):
        """
        Method to add labels to matplotlib plot
        """
        if self.config["add_legend"]["value"]:
            plt.legend()

    def savefig(self, savefig):
        """
        Method to save plot
        """
        plt.savefig(savefig, dpi=self.config["dpi"]["value"], bbox_inches="tight")
        plt.close()

    def show(self):
        """
        Method show th plot
        """
        plt.show()
