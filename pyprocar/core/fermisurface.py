__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import sys
import re
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
from skimage import measure
from matplotlib import colors as mpcolors
from matplotlib import cm
from matplotlib.collections import LineCollection

from ..utils.defaults import settings
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
        loglevel : _type_, optional
            The verbosity level., by default logging.WARNING
    """
    def __init__(self, kpoints, bands, spd, cmap='jet', loglevel=logging.WARNING):
        
        
        # Since some time ago Kpoints are in cartesian coords (ready to use)
        self.kpoints = kpoints
        self.bands = bands
        self.spd = spd
        self.cmap = cmap
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

        self.log.debug("FindEnergy: ...")
        self.energy = energy
        self.log.info("Energy   : " + str(energy))

        # searching for bands crossing the desired energy
        self.useful_bands_by_spins = []
        for i_spin in range(self.bands.shape[2]):
            indices = np.where( np.logical_and(self.bands[:,:,i_spin].min(axis=0) < energy, self.bands[:,:,i_spin].max(axis=0) > energy))
            self.useful_bands_by_spins.append(indices[0])
        self.log.info("set of useful bands    : " + str(self.useful_bands_by_spins))
        
        if len(self.useful_bands_by_spins) == 1:
            bands = self.bands[:,self.useful_bands_by_spins[0],:]
            self.log.debug("new bands.shape : " + str(bands.shape))
            if len(bands) == 0:
                self.log.error("No bands found in that range. Check your data. Returning")
                raise RuntimeError("No bands to plot")
        else:
            bands_up = self.bands[:,self.useful_bands_by_spins[0],0]
            bands_down= self.bands[:,self.useful_bands_by_spins[1],1]
            self.log.debug("new bands_up.shape : " + str(bands_up.shape))
            self.log.debug("new bands_down.shape : " + str(bands_down.shape))
            if len(bands_up) == 0:
                self.log.error("No bands found in that range for spin up. Check your data. Returning")
                raise RuntimeError("No bands to plot")
            if len(bands_down) == 0:
                self.log.error("No bands found in that range for spin down. Check your data. Returning")
                raise RuntimeError("No bands to plot")

        self.log.debug("FindEnergy: ...Done")
        return None

    def plot(self,mode:str, interpolation=500):
        """ This method plots the 2d fermi surface along the z axis
        
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
        self.log.debug("Plot: ...")
        from scipy.interpolate import griddata

        if self.useful_bands_by_spins is None:
            raise RuntimeError("self.find_energy() must be called before Plotting")

        
        # selecting components of K-points
        x, y = self.kpoints[:, 0], self.kpoints[:, 1]
        self.log.debug("k_x[:10], k_y[:10] values" + str([x[:10], y[:10]]))

        # and new, interpolated component
        xmax, xmin = x.max(), x.min()
        ymax, ymin = y.max(), y.min()
        self.log.debug("xlim = " + str([xmin, xmax]) + "  ylim = " + str([ymin, ymax]))
        xnew, ynew = np.mgrid[
            xmin : xmax : interpolation * 1j, ymin : ymax : interpolation * 1j
        ]

        # interpolation
        n_spins = self.bands.shape[2]
        for i_spin in range(n_spins):
            # transpose so bands axis is first
            bands = self.bands[:,self.useful_bands_by_spins[i_spin],i_spin].transpose()
            spd = self.spd[:,self.useful_bands_by_spins[i_spin],i_spin].transpose()
            band_labels = np.unique(self.useful_bands_by_spins[i_spin])
            vmin = spd.min()
        
            vmax = spd.max()
            # print("normalizing to : ", (vmin, vmax))
            norm = mpcolors.Normalize(vmin, vmax)
    
            # Interpolating band energies on to new grid
            bnew = []
            for i_band, band in enumerate(bands):
                self.log.debug("Interpolating ...")
                bnew.append(griddata((x, y), band, (xnew, ynew), method="cubic"))
            bnew = np.array(bnew)

            # Generates colors per band
            
            n_bands = bands.shape[0]
            norm = mpcolors.Normalize(vmin=0, vmax=1)
            cmap = cm.get_cmap(self.cmap)
            solid_color_surface = np.arange(n_bands) / n_bands
            band_colors = np.array([cmap(norm(x)) for x in solid_color_surface[:]]).reshape(-1, 4)
            plots = []
            for i_band,band_energies in enumerate(bnew):
                contours = measure.find_contours(band_energies, self.energy)
                for i_contour,contour in enumerate(contours):
                    points = np.array([contour[:, 0], contour[:, 1]]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    if mode=='plain':
                        lc = LineCollection(segments, colors=settings.ebs.color[i_spin], linestyle=settings.ebs.linestyle[i_spin])
                    if mode=='plain_bands':
                        c = band_colors[i_band]
                        lc = LineCollection(segments, colors=c, linestyle=settings.ebs.linestyle[i_spin])
                        lc.set_array(c)
                        if i_contour == 0:
                            if n_spins == 2:
                                label=f'Band {band_labels[i_band]}- Spin - {i_spin}'
                            else:
                                label=f'Band {band_labels[i_band]}'
                            lc.set_label(label)
                    if mode=='parametric':
                        c = griddata((x, y), spd[i_band,:], (contour[:, 0], contour[:, 1]), method="nearest")
                        lc = LineCollection(segments, cmap=plt.get_cmap(self.cmap), norm=norm)
                        lc.set_array(c)

                    plt.gca().add_collection(lc)

        if mode == 'parametric':
            cbar = plt.colorbar(lc, ax=plt.gca())
            cbar.ax.tick_params(labelsize=10)
            cbar.ax.set_ylabel('Atomic Orbital Projections', labelpad=10, rotation=270)

        plt.axis("equal")
        self.log.debug("Plot: ...Done")
        return plots

    def spin_texture(self, sx, sy, sz, 
                    arrow_projection:str='sz',
                    spin=None, 
                    no_arrow=False,
                    interpolation=300, 
                    arrow_color=None,
                    arrow_size=0.05,
                    arrow_density=6,
                    color_bar:bool=False):
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
        no_arrow : bool, optional
            Determines whether to plot arrows, by default False
        interpolation : int, optional
            The interpolation level, by default 300

        Raises
        ------
        RuntimeError
            Raise error if find energy was not called before plotting.
        """

        self.log.debug("spin_texture: ...")
    
        if self.useful_bands_by_spins is None:
            raise RuntimeError("self.find_energy() must be called before plotting")

        # selecting components of K-points
        x, y = self.kpoints[:, 0], self.kpoints[:, 1]


        bands = self.bands[:,self.useful_bands_by_spins[0],0].transpose()

        band_labels = np.unique(self.useful_bands_by_spins[0])
        sx = sx[:,self.useful_bands_by_spins[0]].transpose()
        sy = sy[:,self.useful_bands_by_spins[0]].transpose()
        sz = sz[:,self.useful_bands_by_spins[0]].transpose()

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
        if no_arrow:
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
            if arrow_size is not None:
                scale  = arrow_size
                scale_units = "xy"
                angles="xy"
            else:
                scale=None
                scale_units = "xy"
                angles="xy"

            if no_arrow:
                # a dictionary to select the right spin component
                spinDict = {0: newSx[::arrow_density], 1: newSy[::arrow_density], 2: newSz[::arrow_density]}
                plt.scatter(
                    points[::arrow_density, 0],
                    points[::arrow_density, 1],
                    c=spinDict[spin],
                    s=50,
                    edgecolor="none",
                    alpha=1.0,
                    marker=".",
                    cmap="seismic",
                    norm=colors.Normalize(-0.5, 0.5),
                )

            else:
                if arrow_color is not None:
                    plt.quiver(
                        points[::arrow_density, 0],  # Arrow position x-component
                        points[::arrow_density, 1],  # Arrow position y-component
                        newSx[::arrow_density],      # Arrow direction x-component
                        newSy[::arrow_density],      # Arrow direction y-component
                        scale=scale,
                        scale_units=scale_units,
                        angles=angles,
                        color=arrow_color
                    )
                else:
                    if arrow_projection == 'z':
                        color = newSz[::arrow_density]
                    elif arrow_projection == 'y':
                        color = newSy[::arrow_density]
                    elif arrow_projection == 'x':
                        color = newSx[::arrow_density]

                    elif arrow_projection == 'x^2':
                        color = newSx[::arrow_density]**2
                    elif arrow_projection == 'y^2':
                        color = newSy[::arrow_density]**2
                    elif arrow_projection == 'z^2':
                        color = newSz[::arrow_density]**2

                    
                    plt.quiver(
                        points[::arrow_density, 0],  # Arrow position x-component
                        points[::arrow_density, 1],  # Arrow position y-component
                        newSx[::arrow_density],      # Arrow direction x-component
                        newSy[::arrow_density],      # Arrow direction y-component
                        color,                           # Color for each arrow
                        scale=scale,
                        scale_units=scale_units,
                        angles=angles,
                        norm=colors.Normalize(-0.5, 0.5),
                    )
                
            
        if color_bar:
            cbar = plt.colorbar()
            if len(arrow_projection.split('^')) == 2:
                tmp = arrow_projection.split('^')
                label = f'S$_{tmp[0]}^{tmp[1]}$ projection'
            else:
                tmp = arrow_projection.split('^')
                label = f'S$_{tmp[0]}$ projection'
            cbar.ax.set_ylabel(label, rotation=270)
        plt.axis("equal")
        font = {"size": 16}
        plt.rc("font", **font)

        self.log.debug("st: ...Done")
        return None

    def add_axes_labels(self):
        """
        Method to add labels to matplotlib plot
        """

        plt.ylabel('$k_{y}$  ($\AA^{-1}$)')
        plt.xlabel('$k_{z}$  ($\AA^{-1}$)')

    def add_legend(self):
        """
        Method to add labels to matplotlib plot
        """
        plt.legend()