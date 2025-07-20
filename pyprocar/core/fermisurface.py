__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import logging
import os
import re
import sys
from enum import Enum
from typing import List

import matplotlib as mpl
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
    elif all(isinstance(x, tuple) for x in band_indices) or isinstance(band_indices, list):
        return band_indices
    elif all(isinstance(x, int) for x in band_indices):
        return [band_indices]
    else:
        raise ValueError(
            f"Invalid band indices: {band_indices}. Band indices must be a list of lists of integers or a list of integers. This represents selecting the bands for each spin.\n"
            "Example: \n [[0,1], [2,3]] means that the first band is selected for the first spin and the second band is selected for the second spin."
        )

class SpinProjection(Enum):
    """An enumeration for defining the spin projection"""
    SZ = "S$_z$"
    SY = "S$_y$"
    SX = "S$_x$"
    SX2 = "S$_x^2$"
    SY2 = "S$_y^2$"
    SZ2 = "S$_z^2$"
    
    @classmethod
    def from_str(cls, spin_projection: str):
        spin_projection = spin_projection.lower()
        if "z" in spin_projection and "2" in spin_projection:
            return cls.SZ2
        elif "y" in spin_projection and "2" in spin_projection:
            return cls.SY2
        elif "x" in spin_projection and "2" in spin_projection:
            return cls.SX2
        elif "z" in spin_projection:
            return cls.SZ
        elif "y" in spin_projection:
            return cls.SY
        elif "x" in spin_projection:
            return cls.SX
        else:
            raise ValueError(f"Invalid spin projection: {spin_projection}")
    

class FermiSurface:
    """A class for plotting and analyzing 2D Fermi surfaces.

    This class provides comprehensive functionality for visualizing 2D Fermi surfaces
    from electronic band structure calculations. It supports plotting Fermi surface
    contours, spin texture analysis, and various customization options for scientific
    visualization. The class handles interpolation of k-point data, contour generation,
    and multiple plotting modes including line segments, scatter plots, and vector fields.

    Parameters
    ----------
    kpoints : np.ndarray
        Array of k-points in Cartesian coordinates with shape (n_kpoints, 3).
        These should be the k-points from the electronic structure calculation.
    bands : np.ndarray
        Array of band energies with shape (n_kpoints, n_bands, n_spins).
        The Fermi energy should already be subtracted from these values.
    spd : np.ndarray
        Array of spin-projected density with shape (n_kpoints, n_bands, n_spins, n_orbitals, n_atoms).
        Contains the orbital and atomic projections for each k-point and band.
    figsize : tuple of float, optional
        Figure size as (width, height) in inches, by default (6, 6).
    ax : matplotlib.axes.Axes or None, optional
        Matplotlib axes object to plot on. If None, a new figure and axes
        will be created, by default None.
    **kwargs
        Additional keyword arguments passed to matplotlib functions.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    handles : list
        List of matplotlib handles for plotted elements.
    kpoints : np.ndarray
        The k-points array.
    bands : np.ndarray
        The band energies array.
    spd : np.ndarray
        The spin-projected density array.
    energy : float or None
        The energy level for Fermi surface analysis.
    useful_bands_by_spins : list or None
        List of band indices that cross the specified energy for each spin.
    x_limits : tuple
        The x-axis limits for plotting.
    y_limits : tuple
        The y-axis limits for plotting.
    clim : tuple
        The color limits for color mapping.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Create sample data
    >>> kpoints = np.random.rand(100, 3)
    >>> bands = np.random.rand(100, 10, 1) - 0.5  # Centered around 0
    >>> spd = np.random.rand(100, 10, 1, 4, 1)
    >>> 
    >>> # Initialize FermiSurface
    >>> fs = FermiSurface(kpoints, bands, spd)
    >>> 
    >>> # Find bands crossing the Fermi level
    >>> fs.find_energy(0.0)
    >>> 
    >>> # Generate and plot contours
    >>> contour_data = fs.generate_contours()
    >>> fs.plot_band_spin_contour_line_segments(contour_data)
    >>> 
    >>> # Customize the plot
    >>> fs.set_xlabel('$k_x$ (Å$^{-1}$)')
    >>> fs.set_ylabel('$k_y$ (Å$^{-1}$)')
    >>> fs.show()

    Notes
    -----
    The class assumes that the input band energies have the Fermi energy already
    subtracted, so that the Fermi level corresponds to energy = 0. The k-points
    should be in Cartesian coordinates and ready for plotting.

    For spin texture analysis, additional spin arrays (sx, sy, sz) are required
    and should be passed to the appropriate spin texture methods.
    """

    def __init__(
        self,
        kpoints,
        bands,
        spd,
        figsize: tuple = (6, 6),
        ax: plt.Axes | None = None,
        **kwargs,
    ):
        """Initialize the FermiSurface object.

        Parameters
        ----------
        kpoints : np.ndarray
            Array of k-points in Cartesian coordinates with shape (n_kpoints, 3).
        bands : np.ndarray
            Array of band energies with shape (n_kpoints, n_bands, n_spins).
        spd : np.ndarray
            Array of spin-projected density with shape (n_kpoints, n_bands, n_spins, n_orbitals, n_atoms).
        figsize : tuple of float, optional
            Figure size as (width, height) in inches, by default (6, 6).
        ax : matplotlib.axes.Axes or None, optional
            Matplotlib axes object to plot on, by default None.
        **kwargs
            Additional keyword arguments.
        """
        if ax is None:
            self.fig = plt.figure(
                figsize=figsize,
            )
            self.ax = self.fig.add_subplot(111)
        else:
            self.fig = plt.gcf()
            self.ax = ax
            
        self.handles = []
        # Since some time ago Kpoints are in cartesian coords (ready to use)
        self.kpoints = kpoints
        self.bands = bands
        self.spd = spd

        self.useful = None  # List of useful bands (filled in findEnergy)
        self.energy = None

        logger.debug("FermiSurface.init: ...")
        logger.info("Kpoints.shape : " + str(self.kpoints.shape))
        logger.info("bands.shape   : " + str(self.bands.shape))
        logger.info("spd.shape     : " + str(self.spd.shape))
        logger.debug("FermiSurface.init: ...Done")
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
    
    
    def select_bands(self, band_indices:tuple[tuple[int, int], tuple[int, int]] = None):
        """Select specific bands for plotting based on band indices.

        Parameters
        ----------
        band_indices : tuple of tuples of int, optional
            Band indices for each spin channel. Format: ((band_indices_spin0,), (band_indices_spin1,)).
            If None, uses all available bands, by default None.

        Returns
        -------
        dict
            Dictionary containing selected bands data with keys:
            - 'bands': List of band arrays for each spin
            - 'spd': List of spin-projected density arrays for each spin  
            - 'band_labels': List of unique band labels for each spin
        """
        bands_data= {
            "bands": [],
            "spd": [],
            "band_labels": [],
        }
        
        band_indices = validate_band_indices(band_indices)
        
        n_spins = self.bands.shape[2]
        for i_spin in range(n_spins):
            if i_spin==1 and len(band_indices) == 1:
                continue
                
            if len(band_indices[i_spin]) == 0:
                continue
            bands = self.bands[:, band_indices[i_spin], i_spin].transpose()
            spd = self.spd[:, band_indices[i_spin], i_spin].transpose()
            band_labels = np.unique(band_indices[i_spin])
            
            if spd.shape[0] == 0:
                continue
            
            
            
            bands_data["bands"].append(bands)
            bands_data["spd"].append(spd)
            bands_data["band_labels"].append(band_labels)
        
        return bands_data
    
    def generate_band_colors(self, bands, i_spin:int = 0, cmap:str = "plasma"):
        """Generate colors for each band using a colormap.

        Parameters
        ----------
        bands : np.ndarray
            Array of band energies with shape (n_bands, n_kpoints).
        i_spin : int, optional
            Spin index (0 or 1), by default 0.
        cmap : str, optional
            Colormap name, by default "plasma".

        Returns
        -------
        np.ndarray
            Array of RGBA colors with shape (n_bands, 4).
        """
        n_bands = bands.shape[0]
        cmap = cm.get_cmap(cmap)
        norm = mpcolors.Normalize(0, 1)
        factor = 0.25 if i_spin == 1 else 0
        
        solid_color_surface = np.arange(n_bands) / n_bands + factor
        band_colors = np.array(
            [cmap(norm(x)) for x in solid_color_surface[:]]
        ).reshape(-1, 4)
        return band_colors

    def generate_contours(self, band_indices: list[list[int]] = None, interpolation=500, ignore_scalars: bool = False):
        """
        Generate 2D Fermi surface contours for selected bands.

        This method interpolates the band energies onto a regular grid in the
        kx-ky plane and extracts contour lines at the specified energy level
        (typically the Fermi energy). It supports selection of specific bands
        and spin channels, and can optionally ignore scalar values associated
        with the bands.

        Parameters
        ----------
        band_indices : list of list of int, optional
            List of band indices to include for each spin channel. If None,
            uses the bands identified by `find_energy()`.
        interpolation : int, optional
            Number of grid points along each axis for interpolation, by default 500.
        ignore_scalars : bool, optional
            If True, scalar values (e.g., projections) are ignored in the output,
            by default False.

        Returns
        -------
        dict
            Dictionary containing interpolated band energies, spin-projected
            densities, and band labels for each spin channel.

        Raises
        ------
        RuntimeError
            If `find_energy()` has not been called prior to this method.
        """
        logger.debug("Plot: ...")
        

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
        
        self.x_limits = (xmin, xmax)
        self.y_limits = (ymin, ymax)

        # Selecting bands to plot
        bands_indices = band_indices if band_indices is not None else self.useful_bands_by_spins
        bands_data = self.select_bands(bands_indices)

        
        bands_spin_contour_data = {}
        self.selected_bands = bands_data
        for i_spin, (bands, spd, band_labels) in enumerate(zip(bands_data["bands"], bands_data["spd"], bands_data["band_labels"])):

            # Interpolating band energies on to new grid
            bnew = []
            logger.debug("Interpolating ...")
            for i_band, band in enumerate(bands):
                bnew.append(griddata((x, y), band, (xnew, ynew), method="cubic"))
            bnew = np.array(bnew)

            # Generates colors per band
            plots = []

            for i_band, band_energies in enumerate(bnew):
                contour_data = {"lines": [],"scalars": [], "labels": []}
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
                    contour_data["lines"].append(segments)
                    
                    if spd is not None and not ignore_scalars:
                        scalars = griddata((x, y), spd[i_band, :], (x_interp, y_interp), method="cubic")
                        contour_data["scalars"].append(scalars)

                    
                bands_spin_contour_data[(i_band, i_spin)] = contour_data
                
        return bands_spin_contour_data
        
    def plot_contour_line_segments(self, contour_data:dict,
                                   label:str = None,
                                   cmap:str = "plasma", 
                                   norm:mpcolors.Normalize = None,
                                   clim:tuple = (None, None),
                                   linewidth:float = 0.2,
                                   linestyle:str = "solid",
                                   color:str = "k",
                                   alpha:float = 1.0,
                                   line_collection_kwargs:dict = None,
                                   ):
        """Plot contour line segments from contour data.

        Parameters
        ----------
        contour_data : dict
            Dictionary containing contour lines and scalars data.
        label : str, optional
            Label for the plot legend, by default None.
        cmap : str, optional
            Colormap name, by default "plasma".
        norm : matplotlib.colors.Normalize, optional
            Normalization instance for color mapping, by default None.
        clim : tuple of float, optional
            Color limits as (vmin, vmax), by default (None, None).
        linewidth : float, optional
            Width of the contour lines, by default 0.2.
        linestyle : str, optional
            Style of the contour lines, by default "solid".
        color : str, optional
            Color of the lines when not using scalar coloring, by default "k".
        alpha : float, optional
            Transparency level (0-1), by default 1.0.
        line_collection_kwargs : dict, optional
            Additional kwargs for LineCollection, by default None.

        Returns
        -------
        list
            List of matplotlib LineCollection handles.
        """
        
        line_handles = []
        
        lines = contour_data["lines"]
        scalars = contour_data["scalars"]
        if len(scalars) > 0:
            vmin = clim[0]
            vmax = clim[1]
            if vmin is None and vmax is None:
                for i_segment, segment_scalars in enumerate(scalars):
                    vmin = min(vmin, segment_scalars.min())
                    vmax = max(vmax, segment_scalars.max())
                clim = (vmin, vmax)
                
        if not hasattr(self, "norm") or not hasattr(self, "clim") or not hasattr(self, "cmap"):
            self.set_scalar_mappable(norm=norm, clim=clim, cmap=cmap)
            

        for i_segment, segments in enumerate(lines):
            lc = LineCollection(segments,  linestyle=linestyle, linewidth=linewidth, alpha=alpha, **line_collection_kwargs)

            if len(scalars) > 0:
                lc.set_array(scalars[i_segment])
                lc.set_cmap(self.cmap)
                lc.set_norm(self.norm)
            else:
                lc.set_color(color)
                if i_segment == 0:
                    lc.set_label(label)
            
            line_handles.append(self.ax.add_collection(lc))
            
        
        self.handles.extend(line_handles)
        return line_handles
    
    def plot_band_spin_contour_line_segments(self, 
                            bands_spin_contour_data:dict, 
                                linestyles:tuple[str, str] = ("solid", "dashed"),
                                colors:tuple[str, str] = None,
                                linewidths:tuple[float, float] = (0.2, 0.2),
                                alphas:tuple[float, float] = (1.0, 1.0),
                                cmap:str = "plasma",
                                norm:mpcolors.Normalize = None,
                                clim:tuple = (None, None),
                                line_collection_kwargs:dict = None
                                   ):
        """Plot contour line segments for multiple bands and spins.

        Parameters
        ----------
        bands_spin_contour_data : dict
            Dictionary containing contour data for each (band, spin) combination.
        linestyles : tuple of str, optional
            Line styles for each spin channel, by default ("solid", "dashed").
        colors : tuple of str, optional
            Colors for each spin channel. If None, uses automatic coloring, by default None.
        linewidths : tuple of float, optional
            Line widths for each spin channel, by default (0.2, 0.2).
        alphas : tuple of float, optional
            Transparency levels for each spin channel, by default (1.0, 1.0).
        cmap : str, optional
            Colormap name, by default "plasma".
        norm : matplotlib.colors.Normalize, optional
            Normalization instance for color mapping, by default None.
        clim : tuple of float, optional
            Color limits as (vmin, vmax), by default (None, None).
        line_collection_kwargs : dict, optional
            Additional kwargs for LineCollection, by default None.

        Returns
        -------
        list
            List of matplotlib LineCollection handles.
        """
        plot_contour_line_segments_kwargs = {}
        
        if not hasattr(self, "norm") or not hasattr(self, "clim") or not hasattr(self, "cmap"):
            self.set_scalar_mappable(norm=norm, clim=clim, cmap=cmap)
        plot_contour_line_segments_kwargs["norm"] = self.norm
        plot_contour_line_segments_kwargs["clim"] = self.clim
        plot_contour_line_segments_kwargs["cmap"] = self.cmap
        plot_contour_line_segments_kwargs["line_collection_kwargs"] = line_collection_kwargs if line_collection_kwargs is not None else {}
        
        band_spin_handles = []
        for (i_band, i_spin), contour_data in bands_spin_contour_data.items():
            label = f"Band {i_band}, Spin {i_spin}"
            bands = self.selected_bands["bands"][i_spin]
            band_colors = self.generate_band_colors(bands, i_spin)
            line_kwargs = plot_contour_line_segments_kwargs.copy()
            line_kwargs["label"] = label
            line_kwargs["linestyle"] = linestyles[i_spin]
            line_kwargs["linewidth"] = linewidths[i_spin]
            line_kwargs["alpha"] = alphas[i_spin]
            if colors is not None:
                line_kwargs["color"] = colors[i_spin]
            else:
                line_kwargs["color"] = band_colors[i_band]
            
            
            handles = self.plot_contour_line_segments(contour_data, **line_kwargs)
            band_spin_handles.extend(handles)
        self.handles.extend(band_spin_handles)
        return band_spin_handles
        
    def generate_spin_texture_contours(self, sx, sy, sz, 
                                band_indices:tuple[int, int] | None = None,
                                point_density:int = 10,
                                spin_projection: SpinProjection | str = "z^2",
                                interpolation=300):
        """This method generates the spin texture contours"""
        logger.debug("spin_texture: ...")
        
        point_density = 50 // point_density
        
        if isinstance(spin_projection, str):
            spin_projection = SpinProjection.from_str(spin_projection)

        if self.useful_bands_by_spins is None:
            raise RuntimeError("self.find_energy() must be called before plotting")

        # selecting components of K-points
        x, y = self.kpoints[:, 0], self.kpoints[:, 1]
        
        
        if band_indices is None:
            bands = self.bands[:, self.useful_bands_by_spins[0], 0].transpose()
            band_labels = np.unique(self.useful_bands_by_spins[0])
            sx = sx[:, self.useful_bands_by_spins[0]].transpose()
            sy = sy[:, self.useful_bands_by_spins[0]].transpose()
            sz = sz[:, self.useful_bands_by_spins[0]].transpose()
        else:
            bands = self.bands[:, band_indices[0], 0].transpose()

            band_labels = np.unique(band_indices[0])
            sx = sx[:, band_indices[0]].transpose()
            sy = sy[:, band_indices[0]].transpose()
            sz = sz[:, band_indices[0]].transpose()

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
        
        
     
        spin_texture_contours = []
        
        # plt.ioff()  # Turn off interactive mode
 
        fig, ax = plt.subplots()
        for z in bnew:
            spin_texture_contours.append(ax.contour(xnew, ynew, z, [self.energy]))
        plt.close(fig)
        
        if len(spin_texture_contours) == 0:
            raise RuntimeError("Could not find any contours at this energy")
        
        
        self.x_limits = [0, 0]
        self.y_limits = [0, 0]
        spin_texture_contour_data = {
            "contours": [],
            "points": [],
            "sx": [],
            "sy": [],
            "sz": [],
            "scalars": [],
        }
        
        for i_band, (contour, spinX, spinY, spinZ) in enumerate(zip(spin_texture_contours, sx, sy, sz)):
            # The previous interp. yields the level curves, nothing more is
            # useful from there
            paths = contour.get_paths()
            if paths:
                verts = [path.vertices for path in paths]
                points = np.concatenate(verts)
                

                logger.debug("Fermi surf. points.shape: " + str(points.shape))
                newSx = griddata((x, y), spinX, (points[:, 0], points[:, 1]))
                newSy = griddata((x, y), spinY, (points[:, 0], points[:, 1]))
                newSz = griddata((x, y), spinZ, (points[:, 0], points[:, 1]))
                
                
                # This is so the density scales the way you think. increasing number means increasing density.
                # The number in the numerator is so it scales reasonable with 0-20
                
                if spin_projection == SpinProjection.SZ:
                    scalars = newSz[::point_density]
                elif spin_projection == SpinProjection.SY:
                    scalars = newSy[::point_density]
                elif spin_projection == SpinProjection.SX:
                    scalars = newSx[::point_density]
                elif spin_projection == SpinProjection.SX2:
                    scalars = newSx[::point_density] ** 2
                elif spin_projection == SpinProjection.SY2:
                    scalars = newSy[::point_density] ** 2
                elif spin_projection == SpinProjection.SZ2:
                    scalars = newSz[::point_density] ** 2
                    
                spin_texture_contour_data["sx"].append(newSx[::point_density])
                spin_texture_contour_data["sy"].append(newSy[::point_density])
                spin_texture_contour_data["sz"].append(newSz[::point_density])
                spin_texture_contour_data["points"].append(points[::point_density])
                spin_texture_contour_data["contours"].append(contour)
                spin_texture_contour_data["scalars"].append(scalars)
                
        
        self.spin_texture_contour_data = spin_texture_contour_data
        return self.spin_texture_contour_data 
    
    def plot_spin_texture_contours(self, spin_texture_contour_data:dict, 
                    alpha:float = 1.0,
                    linewidth:float = 0.2,
                    colors:str = None,
                    linestyles:str = "solid",
                    norm:mpcolors.Normalize = None,
                    clim:tuple = (None, None),
                    cmap:str = "plasma",
                    countour_kwargs: dict = None):
        """Plot spin texture contours from spin texture data.

        Parameters
        ----------
        spin_texture_contour_data : dict
            Dictionary containing spin texture contour data with contours, scalars, etc.
        clim : tuple of float, optional
            Color limits as (vmin, vmax), by default (None, None).
        cmap : str, optional
            Colormap name, by default "plasma".
        alpha : float, optional
            Transparency level (0-1), by default 1.0.
        linewidth : float, optional
            Width of the contour lines, by default 0.2.
        colors : str, optional
            Color for the contours, by default None.
        linestyles : str, optional
            Style of the contour lines, by default "solid".
        norm : matplotlib.colors.Normalize, optional
            Normalization instance for color mapping, by default None.
        clim : tuple of float, optional
            Color limits as (vmin, vmax), by default (None, None).
        cmap : str, optional
            Colormap name, by default "plasma".
        countour_kwargs : dict, optional
            Additional kwargs for matplotlib contour function, by default None.

        Returns
        -------
        list
            List of matplotlib contour handles.
        """
        
        
        if not hasattr(self, "norm") or not hasattr(self, "clim") or not hasattr(self, "cmap"):
            self.set_scalar_mappable(norm=norm, clim=clim, cmap=cmap)
            
        countour_kwargs = {} if countour_kwargs is None else countour_kwargs
        countour_kwargs.setdefault("linewidths", linewidth)
        countour_kwargs.setdefault("colors", colors)
        countour_kwargs.setdefault("linestyles", linestyles)
        countour_kwargs.setdefault("alpha", alpha)
        if colors is not None:
            countour_kwargs.setdefault("colors", colors)
        else:
            countour_kwargs.setdefault("cmap", self.cmap)
            countour_kwargs.setdefault("norm", self.norm)
            countour_kwargs.setdefault("vmin", self.clim[0])
            countour_kwargs.setdefault("vmax", self.clim[1])
        
        self.contour_handles = []
        for z in spin_texture_contour_data["contours"]:
            self.contour_handles.append(self.ax.contour(z, [self.energy], **countour_kwargs))
        
        self.handles.extend(self.contour_handles)

        return self.contour_handles
    
    def plot_spin_texture_scatter(self, 
                                  spin_texture_contour_data:dict,
                                  s:int =50,
                                  edgecolor:str = "none",
                                  alpha:float = 1.0,
                                  marker:str = ".",
                                  color:str = "k",
                                  scatter_kwargs:dict = None,
                                  ):
        """Plot spin texture as scatter points on Fermi surface contours.

        Parameters
        ----------
        spin_texture_contour_data : dict
            Dictionary containing spin texture contour data with points and scalars.
        s : int, optional
            Size of scatter points, by default 50.
        edgecolor : str, optional
            Color of point edges, by default "none".
        alpha : float, optional
            Transparency level (0-1), by default 1.0.
        marker : str, optional
            Marker style for scatter points, by default ".".
        color : str, optional
            Color for points when not using scalar coloring, by default "k".
        scatter_kwargs : dict, optional
            Additional kwargs for matplotlib scatter function, by default None.

        Returns
        -------
        list
            List of matplotlib scatter handles.
        """

        x_limits = [0, 0]
        y_limits = [0, 0]
        self.scatter_handles = []
        for i_band, (sx, sy, sz, points, scalars) in enumerate(zip(spin_texture_contour_data["sx"], 
                                                          spin_texture_contour_data["sy"], 
                                                          spin_texture_contour_data["sz"], 
                                                          spin_texture_contour_data["points"],
                                                          spin_texture_contour_data["scalars"])):
            
            x_limits = [
                    min(x_limits[0], points[:, 0].min()),
                    max(x_limits[1], points[:, 0].max()),
                ]
            y_limits = [
                min(y_limits[0], points[:, 1].min()),
                max(y_limits[1], points[:, 1].max()),
            ]
            
            scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
            scatter_kwargs.setdefault("s", s)
            scatter_kwargs.setdefault("edgecolor", edgecolor)
            scatter_kwargs.setdefault("alpha", alpha)
            scatter_kwargs.setdefault("marker", marker)
            scatter_kwargs.setdefault("c", color)
            scatter_kwargs.setdefault("cmap", self.cmap)
            scatter_kwargs.setdefault("norm", self.norm)
            scatter_kwargs.setdefault("vmin", self.clim[0])
            scatter_kwargs.setdefault("vmax", self.clim[1])
            
            self.scatter_handles.append(self.ax.scatter(
                points[:, 0],
                points[:, 1],
                **scatter_kwargs
            ))
            
        self.x_limits = (
        x_limits[0] - abs(x_limits[0]) * 0.1,
        x_limits[1] + abs(x_limits[1]) * 0.1,
        )
        self.y_limits = (
            y_limits[0] - abs(y_limits[0]) * 0.1,
            y_limits[1] + abs(y_limits[1]) * 0.1,
        )
        self.handles.extend(self.scatter_handles)
        return self.scatter_handles

    def plot_spin_texture_arrows(self, 
                     spin_texture_contour_data:dict,
                     arrow_color: str | list[str]| None = None,
                     scale:float | None = None,
                     scale_units:str = "inches",
                     units:str = "inches",
                     angles:str = "uv",
                     quiver_kwargs: dict = None,
                     ):
        """Plot spin texture as arrows (vectors) on Fermi surface contours.

        This method visualizes the spin texture by drawing arrows that represent
        the in-plane spin components (sx, sy) at points along the Fermi surface
        contours. The arrow direction indicates the spin orientation and can be
        colored by various spin projections.

        Parameters
        ----------
        spin_texture_contour_data : dict
            Dictionary containing spin texture contour data with points, sx, sy, sz, and scalars.
        arrow_color : str, list of str, or None, optional
            Color(s) for the arrows. If str, all arrows use same color. If list, 
            different colors for each band. If None, colors based on scalars, by default None.
        scale : float or None, optional
            Scale factor for arrow length. If None, matplotlib auto-scales, by default None.
        scale_units : str, optional
            Units for the scale parameter, by default "inches".
        units : str, optional
            Units for arrow dimensions, by default "inches".
        angles : str, optional
            How to interpret arrow angles ('uv' for x,y components), by default "uv".
        cmap : str, optional
            Colormap name for scalar coloring, by default "plasma".
        norm : matplotlib.colors.Normalize, optional
            Normalization instance for color mapping, by default None.
        quiver_kwargs : dict, optional
            Additional kwargs for matplotlib quiver function, by default None.

        Returns
        -------
        None
            This method adds quiver plots to the axes but returns None.

        Notes
        -----
        The arrows represent the in-plane spin components (sx, sy) while the color
        can represent any spin projection specified during contour generation.
        """
        
        
        quiver_kwargs = {} if quiver_kwargs is None else quiver_kwargs
        quiver_kwargs.setdefault("scale",  1 / scale)
        quiver_kwargs.setdefault("scale_units", scale_units)
        quiver_kwargs.setdefault("angles", angles)
        
        
            
        x_limits = [0, 0]
        y_limits = [0, 0]
        self.quiver_handles = []
        for i_band, (sx, sy, sz, points, scalars) in enumerate(zip(spin_texture_contour_data["sx"], 
                                                          spin_texture_contour_data["sy"], 
                                                          spin_texture_contour_data["sz"], 
                                                          spin_texture_contour_data["points"],
                                                          spin_texture_contour_data["scalars"])):
            
            x_limits = [
                    min(x_limits[0], points[:, 0].min()),
                    max(x_limits[1], points[:, 0].max()),
                ]
            y_limits = [
                min(y_limits[0], points[:, 1].min()),
                max(y_limits[1], points[:, 1].max()),
            ]

            quiver_args = [
                points[:, 0],  # Arrow position x-component
                points[:, 1],  # Arrow position y-component
                sx[:],  # Arrow direction x-component
                sy[:],  # Arrow direction y-component
            ]
            band_quiver_kwargs = quiver_kwargs.copy()
            
            if isinstance(arrow_color, list):
                band_quiver_kwargs["color"] = arrow_color[i_band]
            elif isinstance(arrow_color, str):
                band_quiver_kwargs["color"] = arrow_color
            else:
                quiver_args.append(scalars)
                band_quiver_kwargs.setdefault("cmap", self.cmap)
                band_quiver_kwargs.setdefault("norm", self.norm)
                band_quiver_kwargs.setdefault("clim", self.clim)
                band_quiver_kwargs["color"] = None

            self.quiver_handles.append(self.ax.quiver(*quiver_args,**band_quiver_kwargs))

        self.x_limits = (
            x_limits[0] - abs(x_limits[0]) * 0.1,
            x_limits[1] + abs(x_limits[1]) * 0.1,
        )
        self.y_limits = (
            y_limits[0] - abs(y_limits[0]) * 0.1,
            y_limits[1] + abs(y_limits[1]) * 0.1,
        )
        self.handles.extend(self.quiver_handles)
   
        return None
        

    def show_colorbar(self, 
                      label:str = "",
                      n_ticks:int = 5,
                      cmap:str = "plasma", 
                      clim:tuple = (None, None),
                      colorbar_kwargs:dict = None):
        """Add a colorbar to the plot.

        Parameters
        ----------
        label : str, optional
            Label for the colorbar, by default "".
        n_ticks : int, optional
            Number of ticks on the colorbar, by default 5.
        cmap : str, optional
            Colormap name, by default "plasma".
        norm : matplotlib.colors.Normalize, optional
            Normalization instance for color mapping, by default None.
        clim : tuple of float, optional
            Color limits as (vmin, vmax), by default (None, None).
        colorbar_kwargs : dict, optional
            Additional kwargs for matplotlib colorbar function, by default None.
        """
        self.colorbar = self.fig.colorbar(
                        self.cm,
                        ax=self.ax, 
                        label=label,
                        **colorbar_kwargs)
        
    def set_scalar_mappable(self, 
                            norm:mpcolors.Normalize = None, 
                            clim:tuple = (None, None), 
                            cmap:str = "plasma"):
        
        vmin = clim[0]
        vmax = clim[1]
        if vmin is None:
            vmin=-0.5
        if vmax is None:
            vmax=0.5
        if norm is None:
            norm = mpcolors.Normalize
            
        norm = norm(vmin, vmax)
        self.norm = norm
        self.clim = clim
        self.cmap = cmap

        self.cm = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        
    def set_colorbar_ticks(self, n_ticks:int = 5, tick_labels = None, tick_positions = None, **kwargs):
        """Set the tick positions and labels for the colorbar.

        Parameters
        ----------
        n_ticks : int, optional
            Number of ticks to use if tick_positions is None, by default 5.
        tick_labels : array-like, optional
            Custom tick labels, by default None.
        tick_positions : array-like, optional
            Custom tick positions, by default None.
        **kwargs
            Additional keyword arguments passed to matplotlib set_yticks.
        """
        if tick_positions is None:
            tick_positions = np.linspace(self.colorbar.vmin, self.colorbar.vmax, n_ticks)
        if tick_labels is None:
            tick_labels = np.linspace(self.colorbar.vmin, self.colorbar.vmax, n_ticks)
        set_y_tick_kwargs = {}
        set_y_tick_kwargs.setdefault("ticks", tick_positions)
        set_y_tick_kwargs.setdefault("labels", tick_labels)
        set_y_tick_kwargs.update(kwargs)
        self.colorbar.ax.set_yticks(**set_y_tick_kwargs)
        
        
    def set_xticks(self, n_ticks:int = 5, tick_labels = None, tick_positions = None, **kwargs):
        """Set the tick positions and labels for the colorbar.

        Parameters
        ----------
        n_ticks : int, optional
            Number of ticks to use if tick_positions is None, by default 5.
        tick_labels : array-like, optional
            Custom tick labels, by default None.
        tick_positions : array-like, optional
            Custom tick positions, by default None.
        **kwargs
            Additional keyword arguments passed to matplotlib set_yticks.
        """
        if tick_positions is None:
            tick_positions = np.linspace(self.x_limits[0], self.x_limits[1], n_ticks)
        if tick_labels is None:
            tick_labels = np.linspace(self.x_limits[0], self.x_limits[1], n_ticks)
        set_x_tick_kwargs = {}
        set_x_tick_kwargs.setdefault("ticks", tick_positions)
        set_x_tick_kwargs.setdefault("labels", tick_labels)
        set_x_tick_kwargs.update(kwargs)
        self.ax.set_xticks(**set_x_tick_kwargs)
        
    def set_yticks(self, n_ticks:int = 5, tick_labels = None, tick_positions = None, **kwargs):
        """Set the tick positions and labels for the colorbar.

        Parameters
        ----------
        n_ticks : int, optional
            Number of ticks to use if tick_positions is None, by default 5.
        tick_labels : array-like, optional
            Custom tick labels, by default None.
        tick_positions : array-like, optional
            Custom tick positions, by default None.
        **kwargs
            Additional keyword arguments passed to matplotlib set_yticks.
        """
        if tick_positions is None:
            tick_positions = np.linspace(self.y_limits[0], self.y_limits[1], n_ticks)
        if tick_labels is None:
            tick_labels = np.linspace(self.y_limits[0], self.y_limits[1], n_ticks)
        set_y_tick_kwargs = {}
        set_y_tick_kwargs.setdefault("ticks", tick_positions)
        set_y_tick_kwargs.setdefault("labels", tick_labels)
        set_y_tick_kwargs.update(kwargs)
        self.ax.set_yticks(**set_y_tick_kwargs)
       
    def set_colorbar_tick_params(self, **kwargs):
        """Set the tick parameters for the colorbar.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to matplotlib tick_params.
        """
        self.colorbar.ax.tick_params(**kwargs)
        
    def set_colorbar_label(self, label:str|None = None, **kwargs):
        """Set the label for the colorbar.

        Parameters
        ----------
        label : str or None, optional
            Label text. If None, uses current label, by default None.
        **kwargs
            Additional keyword arguments passed to matplotlib set_ylabel.
        """
        if label is None:
            label = self.colorbar.ax.get_yaxis().label.get_text()
            
        self.colorbar.ax.set_ylabel(label, **kwargs)
    
    def set_xlim(self, xlimits = None, **kwargs):
        """Set the x-axis limits.

        Parameters
        ----------
        xlimits : tuple of float, optional
            X-axis limits as (xmin, xmax). If None, uses automatically determined limits, by default None.
        **kwargs
            Additional keyword arguments passed to matplotlib set_xlim.
        """
        if xlimits is None:
            xlimits = self.x_limits
        self.ax.set_xlim(xlimits, **kwargs)
    
    def set_ylim(self, ylimits =None, **kwargs):
        """Set the y-axis limits.

        Parameters
        ----------
        ylimits : tuple of float, optional
            Y-axis limits as (ymin, ymax). If None, uses automatically determined limits, by default None.
        **kwargs
            Additional keyword arguments passed to matplotlib set_ylim.
        """
        if ylimits is None:
            ylimits = self.y_limits
        self.ax.set_ylim(ylimits, **kwargs)
    
    def set_xlabel(self, xlabel = '$k_{x}$ ($\AA^{-1}$)', **kwargs):
        """Set the x-axis label.

        Parameters
        ----------
        xlabel : str, optional
            X-axis label text, by default '$k_{x}$ ($\AA^{-1}$)'.
        **kwargs
            Additional keyword arguments passed to matplotlib set_xlabel.
        """
        self.ax.set_xlabel(xlabel, **kwargs)
    
    def set_ylabel(self, ylabel = '$k_{y}$ ($\AA^{-1}$)', **kwargs):
        """Set the y-axis label.

        Parameters
        ----------
        ylabel : str, optional
            Y-axis label text, by default '$k_{y}$ ($\AA^{-1}$)'.
        **kwargs
            Additional keyword arguments passed to matplotlib set_ylabel.
        """
        self.ax.set_ylabel(ylabel, **kwargs)
        
    def set_tick_params(self, axis: str = "both", which: str = "major", reset: bool = False, **kwargs):
        """Set the tick parameters for the axes.

        Parameters
        ----------
        axis : str, optional
            Which axis to apply the parameters to ('x', 'y', or 'both'), by default "both".
        which : str, optional
            Which ticks to apply the parameters to ('major', 'minor', or 'both'), by default "major".
        reset : bool, optional
            Whether to reset all parameters before setting new ones, by default False.
        **kwargs
            Additional keyword arguments passed to matplotlib tick_params.
        """
        self.ax.tick_params(axis=axis, which=which, reset=reset, **kwargs)
        
    def get_colorbar(self):
        """Get the colorbar object.

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The colorbar instance associated with this plot.
        """
        return self.colorbar
    
    def add_legend(self, **kwargs):
        """Add a legend to the plot.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to matplotlib legend function.
        """
        
        self.ax.legend()
        
    def set_aspect(self, aspect: float | str = "equal", **kwargs):
        """Set the aspect ratio of the plot.
        
        Parameters
        """
        self.ax.set_aspect(aspect, **kwargs)

    def savefig(self, savefig, dpi: int | str = "figure", **kwargs):
        """Save the figure to a file.

        Parameters
        ----------
        savefig : str or path-like
            The filename or path where the figure should be saved.
        dpi : int or str, optional
            The resolution in dots per inch. Can be 'figure' to use figure's dpi, by default "figure".
        **kwargs
            Additional keyword arguments passed to matplotlib savefig function.
        """
        self.fig.savefig(savefig, dpi=dpi, bbox_inches="tight", **kwargs)

    def show(self, **kwargs):
        """Display the plot.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to matplotlib show function.
        """
        plt.show(**kwargs)
