__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import copy
import logging
import os
from enum import Enum
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from matplotlib import colors as mpcolors

from pyprocar.core import FermiSurface
from pyprocar.plotter import FermiSlicePlotter
from pyprocar.utils import ROOT, data_utils, welcome

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


class Fermi2DMode(Enum):
    plain = "plain"
    plain_bands = "plain_bands"
    parametric = "parametric"
    spin_texture = "spin_texture"
    
    

def fermi2D(
    code: str,
    dirname: str,
    mode: Fermi2DMode = Fermi2DMode.plain,
    fermi: float = None,
    band_indices: List[List] = None,
    spins: List[int] = None,
    atoms: List[int] = None,
    orbitals: List[int] = None,
    energy: float = 0.0,
    k_z_plane: float = 0.0,
    show: bool = True,
    savefig: str = None,
    extend_zone_directions: List[Union[List[int], tuple]] = None,
    show_colorbar: bool = True,
    plot_line_kwargs: dict = None,
    plot_arrows: bool = True,
    plot_arrows_kwargs: dict = None,
    show_colorbar_kwargs:dict=None,
    use_cache: bool = False,
    verbose: int = 1,
    padding: int = 10,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    ax: plt.Axes = None,
):
    """Plot the 2D Fermi surface in a constant k_z plane.

    This function generates 2D Fermi surface plots by slicing the 3D Fermi surface
    at a specified k_z plane. It supports multiple visualization modes including
    plain contours, parametric coloring, and spin texture analysis.

    Parameters
    ----------
    code : str
        The DFT code used for the calculation. Options include 'vasp', 'qe', 'elk', 
        'abinit', 'siesta', 'lobster', etc.
    dirname : str
        The directory path containing the DFT calculation files.
    mode : str, optional
        The plotting mode. Options are 'plain', 'plain_bands', 'parametric', or 
        'spin_texture', by default 'plain'
    use_cache : bool, optional
        Whether to use cached EBS data if available, by default False
    spin_projection : SpinProjection or str, optional
        The spin projection component for spin texture mode. Options include 
        'x', 'y', 'z', 'x^2', 'y^2', 'z^2', by default 'z^2'
    fermi : float, optional
        The Fermi energy in eV. If None, the Fermi energy from the calculation 
        will be used, by default None
    fermi_shift : float, optional
        The fermi energy shift, by default 0.0
    band_indices : List[List]
        A list of list that contains band indices for a given spin (Not implemented in new version)
    band_colors : List[List]
        A list of list that contains colors for the band index
        corresponding the band_indices for a given spin (Not implemented in new version)
    spins : List[int], optional
        List of spin indices to include. For non-collinear calculations, 
        use [0], by default None (all spins)
    atoms : List[int], optional
        List of atom indices for atomic projections, by default None (all atoms)
    orbitals : List[int], optional
        List of orbital indices for orbital projections, by default None (all orbitals)
    energy : float, optional
        The energy level (relative to Fermi) at which to generate the iso-surface.
        When None, uses 0 (Fermi energy), by default None
    k_z_plane : float, optional
        The k_z coordinate of the plane to slice for the 2D surface, by default 0.0
    k_z_plane_tol : float, optional
        Tolerance for selecting k-points near the k_z plane, by default 0.01
    rot_symm : int, optional
        Rotational symmetry factor to apply around the z-axis, by default 1
    translate : List[int], optional
        Translation vector [x, y, z] to apply to k-points, by default [0, 0, 0]
    rotation : List[int], optional
        Rotation parameters [angle, x, y, z] where angle is in degrees and 
        [x, y, z] is the rotation axis, by default [0, 0, 0, 1]
    point_density : int, optional
        Density of points for spin texture interpolation, by default 10
    interpolation : int, optional
        Number of interpolation points for generating smooth contours, by default 300
    linecollection_kwargs : dict, optional
        Additional keyword arguments for matplotlib LineCollection, by default None
    linestyles : tuple[str, str], optional
        Line styles for different spin channels, by default ('solid', 'dashed')
    colors : tuple[str, str], optional
        Colors for different spin channels, by default None
    linewidths : tuple[float, float], optional
        Line widths for different spin channels, by default (0.2, 0.2)
    alphas : tuple[float, float], optional
        Alpha values (transparency) for different spin channels, by default (1.0, 1.0)
    plot_scatter : bool, optional
        Whether to plot scatter points in spin texture mode, by default True
    plot_scatter_kwargs : dict, optional
        Additional keyword arguments for scatter plot, by default None
    plot_contours : bool, optional
        Whether to plot contour lines, by default True
    plot_contours_kwargs : dict, optional
        Additional keyword arguments for contour plots, by default None
    plot_arrows : bool, optional
        Whether to plot spin direction arrows in spin texture mode, by default True
    plot_arrows_kwargs : dict, optional
        Additional keyword arguments for arrow plots, by default None
    arrow_scale : float or None, optional
        Scaling factor for arrow size in spin texture mode, by default 1.0
    show_colorbar : bool, optional
        Whether to display the colorbar, by default True
    cmap : str, optional
        Colormap name for the plot, by default 'plasma'
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap, by default None
    clim : tuple, optional
        Color limits (vmin, vmax) for the colormap, by default (None, None)
    colorbar_kwargs : dict, optional
        Additional keyword arguments for colorbar creation, by default None
    colorbar_tick_kwargs : dict, optional
        Additional keyword arguments for colorbar tick formatting, by default None
    colorbar_tick_params_kwargs : dict, optional
        Additional keyword arguments for colorbar tick parameters, by default None
    colorbar_label_kwargs : dict, optional
        Additional keyword arguments for colorbar label formatting, by default None
    add_legend : bool, optional
        Whether to add a legend to the plot, by default False
    show : bool, optional
        Whether to display the plot immediately, by default True
    savefig : str, optional
        The filename to save the plot as., by default None
    spin_texture : bool, optional
        Boolean value to determine if spin arrows are plotted, by default False
    exportplt : bool, optional
        Boolean value where to return the matplotlib.pyplot state plt, by default False
    print_plot_opts: bool, optional
        Boolean to print the plotting options
    use_cache: bool, optional
        Boolean to use cache for EBS
    verbose: int, optional
        Verbosity level
    padding: int, optional
        Amount of padding for the Fermi surface calculation, by default 10
    extend_zone_directions: List[Union[List[int], tuple]], optional
        Directions to extend the surface to neighboring Brillouin zones, by default None
    plot_arrows: bool, optional
        Whether to plot arrow vectors on the 2D slice, by default False
    arrow_factor: float, optional
        Scaling factor for arrow sizes, by default 1.0
    cmap: str, optional
        Colormap for the plot, by default "plasma"

    Returns
    -------
    matplotlib.pyplot or FermiSlicePlotter
        Returns the matplotlib.pyplot state plt or the FermiSlicePlotter object

    Raises
    ------
    RuntimeError
        If the translate option is invalid (not length 1 or 3)
    ValueError
        If an invalid mode is specified

    Examples
    --------
    Basic usage with VASP calculation:

    >>> fermi2D(code='vasp', dirname='calculation_dir')

    Plot with parametric coloring for specific atoms and orbitals:

    >>> fermi2D(code='vasp', dirname='calculation_dir', mode='parametric',
    ...         atoms=[0, 1], orbitals=[0, 1, 2])

    Generate spin texture plot:

    >>> fermi2D(code='vasp', dirname='calculation_dir', mode='spin_texture',
    ...         spin_projection='z', plot_arrows=True)
    """

    user_logger.info(f"If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    welcome()
    # Turn interactive plotting off
    plt.ioff()

    user_logger.info("_" * 100)
    user_logger.info("### Parameters ###")
    user_logger.info(f"dirname         : {dirname}")
    user_logger.info(f"mode            : {mode.value}")
    user_logger.info(f"bands           : {band_indices}")
    user_logger.info(f"atoms           : {atoms}")
    user_logger.info(f"orbitals        : {orbitals}")
    user_logger.info(f"spin comp.      : {spins}")
    user_logger.info(f"energy          : {energy}")
    user_logger.info(f"k_z_plane       : {k_z_plane}")
    user_logger.info(f"save figure     : {savefig}")
    user_logger.info("_" * 100)

    modes_txt = " , ".join([mode.value for mode in Fermi2DMode])
    message = f"""
            There are additional plot options that are defined in a configuration file. 
            You can change these configurations by passing the keyword argument to the function
            To print a list of plot options set print_plot_opts=True

            Here is a list modes : {modes_txt}"""
    user_logger.info(message)

    user_logger.info("_" * 100)
    
    # Create Fermi surface using the new implementation
    logger.info("Creating Fermi surface using the new implementation")
    
    fs = FermiSurface.from_code(
        code=code, 
        dirpath=dirname, 
        fermi=fermi,
        fermi_shift=energy,
        padding=padding,
        use_cache=use_cache
    )
    
    logger.info(f"Created Fermi surface: {fs}")

    # Calculate slice properties based on mode and spin texture
    if mode in [Fermi2DMode.plain.value, Fermi2DMode.plain_bands.value]:
        property_name = None
    elif mode == Fermi2DMode.parametric.value:
        property_name = "projected_sum"
        fs.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)

    elif mode == Fermi2DMode.spin_texture.value and fs.ebs.is_non_collinear:
        property_name = "projected_sum_spin_texture"
        fs.get_property(property_name, atoms=atoms, orbitals=orbitals)
        
    elif mode == Fermi2DMode.spin_texture.value and not fs.ebs.is_non_collinear:
        raise ValueError("Spin texture is only available for non-collinear calculations")
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Please choose from {modes_txt}.")

    # Extend surface to neighboring zones if requested
    if extend_zone_directions is not None:
        user_logger.info(f"Extending surface to zones: {extend_zone_directions}")
        fs = fs.extend_surface(zone_directions=extend_zone_directions)

    # Create 2D slice plotter
    normal = (0, 0, 1)
    origin = (0, 0, k_z_plane)
    
    fsplt = FermiSlicePlotter(
        fs,
        normal=normal, 
        origin=origin,
        figsize=figsize,
        dpi=dpi,
        ax=ax
    )
    
    user_logger.info(f"Creating 2D slice at k_z = {k_z_plane}")
    
    plot_arrows_kwargs = {} if plot_arrows_kwargs is None else plot_arrows_kwargs
    plot_line_kwargs = {} if plot_line_kwargs is None else plot_line_kwargs
    
    # Plot the slice
    if mode == Fermi2DMode.plain.value or property_name is None:
        fsplt.plot(plot_arrows=plot_arrows, **plot_line_kwargs)
    else:
        plot_arrows_kwargs = {} if plot_arrows_kwargs is None else plot_arrows_kwargs
        fsplt.plot(
            scalars_name=property_name,
            vectors_name=property_name if mode == Fermi2DMode.spin_texture.value else None,
            plot_arrows=plot_arrows,
            plot_arrows_kwargs=plot_arrows_kwargs,
            **plot_line_kwargs
        )
        
        # Show colorbar for parametric modes
        if show_colorbar:
            show_colorbar_kwargs = {} if show_colorbar_kwargs is None else show_colorbar_kwargs
            fsplt.show_colorbar(**show_colorbar_kwargs)

    if savefig:
        fsplt.fig.savefig(savefig)
        user_logger.info(f"Plot saved to {savefig}")
    elif show:
        fsplt.show()
        
    return fsplt.fig, fsplt.ax
