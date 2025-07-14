__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import copy
import logging
import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from matplotlib import colors as mpcolors

from pyprocar.core import FermiSurface
from pyprocar.plotter import FermiSlicePlotter
from pyprocar.utils import ROOT, data_utils, welcome
from pyprocar.utils.log_utils import set_verbose_level

with open(os.path.join(ROOT, "pyprocar", "cfg", "fermi_surface_2d.yml"), "r") as file:
    plot_opt = yaml.safe_load(file)


user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


def fermi2D(
    code: str,
    dirname: str,
    mode: str = "plain",
    fermi: float = None,
    fermi_shift: float = 0.0,
    band_indices: List[List] = None,
    band_colors: List[List] = None,
    spins: List[int] = None,
    atoms: List[int] = None,
    orbitals: List[int] = None,
    energy: float = None,
    k_z_plane: float = 0.0,
    k_z_plane_tol: float = 0.001,
    rot_symm=1,
    translate: List[int] = [0, 0, 0],
    rotation: List[int] = [0, 0, 0, 1],
    spin_texture: bool = False,
    exportplt: bool = False,
    savefig: str = None,
    print_plot_opts: bool = False,
    use_cache: bool = False,
    verbose: int = 1,
    padding: int = 10,
    extend_zone_directions: List[Union[List[int], tuple]] = None,
    plot_arrows: bool = False,
    arrow_factor: float = 1.0,
    cmap: str = "plasma",
    **kwargs,
):
    """This function plots the 2d fermi surface in the z = 0 plane

    Parameters
    ----------
    code : str,
        This parameter sets the code to parse, by default "vasp"
    dirname : str, optional
        This parameter is the directory of the calculation, by default ''
    fermi : float, optional
        The fermi energy. If none is given, the fermi energy in the directory will be used, by default None
    fermi_shift : float, optional
        The fermi energy shift, by default 0.0
    band_indices : List[List]
        A list of list that contains band indices for a given spin (Not implemented in new version)
    band_colors : List[List]
        A list of list that contains colors for the band index
        corresponding the band_indices for a given spin (Not implemented in new version)
    spins : List[int], optional
        List of spins, by default [0]
    atoms : List[int], optional
        List of atoms, by default None
    orbitals : List[int], optional
        List of orbitals, by default None
    energy : float, optional
        The energy to generate the iso surface.
        When energy is None the 0 is used by default, which is the fermi energy,
        by default None
    k_z_plane : float, optional
        Which K_z plane to generate 2d fermi surface, by default 0.0
    rot_symm : int, optional
        _description_, by default 1
    translate : List[int], optional
        Matrix to translate the kpoints, by default [0, 0, 0]
    rotation : List[int], optional
         Matrix to rotate the kpoints, by default [0, 0, 0, 1]
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
        invalid option --translate
    """
    set_verbose_level(verbose)

    user_logger.info(f"If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    welcome()
    # Turn interactive plotting off
    plt.ioff()

    user_logger.info("_" * 100)
    user_logger.info("### Parameters ###")

    if len(translate) != 3 and len(translate) != 1:
        logger.error(f"Error: --translate option is invalid! ({translate})")
        raise RuntimeError("invalid option --translate")

    user_logger.info(f"dirname         : {dirname}")
    user_logger.info(f"bands           : {band_indices}")
    user_logger.info(f"atoms           : {atoms}")
    user_logger.info(f"orbitals        : {orbitals}")
    user_logger.info(f"spin comp.      : {spins}")
    user_logger.info(f"energy          : {energy}")
    user_logger.info(f"k_z_plane       : {k_z_plane}")
    user_logger.info(f"rot. symmetry   : {rot_symm}")
    user_logger.info(f"origin (trasl.) : {translate}")
    user_logger.info(f"rotation        : {rotation}")
    user_logger.info(f"save figure     : {savefig}")
    user_logger.info(f"spin_texture    : {spin_texture}")
    user_logger.info(f"padding         : {padding}")
    user_logger.info("_" * 100)

    modes = ["plain", "plain_bands", "parametric"]
    modes_txt = " , ".join(modes)
    message = f"""
            There are additional plot options that are defined in a configuration file. 
            You can change these configurations by passing the keyword argument to the function
            To print a list of plot options set print_plot_opts=True

            Here is a list modes : {modes_txt}"""
    user_logger.info(message)

    if print_plot_opts:
        for key, value in plot_opt.items():
            user_logger.info(f"{key} : {value}")

    user_logger.info("_" * 100)
    
    # Create Fermi surface using the new implementation
    logger.info("Creating Fermi surface using the new implementation")
    
    # Set default energy to fermi level if not provided
    if energy is None:
        energy = 0.0
        
    # Reduce bands near the specified energy
    # reduce_bands_near_energy = energy if energy != 0.0 else None
    
    fs = FermiSurface.from_code(
        code=code, 
        dirpath=dirname, 
        fermi=fermi,
        fermi_shift=energy,
        padding=padding,
        use_cache=use_cache
    )
    
    logger.info(f"Created Fermi surface: {fs}")
    user_logger.info(f"Fermi surface has {fs.n_points} points")

    # Calculate slice properties based on mode and spin texture
    if spin_texture:
        if not fs.ebs.is_non_collinear:
            raise ValueError("Spin texture is only available for non-collinear calculations")
        
        property_name = "projected_sum_spin_texture"
        fs.get_property(property_name, atoms=atoms, orbitals=orbitals)
        plot_arrows = True
        
    elif mode == "parametric":
        property_name = "projected_sum"
        fs.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)
        
    elif mode in ["plain", "plain_bands"]:
        # For plain mode, we just use the fermi surface itself without additional properties
        property_name = None
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Extend surface to neighboring zones if requested
    if extend_zone_directions is not None:
        user_logger.info(f"Extending surface to zones: {extend_zone_directions}")
        fs = fs.extend_surface(zone_directions=extend_zone_directions)

    # Create 2D slice plotter
    normal = (0, 0, 1)
    origin = (0, 0, k_z_plane)
    
    fsplt = FermiSlicePlotter(
        normal=normal, 
        origin=origin,
        figsize=(8, 6)
    )
    
    user_logger.info(f"Creating 2D slice at k_z = {k_z_plane}")
    
    # Plot the slice
    if mode == "plain" or property_name is None:
        fsplt.plot(fs, plot_arrows=plot_arrows, cmap=cmap, **kwargs)
    else:
        fsplt.plot(
            fs, 
            scalars_name=property_name,
            vectors_name=property_name if spin_texture else None,
            plot_arrows=plot_arrows,
            plot_arrows_args={"arrow_length_factor": arrow_factor},
            cmap=cmap,
            **kwargs
        )
        
        # Show colorbar for parametric modes
        fsplt.show_colorbar()

    # Handle output
    if exportplt:
        return plt
    else:
        if savefig:
            plt.savefig(savefig)
            user_logger.info(f"Plot saved to {savefig}")
        else:
            fsplt.show()
        return None
