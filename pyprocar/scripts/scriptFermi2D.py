__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import logging
from enum import Enum

import matplotlib.pyplot as plt

from pyprocar.core import FermiSurface
from pyprocar.plotter import FermiSlicePlotter
from pyprocar.utils import welcome

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
    mode: Fermi2DMode | str = Fermi2DMode.plain,
    fermi: float = None,
    band_indices: list[list] = None,
    spins: list[int] = None,
    atoms: list[int] = None,
    orbitals: list[int] = None,
    energy: float = 0.0,
    k_z_plane: float = 0.0,
    show: bool = True,
    savefig: str = None,
    extend_zone_directions: list[list[int] | tuple] = None,
    show_colorbar: bool = True,
    plot_line_kwargs: dict = None,
    plot_arrows: bool = True,
    plot_arrows_kwargs: dict = None,
    cmap: str = "plasma",
    use_cache: bool = False,
    verbose: int = 1,
    padding: int = 10,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 100,
    ax: plt.Axes = None,
):
    """Plot the 2D Fermi surface in a constant k_z plane.

    Slices the 3D Fermi surface at ``k_z = k_z_plane`` and draws the contour lines,
    optionally colored by an atomic/orbital/spin projection or decorated with
    spin-texture arrows.

    Parameters
    ----------
    code : str
        The DFT code used for the calculation, such as 'vasp', 'qe', 'elk',
        'abinit', 'siesta' or 'lobster'.
    dirname : str
        The directory containing the DFT calculation files.
    mode : Fermi2DMode or str, optional
        'plain', 'plain_bands', 'parametric' or 'spin_texture', by default 'plain'.
        'spin_texture' needs a non-collinear calculation.
    fermi : float, optional
        The Fermi energy in eV. If None, the Fermi energy of the calculation is used.
    band_indices : list[list], optional
        Not implemented in the new version; the value is only logged.
    spins : list[int], optional
        Spin indices to project onto in 'parametric' mode. For a non-collinear
        calculation, 0 is the total and 1, 2, 3 are Sx, Sy, Sz.
    atoms : list[int], optional
        Atom indices to project onto, by default all atoms.
    orbitals : list[int], optional
        Orbital indices to project onto, by default all orbitals.
    energy : float, optional
        The iso-energy relative to the Fermi energy, in eV, by default 0.0.
    k_z_plane : float, optional
        The k_z coordinate of the slicing plane, by default 0.0.
    show : bool, optional
        Whether to show the figure when ``savefig`` is not given, by default True.
    savefig : str, optional
        The filename to save the figure to, by default None.
    extend_zone_directions : list[list[int] | tuple], optional
        Directions to extend the surface into neighboring Brillouin zones.
    show_colorbar : bool, optional
        Whether to draw a colorbar in 'parametric' and 'spin_texture' modes,
        by default True.
    plot_line_kwargs : dict, optional
        Keyword arguments for the matplotlib LineCollection, such as
        ``{"colors": "purple", "linewidths": 2.0, "linestyles": "dashed"}``.
    plot_arrows : bool, optional
        Whether to draw spin arrows in 'spin_texture' mode, by default True.
    plot_arrows_kwargs : dict, optional
        Keyword arguments for the matplotlib quiver, such as ``{"scale": 2.0}``.
    cmap : str, optional
        The colormap for the projection colors and arrows, by default 'plasma'.
    use_cache : bool, optional
        Whether to load a cached EBS if one exists, by default False.
    verbose : int, optional
        Verbosity level, by default 1.
    padding : int, optional
        Padding of the k-mesh for the Fermi surface calculation, by default 10.
    figsize : tuple[float, float], optional
        Figure size in inches, by default (8, 6).
    dpi : int, optional
        Figure resolution, by default 100.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes of the plot.

    Raises
    ------
    ValueError
        If the mode is unknown, or 'spin_texture' is requested for a calculation
        that is not non-collinear.

    Examples
    --------
    >>> fermi2D(code='vasp', dirname='calculation_dir')

    Color the contours by the d-orbital character of atom 1:

    >>> fermi2D(code='vasp', dirname='calculation_dir', mode='parametric',
    ...         atoms=[1], orbitals=[4, 5, 6, 7, 8], cmap='viridis')

    Color a non-collinear Fermi surface by Sx:

    >>> fermi2D(code='vasp', dirname='calculation_dir', mode='parametric', spins=[1])

    Draw the spin texture:

    >>> fermi2D(code='vasp', dirname='calculation_dir', mode='spin_texture')
    """

    mode = Fermi2DMode(mode)
    user_logger.info("If you want more detailed logs, set verbose to 2 or more")
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
    user_logger.info(f"Here is a list modes : {modes_txt}")

    user_logger.info("_" * 100)

    # Create Fermi surface using the new implementation
    logger.info("Creating Fermi surface using the new implementation")

    fs = FermiSurface.from_code(
        code=code,
        dirpath=dirname,
        fermi=fermi,
        fermi_shift=energy,
        padding=padding,
        use_cache=use_cache,
    )

    logger.info(f"Created Fermi surface: {fs}")

    # Calculate slice properties based on mode and spin texture
    if mode in [Fermi2DMode.plain, Fermi2DMode.plain_bands]:
        property_name = None
    elif mode == Fermi2DMode.parametric:
        property_name = "projected_sum"
        prop = fs.get_property(property_name, atoms=atoms, orbitals=orbitals, spins=spins)
        fs.set_values(property_name, prop.value)

    elif mode == Fermi2DMode.spin_texture and fs.ebs.is_non_collinear:
        property_name = "projected_sum_spin_texture"
        prop = fs.get_property(property_name, atoms=atoms, orbitals=orbitals)
        fs.set_values(property_name, prop.value)

    elif mode == Fermi2DMode.spin_texture and not fs.ebs.is_non_collinear:
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

    fsplt = FermiSlicePlotter(fs, normal=normal, origin=origin, figsize=figsize, dpi=dpi, ax=ax)

    user_logger.info(f"Creating 2D slice at k_z = {k_z_plane}")

    fsplt.plot(
        scalars_name=property_name,
        vectors_name=property_name if mode == Fermi2DMode.spin_texture else None,
        scalars_cmap=cmap,
        vectors_cmap=cmap,
        scalars_show_colorbar="single" if show_colorbar and property_name else "none",
        plot_arrows=plot_arrows,
        line_kwargs=plot_line_kwargs,
        quiver_kwargs=plot_arrows_kwargs,
    )
    if savefig:
        fsplt.savefig(savefig)
        user_logger.info(f"Plot saved to {savefig}")
    elif show:
        fsplt.show()

    return fsplt.fig, fsplt.ax
