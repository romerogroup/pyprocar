__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import copy
import logging
import os
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from matplotlib import colors as mpcolors

from pyprocar import io
from pyprocar.core import ElectronicBandStructure, FermiSurface, ProcarSymmetry
from pyprocar.core.fermisurface import SpinProjection
from pyprocar.utils import ROOT, data_utils, welcome

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)

class FermiSurface2DMode(Enum):
    """
    An enumeration for defining the modes of 2D Fermi surface representations.
    """
    PLAIN = "plain"
    PLAIN_BANDS = "plain_bands"
    PARAMETRIC = "parametric"
    SPIN_TEXTURE = "spin_texture"
    
    @classmethod
    def from_str(cls, mode: str):
        """
        Returns the FermiSurface2DMode enum value from a string.
        """
        if mode.lower() == cls.PLAIN.value:
            return cls.PLAIN
        elif mode.lower() == cls.PLAIN_BANDS.value:
            return cls.PLAIN_BANDS
        elif mode.lower() == cls.PARAMETRIC.value:
            return cls.PARAMETRIC
        elif mode.lower() == cls.SPIN_TEXTURE.value:
            return cls.SPIN_TEXTURE
        else:
            raise ValueError(f"Invalid mode: {mode}")

def fermi2D(
    code: str,
    dirname: str,
    mode: str = "plain",
    use_cache: bool = False,
    spin_projection: SpinProjection | str = "z^2",
    fermi: float = None,
    fermi_shift: float = 0.0,
    band_indices: List[List] = None,
    band_colors: List[List] = None,
    spins: List[int] = None,
    atoms: List[int] = None,
    orbitals: List[int] = None,
    energy: float = None,
    k_z_plane: float = 0.0,
    k_z_plane_tol: float = 0.01,
    rot_symm=1,
    translate: List[int] = [0, 0, 0],
    rotation: List[int] = [0, 0, 0, 1],
    point_density: int = 10,
    interpolation: int = 300,
    linecollection_kwargs: dict = None,
    linestyles: tuple[str, str] = ("solid", "dashed"),
    colors: tuple[str, str] = None,
    linewidths: tuple[float, float] = (0.2, 0.2),
    alphas: tuple[float, float] = (1.0, 1.0),
    plot_scatter: bool = True,
    plot_scatter_kwargs: dict = None,
    plot_contours: bool = True,
    plot_contours_kwargs: dict = None,
    plot_arrows: bool = True,
    plot_arrows_kwargs: dict = None,
    arrow_scale: float | None = 1.0,
    show_colorbar: bool = True,
    cmap: str = "plasma",
    norm: mpcolors.Normalize = None,
    use_norm: bool = False,
    clim: tuple = (None, None),
    colorbar_kwargs: dict = None,
    colorbar_tick_kwargs: dict = None,
    colorbar_tick_params_kwargs: dict = None,
    colorbar_label_kwargs: dict = None,
    add_legend: bool = False,
    show: bool = True,
    savefig: str = None,
    dpi: int | str = "figure",
    savefig_kwargs: dict = None,
    xlabel_kwargs: dict = None,
    ylabel_kwargs: dict = None,
    xlim_kwargs: dict = None,
    ylim_kwargs: dict = None,
    x_major_tick_params_kwargs: dict = None,
    y_major_tick_params_kwargs: dict = None,
    x_minor_tick_params_kwargs: dict = None,
    y_minor_tick_params_kwargs: dict = None,
    major_tick_params_kwargs: dict = None,
    minor_tick_params_kwargs: dict = None,
    x_ticks_kwargs: dict = None,
    y_ticks_kwargs: dict = None,
    figsize: tuple = (6, 6),
    aspect: float | str = "equal",
    set_aspect_kwargs: dict = None,
    ax: plt.Axes | None = None,
    **kwargs,
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
        Energy shift to apply to the Fermi level in eV, by default 0.0
    band_indices : List[List], optional
        List of band indices for each spin channel to include in the plot, 
        by default None (all bands)
    band_colors : List[List], optional
        List of colors corresponding to each band index for each spin channel, 
        by default None
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
        Filename to save the figure. If None, the figure is not saved, by default None
    dpi : int or str, optional
        Resolution for saved figure. Can be integer DPI or 'figure', by default 'figure'
    savefig_kwargs : dict, optional
        Additional keyword arguments for figure saving, by default None
    xlabel_kwargs : dict, optional
        Additional keyword arguments for x-axis label formatting, by default None
    ylabel_kwargs : dict, optional
        Additional keyword arguments for y-axis label formatting, by default None
    xlim_kwargs : dict, optional
        Additional keyword arguments for x-axis limits, by default None
    ylim_kwargs : dict, optional
        Additional keyword arguments for y-axis limits, by default None
    x_major_tick_params_kwargs : dict, optional
        Additional keyword arguments for x-axis major tick parameters, by default None
    y_major_tick_params_kwargs : dict, optional
        Additional keyword arguments for y-axis major tick parameters, by default None
    x_minor_tick_params_kwargs : dict, optional
        Additional keyword arguments for x-axis minor tick parameters, by default None
    y_minor_tick_params_kwargs : dict, optional
        Additional keyword arguments for y-axis minor tick parameters, by default None
    major_tick_params_kwargs : dict, optional
        Additional keyword arguments for major tick parameters, by default None
    minor_tick_params_kwargs : dict, optional
        Additional keyword arguments for minor tick parameters, by default None
    x_ticks_kwargs : dict, optional
        Additional keyword arguments for x-axis tick parameters, by default None
    y_ticks_kwargs : dict, optional
        Additional keyword arguments for y-axis tick parameters, by default None
    figsize : tuple, optional
        Figure size as (width, height) in inches, by default (6, 6)
    aspect: float | str, optional
        Aspect ratio of the plot, by default "equal"
    set_aspect_kwargs : dict, optional
        Additional keyword arguments for set_aspect, by default None
    ax : matplotlib.pyplot.Axes, optional
        Existing axes object to plot on. If None, creates new figure, by default None
    **kwargs
        Additional keyword arguments passed to the FermiSurface class

    Returns
    -------
    FermiSurface or None
        Returns the FermiSurface object if show=False, otherwise returns None

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

    if len(translate) != 3 and len(translate) != 1:
        logger.error(f"Error: --translate option is invalid! ({translate})")
        raise RuntimeError("invalid option --translate")
    
    mode = FermiSurface2DMode.from_str(mode)

    user_logger.info(f"dirname         : {dirname}")
    user_logger.info(f"mode            : {mode.value}")
    user_logger.info(f"bands           : {band_indices}")
    user_logger.info(f"atoms           : {atoms}")
    user_logger.info(f"orbitals        : {orbitals}")
    user_logger.info(f"spin comp.      : {spins}")
    user_logger.info(f"energy          : {energy}")
    user_logger.info(f"rot. symmetry   : {rot_symm}")
    user_logger.info(f"origin (trasl.) : {translate}")
    user_logger.info(f"rotation        : {rotation}")
    user_logger.info(f"save figure     : {savefig}")
    user_logger.info("_" * 100)

    ebs_pkl_filepath = os.path.join(dirname, "ebs.pkl")
    if not use_cache and os.path.exists(ebs_pkl_filepath):
        logger.info(f"Removing existing EBS file: {ebs_pkl_filepath}")
        os.remove(ebs_pkl_filepath)

    if not os.path.exists(ebs_pkl_filepath):
        logger.info(f"Parsing EBS from {dirname}")

        parser = io.Parser(code=code, dirpath=dirname)
        ebs = parser.ebs
        structure = parser.structure

        if structure.rotations is not None:
            logger.info(
                f"Detected symmetry operations ({structure.rotations.shape}). Applying to ebs to get full BZ"
            )
            ebs.ibz2fbz(structure.rotations)

        data_utils.save_pickle(ebs, ebs_pkl_filepath)

    else:
        logger.info(f"Loading EBS from cached Pickle files in {dirname}")

        ebs = data_utils.load_pickle(ebs_pkl_filepath)

    codes_with_scf_fermi = ["qe", "elk"]
    if code in codes_with_scf_fermi and fermi is None:
        logger.info(f"No fermi given, using the found fermi energy: {ebs.efermi}")
        fermi = ebs.efermi

    if fermi is not None:
        logger.info(f"Shifting Fermi energy to zero: {fermi}")
        ebs.bands -= fermi
        ebs.bands += fermi_shift
        fermi_level = fermi_shift
    else:
        user_logger.warning(
            "`fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy."
        )

    logger.debug(f"EBS: {str(ebs)}")

    # Shifting all kpoint to first Brillouin zone
    bound_ops = -1.0 * (ebs.kpoints > 0.5) + 1.0 * (ebs.kpoints <= -0.5)
    ebs.kpoints = ebs.kpoints + bound_ops
    
    kpoints = ElectronicBandStructure.reduced_to_cartesian(ebs.kpoints, 2*np.pi*ebs.reciprocal_lattice)
    # kpoints = ebs.kpoints_cartesian
    
    

    if spins is None:
        spins = np.arange(ebs.bands.shape[-1])
    if energy is None:
        energy = 0

    ### End of parsing ###
    # Selecting kpoints in a constant k_z plane
    i_kpoints_near_z_0 = np.where(
        np.logical_and(
            kpoints[:, 2] < k_z_plane + k_z_plane_tol,
            kpoints[:, 2] > k_z_plane - k_z_plane_tol,
        )
    )

    user_logger.info(f"Initial kpoints shape: {ebs.kpoints.shape}")
    user_logger.info(f"Initial bands shape: {ebs.bands.shape}")
    user_logger.info(f"Initial projected shape: {ebs.projected.shape}")

    kpoints = kpoints[i_kpoints_near_z_0, :][0]
    ebs.bands = ebs.bands[i_kpoints_near_z_0, :][0]
    ebs.projected = ebs.projected[i_kpoints_near_z_0, :][0]

    user_logger.warning(
        f"Make sure the kmesh has the correct number of kz points"
        f"with kz={k_z_plane} +- {k_z_plane_tol}."
    )

    user_logger.info(f"Kpoints in the kz={k_z_plane} plane: {kpoints.shape}")
    user_logger.info(f"Bands in the kz={k_z_plane} plane: {ebs.bands.shape}")
    user_logger.info(f"Projected in the kz={k_z_plane} plane: {ebs.projected.shape}")

    if mode != FermiSurface2DMode.SPIN_TEXTURE:
        # processing the data
        if orbitals is None and ebs.projected is not None:
            orbitals = np.arange(ebs.norbitals, dtype=int)
        if atoms is None and ebs.projected is not None:
            atoms = np.arange(ebs.natoms, dtype=int)

        user_logger.info(f"Spins for projections: {spins}")
        user_logger.info(f"Atoms for projections: {atoms}")
        user_logger.info(f"Orbitals for projections: {orbitals}")

        projected = ebs.ebs_sum(
            spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False
        )
        projected = projected[:, :, spins]

        logger.info(f"projection shape after ebs_sum: {projected.shape}")
    else:
        # first get the sdp reduced array for all spin components.
        stData = []
        ebsX = copy.deepcopy(ebs)
        ebsY = copy.deepcopy(ebs)
        ebsZ = copy.deepcopy(ebs)

        projected = ebs.ebs_sum(
            spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False
        )
        ebsX.projected = copy.deepcopy(projected)[:, :, [1]][:, :, 0]
        ebsY.projected = copy.deepcopy(projected)[:, :, [2]][:, :, 0]
        ebsZ.projected = copy.deepcopy(projected)[:, :, [3]][:, :, 0]

        logger.info(f"ebsX.projected shape after ebs_sum: {ebsX.projected.shape}")
        logger.info(f"ebsY.projected shape after ebs_sum: {ebsY.projected.shape}")
        logger.info(f"ebsZ.projected shape after ebs_sum: {ebsZ.projected.shape}")

        stData.append(ebsX.projected)
        stData.append(ebsY.projected)
        stData.append(ebsZ.projected)

    if ebs.is_non_collinear:
        spin_channels = [0]
    else:
        spin_channels = spins

    bands = ebs.bands[..., spin_channels]
    character = projected

    if mode == FermiSurface2DMode.SPIN_TEXTURE:
        sx, sy, sz = stData[0], stData[1], stData[2]
        symm = ProcarSymmetry(kpoints, bands, sx=sx, sy=sy, sz=sz, character=character)
    else:
        symm = ProcarSymmetry(kpoints, bands, character=character)

    symm.translate(translate)
    symm.general_rotation(rotation[0], rotation[1:])
    # symm.MirrorX()
    symm.rot_symmetry_z(rot_symm)

    fs = FermiSurface(
        symm.kpoints,
        symm.bands,
        symm.character,
        figsize=figsize,
        **kwargs,
    )
    fs.find_energy(energy)

    if mode in [FermiSurface2DMode.PLAIN, FermiSurface2DMode.PLAIN_BANDS]:
        
        bands_spin_contour_data = fs.generate_contours(band_indices=band_indices, 
                                                       interpolation=interpolation, 
                                                       ignore_scalars=True)
        
        linecollection_kwargs = {} if linecollection_kwargs is None else linecollection_kwargs
        fs.plot_band_spin_contour_line_segments(bands_spin_contour_data=bands_spin_contour_data,
                                                linestyles=linestyles,
                                                colors=colors,
                                                linewidths=linewidths,
                                                alphas=alphas,
                                                cmap=cmap, 
                                                norm=norm,
                                                clim=clim,
                                                line_collection_kwargs=linecollection_kwargs,
                                                )
    

    elif mode == FermiSurface2DMode.PARAMETRIC:
        bands_spin_contour_data = fs.generate_contours(band_indices=band_indices, interpolation=interpolation, ignore_scalars=False)
        linecollection_kwargs = {} if linecollection_kwargs is None else linecollection_kwargs
        fs.plot_band_spin_contour_line_segments(bands_spin_contour_data=bands_spin_contour_data,
                                                linestyles=linestyles,
                                                colors=colors,
                                                linewidths=linewidths,
                                                alphas=alphas,
                                                cmap=cmap, 
                                                norm=norm,
                                                clim=clim,
                                                line_collection_kwargs=linecollection_kwargs,
                                                )
    
        if show_colorbar and mode not in [FermiSurface2DMode.PLAIN, FermiSurface2DMode.PLAIN_BANDS]:
            
            colorbar_kwargs = {} if colorbar_kwargs is None else colorbar_kwargs
            fs.show_colorbar(
                label="Atomic Orbital Projection",
                cmap=cmap,
                norm=norm,
                clim=clim,
                colorbar_kwargs=colorbar_kwargs,
            )
            

    elif mode == FermiSurface2DMode.SPIN_TEXTURE:
        spin_texture_contour_data = fs.generate_spin_texture_contours(
            sx=sx,
            sy=sy,
            sz=sz,
            band_indices=band_indices,
            point_density=point_density,
            spin_projection=spin_projection,
            interpolation=interpolation,
        )
        
        fs.set_scalar_mappable(norm=norm, clim=clim, cmap=cmap)
        
        if plot_contours:
            plot_contours_kwargs = {} if plot_contours_kwargs is None else plot_contours_kwargs
            fs.plot_spin_texture_contours(spin_texture_contour_data, **plot_contours_kwargs)

        if plot_arrows:
            plot_arrows_kwargs = {} if plot_arrows_kwargs is None else plot_arrows_kwargs
            if arrow_scale is not None:
                plot_arrows_kwargs.setdefault("scale", arrow_scale)
            fs.plot_spin_texture_arrows(spin_texture_contour_data, **plot_arrows_kwargs)
        if plot_scatter:
            plot_scatter_kwargs = {} if plot_scatter_kwargs is None else plot_scatter_kwargs
            fs.plot_spin_texture_scatter(spin_texture_contour_data, **plot_scatter_kwargs)
            
            
        if show_colorbar:
            
            colorbar_kwargs = {} if colorbar_kwargs is None else colorbar_kwargs
            fs.show_colorbar(
                label=SpinProjection.from_str(spin_projection).value,
                cmap=cmap,
                clim=clim,
                colorbar_kwargs=colorbar_kwargs,
            )
            
            
    if hasattr(fs, "colorbar"):
        colorbar_tick_kwargs = {} if colorbar_tick_kwargs is None else colorbar_tick_kwargs
        fs.set_colorbar_ticks(**colorbar_tick_kwargs)
        
        colorbar_tick_params_kwargs = {} if colorbar_tick_params_kwargs is None else colorbar_tick_params_kwargs
        fs.set_colorbar_tick_params(**colorbar_tick_params_kwargs)
        
        colorbar_label_kwargs = {} if colorbar_label_kwargs is None else colorbar_label_kwargs
        fs.set_colorbar_label(**colorbar_label_kwargs)
        
    xlabel_kwargs = {} if xlabel_kwargs is None else xlabel_kwargs
    ylabel_kwargs = {} if ylabel_kwargs is None else ylabel_kwargs
    xlim_kwargs = {} if xlim_kwargs is None else xlim_kwargs
    ylim_kwargs = {} if ylim_kwargs is None else ylim_kwargs
 
    fs.set_xlabel(**xlabel_kwargs)
    fs.set_ylabel(**ylabel_kwargs)
    fs.set_xlim(**xlim_kwargs)
    fs.set_ylim(**ylim_kwargs)
    
    if x_ticks_kwargs is not None:
        fs.set_xticks(**x_ticks_kwargs)
    if y_ticks_kwargs is not None:
        fs.set_yticks(**y_ticks_kwargs)
    
    if x_major_tick_params_kwargs is not None:
        fs.set_tick_params(axis = "x", which = "major", **x_major_tick_params_kwargs)
    if x_minor_tick_params_kwargs is not None:
        fs.set_tick_params(axis = "x", which = "minor", **x_minor_tick_params_kwargs)
    if y_major_tick_params_kwargs is not None:
        fs.set_tick_params(axis = "y", which = "major", **y_major_tick_params_kwargs)
    if y_minor_tick_params_kwargs is not None:
        fs.set_tick_params(axis = "y", which = "minor", **y_minor_tick_params_kwargs)
    
    if major_tick_params_kwargs is not None:
        fs.set_tick_params(which = "major", **major_tick_params_kwargs)
    if minor_tick_params_kwargs is not None:
        fs.set_tick_params(which = "minor", **minor_tick_params_kwargs)
    
    
    
    set_aspect_kwargs = {} if set_aspect_kwargs is None else set_aspect_kwargs
    
    fs.set_aspect(aspect=aspect, **set_aspect_kwargs)
    
    if add_legend and mode in [FermiSurface2DMode.PLAIN, FermiSurface2DMode.PLAIN_BANDS]:
        fs.add_legend()
        
    if savefig:
        savefig_kwargs = {} if savefig_kwargs is None else savefig_kwargs
        fs.savefig(savefig, dpi=dpi, **savefig_kwargs)

    if show:
        fs.show()
    else:
        return fs