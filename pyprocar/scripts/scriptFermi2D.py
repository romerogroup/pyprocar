__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import copy
import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import cm
from matplotlib import colors as mpcolors

from pyprocar import io
from pyprocar.core import FermiSurface, ProcarSymmetry
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
    k_z_plane_tol: float = 0.01,
    rot_symm=1,
    translate: List[int] = [0, 0, 0],
    rotation: List[int] = [0, 0, 0, 1],
    spin_texture: bool = False,
    exportplt: bool = False,
    savefig: str = None,
    print_plot_opts: bool = False,
    use_cache: bool = True,
    verbose: int = 1,
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
    lobster : bool, optional
        A boolean value to determine to use lobster, by default False
    band_indices : List[List]
        A list of list that contains band indices for a given spin
    band_colors : List[List]
            A list of list that contains colors for the band index
            corresponding the band_indices for a given spin
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

    Returns
    -------
    matplotlib.pyplot
        Returns the matplotlib.pyplot state plt

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
    user_logger.info(f"rot. symmetry   : {rot_symm}")
    user_logger.info(f"origin (trasl.) : {translate}")
    user_logger.info(f"rotation        : {rotation}")
    user_logger.info(f"save figure     : {savefig}")
    user_logger.info(f"spin_texture    : {spin_texture}")
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
    kpoints = ebs.kpoints_cartesian

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

    if spin_texture is False:
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

    # Once the PROCAR is parsed and reduced to 2x2 arrays, we can apply
    # symmetry operations to unfold the Brillouin Zone
    # kpoints = data.kpoints
    # bands = data.bands
    # character = data.spd
    if ebs.is_non_collinear:
        spin_channels = [0]
    else:
        spin_channels = spins

    bands = ebs.bands[..., spin_channels]
    character = projected

    if spin_texture is True:
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
        band_indices=band_indices,
        band_colors=band_colors,
        **kwargs,
    )
    fs.find_energy(energy)

    if not spin_texture:
        fs.plot(mode=mode, interpolation=300)
    else:
        fs.spin_texture(sx=symm.sx, sy=symm.sy, sz=symm.sz, spin=spins[0])

    fs.add_axes_labels()
    fs.add_legend()

    if exportplt:
        return plt

    else:
        if savefig:
            fs.savefig(savefig)
        else:
            fs.show()
        return
