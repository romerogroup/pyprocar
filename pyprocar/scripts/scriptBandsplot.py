_author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pyprocar import io
from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.plotter import EBSPlot
from pyprocar.utils import data_utils, np_utils, welcome
from pyprocar.utils.info import orbital_names
from pyprocar.utils.log_utils import set_verbose_level

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


def bandsplot(
    code: str,
    dirname: str,
    mode: str = "plain",
    spins: List[int] = None,
    atoms: List[int] = None,
    orbitals: List[int] = None,
    items: dict = {},
    fermi: float = None,
    fermi_shift: float = 0,
    interpolation_factor: int = 1,
    interpolation_type: str = "cubic",
    projection_mask: np.ndarray = None,
    kticks=None,
    knames=None,
    kdirect: bool = True,
    elimit: List[float] = None,
    ax: plt.Axes = None,
    show: bool = True,
    savefig: str = None,
    print_plot_opts: bool = False,
    export_data_file: str = None,
    export_append_mode: bool = True,
    ktick_limit: List[float] = None,
    x_limit: List[float] = None,
    use_cache: bool = False,
    quiet_welcome: bool = False,
    **kwargs,
):
    """A function to plot the band structutre

    Parameters
    ----------
    code : str, optional
        String to of the code used, by default "vasp"
    dirname : str, optional
        The directory name of the calculation, by default None
    mode : str, optional
        Sting for the mode of the calculation, by default "plain"
    spins : List[int], optional
        A list of spins, by default None
    atoms : List[int], optional
        A list of atoms, by default None
    orbitals : List[int], optional
        A list of orbitals, by default None
    items : dict, optional
        A dictionary where the keys are the atoms and the values a list of orbitals, by default {}
    fermi : float, optional
        Float for the fermi energy, by default None. By default the fermi energy will be shifted by the fermi value that is found in the directory.
        For band structure calculations, due to convergence issues, this fermi energy might not be accurate. If so add the fermi energy from the self-consistent calculation.
    fermi_shift : float, optional
        Float to shift the fermi energy, by default 0.
    interpolation_factor : int, optional
        The interpolation_factor, by default 1
    interpolation_type : str, optional
        The interpolation type, by default "cubic"
    projection_mask : np.ndarray, optional
        A custom projection mask, by default None
    kticks : _type_, optional
        A list of kticks, by default None
    knames : _type_, optional
        A list of kanems, by default None
    elimit : List[float], optional
        A list of floats to decide the energy window, by default None
    ax : plt.Axes, optional
        A matplotlib axes, by default None
    show : bool, optional
        Boolean if to show the plot, by default True
    savefig : str, optional
        String to save the plot, by default None
    export_data_file : str, optional
        The file name to export the data to. If not provided the
        data will not be exported.
    export_append_mode : bool, optional
        Boolean to append the mode to the file name. If not provided the
        data will be overwritten.
    print_plot_opts: bool, optional
        Boolean to print the plotting options
    quiet_welcome: bool, optional
        Boolean to not print the welcome message
    use_cache: bool, optional
        Boolean to use cache for EBS

    """

    user_logger.info(f"If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    if not quiet_welcome:
        welcome()

    default_config = ConfigFactory.create_config(PlotType.BAND_STRUCTURE)
    config = ConfigManager.merge_configs(default_config, kwargs)

    user_logger.info("_" * 100)
    modes_txt = " , ".join(config.modes)
    message = f"""
            There are additional plot options that are defined in the configuration file. 
            You can change these configurations by passing the keyword argument to the function.
            To print a list of all plot options set `print_plot_opts=True`

            Here is a list modes : {modes_txt}
            """
    if not quiet_welcome:
        user_logger.info(message)

    if print_plot_opts:
        for key, value in default_config.as_dict().items():
            user_logger.info(f"{key} : {value}")

    user_logger.info("_" * 100)

    ebs_pkl_filepath = os.path.join(dirname, "ebs.pkl")
    structure_pkl_filepath = os.path.join(dirname, "structure.pkl")
    kpath_pkl_filepath = os.path.join(dirname, "kpath.pkl")

    if not use_cache:
        user_logger.warning(f"Not using cache, removing existing cache files")
        if os.path.exists(structure_pkl_filepath):
            logger.info(f"Removing existing structure file: {structure_pkl_filepath}")
            os.remove(structure_pkl_filepath)
        if os.path.exists(kpath_pkl_filepath):
            logger.info(f"Removing existing kpath file: {kpath_pkl_filepath}")
            os.remove(kpath_pkl_filepath)
        if os.path.exists(ebs_pkl_filepath):
            logger.info(f"Removing existing EBS file: {ebs_pkl_filepath}")
            os.remove(ebs_pkl_filepath)

    if not os.path.exists(ebs_pkl_filepath):
        logger.info(f"Parsing EBS from {dirname}")

        parser = io.Parser(code=code, dirpath=dirname)
        ebs = parser.ebs
        structure = parser.structure
        kpath = ebs.kpath

        data_utils.save_pickle(ebs, ebs_pkl_filepath)
        data_utils.save_pickle(structure, structure_pkl_filepath)
    else:
        logger.info(
            f"Loading EBS, Structure, and Kpath from cached Pickle files in {dirname}"
        )

        ebs = data_utils.load_pickle(ebs_pkl_filepath)
        structure = data_utils.load_pickle(structure_pkl_filepath)
        kpath = ebs.kpath

    codes_with_scf_fermi = ["qe", "elk"]
    if code in codes_with_scf_fermi and fermi is None:
        logger.info(f"No fermi given, using the found fermi energy: {ebs.efermi}")

        fermi = ebs.efermi

    if fermi is not None:
        logger.info(f"Shifting Fermi energy to zero: {fermi}")

        ebs.bands -= fermi
        ebs.bands += fermi_shift
        fermi_level = fermi_shift
        y_label = r"E - E$_F$ (eV)"
    else:
        y_label = r"E (eV)"
        user_logger.warning(
            "`fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy."
        )

    # fixing the spin, to plot two channels into one (down is negative)
    if np.array_equal(spins, [-1, 1]) or np.array_equal(spins, [1, -1]):
        if ebs.fix_collinear_spin():
            spins = [0]

    ebs_plot = EBSPlot(ebs, kpath, ax, spins, kdirect=kdirect, config=config)

    projection_labels = []
    labels = []
    if mode == "plain":
        user_logger.info("Plotting bands in plain mode")
        ebs_plot.plot_bands()

    elif mode == "ipr":
        user_logger.info("Plotting bands in IPR mode")
        weights = ebs_plot.ebs.ebs_ipr()
        if config.weighted_color:
            color_weights = weights
        else:
            color_weights = None
        if config.weighted_width:
            width_weights = weights
        else:
            width_weights = None
        color_mask = projection_mask
        width_mask = projection_mask

        ebs_plot.plot_parameteric(
            color_weights=color_weights,
            width_weights=width_weights,
            color_mask=color_mask,
            width_mask=width_mask,
            spins=spins,
            elimit=elimit,
        )
        ebs_plot.set_colorbar_title(title="Inverse Participation Ratio")

    elif mode in ["overlay", "overlay_species", "overlay_orbitals"]:
        weights = []
        if mode == "overlay_species":
            if orbitals is None:
                orbitals = list(np.arange(len(ebs_plot.ebs.projected[0][0]), dtype=int))

            user_logger.info("Plotting bands in overlay species mode")
            for ispc in structure.species:
                labels.append(ispc)
                atoms = np.where(structure.atoms == ispc)[0]

                projection_label = f"atom-{ispc}_orbitals-" + ",".join(
                    str(x) for x in orbitals
                )
                projection_labels.append(projection_label)
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )
                weights.append(w)
        if mode == "overlay_orbitals":
            user_logger.info("Plotting bands in overlay orbitals mode")
            for iorb, orb in enumerate(["s", "p", "d", "f"]):
                if orb == "f" and not ebs_plot.ebs.norbitals > 9:
                    continue
                orbitals = orbital_names[orb]
                labels.append(orb)

                atom_label = ""
                if atoms:
                    atom_labels = ",".join(str(x) for x in atoms)
                    atom_label = f"atom-{atom_labels}_"
                projection_label = f"{atom_label}orbitals-{orb}"
                projection_labels.append(projection_label)
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )

                weights.append(w)

        elif mode == "overlay":
            user_logger.info("Plotting bands in overlay mode")
            if isinstance(items, dict):
                items = [items]

            if isinstance(items, list):
                for it in items:
                    for ispc in it:
                        atoms = np.where(structure.atoms == ispc)[0]
                        if isinstance(it[ispc][0], str):
                            orbitals = []
                            for iorb in it[ispc]:
                                orbitals = np.append(
                                    orbitals, orbital_names[iorb]
                                ).astype(int)
                            labels.append(ispc + "-" + "".join(it[ispc]))
                        else:
                            orbitals = it[ispc]
                            labels.append(
                                ispc + "-" + "_".join(str(x) for x in it[ispc])
                            )

                        atom_labels = ",".join(str(x) for x in atoms)
                        orbital_labels = ",".join(str(x) for x in orbitals)
                        projection_label = (
                            f"atoms-{atom_labels}_orbitals-{orbital_labels}"
                        )
                        projection_labels.append(projection_label)
                        w = ebs_plot.ebs.ebs_sum(
                            atoms=atoms,
                            principal_q_numbers=[-1],
                            orbitals=orbitals,
                            spins=spins,
                        )
                        weights.append(w)
        ebs_plot.plot_parameteric_overlay(
            spins=spins, weights=weights, labels=projection_labels
        )
    else:

        if atoms is not None and isinstance(atoms[0], str):
            atoms_str = atoms
            atoms = []
            for iatom in np.unique(atoms_str):
                atoms = np.append(atoms, np.where(structure.atoms == iatom)[0]).astype(
                    np_utils.INT_DTYPE
                )

        if orbitals is not None and isinstance(orbitals[0], str):
            orbital_str = orbitals

            orbitals = []
            for iorb in orbital_str:
                orbitals = np.append(orbitals, orbital_names[iorb]).astype(np_utils.INT_DTYPE)

        projection_labels = []
        projection_label = ""
        atoms_labels = ""
        if atoms:
            atoms_labels = ",".join(str(x) for x in atoms)
            projection_label += f"atoms-{atoms_labels}"
        orbital_labels = ""
        if orbitals:
            orbital_labels = ",".join(str(x) for x in orbitals)
            if len(projection_label) != 0:
                projection_label += "_"
        projection_label += f"orbitals-{orbital_labels}"
        projection_labels.append(projection_label)

        weights = ebs_plot.ebs.ebs_sum(
            atoms=atoms, principal_q_numbers=[-1], orbitals=orbitals, spins=spins
        )
        if config.weighted_color:
            color_weights = weights
        else:
            color_weights = None
        if config.weighted_width:
            width_weights = weights
        else:
            width_weights = None
        color_mask = projection_mask
        width_mask = projection_mask
        if mode == "parametric":
            user_logger.info("Plotting bands in parametric mode")
            ebs_plot.plot_parameteric(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins,
                labels=projection_labels,
            )
            ebs_plot.set_colorbar_title()
        elif mode == "scatter":
            user_logger.info("Plotting bands in scatter mode")
            ebs_plot.plot_scatter(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins,
                labels=projection_labels,
            )
            ebs_plot.set_colorbar_title()
        elif mode == "atomic":
            user_logger.info("Plotting bands in atomic mode")
            if ebs.kpoints.shape[0] != 1:
                raise Exception("Must use a single kpoint")
            if color_weights is not None:
                color_weights = np.vstack((color_weights, color_weights))
            ebs_plot.plot_atomic_levels(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins,
                elimit=elimit,
                labels=projection_labels,
            )

            ebs_plot.set_xlabel(label=config.x_label)
            ebs_plot.set_colorbar_title()

        else:
            user_logger.warning(
                f"Selected mode {mode} not valid. Please check the spelling"
            )

    ebs_plot.set_xticks(kticks, knames)
    ebs_plot.set_yticks(interval=elimit)
    ebs_plot.set_xlim(interval=x_limit, ktick_interval=ktick_limit)
    ebs_plot.set_ylim(elimit)
    ebs_plot.set_ylabel(label=y_label)
    ebs_plot.set_xlabel(label=config.x_label)

    if fermi is not None:
        ebs_plot.draw_fermi(fermi_level=fermi_level)

    ebs_plot.set_title()
    ebs_plot.grid()

    ebs_plot.legend(labels)

    if savefig is not None:
        ebs_plot.save(savefig)
    if show:
        ebs_plot.show()

    if export_data_file is not None:
        if export_append_mode:
            file_basename, file_type = export_data_file.split(".")
            filename = f"{file_basename}_{mode}.{file_type}"
        else:
            filename = export_data_file
        ebs_plot.export_data(filename)

    return ebs_plot.fig, ebs_plot.ax
