import logging
import os

import numpy as np
import yaml

from pyprocar import io
from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.plotter import EBSPlot
from pyprocar.utils import ROOT, data_utils, welcome
from pyprocar.utils.info import orbital_names
from pyprocar.utils.log_utils import set_verbose_level

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


with open(os.path.join(ROOT, "pyprocar", "cfg", "unfold.yml"), "r") as file:
    plot_opt = yaml.safe_load(file)


def unfold(
    code="vasp",
    dirname=".",
    mode="plain",
    unfold_mode="both",
    transformation_matrix=np.diag([2, 2, 2]),
    spins=None,
    atoms=None,
    orbitals=None,
    items=None,
    projection_mask=None,
    unfold_mask=None,
    fermi=None,
    fermi_shift=0,
    interpolation_factor=1,
    interpolation_type="cubic",
    vmax=None,
    vmin=None,
    kticks=None,
    knames=None,
    kdirect=True,
    elimit=None,
    ax=None,
    show=True,
    savefig=None,
    old=False,
    savetab="unfold_result.csv",
    print_plot_opts: bool = False,
    use_cache: bool = False,
    verbose: int = 1,
    **kwargs,
):
    """

    Parameters
    ----------
    fname: PROCAR filename.
    poscar: POSCAR filename
    outcar: OUTCAR filename, for reading fermi energy. You can also use efermi and set outcar=None
    supercell_matrix: supercell matrix from primitive cell to supercell
    ispin: For non-spin polarized system, ispin=None.
       For spin polarized system: ispin=1 is spin up, ispin=2 is spin down.
    fermi: Fermi energy
    fermi_shift: Shift the bands by the Fermi energy.
    elimit: range of energy to be plotted.
    kticks: the indices of K points which has labels given in knames.
    knames: see kticks
    print_kpts: print all the kpoints to screen. This is to help find the kticks and knames.
    show_band: whether to plot the bands before unfolding.
    width: the width of the unfolded band.
    color: color of the unfoled band.
    savetab: the csv file name of which  the table of unfolding result will be written into.
    savefig: the file name of which the figure will be saved.
    exportplt: flag to export plot as matplotlib.pyplot object.
    use_cache: flag to use cache for parsed data.
    verbose: verbosity level.
    """
    set_verbose_level(verbose)

    user_logger.info(f"If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    welcome()
    if vmin is not None and vmax is not None:
        kwargs["clim"] = (vmin, vmax)
    default_config = ConfigFactory.create_config(PlotType.UNFOLD)
    config = ConfigManager.merge_configs(default_config, kwargs)
    modes_txt = " , ".join(config.modes)

    message = f"""
            There are additional plot options that are defined in a configuration file.
            You can change these configurations by passing the keyword argument to the function
            To print a list of plot options set print_plot_opts=True

            Here is a list modes : {modes_txt}
            """
    user_logger.info(message)
    if print_plot_opts:
        for key, value in plot_opt.items():
            user_logger.info(f"{key} : {value}")

    user_logger.info("_" * 100)

    # Creating pickle files for cache parsed data
    ebs_pkl_filepath = os.path.join(dirname, "ebs.pkl")
    structure_pkl_filepath = os.path.join(dirname, "structure.pkl")
    kpath_pkl_filepath = os.path.join(dirname, "kpath.pkl")

    if not use_cache:
        if os.path.exists(ebs_pkl_filepath):
            logger.info(f"Removing existing EBS file: {ebs_pkl_filepath}")
            os.remove(ebs_pkl_filepath)
        if os.path.exists(structure_pkl_filepath):
            logger.info(f"Removing existing structure file: {structure_pkl_filepath}")
            os.remove(structure_pkl_filepath)
        if os.path.exists(kpath_pkl_filepath):
            logger.info(f"Removing existing kpath file: {kpath_pkl_filepath}")
            os.remove(kpath_pkl_filepath)

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

    if fermi is not None:
        ebs.bands -= fermi
        ebs.bands += fermi_shift
        fermi_level = fermi_shift
        y_label = r"E - E$_F$ (eV)"
    else:
        y_label = r"E (eV)"
        print(
            """
            WARNING : `fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy.
            ----------------------------------------------------------------------------------------------------------
            """
        )

    ebs_plot = EBSPlot(ebs, kpath, ax, spins, config=config)

    labels = None

    if mode is not None:
        if ebs.projected_phase is None:
            raise ValueError(
                "The provided electronic band structure file does not include phases"
            )
        ebs_plot.ebs.unfold(
            transformation_matrix=transformation_matrix, structure=structure
        )
    if unfold_mode == "both":
        logger.info("Unfolding bands in both modes")

        width_weights = ebs_plot.ebs.weights
        width_mask = unfold_mask
        color_weights = ebs_plot.ebs.weights
        color_mask = unfold_mask
    elif unfold_mode == "thickness":
        logger.info("Unfolding bands in thickness mode")

        width_weights = ebs_plot.ebs.weights
        width_mask = unfold_mask
        color_weights = None
        color_mask = None
    elif unfold_mode == "color":
        logger.info("Unfolding bands in color mode")

        width_weights = None
        width_mask = None
        color_weights = ebs_plot.ebs.weights
        color_mask = unfold_mask
    else:
        raise ValueError(
            f"Invalid unfold_mode was selected: {unfold_mode} please select from the following 'both', 'thickness','color'"
        )

    if color_weights is not None:
        logger.debug(f"color_weights shape: {color_weights.shape}")
    if width_weights is not None:
        logger.debug(f"width_weights shape: {width_weights.shape}")

    labels = []
    if mode == "plain":
        logger.info("Plotting bands in plain mode")

        ebs_plot.plot_bands()
        ebs_plot.plot_parameteric(
            color_weights=color_weights,
            width_weights=width_weights,
            color_mask=color_mask,
            width_mask=width_mask,
            spins=spins,
        )
        ebs_plot.handles = ebs_plot.handles[: ebs_plot.nspins]
    elif mode in ["overlay", "overlay_species", "overlay_orbitals"]:

        weights = []

        if mode == "overlay_species":
            logger.info("Plotting bands in overlay species mode")

            for ispc in structure.species:
                labels.append(ispc)
                atoms = np.where(structure.atoms == ispc)[0]
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )
                weights.append(w)
        if mode == "overlay_orbitals":
            logger.info("Plotting bands in overlay orbitals mode")

            for iorb in ["s", "p", "d", "f"]:
                if iorb == "f" and not ebs_plot.ebs.norbitals > 9:
                    continue
                labels.append(iorb)
                orbitals = orbital_names[iorb]
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )
                weights.append(w)

        elif mode == "overlay":
            logger.info("Plotting bands in overlay mode")

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
                            labels.append(ispc + "-" + "_".join(it[ispc]))
                        w = ebs_plot.ebs.ebs_sum(
                            atoms=atoms,
                            principal_q_numbers=[-1],
                            orbitals=orbitals,
                            spins=spins,
                        )
                        weights.append(w)
        ebs_plot.plot_parameteric_overlay(spins=spins, weights=weights, labels=labels)
    else:
        if atoms is not None and isinstance(atoms[0], str):
            atoms_str = atoms
            atoms = []
            for iatom in np.unique(atoms_str):
                atoms = np.append(atoms, np.where(structure.atoms == iatom)[0]).astype(
                    int
                )

        if orbitals is not None and isinstance(orbitals[0], str):
            orbital_str = orbitals

            orbitals = []
            for iorb in orbital_str:
                orbitals = np.append(orbitals, orbital_names[iorb]).astype(int)

        projection_labels = []
        projection_label = ""
        atoms_labels = ""
        if atoms is not None:
            atoms_labels = ",".join(str(x) for x in atoms)
            projection_label += f"atoms-{atoms_labels}"
        orbital_labels = ""
        if orbitals is not None:
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
        width_mask = unfold_mask
        width_weights = ebs_plot.ebs.weights
        if mode == "parametric":
            logger.info("Plotting bands in parametric mode")

            ebs_plot.plot_parameteric(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins,
                labels=projection_labels,
            )
        elif mode == "scatter":
            logger.info("Plotting bands in scatter mode")

            ebs_plot.plot_scatter(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                spins=spins,
                labels=projection_labels,
            )

        else:
            user_logger.warning(
                f"Selected mode {mode} not valid. Please check the spelling"
            )

    ebs_plot.set_xticks(kticks, knames)
    ebs_plot.set_yticks(interval=elimit)
    ebs_plot.set_xlim()
    ebs_plot.set_ylim(elimit)
    if fermi is not None:
        ebs_plot.draw_fermi(fermi_level=fermi_level)
    ebs_plot.set_ylabel(label=y_label)

    ebs_plot.grid()
    ebs_plot.legend(labels)
    if savefig is not None:
        ebs_plot.save(savefig)
    if show:
        ebs_plot.show()
    return ebs_plot.fig, ebs_plot.ax


#     if efermi is not None:
#         fermi = efermi
#     elif outcar is not None:
#         outcarparser = UtilsProcar()
#         fermi = outcarparser.FermiOutcar(outcar)
#     else:
#         raise Warning("Fermi energy is not given, neither an OUTCAR contains it.")

#     uf = ProcarUnfolder(
#         procar=fname, poscar=poscar, supercell_matrix=supercell_matrix, ispin=ispin
#     )
#     if print_kpts:
#         for ik, k in enumerate(uf.procar.kpoints):
#             print(ik, k)
#     axes = uf.plot(
#         efermi=fermi,
#         ispin=ispin,
#         shift_efermi=shift_efermi,
#         ylim=elimit,
#         ktick=kticks,
#         kname=knames,
#         color=color,
#         width=width,
#         savetab=savetab,
#         show_band=show_band,
#     )

#     if exportplt:
#         return plt

#     else:
#         if savefig:
#             plt.savefig(savefig, bbox_inches="tight")
#             plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
#         else:
#             plt.show()
#         return


# # if __name__ == '__main__':
# #     """
# #     An example of how to use
# #     """
# #     import pyprocar
# #     import numpy as np
# #     pyprocar.unfold(
# #         fname='PROCAR',
# #         poscar='POSCAR',
# #         outcar='OUTCAR',
# #         supercell_matrix=np.diag([2, 2, 2]),
# #         efermi=None,
# #         shift_efermi=True,
# #         ispin=0,
# #         elimit=(-5, 15),
# #         kticks=[0, 36, 54, 86, 110, 147, 165, 199],
# #         knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
# #         print_kpts=False,
# #         show_band=True,
# #         savefig='unfolded_band.png')
