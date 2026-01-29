from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.cfg.unfold import UnfoldingConfig
from pyprocar.core import ElectronicBandStructure
from pyprocar.core.atomic_orbital_index import LEGACY_ORBITAL_NAMES
from pyprocar.utils import ROOT, welcome
from pyprocar.utils.log_utils import set_verbose_level

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)


with open(os.path.join(ROOT, "pyprocar", "cfg", "unfold.yml")) as file:
    plot_opt: dict[str, Any] = yaml.safe_load(file)


def unfold(
    code: str = "vasp",
    dirname: str = ".",
    mode: str = "plain",
    unfold_mode: str = "both",
    transformation_matrix: npt.NDArray[np.float64] | None = None,
    spins: list[int] | None = None,
    atoms: list[int] | list[str] | None = None,
    orbitals: list[int] | list[str] | None = None,
    items: dict[str, Any] | None = None,
    projection_mask: npt.NDArray[np.floating[Any]] | None = None,
    unfold_mask: npt.NDArray[np.floating[Any]] | None = None,
    fermi: float | None = None,
    fermi_shift: float = 0,
    interpolation_factor: int = 1,
    interpolation_type: str = "cubic",
    vmax: float | None = None,
    vmin: float | None = None,
    kticks: list[int] | None = None,
    knames: list[str] | None = None,
    kdirect: bool = True,
    elimit: list[float] | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savefig: str | None = None,
    old: bool = False,
    savetab: str = "unfold_result.csv",
    print_plot_opts: bool = False,
    use_cache: bool = False,
    verbose: int = 1,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Unfold and plot band structure.

    Parameters
    ----------
    code : str
        String for the code used, by default "vasp"
    dirname : str
        The directory name of the calculation, by default "."
    mode : str
        String for the mode of the calculation, by default "plain"
    unfold_mode : str
        The unfolding mode, by default "both"
    transformation_matrix : np.ndarray, optional
        Supercell matrix from primitive cell to supercell
    spins : list of int, optional
        A list of spins, by default None
    atoms : list, optional
        A list of atoms, by default None
    orbitals : list, optional
        A list of orbitals, by default None
    items : dict, optional
        A dictionary where the keys are atoms and values are orbital lists
    projection_mask : np.ndarray, optional
        A custom projection mask, by default None
    unfold_mask : np.ndarray, optional
        A custom unfold mask, by default None
    fermi : float, optional
        Fermi energy, by default None
    fermi_shift : float
        Shift for the Fermi energy, by default 0.
    interpolation_factor : int
        Interpolation factor, by default 1
    interpolation_type : str
        Interpolation type, by default "cubic"
    vmax : float, optional
        Maximum value for color scale
    vmin : float, optional
        Minimum value for color scale
    kticks : list of int, optional
        K-point tick indices, by default None
    knames : list of str, optional
        K-point tick names, by default None
    kdirect : bool
        Whether k-points are in direct coordinates, by default True
    elimit : list of float, optional
        Energy limits for plotting, by default None
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on, by default None
    show : bool
        Whether to show the plot, by default True
    savefig : str, optional
        Filename to save figure, by default None
    old : bool
        Whether to use old unfolding method, by default False
    savetab : str
        Filename for unfolding result CSV, by default "unfold_result.csv"
    print_plot_opts : bool
        Whether to print plot options, by default False
    use_cache : bool
        Whether to use cached data, by default False
    verbose : int
        Verbosity level, by default 1
    """
    # Unused parameters retained for API compatibility
    _ = (interpolation_factor, interpolation_type, kdirect, old, savetab, projection_mask, ax)

    if transformation_matrix is None:
        transformation_matrix = np.diag([2, 2, 2]).astype(np.float64)

    set_verbose_level(verbose)

    user_logger.info("If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    welcome()
    if vmin is not None and vmax is not None:
        kwargs["clim"] = (vmin, vmax)
    default_config = ConfigFactory.create_config(PlotType.UNFOLD)
    config = ConfigManager.merge_configs(default_config, kwargs)
    assert isinstance(config, UnfoldingConfig)
    modes_txt = " , ".join(config.modes)

    _message = f"""
            There are additional plot options that are defined in a configuration file.
            You can change these configurations by passing the keyword argument to the function
            To print a list of plot options set print_plot_opts=True

            Here is a list modes : {modes_txt}
            """
    user_logger.info(_message)
    if print_plot_opts:
        for key, value in plot_opt.items():
            user_logger.info(f"{key} : {value}")

    user_logger.info("_" * 100)

    ebs = ElectronicBandStructure.from_code(code, dirname, use_cache=use_cache)
    _kpath: Any = getattr(ebs, "kpath", None)
    structure = ebs.structure
    assert structure is not None, "No structure data found in electronic band structure"

    fermi_level: float = 0.0
    if fermi is not None:
        ebs.shift_bands(-1 * fermi, inplace=True)
        ebs.shift_bands(fermi_shift, inplace=True)
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
    ebs = ebs.unfold(transformation_matrix=transformation_matrix, structure=structure)

    # NOTE: BandsStructurePlotter is a legacy class that no longer exists in the codebase.
    # This script needs to be rewritten to use the current BandStructurePlotter API.
    # Using Any type to allow type checking of the rest of the file.
    ebs_plot: Any = None  # BandsStructurePlotter(ebs, kpath, ax, spins, config=config)

    labels: list[str] | None = None

    if ebs.projected_phase is None:
        raise ValueError("The provided electronic band structure file does not include phases")

    if unfold_mode == "both":
        logger.info("Unfolding bands in both modes")

        width_weights: npt.NDArray[np.float64] | None = ebs_plot.ebs.weights
        width_mask: npt.NDArray[np.floating[Any]] | None = unfold_mask
        color_weights: npt.NDArray[np.float64] | None = ebs_plot.ebs.weights
        color_mask: npt.NDArray[np.floating[Any]] | None = unfold_mask
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
        ebs_plot.handles = ebs_plot.handles[: ebs_plot.n_spins]
    elif mode in ["overlay", "overlay_species", "overlay_orbitals"]:
        overlay_weights: list[npt.NDArray[np.float64]] = []

        if mode == "overlay_species":
            logger.info("Plotting bands in overlay species mode")

            for ispc in structure.species:
                labels.append(ispc)
                species_atoms: npt.NDArray[np.intp] = np.where(structure.atoms == ispc)[0]
                w: npt.NDArray[np.float64] = ebs_plot.ebs.ebs_sum(
                    atoms=species_atoms,
                    orbitals=orbitals,
                    spins=spins,
                )
                overlay_weights.append(w)
        if mode == "overlay_orbitals":
            logger.info("Plotting bands in overlay orbitals mode")

            shell_indices: dict[str, list[int]] = {
                "s": [0],
                "p": [1, 2, 3],
                "d": [4, 5, 6, 7, 8],
                "f": [9, 10, 11, 12, 13, 14, 15],
            }
            resolved_atoms: list[int] | npt.NDArray[np.intp] | None = None
            if (
                atoms is not None
                and isinstance(atoms, list)
                and all(isinstance(a, int) for a in atoms)
            ):
                resolved_atoms = atoms  # pyright: ignore[reportAssignmentType]

            for iorb in ["s", "p", "d", "f"]:
                if iorb == "f" and not ebs_plot.ebs.n_orbitals > 9:
                    continue
                labels.append(iorb)
                orb_indices: list[int] = shell_indices[iorb]
                w = ebs_plot.ebs.ebs_sum(
                    atoms=resolved_atoms,
                    orbitals=orb_indices,
                    spins=spins,
                )
                overlay_weights.append(w)

        elif mode == "overlay":
            logger.info("Plotting bands in overlay mode")

            items_list: list[dict[str, Any]] = [items] if items else []

            for it in items_list:
                for ispc in it:
                    overlay_atoms: npt.NDArray[np.intp] = np.where(structure.atoms == ispc)[0]
                    if isinstance(it[ispc][0], str):
                        overlay_orb_arr: npt.NDArray[np.intp] = np.array([], dtype=np.intp)
                        for iorb_name in it[ispc]:
                            legacy_val = LEGACY_ORBITAL_NAMES.get(iorb_name, [])
                            if isinstance(legacy_val, int):
                                legacy_val = [legacy_val]
                            overlay_orb_arr = np.append(overlay_orb_arr, legacy_val).astype(np.intp)
                        labels.append(ispc + "-" + "".join(it[ispc]))
                    else:
                        overlay_orb_arr = np.array(it[ispc], dtype=np.intp)
                        labels.append(ispc + "-" + "_".join(it[ispc]))
                    w = ebs_plot.ebs.ebs_sum(
                        atoms=overlay_atoms,
                        orbitals=list(overlay_orb_arr),
                        spins=spins,
                    )
                    overlay_weights.append(w)
        ebs_plot.plot_parameteric_overlay(spins=spins, weights=overlay_weights, labels=labels)
    else:
        # Resolve atom names to indices
        param_atoms: npt.NDArray[np.intp] | list[int] | None = None
        if atoms is not None and isinstance(atoms[0], str):
            param_atoms_arr = np.array([], dtype=np.intp)
            for iatom in np.unique(atoms):
                param_atoms_arr = np.append(
                    param_atoms_arr, np.where(structure.atoms == iatom)[0]
                ).astype(int)
            param_atoms = param_atoms_arr
        elif atoms is not None:
            param_atoms = [int(a) for a in atoms]

        # Resolve orbital names to indices
        param_orbitals: npt.NDArray[np.intp] | list[int] | None = None
        if orbitals is not None and isinstance(orbitals[0], str):
            param_orb_arr = np.array([], dtype=np.intp)
            for iorb_name in orbitals:
                legacy_val = LEGACY_ORBITAL_NAMES.get(str(iorb_name), [])
                if isinstance(legacy_val, int):
                    legacy_val = [legacy_val]
                param_orb_arr = np.append(param_orb_arr, legacy_val).astype(int)
            param_orbitals = param_orb_arr
        elif orbitals is not None:
            param_orbitals = [int(o) for o in orbitals]

        projection_labels: list[str] = []
        projection_label = ""
        atoms_labels_str = ""
        if param_atoms is not None:
            atoms_labels_str = ",".join(str(x) for x in param_atoms)
            projection_label += f"atoms-{atoms_labels_str}"
        orbital_labels_str = ""
        if param_orbitals is not None:
            orbital_labels_str = ",".join(str(x) for x in param_orbitals)
            if len(projection_label) != 0:
                projection_label += "_"
        projection_label += f"orbitals-{orbital_labels_str}"
        projection_labels.append(projection_label)

        param_sum_weights: npt.NDArray[np.float64] = ebs_plot.ebs.ebs_sum(
            atoms=param_atoms, orbitals=param_orbitals, spins=spins
        )

        if config.weighted_color:
            color_weights = param_sum_weights
        else:
            color_weights = None
        if config.weighted_width:
            width_weights = param_sum_weights
        else:
            width_weights = None

        color_mask = unfold_mask
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
            user_logger.warning(f"Selected mode {mode} not valid. Please check the spelling")

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
