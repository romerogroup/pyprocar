_author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import os
from enum import Enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.core import ElectronicBandStructurePath
from pyprocar.plotter.bs_plot import BandStructurePlotter
from pyprocar.utils import data_utils, welcome
from pyprocar.utils.info import orbital_names
from pyprocar.utils.log_utils import set_verbose_level

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)



class BandStructureMode(Enum):
    """
    An enumeration for defining the modes of Band Structure representations.

    Attributes
    ----------
    PLAIN : str
        Represents the band structure in a simple, where the colors are the different bands.
    PARAMETRIC : str
        Represents the band structure in a parametric form, summing over the projections.
    SACATTER : str
        Represents the band structure in a scatter plot, where the colors are the different bands.
    ATOMIC : str
        Represents the band structure in an atomic level plot, plots singlr kpoint bands.
    OVERLAY : str
        Represents the band structure in an overlay plot, where the colors are the selected projections
    OVERLAY_SPECIES : str
        Represents the band structure in an overlay plot, where the colors are 
        the different projection of the species.
    OVERLAY_ORBITALS : str
        Represents the band structure in an overlay plot, where  the colors are 
        the different projection of the orbitals.
    """
    PLAIN = "plain"
    PARAMETRIC = "parametric"
    SACATTER = "scatter"
    ATOMIC = "atomic"
    OVERLAY = "overlay"
    OVERLAY_SPECIES = "overlay_species"
    OVERLAY_ORBITALS = "overlay_orbitals"
    IPR = "ipr"
    
    @classmethod
    def from_str(cls, mode: str):
        try:
            return cls[mode.upper()]
        except KeyError:
            raise ValueError(f"Invalid mode: {mode}. The modes available are: {cls.to_list()}")
    
    @classmethod
    def to_list(cls):
        return [mode.value for mode in cls]

    @classmethod
    def get_overlay_modes(cls):
        return [cls.OVERLAY, cls.OVERLAY_SPECIES, cls.OVERLAY_ORBITALS]



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
    plot_kwargs: dict = {},
    scatter_kwargs: dict = {},
    parametric_kwargs: dict = {},
    quiver_kwargs: dict = {},
    atomic_levels_kwargs: dict = {},
    atomic_kwargs: dict = {},
    overlay_kwargs: dict = {},
    ipr_kwargs: dict = {},
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
    
    if quiet_welcome:
        user_logger.setLevel(logging.ERROR)

    user_logger.info(f"If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)


    welcome()

    default_config = ConfigFactory.create_config(PlotType.BAND_STRUCTURE)
    config = ConfigManager.merge_configs(default_config, kwargs)

    user_logger.info("_" * 100)
    modes_txt = " , ".join(BandStructureMode.to_list())
    message = f"""
            There are additional plot options that are defined in the configuration file. 
            You can change these configurations by passing the keyword argument to the function.
            To print a list of all plot options set `print_plot_opts=True`

            Here is a list modes : {modes_txt}
            """

    if print_plot_opts:
        for key, value in default_config.as_dict().items():
            user_logger.info(f"{key} : {value}")

    user_logger.info("_" * 100)

    ebs = ElectronicBandStructurePath.from_code(code, dirname, use_cache=use_cache)
    structure=ebs.structure
    
    # Covers when kpath is single kpoint
    kpath=None
    if hasattr(ebs, "kpath"):
        kpath=ebs.kpath
    

    codes_with_scf_fermi = ["qe", "elk"]
    if code in codes_with_scf_fermi and fermi is None:
        logger.info(f"No fermi given, using the found fermi energy: {ebs.fermi}")
        fermi = ebs.fermi

    if fermi is not None:
        logger.info(f"Shifting Fermi energy to zero: {fermi}")

        ebs.shift_bands(-1*fermi, inplace=True)
        ebs.shift_bands(fermi_shift, inplace=True)
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

    plotter = BandStructurePlotter(ax=ax)

    projection_labels = []
    labels = []
    mode = BandStructureMode.from_str(mode)

    # Prepare bands and optionally restrict spin channels
    bands = ebs.bands
    if spins is not None:
        try:
            bands = bands[..., spins]
        except Exception:
            pass
        
    n_spin_channels = bands.shape[-1]
    n_bands = bands.shape[1]
    n_kpoints = bands.shape[0]
    
    
    
    if mode == BandStructureMode.PLAIN:
        user_logger.info("Plotting bands in plain mode")
        for i_spin_channel in range(n_spin_channels):
                plotter.plot(kpath, bands[..., i_spin_channel], **plot_kwargs)

    elif mode == BandStructureMode.IPR:
        user_logger.info("Plotting bands in IPR mode")
        weights = ebs.get_ebs_ipr()
        if spins is not None and weights is not None and weights.ndim >= 3:
            try:
                weights = weights[..., spins]
            except Exception:
                pass
        plotter.plot_parametric(kpath, bands, weights)
        plotter.set_colorbar_title(title="Inverse Participation Ratio")

    elif mode in BandStructureMode.get_overlay_modes():
        weights = []
        if mode == BandStructureMode.OVERLAY_SPECIES:
            if orbitals is None:
                try:
                    n_orbitals = ebs.n_orbitals
                except Exception:
                    n_orbitals = ebs.projected.shape[-1]
                orbitals = list(np.arange(n_orbitals, dtype=int))

            user_logger.info("Plotting bands in overlay species mode")
            for ispc in structure.species:
                labels.append(ispc)
                atoms = np.where(structure.atoms == ispc)[0]

                projection_label = f"atom-{ispc}_orbitals-" + ",".join(
                    str(x) for x in orbitals
                )
                projection_labels.append(projection_label)
                w = ebs.ebs_sum(
                    atoms=atoms,
                    orbitals=orbitals,
                    spins=spins,
                )
                if spins is not None and w is not None and w.ndim >= 3:
                    try:
                        w = w[..., spins]
                    except Exception:
                        pass
                weights.append(w)
        elif mode == BandStructureMode.OVERLAY_ORBITALS:
            user_logger.info("Plotting bands in overlay orbitals mode")
            for iorb, orb in enumerate(["s", "p", "d", "f"]):
                if orb == "f" and not ebs.n_orbitals > 9:
                    continue
                orbitals = orbital_names[orb]
                labels.append(orb)

                atom_label = ""
                if atoms:
                    atom_labels = ",".join(str(x) for x in atoms)
                    atom_label = f"atom-{atom_labels}_"
                projection_label = f"{atom_label}orbitals-{orb}"
                projection_labels.append(projection_label)
                w = ebs.ebs_sum(
                    atoms=atoms,
                    orbitals=orbitals,
                    spins=spins,
                )
                if spins is not None and w is not None and w.ndim >= 3:
                    try:
                        w = w[..., spins]
                    except Exception:
                        pass
                weights.append(w)

        elif mode == BandStructureMode.OVERLAY:
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
                        w = ebs.ebs_sum(
                            atoms=atoms,
                            orbitals=orbitals,
                            spins=spins,
                        )
                        if spins is not None and w is not None and w.ndim >= 3:
                            try:
                                w = w[..., spins]
                            except Exception:
                                pass
                        weights.append(w)
        plotter.plot_overlay(kpath, bands, weights=weights, labels=projection_labels)
    elif mode in [BandStructureMode.PARAMETRIC, BandStructureMode.SACATTER, BandStructureMode.ATOMIC]:

        if atoms is not None and isinstance(atoms[0], str):
            atoms_str = atoms
            atoms = []
            for iatom in np.unique(atoms_str):
                atoms = np.append(atoms, np.where(structure.atoms == iatom)[0]).astype(
                    np.int
                )

        if orbitals is not None and isinstance(orbitals[0], str):
            orbital_str = orbitals

            orbitals = []
            for iorb in orbital_str:
                orbitals = np.append(orbitals, orbital_names[iorb]).astype(np.int)

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

        weights = ebs.ebs_sum(
            atoms=atoms, orbitals=orbitals, spins=spins
        )
        if spins is not None and weights is not None and weights.ndim >= 3:
            try:
                weights = weights[..., spins]
            except Exception:
                pass
        if mode == BandStructureMode.PARAMETRIC:
            user_logger.info("Plotting bands in parametric mode")
            plotter.plot_parametric(kpath, bands, weights)
            plotter.set_colorbar_title()
        elif mode == BandStructureMode.SACATTER:
            user_logger.info("Plotting bands in scatter mode")
            plotter.plot_scatter(kpath, bands, weights)
            plotter.set_colorbar_title()
        elif mode == BandStructureMode.ATOMIC:
            user_logger.info("Plotting bands in atomic mode")
            if ebs.kpoints.shape[0] != 1:
                raise Exception("Must use a single kpoint")
            plotter.plot_atomic_levels(
                bands=bands,
                elimit=elimit,
            )

            plotter.set_xlabel(label=config.x_label)
            plotter.set_colorbar_title()

    plotter.set_xticks(kticks, knames)
    plotter.set_yticks(interval=elimit)
    if x_limit is not None:
        plotter.set_xlim(x_limit)
    plotter.set_ylim(elimit)
    plotter.set_ylabel(label=y_label)
    plotter.set_xlabel(label=config.x_label)

    if fermi is not None:
        plotter.draw_fermi(fermi_level=fermi_level)

    plotter.set_title()
    plotter.grid()

    plotter.legend(labels)

    if savefig is not None:
        plotter.save(savefig)
    if show:
        plotter.show()

    if export_data_file is not None:
        if export_append_mode:
            file_basename, file_type = export_data_file.split(".")
            filename = f"{file_basename}_{mode}.{file_type}"
        else:
            filename = export_data_file
        plotter.export_data(filename)

    return plotter.fig, plotter.ax
