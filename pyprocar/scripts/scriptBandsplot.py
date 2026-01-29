from __future__ import annotations

_author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.cfg.band_structure import BandStructureConfig
from pyprocar.core import ElectronicBandStructurePath, KPath
from pyprocar.core.atomic_orbital_index import LEGACY_ORBITAL_NAMES
from pyprocar.plotter.bs_plot import BandStructurePlotter
from pyprocar.utils import np_utils, welcome

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
    def from_str(cls, mode: str) -> BandStructureMode:
        try:
            return cls[mode.upper()]
        except KeyError:
            raise ValueError(f"Invalid mode: {mode}. The modes available are: {cls.to_list()}")

    @classmethod
    def to_list(cls) -> list[str]:
        return [mode.value for mode in cls]

    @classmethod
    def get_overlay_modes(cls) -> list[BandStructureMode]:
        return [cls.OVERLAY, cls.OVERLAY_SPECIES, cls.OVERLAY_ORBITALS]


def bandsplot(
    code: str,
    dirname: str,
    mode: str = "plain",
    spins: list[int] | None = None,
    atoms: list[int] | list[str] | None = None,
    orbitals: list[int] | list[str] | None = None,
    items: dict[str, Any] | None = None,
    fermi: float | None = None,
    fermi_shift: float = 0,
    interpolation_factor: int = 1,
    interpolation_type: str = "cubic",
    projection_mask: npt.NDArray[np.floating[Any]] | None = None,
    kticks: list[int] | None = None,
    knames: list[str] | None = None,
    kdirect: bool = True,
    elimit: list[float] | None = None,
    ax: Axes | None = None,
    show: bool = True,
    savefig: str | None = None,
    print_plot_opts: bool = False,
    export_data_file: str | None = None,
    export_append_mode: bool = True,
    ktick_limit: list[float] | None = None,
    x_limit: list[float] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    parametric_kwargs: dict[str, Any] | None = None,
    quiver_kwargs: dict[str, Any] | None = None,
    atomic_levels_kwargs: dict[str, Any] | None = None,
    atomic_kwargs: dict[str, Any] | None = None,
    overlay_kwargs: dict[str, Any] | None = None,
    ipr_kwargs: dict[str, Any] | None = None,
    use_cache: bool = False,
    quiet_welcome: bool = False,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
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
    # Unused parameters retained for API compatibility
    _ = (interpolation_factor, interpolation_type, projection_mask, kdirect, ktick_limit)

    # Initialize mutable defaults
    if items is None:
        items = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if parametric_kwargs is None:
        parametric_kwargs = {}
    if quiver_kwargs is None:
        quiver_kwargs = {}
    if atomic_levels_kwargs is None:
        atomic_levels_kwargs = {}
    if atomic_kwargs is None:
        atomic_kwargs = {}
    if overlay_kwargs is None:
        overlay_kwargs = {}
    if ipr_kwargs is None:
        ipr_kwargs = {}

    if quiet_welcome:
        user_logger.setLevel(logging.ERROR)

    user_logger.info("If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    welcome()

    config = ConfigFactory.create_config(PlotType.BAND_STRUCTURE)
    assert isinstance(config, BandStructureConfig)
    config = ConfigManager.merge_configs(config, kwargs)
    assert isinstance(config, BandStructureConfig)

    user_logger.info("_" * 100)
    modes_txt = " , ".join(BandStructureMode.to_list())
    _message = f"""
            There are additional plot options that are defined in the configuration file.
            You can change these configurations by passing the keyword argument to the function.
            To print a list of all plot options set `print_plot_opts=True`

            Here is a list modes : {modes_txt}
            """

    if print_plot_opts:
        for key, value in config.as_dict().items():
            user_logger.info(f"{key} : {value}")

    user_logger.info("_" * 100)

    ebs = ElectronicBandStructurePath.from_code(code, dirname, use_cache=use_cache)
    assert isinstance(ebs, ElectronicBandStructurePath)
    structure = ebs.structure
    assert structure is not None, "No structure data found in electronic band structure"

    kpath: KPath = ebs.kpath

    codes_with_scf_fermi = ["qe", "elk"]
    if code in codes_with_scf_fermi and fermi is None:
        logger.info(f"No fermi given, using the found fermi energy: {ebs.fermi}")
        fermi = ebs.fermi

    fermi_level: float = 0.0
    if fermi is not None:
        logger.info(f"Shifting Fermi energy to zero: {fermi}")

        ebs.shift_bands(-1 * fermi, inplace=True)
        ebs.shift_bands(fermi_shift, inplace=True)
        fermi_level = fermi_shift
        y_label = r"E - E$_F$ (eV)"
    else:
        y_label = r"E (eV)"
        user_logger.warning(
            "`fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy."
        )

    # fixing the spin, to plot two channels into one (down is negative)
    if spins is not None and (np.array_equal(spins, [-1, 1]) or np.array_equal(spins, [1, -1])):
        if ebs.fix_collinear_spin():
            spins = [0]

    plotter = BandStructurePlotter(ax=ax)

    projection_labels: list[str] = []
    labels: list[str] = []
    band_mode = BandStructureMode.from_str(mode)

    # Prepare bands and optionally restrict spin channels
    bands_prop = ebs.bands
    assert bands_prop is not None, "No bands data found in electronic band structure"
    bands: npt.NDArray[np.float64] = bands_prop.value
    if spins is not None:
        try:
            bands = bands[..., spins]
        except Exception:
            pass

    n_spin_channels: int = bands.shape[-1]

    if band_mode == BandStructureMode.PLAIN:
        user_logger.info("Plotting bands in plain mode")
        for i_spin_channel in range(n_spin_channels):
            plotter.plot(kpath, bands[..., i_spin_channel], **plot_kwargs)

    elif band_mode == BandStructureMode.IPR:
        user_logger.info("Plotting bands in IPR mode")
        ipr_prop = ebs.ebs_ipr
        ipr_weights: npt.NDArray[np.float64] | None = (
            ipr_prop.value if ipr_prop is not None else None
        )
        if spins is not None and ipr_weights is not None and ipr_weights.ndim >= 3:
            try:
                ipr_weights = ipr_weights[..., spins]
            except Exception:
                pass
        plotter.plot_parametric(kpath, bands, ipr_weights)
        plotter.set_colorbar_title(title="Inverse Participation Ratio")

    elif band_mode in BandStructureMode.get_overlay_modes():
        overlay_weights: list[npt.NDArray[np.float64]] = []
        if band_mode == BandStructureMode.OVERLAY_SPECIES:
            resolved_orbitals: list[int] | npt.NDArray[np.intp] | None = None
            if orbitals is None:
                try:
                    n_orbitals = ebs.n_orbitals
                except Exception:
                    projected = ebs.projected
                    assert projected is not None
                    n_orbitals = projected.shape[-1]
                resolved_orbitals = [int(x) for x in range(n_orbitals)]
            elif isinstance(orbitals, list) and all(isinstance(o, int) for o in orbitals):
                resolved_orbitals = orbitals  # pyright: ignore[reportAssignmentType]
            else:
                resolved_orbitals = None

            user_logger.info("Plotting bands in overlay species mode")
            for ispc in structure.species:
                labels.append(ispc)
                species_atoms: npt.NDArray[np.intp] = np.where(structure.atoms == ispc)[0]

                orbitals_str = (
                    ",".join(str(x) for x in resolved_orbitals)
                    if resolved_orbitals is not None
                    else ""
                )
                projection_label = f"atom-{ispc}_orbitals-{orbitals_str}"
                projection_labels.append(projection_label)
                w = ebs.ebs_sum(
                    atoms=species_atoms,
                    orbitals=resolved_orbitals,
                    spins=spins,
                )
                if spins is not None and w.ndim >= 3:
                    try:
                        w = w[..., spins]
                    except Exception:
                        pass
                overlay_weights.append(w)
        elif band_mode == BandStructureMode.OVERLAY_ORBITALS:
            user_logger.info("Plotting bands in overlay orbitals mode")
            # Use legacy orbital name mapping for shell → indices
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

            for _iorb, orb in enumerate(["s", "p", "d", "f"]):
                if orb == "f" and not ebs.n_orbitals > 9:
                    continue
                orb_indices: list[int] = shell_indices[orb]
                labels.append(orb)

                atom_label = ""
                if resolved_atoms:
                    atom_labels_str = ",".join(str(x) for x in resolved_atoms)
                    atom_label = f"atom-{atom_labels_str}_"
                projection_label = f"{atom_label}orbitals-{orb}"
                projection_labels.append(projection_label)
                w = ebs.ebs_sum(
                    atoms=resolved_atoms,
                    orbitals=orb_indices,
                    spins=spins,
                )
                if spins is not None and w.ndim >= 3:
                    try:
                        w = w[..., spins]
                    except Exception:
                        pass
                overlay_weights.append(w)

        elif band_mode == BandStructureMode.OVERLAY:
            user_logger.info("Plotting bands in overlay mode")
            items_list: list[dict[str, Any]] = [items]

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
                        labels.append(ispc + "-" + "_".join(str(x) for x in it[ispc]))

                    atom_labels_str2 = ",".join(str(x) for x in overlay_atoms)
                    orbital_labels_str = ",".join(str(x) for x in overlay_orb_arr)
                    projection_label = f"atoms-{atom_labels_str2}_orbitals-{orbital_labels_str}"
                    projection_labels.append(projection_label)
                    w = ebs.ebs_sum(
                        atoms=overlay_atoms,
                        orbitals=list(overlay_orb_arr),
                        spins=spins,
                    )
                    if spins is not None and w.ndim >= 3:
                        try:
                            w = w[..., spins]
                        except Exception:
                            pass
                    overlay_weights.append(w)
        plotter.plot_overlay(kpath, bands, weights=overlay_weights, labels=projection_labels)
    elif band_mode in [
        BandStructureMode.PARAMETRIC,
        BandStructureMode.SACATTER,
        BandStructureMode.ATOMIC,
    ]:
        # Resolve atom names to indices
        param_atoms: npt.NDArray[np.intp] | list[int] | None = None
        if atoms is not None and isinstance(atoms[0], str):
            param_atoms_arr = np.array([], dtype=np.intp)
            for iatom in np.unique(atoms):
                param_atoms_arr = np.append(
                    param_atoms_arr, np.where(structure.atoms == iatom)[0]
                ).astype(np_utils.INT_DTYPE)
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
                param_orb_arr = np.append(param_orb_arr, legacy_val).astype(np_utils.INT_DTYPE)
            param_orbitals = param_orb_arr
        elif orbitals is not None:
            param_orbitals = [int(o) for o in orbitals]

        projection_labels = []
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

        param_weights = ebs.ebs_sum(atoms=param_atoms, orbitals=param_orbitals, spins=spins)
        if spins is not None and param_weights.ndim >= 3:
            try:
                param_weights = param_weights[..., spins]
            except Exception:
                pass
        if band_mode == BandStructureMode.PARAMETRIC:
            user_logger.info("Plotting bands in parametric mode")
            plotter.plot_parametric(kpath, bands, param_weights)
            plotter.set_colorbar_title()
        elif band_mode == BandStructureMode.SACATTER:
            user_logger.info("Plotting bands in scatter mode")
            plotter.plot_scatter(kpath, bands, param_weights)
            plotter.set_colorbar_title()
        elif band_mode == BandStructureMode.ATOMIC:
            user_logger.info("Plotting bands in atomic mode")
            if ebs.kpoints.shape[0] != 1:
                raise Exception("Must use a single kpoint")
            elimit_tuple: tuple[float, float] | None = None
            if elimit is not None and len(elimit) >= 2:
                elimit_tuple = (elimit[0], elimit[1])
            plotter.plot_atomic_levels(
                bands=bands,
                elimit=elimit_tuple,
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
