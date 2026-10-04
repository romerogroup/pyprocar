import logging
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from pyprocar.cfg import ConfigFactory, ConfigManager, PlotType
from pyprocar.cfg.unfold import UnfoldingConfig, UnfoldMode, UnfoldPlotMode
from pyprocar.core import ElectronicBandStructurePath, Structure
from pyprocar.core.projection import selection_resolver
from pyprocar.core.property_store import Property
from pyprocar.plotter.bs_plot import BandStructurePlotter
from pyprocar.scripts._selection import (
    as_lim,
    orbital_indices,
    per_channel,
    projection_components,
    resolve_spins,
    take_channels,
)
from pyprocar.utils.log_utils import set_verbose_level, warn_user
from pyprocar.utils.splash import welcome

user_logger = logging.getLogger("user")

OVERLAY_MODES = (
    UnfoldPlotMode.OVERLAY,
    UnfoldPlotMode.OVERLAY_SPECIES,
    UnfoldPlotMode.OVERLAY_ORBITALS,
)
REMOVED_PARAMETERS = (
    "kdirect",
    "projection_mask",
    "unfold_mask",
    "interpolation_factor",
    "interpolation_type",
    "old",
    "savetab",
)


def unfold(
    code: str = "vasp",
    dirname: str = ".",
    mode: str = "plain",
    unfold_mode: str = "both",
    transformation_matrix=np.diag([2, 2, 2]),
    spins: list[int] | None = None,
    atoms: list[int] | None = None,
    orbitals: list[int] | None = None,
    items: dict | list[dict] | None = None,
    fermi: float | None = None,
    fermi_shift: float = 0,
    vmax: float | None = None,
    vmin: float | None = None,
    kticks=None,
    knames=None,
    elimit: list[float] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    savefig: str | None = None,
    print_plot_opts: bool = False,
    use_cache: bool = False,
    verbose: int = 1,
    **kwargs,
):
    """Plot the band structure of a supercell unfolded onto the primitive cell.

    Each supercell band at each k-point gets an unfolding weight between 0 and 1:
    the fraction of the band that has the primitive cell's Bloch character at
    that k-point. The supercell calculation must list its k-points along the
    primitive cell path, expressed in the supercell's reciprocal basis, and VASP
    must write the projection phases (``LORBIT = 12``).

    Parameters
    ----------
    code : str
        The DFT code of the supercell calculation. Only VASP writes phases.
    dirname : str
        The supercell calculation directory.
    mode : str
        ``plain`` draws the unfolded bands. ``parametric`` and ``scatter`` color
        them by the projection on ``atoms``, ``orbitals`` and ``spins``.
        ``overlay``, ``overlay_species`` and ``overlay_orbitals`` fill each
        projection with a thickness of projection times unfolding weight.
    unfold_mode : str
        How the unfolding weight is drawn: ``thickness`` (line width or marker
        size), ``color`` (plain mode only), or ``both``. In the parametric,
        scatter and overlay modes the projection sets the color, so the weight
        sets only the thickness.
    transformation_matrix : np.ndarray
        The (3, 3) integer matrix that turns the primitive cell into the supercell.
    spins : list of int, optional
        The spin channels to draw.
    atoms, orbitals : list, optional
        The projection selection, as indices or as species and orbital names.
    items : dict or list of dict, optional
        The species-to-orbitals mapping of the ``overlay`` mode.
    fermi : float, optional
        The Fermi energy. The bands are shifted so that it sits at ``fermi_shift``.
    fermi_shift : float
        Where the Fermi energy sits after the shift.
    vmin, vmax : float, optional
        The color range. The default is 0 to 1.
    kticks, knames : list, optional
        The k-point indices and names of the x ticks. The KPOINTS file sets
        them when omitted.
    elimit : list of float, optional
        The energy range.
    ax : matplotlib.axes.Axes, optional
        The axes to draw on.
    show : bool
        Show the figure.
    savefig : str, optional
        Save the figure to this file.
    print_plot_opts : bool
        Log the plot options that ``**kwargs`` can set.
    use_cache : bool
        Read and write the parsed band structure as ``ebs.pkl`` in ``dirname``.
    verbose : int
        The verbosity level.
    **kwargs
        Plot options of :class:`pyprocar.cfg.unfold.UnfoldingConfig`, such as
        ``title``, ``cmap`` or ``linewidth``.

    Returns
    -------
    tuple
        The matplotlib figure and axes.
    """
    removed = [name for name in REMOVED_PARAMETERS if name in kwargs]
    if removed:
        raise TypeError(
            f"unfold() no longer takes {', '.join(removed)}; these parameters were removed"
            + " because they had no effect"
        )
    set_verbose_level(verbose)
    welcome()

    plot_mode = UnfoldPlotMode(mode)
    if plot_mode is UnfoldPlotMode.ATOMIC:
        raise ValueError("unfold has no atomic mode; use bandsplot for single k-point levels")
    weight_mode = UnfoldMode(unfold_mode)
    if weight_mode is UnfoldMode.COLOR and plot_mode is not UnfoldPlotMode.PLAIN:
        raise ValueError(
            f"unfold_mode='color' needs mode='plain'; {mode} mode colors by the projection,"
            + " so use unfold_mode='thickness'"
        )

    default_config = cast(UnfoldingConfig, ConfigFactory.create_config(PlotType.UNFOLD))
    if vmin is not None or vmax is not None:
        kwargs["clim"] = as_lim((vmin, vmax), default_config.clim or (0.0, 1.0))
    config = cast(UnfoldingConfig, ConfigManager.merge_configs(default_config, kwargs))
    if print_plot_opts:
        for key, value in default_config.as_dict().items():
            user_logger.info(f"{key} : {value}")

    ebs = cast(
        ElectronicBandStructurePath,
        ElectronicBandStructurePath.from_code(code, dirname, use_cache=use_cache),
    )
    if ebs.projected_phase is None:
        raise ValueError(
            f"{dirname} has no projection phases; unfolding needs a PROCAR written with LORBIT = 12"
        )
    structure = cast(Structure, ebs.structure)

    if fermi is not None:
        ebs.shift_bands(fermi_shift - fermi, inplace=True)
        y_label = r"E - E$_F$ (eV)"
    else:
        y_label = r"E (eV)"
        warn_user(
            "`fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy."
        )

    ebs.unfold(transformation_matrix=transformation_matrix, structure=structure)

    selection = resolve_spins(ebs.is_non_collinear, ebs.n_spin_channels, spins, plain=False)
    channels, projection_spins = selection.channels, selection.projection_spins
    n_channels = len(channels)
    bands = take_channels(cast(Property, ebs.bands), channels)
    weights = take_channels(cast(Property, ebs.weights), channels)

    if atoms is not None and isinstance(atoms[0], str):
        species = [str(name) for name in atoms]
        atoms = list(selection_resolver(ebs).resolve(species=species).atoms)
    orbitals = orbital_indices(orbitals, ebs)

    plotter = BandStructurePlotter(ax=ax)
    linestyle = per_channel(config.linestyle, n_channels)
    plotter.plot(
        bands,
        color=config.color,
        alpha=per_channel(config.opacity, n_channels),
        linestyle=linestyle,
        linewidth=per_channel(config.linewidth, n_channels),
    )

    if plot_mode not in OVERLAY_MODES:
        if plot_mode is UnfoldPlotMode.PLAIN:
            scalars = weights if weight_mode is not UnfoldMode.THICKNESS else None
            colorbar_title = "Unfolding weight"
        else:
            projection = ebs.compute_projected_sum(
                atoms=atoms, orbitals=orbitals, spins=projection_spins
            )
            scalars = take_channels(projection, channels)
            colorbar_title = config.colorbar_title
        plotter.plot(
            bands,
            scalars_data=scalars,
            widths_data=weights if weight_mode is not UnfoldMode.COLOR else None,
            scalars_mode="scatter" if plot_mode is UnfoldPlotMode.SCATTER else "parametric",
            scalars_cmap=config.cmap,
            scalars_clim=config.clim,
            color=per_channel(config.spin_colors, n_channels) if scalars is None else None,
            linestyle=linestyle,
        )
        plotter.set_colorbar_label(colorbar_title)
    else:
        user_logger.info(f"Plotting unfolded bands in {plot_mode.value} mode")
        kind = plot_mode.value.split("_", 1)[1] if "_" in plot_mode.value else "items"
        components = projection_components(
            ebs, kind, atoms=atoms, orbitals=orbitals, items=items, spins=projection_spins
        )
        plotter.plot_overlay(
            ebs.kpath,
            bands.value,
            weights=[take_channels(c, channels).value * weights.value for c in components],
            labels=[str(c.label) for c in components],
            linewidth=0.0,
        )

    if kticks is not None or knames is not None:
        plotter.set_xticks(kticks, knames)
    plotter.set_yticks(interval=elimit)
    if elimit is not None:
        plotter.set_ylim(elimit)
    plotter.set_ylabel(label=y_label)
    plotter.set_xlabel(label=config.x_label)

    if fermi is not None:
        plotter.draw_fermi(
            fermi_level=fermi_shift,
            color=config.fermi_color,
            linestyle=config.fermi_linestyle,
            linewidth=config.fermi_linewidth,
        )

    plotter.set_title(config.title if config.title is not None else "Unfolded Band Structure")
    plotter.grid()

    if savefig is not None:
        plotter.save(savefig)
    if show:
        plotter.show()

    return plotter.fig, plotter.ax
