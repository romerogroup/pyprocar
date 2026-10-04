__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import itertools
import logging
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from pyprocar.cfg import ConfigFactory, ConfigManager
from pyprocar.cfg.base import PlotType
from pyprocar.cfg.dos import DensityOfStatesConfig
from pyprocar.core import DensityOfStates
from pyprocar.core.property_store import Property
from pyprocar.plotter.dos_plot import AxesOrientation, DOSPlotter
from pyprocar.scripts._selection import (
    as_lim,
    orbital_indices,
    per_channel,
    projection_components,
    resolve_spins,
    signed_clim,
    take_channels,
)
from pyprocar.utils.log_utils import set_verbose_level, warn_user
from pyprocar.utils.splash import welcome

user_logger = logging.getLogger("user")
logger = logging.getLogger(__name__)

PARAMETRIC_MODES = ("parametric", "parametric_line")
DOS_MODES = (
    "plain",
    *PARAMETRIC_MODES,
    "stack",
    "stack_species",
    "stack_orbitals",
    "overlay",
    "overlay_species",
    "overlay_orbitals",
)


def dosplot(
    code: str = "vasp",
    dirname: str | None = None,
    mode: str = "plain",
    orientation: str = "horizontal",
    spins: list[int] | None = None,
    atoms: list[int] | None = None,
    orbitals: list[int] | None = None,
    items: dict | None = None,
    normalize_dos_mode: str | None = None,
    fermi: float | None = None,
    fermi_shift: float = 0,
    elimit: list[float | None] | None = None,
    dos_limit: list[float | None] | None = None,
    savefig: str | None = None,
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
    show: bool = True,
    print_plot_opts: bool = False,
    use_cache: bool = False,
    verbose: int = 1,
    **kwargs,
):
    """
    This function plots the density of states in different formats

    Parameters
    ----------

    filename : str, optional (default ``'vasprun.xml'``)
        The most important argument needed dosplot is
        **filename**. **filename** defines the path to `vasprun.xml`
        from the density of states calculation. If plotting is being
        carried out in the directory of the calculation, one does not
        need to specify this argument.

        e.g. ``filename='~/SrVO3/DOS/vasprun.xml'``

    dirname : str, optional (default ``'vasprun.xml'``)
        This is used for qe and lobster codes. It specifies the directory the dosplot
        calculation was performed.

        e.g. ``dirname='~/SrVO3/dos'``

    mode : str, optional (default ``'plain'``)
        **mode** defines the mode of the plot. This parameter will be
        explained in details with exmaples in the tutorial.
        options are ``'plain'``, ``'parametric'``,
        ``'parametric_line'``, ``'stack'``,
        ``'stack_orbitals'``, ``'stack_species'``.

        e.g. ``mode='stack'``


    orientation : str, optional (default ``horizontal'``)
        The orientation of the DOS plot.  options are
        ``'horizontal', 'vertical'``

        e.g. ``orientation='vertical'``

    spins : list int, optional
        ``spins`` defines plotting of different spins channels present
        in the calculation, If the calculation is spin non-polorized
        the spins will be set by default to ``spins=[0]``. if the
        calculation is spin polorized this parameter can be set to 0
        or 1 or both.

        e.g. ``spins=[0, 1]``

    atoms : list int, optional
        ``atoms`` define the projection of the atoms in the Density of
        States. In other words it selects only the contribution of the
        atoms provided. Atoms has to be a python list(or numpy array)
        containing the atom indices. Atom indices has to be order of
        the input files of DFT package. ``atoms`` is only relevant in
        ``mode='parametric'``, ``mode='parametric_line'``,
        ``mode='stack_orbitals'``. keep in mind that python counting
        starts from zero.
        e.g. for SrVO\ :sub:`3`\  we are choosing only the oxygen
        atoms. ``atoms=[2, 3, 4]``, keep in mind that python counting
        starts from zero, for a **POSCAR** similar to following::

            Sr1 V1 O3
            1.0
            3.900891 0.000000 0.000000
            0.000000 3.900891 0.000000
            0.000000 0.000000 3.900891
            Sr V O
            1 1 3
            direct
            0.500000 0.500000 0.500000 Sr atom 0
            0.000000 0.000000 0.000000 V  atom 1
            0.000000 0.500000 0.000000 O  atom 2
            0.000000 0.000000 0.500000 O  atom 3
            0.500000 0.000000 0.000000 O  atom 4

        if nothing is specified this parameter will consider all the
        atoms present.

    orbitals : list int, optional
        ``orbitals`` define the projection of orbitals in the density
        of States. In other words it selects only the contribution of
        the orbitals provided. Orbitals has to be a python list(or
        numpy array) containing the Orbital indices. Orbitals indices
        has to be order of the input files of DFT package. The
        following table represents the indecies for different orbitals
        in **VASP**.

        .. code-block::
            :linenos:

            +-----+-----+----+----+-----+-----+-----+-----+-------+
            |  s  | py  | pz | px | dxy | dyz | dz2 | dxz | x2-y2 |
            +-----+-----+----+----+-----+-----+-----+-----+-------+
            |  0  |  1  |  2 |  3 |  4  |  5  |  6  |  7  |   8   |
            +-----+-----+----+----+-----+-----+-----+-----+-------+

        ``orbitals`` is only relavent in ``mode='parametric'``,
        ``mode='parametric_line'``, ``mode='stack_species'``.

        e.g. ``orbitals=[1,2,3]`` will only select the p orbitals
        while ``orbitals=[4,5,6,7,8]`` will select the d orbitals.

        If nothing is specified pyprocar will select all the present
        orbitals.

    normalize_dos_mode : str, optional
        This defines the mode of the normalization of the density of states. The default is None.
        If None, the density of states will not be normalized.

    elimit : list float, optional
        Energy window limit asked to plot. ``elimit`` has to be a two
        element python list(or numpy array).

        e.g. ``elimit=[-2, 2]``
        The default is set to the minimum and maximum of the energy
        window.

    dos_limit : list float, optional
       ``dos_limit`` defines the density of states axis limits on the
       graph. It is automatically set to select 10% higher than the
       maximum of density of states in the specified energy window.

       e.g. ``dos_limit=[0, 30]``

    labels : list str, optional
        ``labels`` is a list of strings that will be used as the
        legend of the plot. The length of the list should be equal to
        the number of curves being plotted. If not provided the
        default labels will be used.

    savefig : str , optional (default None)
        ``savefig`` defines the file that the plot is going to be
        saved in. ``savefig`` accepts all the formats accepted by
        matplotlib such as png, pdf, jpg, ...
        If not provided the plot will be shown in the
        interactive matplotlib mode.

        e.g. ``savefig='DOS.png'``, ``savefig='DOS.pdf'``

    plot_total : bool, optional (default ``True``)
        If the total density of states is plotted as well as other
        options. The entry should be python boolian.

        e.g. ``plot_total=True``

    code : str, optional (default ``'vasp'``)
        Defines the Density Functional Theory code used for the
        calculation. The default of this argument is vasp, so if the
        cal is done in vasp one does not need to define this argumnet.

        e.g. ``code=vasp``, ``code=elk``, ``code=abinit``

    items : dict, optional
        ``items`` is only relavent for ``mode='stack'``. stack will
        plot the items defined with stacked filled areas under
        curve. For clarification visit the examples in the
        tutorial. ``items`` need to be provided as a python
        dictionary, with keys being specific species and values being
        projections of ``orbitals``. The following examples can
        clarify the python lingo.

        e.g.  ``items={'Sr':[0],'O':[1,2,3],'V':[4,5,6,7,8]}`` or
        ``items=dict(Sr=[0],O=[1,2,3],V=[4,5,6,7,8])``. The two
        examples are equivalent to each other. This will plot the
        following curves stacked on top of each other. projection of s
        orbital in Sr, projection of p orbitals in O and projection of
        d orbitals in V.
        The default is set to take every atom and every orbital. Which
        will be equivalent to ``mode='stack_species'``

    fermi : float, optional
        ``fermi`` defines the fermi energy. If not provided the
        fermi energy will be read from the calculation directory


    ax : matplotlib ax object, optional
        ``ax`` is a matplotlib axes. In case one wants to put plot
        generated from this plot in a different figure and treat the
        output as a subplot in a larger plot.

        e.g. ::

            >>> # Creates a figure with 3 rows and 2 colomuns
            >>> fig, axs = plt.subplots(3, 2)
            >>> x = np.linspace(-np.pi, np.pi, 1000)
            >>> y = np.sin(x)
            >>> axs[0, 0].plot(x, y)
            >>> pyprocar.dosplot(mode='plain',ax=axs[2, 2]),elimit=[-2,2])
            >>> plt.show()

    plt_show : bool, optional (default ``True``)
        whether to show the generated plot or skip to the saving.

        e.g. ``plt_show=True``

    print_plot_opts: bool, optional
        Boolean to print the plotting options

    use_cache: bool, optional
        Boolean to use cache for DOS

    verbose: int, optional
        Verbosity level

    Returns
    -------
    fig : matplotlib figure
        The generated figure

    ax : matplotlib ax object
        The generated ax for this density of states.
        If one chooses ``plt_show=False``, one can modify the plot
        using this returned object.
        e.g. ::

        >>> fig, ax = pyprocar.dosplot(mode='plain', plt_show=False)
        >>> ax.set_ylim(-2,2)
        >>> fig.show()

    """
    set_verbose_level(verbose)
    user_logger.info("If you want more detailed logs, set verbose to 2 or more")
    user_logger.info("_" * 100)

    welcome()
    default_config = ConfigFactory.create_config(PlotType.DENSITY_OF_STATES)
    config = cast(DensityOfStatesConfig, ConfigManager.merge_configs(default_config, kwargs))

    user_logger.info("_" * 100)
    if print_plot_opts:
        for key, value in default_config.as_dict().items():
            user_logger.info(f"{key} : {value}")
    user_logger.info("_" * 100)

    if mode not in DOS_MODES:
        raise ValueError(f"The mode needs to be one of {DOS_MODES}, got {mode!r}")

    if dirname is None:
        raise ValueError("dirname is required")
    dos = DensityOfStates.from_code(code, dirname, use_cache=use_cache)
    orbitals = orbital_indices(orbitals, dos)

    codes_with_scf_fermi = ["qe", "elk", "abinit"]
    if code in codes_with_scf_fermi and fermi is None:
        logger.info(f"No fermi given, using the found fermi energy: {dos.fermi}")
        fermi = dos.fermi

    if fermi is not None:
        logger.info(f"Shifting Fermi energy to zero: {fermi}")
        dos.update_points(dos.energies - fermi + fermi_shift)
        energy_label = r"Energy - E$_F$ (eV)"
    else:
        energy_label = r"Energy (eV)"
        warn_user(
            "`fermi` is not set! Set `fermi={value}`. The plot did not shift the energy by the Fermi energy."
        )

    selection = resolve_spins(
        dos.is_non_collinear, dos.n_spin_channels, spins, plain=mode == "plain"
    )
    channels, projection_spins = selection.channels, selection.projection_spins

    total = take_channels(dos.total, channels)
    if normalize_dos_mode:
        total.value = np.take(dos.normalize(normalize_dos_mode, dos.total.value), channels, axis=-1)
    if selection.joined:
        total = Property(
            name=total.name,
            value=total.value.sum(axis=-1, keepdims=True),
            units=total.units,
            label=total.label,
            point_set=dos,
            metadata={**total.metadata, "label": ["Total"]},
        )
    n_channels = total.value.shape[-1]

    plotter = DOSPlotter(orientation=orientation, ax=ax)
    line_style: dict[str, Any] = {
        "color": per_channel(config.spin_colors if n_channels > 1 else config.color, n_channels),
        "linestyle": per_channel(config.linestyle, n_channels),
        "linewidth": per_channel(config.linewidth, n_channels),
    }

    if mode == "plain" or (mode not in PARAMETRIC_MODES and config.plot_total):
        user_logger.info(f"Plotting DOS total for {mode} mode")
        plotter.plot(total, **line_style)

    if mode in PARAMETRIC_MODES:
        user_logger.info(f"Plotting DOS in {mode} mode")
        scalars = cast(
            Property,
            dos.compute_projected_sum(
                atoms=atoms,
                orbitals=orbitals,
                spins=projection_spins,
                norm_mode="total_projection",
                label=config.colorbar_title,
            ),
        )
        plotter.plot(
            total,
            scalars_data=scalars,
            scalars_mode="fill" if mode == "parametric" else "line",
            scalars_cmap=config.cmap,
            scalars_clim=config.clim or signed_clim(scalars),
            **(line_style if mode == "parametric_line" else {}),
        )
    elif mode != "plain":
        user_logger.info(f"Plotting DOS in {mode} mode")
        components = projection_components(
            dos,
            mode.split("_", 1)[1] if "_" in mode else "items",
            atoms=atoms,
            orbitals=orbitals,
            items=items,
            spins=projection_spins,
            norm_mode=normalize_dos_mode or "raw",
        )
        colors = config.colors
        if mode.startswith("stack"):
            _stack(plotter, components, colors)
        else:
            for component, color in zip(components, itertools.cycle(colors)):
                component_style: dict[str, Any] = {**line_style, "color": color}
                plotter.plot(component, **component_style)

    if fermi is not None:
        plotter.draw_fermi(
            fermi_shift,
            color=config.fermi_color,
            linestyle=config.fermi_linestyle,
            linewidth=config.fermi_linewidth,
        )

    plotter.set_energy_label(energy_label)
    plotter.set_dos_label("DOS")
    ax = cast(plt.Axes, plotter.ax)
    x_axis, y_axis = (plotter.set_xlim, ax.get_xlim), (plotter.set_ylim, ax.get_ylim)
    horizontal = plotter.orientation is AxesOrientation.HORIZONTAL
    energy_axis, dos_axis = (x_axis, y_axis) if horizontal else (y_axis, x_axis)
    for (set_limits, get_limits), requested in ((energy_axis, elimit), (dos_axis, dos_limit)):
        set_limits(as_lim(requested, get_limits()))
    plotter.set_title(config.title)

    if labels:
        plotter.legend(labels=labels)
    elif ax.get_legend_handles_labels()[1]:
        plotter.legend()

    if savefig is not None:
        plotter.fig.savefig(savefig, dpi=config.dpi, bbox_inches="tight")
    if show:
        plotter.show()

    return plotter.fig, ax


def _stack(plotter: DOSPlotter, components, colors) -> None:
    """Fill each component on top of the previous ones; later channels stack downwards."""
    energies = components[0].points
    n_energies, n_channels = components[0].to_array().shape
    signs = np.r_[1.0, -np.ones(n_channels - 1)]
    baseline = np.zeros((n_energies, n_channels))
    for component, color in zip(components, itertools.cycle(colors)):
        values = component.to_array() * signs
        top = baseline + values
        for channel in range(values.shape[1]):
            plotter.fill_between(
                energies,
                top[:, channel],
                baseline[:, channel],
                color=color,
                alpha=0.7,
                label=component.metadata.get("label", [component.label])[channel],
            )
        baseline = top
