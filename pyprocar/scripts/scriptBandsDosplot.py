__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import inspect
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pyprocar.utils.info import orbital_names

from ..io import qe, vasp
from ..plotter import DOSPlot, EBSPlot
from ..utils import welcome
from ..utils.defaults import settings
from .scriptBandsplot import bandsplot
from .scriptDosplot import dosplot

bands_settings = {
    key: value
    for key, value in zip(
        inspect.getfullargspec(bandsplot).args,
        inspect.getfullargspec(bandsplot).defaults,
    )
}
dos_settings = {
    key: value
    for key, value in zip(
        inspect.getfullargspec(dosplot).args, inspect.getfullargspec(dosplot).defaults
    )
}


def bandsdosplot(
    bands_settings: dict = bands_settings,
    dos_settings: dict = dos_settings,
    dos_limit: List[int] = None,
    elimit: List[int] = None,
    k_limit=None,
    grid: bool = False,
    code: str = "vasp",
    lobster: bool = False,
    savefig: str = None,
    title: str = None,
    title_fontsize: float = 16,
    discontinuities=None,
    draw_fermi: bool = True,
    plot_color_bar: bool = True,
    repair: bool = True,
    show: bool = True,
    dpi: int = 300,
    figsize=(8, 6),
    **kwargs,
):
    """A function to plot the band structure and the density of states in the same plot

    Parameters
    ----------
    bands_settings : dict, optional
        A dictionary containing the keyword arguments from bandsplot, by default bands_settings
    dos_settings : dict, optional
         A dictionary containing the keyword arguments from dosplot, by default dos_settings
    dos_limit : List[int], optional
        The dos window to plot, by default None
    elimit : List[int], optional
        The energy window to plot, by default None
    k_limit : _type_, optional
        The kpath points to plot, by default None
    grid : bool, optional
        Boolean to plot a grid, by default False
    code : str, optional
        The code to use, by default "vasp"
    lobster : bool, optional
        Boolean if this is a lobster calculation, by default False
    savefig : str, optional
        The filename to to save the plot as., by default None
    title : str, optional
        String for the title name, by default None
    title_fontsize : float, optional
        Float for the title size, by default 16
    discontinuities : _type_, optional
        _description_, by default None
    draw_fermi : bool, optional
        Boolean to plot the fermi level, by default True
    plot_color_bar : bool, optional
        Boolean to plot the color bar, by default True
    repair : bool, optional
        Boolean to repair the PROCAR file, by default True
    show : bool, optional
        Boolean to show the plot, by default True
    """

    welcome()

    # inital settings
    bands_settings["code"] = code
    dos_settings["code"] = code
    dos_settings["orientation"] = "vertical"
    bands_settings["show"] = False
    dos_settings["show"] = False

    # parses old elements
    # bands_settings, dos_settings = parse_kwargs(kwargs,bands_settings, dos_settings)
    

    # plots bandsplot and dosplot
    
    plt.close("all")
    # fig = plt.figure(figsize=figsize, clear=True, dpi=dpi)
    
    # fig, axes = plt.subplots(1, 2, figsize=figsize, clear=True, dpi=dpi)
    
    if "plain" != dos_settings["mode"] and "plain" != bands_settings["mode"]:
        bands_settings["plot_color_bar"] = False
        
    # Make the two axes share the y-axis (energy axis)
    fig, axes = plt.subplots(1, 2, figsize=figsize, clear=True, dpi=dpi, sharey=True)
    ebs_plot_fig, ebs_plot_ax = bandsplot(ax=axes[0], **bands_settings)
    edos_plot_fig, edos_plot_ax = dosplot(ax=axes[1], **dos_settings)

    # # combines bandsplot and dos plot
    # ax_ebs, ax_dos = combine_axes(
    #     ebs_plot_fig, edos_plot_fig, fig, plot_color_bar=plot_color_bar
    # )
    ax_ebs = axes[0]
    ax_dos = axes[1]
    ax_dos.set_ylabel("")
    
    # axes opitions
    if elimit is not None:
        ax_dos.set_ylim(elimit)
        ax_ebs.set_ylim(elimit)
    if dos_limit is not None:
        ax_dos.set_xlim(dos_limit)
    if k_limit is not None:
        ax_ebs.set_xlim(k_limit)
    if grid:
        ax_ebs.grid()
        ax_dos.grid()
    if draw_fermi:
        ax_ebs.axhline(
            y=0,
            color=settings.dos.fermi_color,
            linestyle=settings.dos.fermi_linestyle,
            linewidth=settings.dos.fermi_linewidth,
        )
        ax_dos.axhline(
            y=0,
            color=settings.dos.fermi_color,
            linestyle=settings.dos.fermi_linestyle,
            linewidth=settings.dos.fermi_linewidth,
        )

    if title is not None:
        fig.suptitle(title, title_fontsize=title_fontsize)

    if savefig:
        fig.set_size_inches(figsize)
        plt.savefig(savefig, dpi=dpi)
        plt.clf()
    if show:
        plt.show()

    return fig, ax_ebs, ax_dos


def combine_axes(fig_ebs, fig_dos, fig, plot_color_bar=True):

    # Changes link of axes to old to new figure. Then adds the axes to the current figure

    ax_ebs = fig_ebs.axes[0]
    ax_dos = fig_dos.axes[0]

    ax_ebs.figure = fig
    fig.axes.append(ax_ebs)
    fig.add_axes(ax_ebs)

    ax_dos.figure = fig
    fig.axes.append(ax_dos)
    fig.add_axes(ax_dos)

    ax_color_bar = None
    if len(fig_ebs.axes) != 1 and plot_color_bar:
        ax_color_bar = fig_ebs.axes[1]
        ax_color_bar.figure = fig
        fig.axes.append(ax_color_bar)
        fig.add_axes(ax_color_bar)
    elif len(fig_dos.axes) != 1 and plot_color_bar:
        ax_color_bar = fig_dos.axes[1]
        ax_color_bar.figure = fig
        fig.axes.append(ax_color_bar)
        fig.add_axes(ax_color_bar)

    # Changeing location of dos plot
    dos_position = list(fig.axes[1].get_position().bounds)
    ebs_position = list(fig.axes[0].get_position().bounds)
    dos_position[0] = ebs_position[0] + ebs_position[3] + 0.025

    fig.axes[1].set_position(dos_position)

    # Formating dos plot to be comatible with band structure plot
    fig.axes[1].axes.set_ylabel("")
    fig.axes[1].axes.set_yticklabels([])
    fig.axes[1].sharey(fig.axes[0])

    fig.axes[1].axes.get_yaxis().set_visible(False)

    # Handles existing colorbars
    if ax_color_bar is not None:
        dos_position = list(fig.axes[1].get_position().bounds)
        color_bar_position = list(fig.axes[2].get_position().bounds)

        color_bar_position[0] = dos_position[0] + dos_position[3] - 0.1
        fig.axes[2].set_position(color_bar_position)

    return fig.axes[0], fig.axes[1]


def parse_kwargs(kwargs, bands_settings, dos_settings):
    for key, value in kwargs.items():
        if key == "dos_file":
            dos_settings["filename"] = value
        if key == "dos_dirname":
            dos_settings["dirname"] = value
        if key == "bands_dirname":
            bands_settings["dirname"] = value
        if key == "kpoints":
            bands_settings["kpoints"] = value
        if key == "bands_mode":
            bands_settings["mode"] = value
        if key == "dos_mode":
            dos_settings["mode"] = value
        if key == "dos_plot_total":
            dos_settings["plot_total"] = value
        if key == "fermi":
            bands_settings["fermi"] = value

        # if key is "mask":
        #     bands_settings["mask"] = value
        # if key is "markersize":
        #     bands_settings["maerkersize"] = value
        # if key is "marker":
        #     bands_settings["marker"] = value

        if key == "atoms":
            bands_settings["atoms"] = value
            dos_settings["atoms"] = value
        if key == "orbitals":
            bands_settings["orbitals"] = value
            dos_settings["orbitals"] = value

        if key == "bands_spin":
            dos_settings["spins"] = value
        if key == "dos_spin":
            dos_settings["spins"] = value
        if key == "dos_labels":
            dos_settings["spin_labels"] = value
        if key == "dos_spin_colors":
            dos_settings["spin_colors"] = value
        if key == "dos_colors":
            dos_settings["colors"] = value
        if key == "dos_title":
            dos_settings["title"] = value
        if key == "items":
            bands_settings["items"] = value
            dos_settings["items"] = value
        if key == "dos_interpolation_factor":
            dos_settings["interpolation_factor"] = value

        if key == "vmin":
            bands_settings["vmin"] = value
            dos_settings["vmin"] = value
        if key == "vmax":
            bands_settings["vmax"] = value
            dos_settings["vmax"] = value
        if key == "cmap":

            dos_settings["cmap"] = value

        if key == "kdirect":
            bands_settings["kdirect"] = value

    return bands_settings, dos_settings
