from ..splash import welcome

from ..io import vasp, qe 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..utils.info import orbital_names
from .scriptBandsDosplot_old import bandsdosplot_old

from .scriptDosplot import dosplot
from .scriptBandsplot import bandsplot
from ..plotter import EDOSPlot, EBSPlot

from ..utils.defaults import settings

import inspect
bands_settings = {key:value for key,value in zip(inspect.getfullargspec(bandsplot).args,inspect.getfullargspec(bandsplot).defaults)}
dos_settings = {key:value for key,value in zip(inspect.getfullargspec(dosplot).args,inspect.getfullargspec(dosplot).defaults)}


def bandsdosplot(
    bands_settings = bands_settings,
    dos_settings = dos_settings,
    dos_limit=None,
    elimit=None,
    k_limit = None,
    grid=False,
    code="vasp",
    savefig=None,
    title=None,
    title_fontsize = 16,
    discontinuities=None,
    draw_fermi = True,
    plot_color_bar=True,
    repair=True,
    old = False,
    show=True,
    **kwargs
    ):
    """This function creates plots containing both DOS and bands."""

    welcome()

    if old or code not in ('vasp', "qe"):
        bandsdosplot_old(**locals())

    #inital settings
    bands_settings['code'] = code
    dos_settings['code'] = code
    dos_settings['orientation'] = 'vertical'
    bands_settings['show'] = False
    dos_settings['show'] = False

    # parses old elements
    bands_settings, dos_settings = parse_kwargs(kwargs,bands_settings, dos_settings)

    #plots bandsplot and dosplot
    ebs_plot = bandsplot(**bands_settings)
    edos_plot = dosplot(**dos_settings)

    plt.close('all')
    fig = plt.figure(figsize = (16.5,5.5), clear = True)

    # combines bandsplot and dos plot
    ax_ebs,ax_dos = combine_axes(ebs_plot.fig,edos_plot.fig,fig, plot_color_bar = plot_color_bar)

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
    if draw_fermi is True:
        ax_ebs.axhline(y=0, color=settings.edos.fermi_color, linestyle=settings.edos.fermi_linestyle, linewidth=settings.edos.fermi_linewidth)
        ax_dos.axhline(y=0, color=settings.edos.fermi_color, linestyle=settings.edos.fermi_linestyle, linewidth=settings.edos.fermi_linewidth)
    
    if title is not None:
        fig.suptitle(title, title_fontsize=title_fontsize)
        

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
        plt.clf()
    if show:
        plt.show()

def combine_axes(fig_ebs,fig_dos,fig, plot_color_bar = True):

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

    #Changeing location of dos plot
    dos_position = list(fig.axes[1].get_position().bounds)
    ebs_position = list(fig.axes[0].get_position().bounds)
    dos_position[0] = ebs_position[0] + ebs_position[3]  + 0.025

    fig.axes[1].set_position(dos_position)

    #Formating dos plot to be comatible with band structure plot
    fig.axes[1].sharey(fig.axes[0])
    fig.axes[1].axes.set_ylabel("")
    fig.axes[1].axes.set_yticklabels([])

    #Handles existing colorbars
    if ax_color_bar is not None:
        dos_position = list(fig.axes[1].get_position().bounds)
        color_bar_position = list(fig.axes[2].get_position().bounds)

        color_bar_position[0] = dos_position[0] + dos_position[3]  + 0.025
        fig.axes[2].set_position(color_bar_position)

    return fig.axes[0],fig.axes[1]

def parse_kwargs(kwargs,bands_settings, dos_settings):
    for key, value in kwargs.items():
        if key is "dos_file":
            dos_settings["filename"] = value
        if key is "dos_dirname":
            dos_settings["dirname"] = value
        if key is "bands_dirname":
            bands_settings["dirname"] = value
        if key is "procar":
            bands_settings["procar"] = value
            dos_settings["procar"] = value
        if key is "outcar":
            bands_settings["outcar"] = value
            dos_settings["outcar"] = value
        if key is "poscar":
            bands_settings["poscar"] = value
            dos_settings["poscar"] = value
        if key is "kpoints":
            bands_settings["kpoints"] = value
        if key is "abinit_output":
            bands_settings["abinit_output"] = value
        if key is "bands_mode":
            bands_settings["mode"] = value
        if key is "dos_mode":
            dos_settings["mode"] = value
        if key is "dos_plot_total":
            dos_settings["plot_total"] = value
        if key is "fermi":
            bands_settings["fermi"] = value

        # if key is "mask":
        #     bands_settings["mask"] = value
        # if key is "markersize":
        #     bands_settings["maerkersize"] = value
        # if key is "marker":
        #     bands_settings["marker"] = value

        if key is "atoms":
            bands_settings["atoms"] = value
            dos_settings["atoms"] = value
        if key is "orbitals":
            bands_settings["orbitals"] = value
            dos_settings["orbitals"] = value

        if key is "bands_spin":
           dos_settings["spins"] = value
        if key is "dos_spin":
            dos_settings["spins"] = value
        if key is "dos_labels":
            dos_settings["spin_labels"] = value
        if key is "dos_spin_colors":
            dos_settings["spin_colors"] = value
        if key is "dos_colors":
            dos_settings["colors"] = value
        if key is "dos_title":
            dos_settings["title"] = value
        if key is "items":
            bands_settings["items"] = value
            dos_settings["items"] = value
        if key is "dos_interpolation_factor":
            dos_settings["interpolation_factor"] = value

        if key is "vmin":
            bands_settings["vmin"] = value
            dos_settings["vmin"] = value
        if key is "vmax":
            bands_settings["vmax"] = value
            dos_settings["vmax"] = value
        if key is "cmap":

            dos_settings["cmap"] = value

        if key is "kdirect":
            bands_settings["kdirect"] = value



    return bands_settings, dos_settings



        
            


