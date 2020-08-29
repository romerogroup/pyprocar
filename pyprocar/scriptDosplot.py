"""
Created on May 17 2020
@author: Pedram Tavadze
"""
# from .elkparser import ElkParser
from .splash import welcome
from .doscarplot import DosPlot
from .vaspxml import VaspXML
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
plt.rcParams["mathtext.default"] = "regular"
# Roman ['rm', 'cal', 'it', 'tt', 'sf',
#        'bf', 'default', 'bb', 'frak',
#        'circled', 'scr', 'regular']
plt.rcParams["font.family"] = "Georgia"
plt.rc("font", size=18)  # controls default text sizes
plt.rc("axes", titlesize=22)  # fontsize of the axes title
plt.rc("axes", labelsize=22)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=22)  # fontsize of the tick labels
plt.rc("ytick", labelsize=22)  # fontsize of the tick labels
# plt.rc('legend', fontsize=22)    # legend fontsize
# plt.rc('figure', titlesize=22)  # fontsize of the figure title


def dosplot(
        filename="vasprun.xml",
        mode="plain",
        interpolation_factor=None,
        orientation="horizontal",
        spin_colors=None,
        colors=None,
        spins=None,
        atoms=None,
        orbitals=None,
        elimit=None,
        dlimit=None,
        cmap="jet",
        vmax=None,
        vmin=None,
        grid=False,
        savefig=None,
        title=None,
        plot_total=True,
        code="vasp",
        labels=None,
        items={},
        ax=None,
        plt_show=True,
):
    """This function plots density of states

    """

    if mode not in [
            'plain', 'parametric_line', 'parametric', 'stack_species',
            'stack_orbitals', 'stack'
    ]:
        raise ValueError(
            "Mode should be choosed from ['plain', 'parametric_line','parametric','stack_species','stack_orbitals','stack']"
        )

    welcome()

    # Verbose section
    print("Script initiated")
    print("code          : ", code)
    print("File name     : ", filename)
    print("mode          : ", mode)
    print("spins         : ", spins)
    print("atoms list    : ", atoms)
    print("orbs. list    : ", orbitals)
    print("energy range  : ", elimit)
    print("colormap      : ", cmap)
    print("vmax          : ", vmax)
    print("vmin          : ", vmin)
    print("grid enabled  : ", grid)
    print("savefig       : ", savefig)
    print("title         : ", title)

    total = plot_total
    code = code.lower()
    if code == "vasp":
        vaspxml = VaspXML(filename=filename,
                          dos_interpolation_factor=interpolation_factor)
        dos_plot = DosPlot(dos=vaspxml.dos, structure=vaspxml.structure)
        if atoms is None:
            atoms = list(np.arange(vaspxml.structure.natoms, dtype=int))
        if spins is None:
            spins = list(np.arange(len(vaspxml.dos.total)))
        if orbitals is None:
            orbitals = list(
                np.arange(len(vaspxml.dos.projected[0][0]), dtype=int))
        if elimit is None:
            elimit = [vaspxml.dos.energies.min(), vaspxml.dos.energies.max()]
    if mode == "plain":
        fig, ax1 = dos_plot.plot_total(
            spins=spins,
            spin_colors=spin_colors,
            ax=ax,
            orientation=orientation,
            labels=labels,
        )

    elif mode == "parametric_line":
        if not total:
            fig, ax1 = dos_plot.plot_parametric_line(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                ax=ax,
                orientation=orientation,
                labels=labels,
            )

        else:
            fig, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax,
                orientation=orientation,
            )
            _, ax1 = dos_plot.plot_parametric_line(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                ax=ax1,
                orientation=orientation,
                labels=labels,
            )
    elif mode == "parametric":
        if not total:
            fig, ax1 = dos_plot.plot_parametric(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                elimit=elimit,
                ax=ax,
                orientation=orientation,
                labels=labels,
            )
        else:
            fig, ax1 = dos_plot.plot_parametric(
                atoms=atoms,
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                elimit=elimit,
                ax=ax,
                orientation=orientation,
                labels=labels,
            )

            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
            )

    elif mode == "stack_species":
        if not total:
            fig, ax1 = dos_plot.plot_stack_species(
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )
        else:
            fig, ax1 = dos_plot.plot_stack_species(
                spins=spins,
                orbitals=orbitals,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )
            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
            )

    elif mode == "stack_orbitals":
        if not total:
            fig, ax1 = dos_plot.plot_stack_orbitals(
                spins=spins,
                atoms=atoms,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

        else:
            fig, ax1 = dos_plot.plot_stack_orbitals(
                spins=spins,
                atoms=atoms,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
            )

    elif mode == "stack":
        if not total:
            fig, ax1 = dos_plot.plot_stack(
                items=items,
                spins=spins,
                spin_colors=spin_colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

        else:
            fig, ax1 = dos_plot.plot_stack(
                items=items,
                spins=spins,
                spin_colors=colors,
                colors=colors,
                elimit=elimit,
                figsize=(12, 6),
                ax=ax,
                orientation=orientation,
            )

            _, ax1 = dos_plot.plot_total(
                spins=spins,
                spin_colors=[(0, 0, 0), (0, 0, 0)],
                ax=ax1,
                orientation=orientation,
            )

    cond1 = vaspxml.dos.energies >= elimit[0]
    cond2 = vaspxml.dos.energies <= elimit[1]
    cond = np.all([cond1, cond2], axis=0)

    if dlimit is not None:
        ylim = dlimit
    else:
        if len(spins) > 1:
            ylim = [
                vaspxml.dos.total[1, cond].max() * -1.1,
                vaspxml.dos.total[0, cond].max() * 1.1
            ]
        else:
            ylim = [0, vaspxml.dos.total[spins[0], cond].max() * 1.1]

    if orientation == "horizontal":
        ax1.set_xlabel(r"$E-E_f$ [eV]")
        ax1.set_ylabel("Density of States [a.u.]")
        ax1.set_xlim(elimit)
        ax1.set_ylim(ylim)

    elif orientation == "vertical":
        ax1.set_ylabel(r"$E-E_f$ [eV]")
        ax1.set_xlabel("Density of States [a.u.]")
        ax1.set_ylim(elimit)
        ax1.set_xlim(ylim)  # we use ylim because the plot is vertiacal

    ax1.axhline(color="black", linestyle="--")
    ax1.axvline(color="black", linestyle="--")

    fig.tight_layout()
    if grid:
        ax1.grid()
    if labels or "stack" in mode:
        ax1.legend()
    if title:
        ax1.set_title(title, fontsize=17)

    if savefig:

        fig.savefig(savefig, bbox_inches="tight")
        plt.close(
        )  # Added by Nicholas Pike to close memory issue of looping and creating many figures
        return None, None
    else:
        plt.show(block=plt_show)

    return fig, ax1


#
#
## if __name__ == "__main__":
## bandsplot(mode='parametric',elimit=[-6,6],orbitals=[4,5,6,7,8],vmin=0,vmax=1, code='elk')
## knames=['$\Gamma$', '$X$', '$M$', '$\Gamma$', '$R$','$X$'],
## kticks=[0, 8, 16, 24, 38, 49])
