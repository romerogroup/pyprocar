from .elkparser import ElkParser
from .splash import welcome
from .doscarplot import DosPlot
import numpy as np
import matplotlib.pyplot as plt


#import matplotlib 
plt.rcParams['mathtext.default'] = 'regular' #Roman ['rm', 'cal', 'it', 'tt', 'sf', 
#                                                   'bf', 'default', 'bb', 'frak',
#                                                   'circled', 'scr', 'regular']
plt.rcParams["font.family"] = "Georgia"
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick',labelsize=22)    # fontsize of the tick labels
plt.rc('ytick',labelsize=22)    # fontsize of the tick labels
#plt.rc('legend', fontsize=22)    # legend fontsize
#plt.rc('figure', titlesize=22)  # fontsize of the figure title



def dosplot(
    file='vasprun.xml',
    mode="plain",
    orientation='horizontal',
    spin_colors=None,
    colors=None,
    spins=None,
    atoms=None,
    orbitals=None,
    elimit=None,
    mask=None,
    markersize=0.02,
    cmap="jet",
    vmax=None,
    vmin=None,
    grid=True,
    marker="o",
    savefig=None,
    title=None,
    plot_total=None,
    code="vasp",
    labels = None,
    items = {},
    ax=None
):

    """This function plots density of states
  """
    welcome()
    total = plot_total
    code = code.lower()
    if code == "vasp":
        dos_plot = DosPlot(file)
        vaspxml = dos_plot.VaspXML
        if atoms is None:
            atoms = list(np.arange(vaspxml.initial_structure.natom,
                                   dtype=int))
        if spins is None:
            spins = list(np.arange(vaspxml.dos_total.ncols))
        if orbitals is None:
            orbitals = list(np.arange(
                        (len(vaspxml.dos_projected[0].labels)-1)//2,
                        dtype=int))
        if elimit is None:
            elimit = [vaspxml.dos_total.energies.min(),
                      vaspxml.dos_total.energies.max()]

#    elif code == "abinit":
#        procarFile = ProcarParser()
#        abinitFile = AbinitParser(abinit_output=abinit_output)
#
#    elif code == "elk":
#        # reciprocal lattice already taken care of
#        procarFile = ElkParser(kdirect=kdirect)

    print("Script initiated")
    print("code          : ", code)
    print("input file    : ", file)
    print("Mode          : ", mode)
    print("spin comp.    : ", spins)
    print("atoms list   : ", atoms)
    print("orbs. list   : ", orbitals)

    print("Fermi Energy   : ", vaspxml.fermi)
    print("Energy range  : ", elimit)

    print("Colormap      : ", cmap)
    print("MarkerSize    : ", markersize)

    print("vmax          : ", vmax)
    print("vmin          : ", vmin)
    print("grid enabled  : ", grid)

    print("Savefig       : ", savefig)
    print("title         : ", title)

    
    if mode == "plain":
        fig, ax1 = dos_plot.plot_total(spins=spins,
                                      markersize=markersize,
                                      marker=marker,
                                      spin_colors=spin_colors,
                                      ax=ax,
                                      orientation=orientation,
                                      )
        dos = dos_plot.VaspXML.dos_total

    elif mode == 'parametric1':
        if not total:
            fig, ax1 = dos_plot.plot_parametric_line(
                    atoms=atoms,
                    spins=spins,
                    orbitals=orbitals,
                    markersize=markersize,
                    marker=marker,
                    spin_colors=spin_colors,
                    ax=ax,
                    orientation=orientation,
                    labels=labels
                    )
            dos = dos_plot.VaspXML.dos_parametric(atoms=atoms,
                                                  spin=spins,
                                                  orbitals=orbitals,
                                                  )
        else:
            fig, ax1 = dos_plot.plot_total(spins=spins,
                                          markersize=markersize,
                                          marker=marker,
                                          spin_colors=[(0,0,0),(0,0,0)],
                                          ax=ax,
                                          orientation=orientation,
                                          )
            dos = dos_plot.VaspXML.dos_total
            _, ax1 = dos_plot.plot_parametric_line(
                    atoms=atoms,
                    spins=spins,
                    orbitals=orbitals,
                    markersize=markersize,
                    marker=marker,
                    spin_colors=spin_colors,
                    ax=ax1,
                    orientation=orientation,
                    labels=labels,
                    )
    elif mode == 'parametric2':
        if not total:
            fig, ax1 = dos_plot.plot_parametric(
                    atoms=atoms,
                    spins=spins,
                    orbitals=orbitals,
                    markersize=markersize,
                    marker=marker,
                    spin_colors=spin_colors,
                    cmap=cmap,
                    elimit=elimit,
                    ax=ax,
                    orientation=orientation,
                    labels=labels,
                    )
            dos = dos_plot.VaspXML.dos_total

        else:
            fig, ax1 = dos_plot.plot_parametric(
                    atoms=atoms,
                    spins=spins,
                    orbitals=orbitals,
                    markersize=markersize,
                    marker=marker,
                    spin_colors=spin_colors,
                    cmap=cmap,
                    elimit=elimit,
                    ax=ax,
                    orientation=orientation,
                    labels=labels,
                    )
            dos = dos_plot.VaspXML.dos_total
            _, ax1 = dos_plot.plot_total(spins=spins,
                                        markersize=markersize,
                                        marker=marker,
                                        spin_colors=[(0,0,0),(0,0,0)],
                                        ax=ax1,
                                        orientation=orientation,
                                        )

    elif mode == 'stack_species':
        if not total:
            fig, ax1 = dos_plot.plot_stack_species(spins=spins,
                                                    orbitals=orbitals,
                                                    markersize=markersize,
                                                    marker=marker,
                                                    spin_colors=spin_colors,
                                                    colors=colors,
                                                    elimit=elimit,
                                                    figsize=(12, 6),
                                                    ax=ax,
                                                    orientation=orientation,
                                                    )
            dos = dos_plot.VaspXML.dos_total
        else:
            fig, ax1 = dos_plot.plot_stack_species(spins=spins,
                                                    orbitals=orbitals,
                                                    markersize=markersize,
                                                    marker=marker,
                                                    spin_colors=spin_colors,
                                                    colors=colors,
                                                    elimit=elimit,
                                                    figsize=(12, 6),
                                                    ax=ax,
                                                    orientation=orientation,
                                                    )
            dos = dos_plot.VaspXML.dos_total
            _, ax1 = dos_plot.plot_total(spins=spins,
                                        markersize=markersize,
                                        marker=marker,
                                        spin_colors=[(0,0,0),(0,0,0)],
                                        ax=ax1,
                                        orientation=orientation,
                                        )

    elif mode == 'stack_orbitals':
        if not total:
            fig, ax1 = dos_plot.plot_stack_orbitals(spins=spins,
                                                     atoms=atoms,
                                                     markersize=markersize,
                                                     marker=marker,
                                                     spin_colors=spin_colors,
                                                     colors=colors,
                                                     elimit=elimit,
                                                     figsize=(12, 6),
                                                     ax=ax,
                                                     orientation=orientation,
                                                     )
            dos = dos_plot.VaspXML.dos_total
        else:
            fig, ax1 = dos_plot.plot_stack_orbitals(spins=spins,
                                                     atoms=atoms,
                                                     markersize=markersize,
                                                     marker=marker,
                                                     spin_colors=spin_colors,
                                                     colors=colors,
                                                     elimit=elimit,
                                                     figsize=(12, 6),
                                                     ax=ax,
                                                     orientation=orientation,
                                                    )
            dos = dos_plot.VaspXML.dos_total
            _, ax1 = dos_plot.plot_total(spins=spins,
                                        markersize=markersize,
                                        marker=marker,
                                        spin_colors=[(0, 0, 0), (0, 0, 0)],
                                        ax=ax1,
                                        orientation=orientation,
                                        )

    elif mode == 'stack':
        if not total:
            fig, ax1 = dos_plot.plot_stack(
                    items=items,
                    spins=spins,
                    markersize=markersize,
                    marker=marker,
                    spin_colors=spin_colors,
                    colors=colors,
                    elimit=elimit,
                    figsize=(12, 6),
                    ax=ax,
                    orientation=orientation,
                    )

            dos = dos_plot.VaspXML.dos_total
        else:
            fig, ax1 = dos_plot.plot_stack(
                    items=items,
                    spins=spins,
                    markersize=markersize,
                    marker=marker,
                    spin_colors=colors,
                    colors=colors,
                    elimit=elimit,
                    figsize=(12, 6),
                    ax=ax,
                    orientation=orientation,
                    )
            dos = dos_plot.VaspXML.dos_total
            _, ax1 = dos_plot.plot_total(spins=spins,
                                        markersize=markersize,
                                        marker=marker,
                                        spin_colors=[(0, 0, 0), (0, 0, 0)],
                                        ax=ax1,
                                        orientation=orientation,
                                        )

                
    cond1 = dos.energies > elimit[0]
    cond2 = dos.energies < elimit[1]
    cond = np.all([cond1, cond2], axis=0)

    if len(spins) > 1:
        ylim=[dos.values[cond][:, 1].max() * -1.1,
              dos.values[cond][:, 0].max() * 1.1]
    else:
        ylim=[0, dos.dos[cond][:, spins[0]+1].max() * 1.1]

    if orientation == 'horizontal':
        ax1.set_xlabel(r"$E-E_f$ [eV]")
        ax1.set_ylabel('Density of States [a.u.]')
        ax1.set_xlim(elimit)
        ax1.set_ylim(ylim)
        
    elif orientation == 'vertical':
        ax1.set_ylabel(r"$E-E_f$ [eV]")
        ax1.set_xlabel('Density of States [a.u.]')
        ax1.set_ylim(elimit)
        ax1.set_xlim(ylim)  # we use ylim because the plot is vertiacal

    
    ax1.axhline(color="black", linestyle="--")
    ax1.axvline(color="black", linestyle="--")   
    

    
    fig.tight_layout()
    if grid:
        ax1.grid()

    ax1.legend()
    if title:
        ax1.set_title(title, fontsize=22)

    else:
        if savefig:
            fig.savefig(savefig, bbox_inches="tight")
            plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
            return None,None
        else:
            plt.show()
    
    return fig,ax1
#
#
## if __name__ == "__main__":
## bandsplot(mode='parametric',elimit=[-6,6],orbitals=[4,5,6,7,8],vmin=0,vmax=1, code='elk')
## knames=['$\Gamma$', '$X$', '$M$', '$\Gamma$', '$R$','$X$'],
## kticks=[0, 8, 16, 24, 38, 49])



