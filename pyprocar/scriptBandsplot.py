import os
import re
import matplotlib.pyplot as plt
import numpy as np
from .io import vasp
from .plotter import EBSPlot
from .abinitparser import AbinitParser
from .elkparser import ElkParser
from .qeparser import QEParser
from .lobsterparser import LobsterParser
from .procarparser import ProcarParser
from .procarplot import ProcarPlot
from .procarselect import ProcarSelect
from .splash import welcome
from .utilsprocar import UtilsProcar

from .scriptBandsplot_old import bandsplot_old

def bandsplot(
        procar="PROCAR",
        abinit_output="abinit.out",
        poscar=None,
        outcar=None,
        kpoints=None,
        elkin="elk.in",
        mode="plain",
        linestyles=None,
        spins=None,
        atoms=None,
        orbitals=None,
        fermi=None,
        interpolation_factor=1,
        projection_mask=None,
        unfold_mask=None,
        colors=None,
        weighted_width=False,
        weighted_color=True,
        cmap="viridis",
        marker="o",
        markersize=0.02,
        linewidths=None,
        opacities=None,
        labels=None,
        vmax=None,
        vmin=None,
        grid=False,
        kticks=None,
        knames=None,
        elimit=None,
        ax=None,
        show=True,
        legend=False,
        savefig=None,
        plot_color_bar=True,
        title=None,
        kdirect=True,
        code="vasp",
        lobstercode="qe",
        unfold_mode=None,
        transformation_matrix=None,
        verbose=True,
        old=False,
):
    """


    Parameters
    ----------
    procar : TYPE, optional
        DESCRIPTION. The default is "PROCAR".
    abinit_output : TYPE, optional
        DESCRIPTION. The default is "abinit.out".
    outcar : TYPE, optional
        DESCRIPTION. The default is "OUTCAR".
    kpoints : TYPE, optional
        DESCRIPTION. The default is "KPOINTS".
    elkin : TYPE, optional
        DESCRIPTION. The default is "elk.in".
    mode : TYPE, optional
        DESCRIPTION. The default is "plain".
    spin_mode : TYPE, optional
        plain, magnetization, density, "spin_up", "spin_down", "both", "sx",
        "sy", "sz", "spin_texture"
        DESCRIPTION. The default is "plain".
    spins : TYPE, optional
        DESCRIPTION.
    atoms : TYPE, optional
        DESCRIPTION. The default is None.
    orbitals : TYPE, optional
        DESCRIPTION. The default is None.
    fermi : TYPE, optional
        DESCRIPTION. The default is None.
    mask : TYPE, optional
        DESCRIPTION. The default is None.
    colors : TYPE, optional
        DESCRIPTION.
    cmap : TYPE, optional
        DESCRIPTION. The default is "jet".
    marker : TYPE, optional
        DESCRIPTION. The default is "o".
    markersize : TYPE, optional
        DESCRIPTION. The default is 0.02.
    linewidth : TYPE, optional
        DESCRIPTION. The default is 1.
    vmax : TYPE, optional
        DESCRIPTION. The default is None.
    vmin : TYPE, optional
        DESCRIPTION. The default is None.
    grid : TYPE, optional
        DESCRIPTION. The default is False.
    kticks : TYPE, optional
        DESCRIPTION. The default is None.
    knames : TYPE, optional
        DESCRIPTION. The default is None.
    elimit : TYPE, optional
        DESCRIPTION. The default is None.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    show : TYPE, optional
        DESCRIPTION. The default is True.
    savefig : TYPE, optional
        DESCRIPTION. The default is None.
    plot_color_bar : TYPE, optional
        DESCRIPTION. The default is True.
    title : TYPE, optional
        DESCRIPTION. The default is None.
    kdirect : TYPE, optional
        DESCRIPTION. The default is True.
    code : TYPE, optional
        DESCRIPTION. The default is "vasp".
    lobstercode : TYPE, optional
        DESCRIPTION. The default is "qe".
    verbose : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    if old:
        bandsplot_old(**locals())



    # import matplotlib
    # Roman ['rm', 'cal', 'it', 'tt', 'sf',
    plt.rcParams["mathtext.default"] = "regular"
    #                                                   'bf', 'default', 'bb', 'frak',
    #                                                   'circled', 'scr', 'regular']
    plt.rcParams["font.family"] = "Arial"
    plt.rc("font", size=22)  # controls default text sizes
    plt.rc("axes", titlesize=22)  # fontsize of the axes title
    plt.rc("axes", labelsize=22)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=22)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=22)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=22)    # legend fontsize
    # plt.rc('figure', titlesize=22)  # fontsize of the figure title

    # Turn interactive plotting off
    plt.ioff()

    # Verbose section

    # First handling the options, to get feedback to the user and check
    # that the input makes sense.
    # It is quite long
    structure = None
    reciprocal_lattice = None
    kpath = None

    if code == "vasp":
        if outcar is not None:
            outcar = vasp.Outcar(outcar)
            if fermi is None:
                fermi = outcar.efermi
            reciprocal_lattice = outcar.reciprocal_lattice
        if poscar is not None:
            poscar = vasp.Poscar(poscar)
            structure = poscar.structure
            if reciprocal_lattice is None:
                reciprocal_lattice = poscar.reciprocal_lattice

        if kpoints is not None:
            kpoints = vasp.Kpoints(kpoints)
            kpath = kpoints.kpath

        procar = vasp.Procar(procar, structure, reciprocal_lattice,
                             kpath, fermi, interpolation_factor=interpolation_factor)
        ebs_plot = EBSPlot(procar.ebs, kpath, ax, spins,
                           colors, opacities, linestyles, linewidths, labels)

    if unfold_mode is not None:
        if procar.has_phase and unfold_mode != "kpath":
            ebs_plot.ebs.unfold(
                transformation_matrix=transformation_matrix, structure=structure)
        # elif unfold_mode != "kpath":
        #     print("unfolding mode chosen : {}".format(unfold_mode))
        #     raise Exception(
        #         "Unfolding requires phases of band projections. If using VASP, use LORBIT=12")
        for isegment in range(ebs_plot.kpath.nsegments):
            for ip in range(2):
                ebs_plot.kpath.special_kpoints[isegment][ip] = np.dot(
                    np.linalg.inv(transformation_matrix),ebs_plot.kpath.special_kpoints[isegment][ip])

    if mode == "plain":
        ebs_plot.plot_bands()
    else:
        if atoms is not None and type(atoms[0]) is str:
            atoms_str = atoms
            atoms = []
            for iatom in np.unique(atoms_str):
                atoms = np.append(atoms, np.where(
                    structure.atoms == iatom)[0]).astype(np.int)

        if orbitals is not None and type(orbitals[0]) is str:
            orbital_str = orbitals
            orbital_names = {'s': 0,
                             'p': [1, 2, 3],
                             'd': [4, 5, 6, 7, 8],
                             'f': [9, 10, 11, 12, 13, 14, 15],
                             "py": 1,
                             "pz": 2,
                             "px": 3,
                             "dxy": 4,
                             "dyz": 5,
                             "dz2": 6,
                             "dxz": 7,
                             "x2-y2": 8,
                             "fy3x2": 9,
                             "fxyz": 10,
                             "fyz2": 11,
                             "fz3": 12,
                             "fxz2": 13,
                             "fzx2": 14,
                             }
            orbitals = []
            for iorb in orbital_str:
                orbitals = np.append(
                    orbitals, orbital_names[iorb]).astype(np.int)
        weights = ebs_plot.ebs.ebs_sum(
            atoms=atoms, principal_q_numbers=[-1], orbitals=orbitals, spins=spins)

        if weighted_color:
            color_weights = weights
        else:
            color_weights = None
        if weighted_width:
            width_weights = weights
        else:
            width_weights = None
        color_mask = projection_mask
        width_mask = projection_mask
        if unfold_mode == "thickness":
            width_weights = ebs_plot.ebs.weights
            width_mask = unfold_mask
        elif unfold_mode == "color":
            color_weights = ebs_plot.ebs.weights
            color_mask = unfold_mask
        elif unfold_mode is not None:
            if weighted_width:
                width_weights = ebs_plot.ebs.weights
                width_mask = unfold_mask
            if weighted_color:
                color_weights = ebs_plot.ebs.weights
                color_mask = unfold_mask

        if mode == "parametric":
            ebs_plot.plot_parameteric(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                cmap=cmap,
                plot_color_bar=plot_color_bar,
                vmin=vmin,
                vmax=vmax)
        elif mode == "scatter":
            ebs_plot.plot_scatter(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                cmap=cmap,
                plot_color_bar=plot_color_bar,
                vmin=vmin,
                vmax=vmax)
        else:
            print("Selected mode %s not valid. Please check the spelling " % mode)

    ebs_plot.set_xticks()
    ebs_plot.set_yticks(interval=elimit)
    ebs_plot.set_xlim()
    ebs_plot.set_ylim(elimit)
    ebs_plot.draw_fermi()
    ebs_plot.set_ylabel()
    if legend:
        ebs_plot.legend()
    if savefig is not None:
        plt.savefig(savefig)
    if show:
        plt.show()
    return ebs_plot


