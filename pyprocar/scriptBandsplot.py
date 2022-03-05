import os
import re

# import matplotlib.pyplot as plt
import numpy as np
from .utils.info import orbital_names
from . import io
from .plotter import EBSPlot
from .splash import welcome
from .scriptBandsplot_old import bandsplot_old
from .utils.defaults import settings


def bandsplot(
    procar="PROCAR",
    abinit_output="abinit.out",
    dirname = None,
    poscar=None,
    outcar=None,
    kpoints=None,
    elkin="elk.in",
    code="vasp",
    mode="plain",
    spins=None,
    atoms=None,
    orbitals=None,
    items=None,
    fermi=None,
    interpolation_factor=1,
    interpolation_type="cubic",
    projection_mask=None,
    vmax=None,
    vmin=None,
    kticks=None,
    knames=None,
    kdirect=True,
    elimit=None,
    ax=None,
    show=True,
    savefig=None,
    old=False,
    **kwargs,
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
    if old or code not in ('vasp', "qe"):
        procarfile = procar
        bandsplot_old(**locals())

    # Turn interactive plotting off
    # plt.ioff()

    # Verbose section

    structure = None
    reciprocal_lattice = None
    kpath = None
    ebs = None
    kpath = None

    settings.modify(kwargs)

    ebs, kpath, structure, reciprocal_lattice = parse(
        code, dirname ,outcar, poscar, procar, reciprocal_lattice, kpoints,
        interpolation_factor, fermi)

    ebs_plot = EBSPlot(ebs, kpath, ax, spins)

 
    labels = []
    if mode == "plain":
        ebs_plot.plot_bands()

    elif mode in ["overlay", "overlay_species", "overlay_orbitals"]:
        weights = []
        if mode == "overlay_species":
            for ispc in structure.species:
                labels.append(ispc)
                atoms = np.where(structure.atoms == ispc)[0]
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )
                weights.append(w)
        if mode == "overlay_orbitals":
            for iorb in ["s", "p", "d", "f"]:
                if iorb == "f" and not ebs_plot.ebs.norbitals > 9:
                    continue
                labels.append(iorb)
                orbitals = orbital_names[iorb]
                w = ebs_plot.ebs.ebs_sum(
                    atoms=atoms,
                    principal_q_numbers=[-1],
                    orbitals=orbitals,
                    spins=spins,
                )
                weights.append(w)

        elif mode == "overlay":
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
                                ).astype(np.int)
                            labels.append(ispc + "-" + "".join(it[ispc]))
                        else:
                            orbitals = it[ispc]
                            labels.append(ispc + "-" + "_".join(it[ispc]))
                        w = ebs_plot.ebs.ebs_sum(
                            atoms=atoms,
                            principal_q_numbers=[-1],
                            orbitals=orbitals,
                            spins=spins,
                        )
                        weights.append(w)
        ebs_plot.plot_parameteric_overlay(
            spins=spins, vmin=vmin, vmax=vmax, weights=weights
        )
    else:
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
        weights = ebs_plot.ebs.ebs_sum(
            atoms=atoms, principal_q_numbers=[-1], orbitals=orbitals, spins=spins
        )

        if settings.ebs.weighted_color:
            color_weights = weights
        else:
            color_weights = None
        if settings.ebs.weighted_width:
            width_weights = weights
        else:
            width_weights = None
        color_mask = projection_mask
        width_mask = projection_mask
        if mode == "parametric":
            ebs_plot.plot_parameteric(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                vmin=vmin,
                vmax=vmax,
            )
        elif mode == "scatter":
            ebs_plot.plot_scatter(
                color_weights=color_weights,
                width_weights=width_weights,
                color_mask=color_mask,
                width_mask=width_mask,
                vmin=vmin,
                vmax=vmax,
            )

        else:
            print("Selected mode %s not valid. Please check the spelling " % mode)

    ebs_plot.set_xticks(kticks, knames)
    ebs_plot.set_yticks(interval=elimit)
    ebs_plot.set_xlim()
    ebs_plot.set_ylim(elimit)
    ebs_plot.draw_fermi(
        color=settings.ebs.fermi_color,
        linestyle=settings.ebs.fermi_linestyle,
        linewidth=settings.ebs.fermi_linewidth,
    )
    ebs_plot.set_ylabel()
    if settings.ebs.grid:
        ebs_plot.grid()
    if settings.ebs.legend and len(labels) != 0:
        ebs_plot.legend(labels)
    if savefig is not None:
        ebs_plot.save(savefig)
    if show:
        ebs_plot.show()
    return ebs_plot


def parse(code='vasp',
          dirname = "",
          outcar=None,
          poscar=None,
          procar=None,
          reciprocal_lattice=None,
          kpoints=None,
          interpolation_factor=1,
          fermi=None):
    ebs = None
    kpath = None
    structure = None

    if code == "vasp":
        if outcar is not None:
            outcar = io.vasp.Outcar(outcar)
            if fermi is None:
                fermi = outcar.efermi
            reciprocal_lattice = outcar.reciprocal_lattice
        if poscar is not None:
            poscar = io.vasp.Poscar(poscar)
            structure = poscar.structure
            if reciprocal_lattice is None:
                reciprocal_lattice = poscar.structure.reciprocal_lattice

        if kpoints is not None:
            kpoints = io.vasp.Kpoints(kpoints)
            kpath = kpoints.kpath

        procar = io.vasp.Procar(procar,
                             structure,
                             reciprocal_lattice,
                             kpath,
                             fermi,
                             interpolation_factor=interpolation_factor)
        ebs = procar.ebs
        
        
    elif code == "qe":
        if dirname is None:
            dirname = "bands"
        parser = io.qe.QEParser(scfIn_filename = "scf.in", dirname = dirname, bandsIn_filename = "bands.in", 
                             pdosIn_filename = "pdos.in", kpdosIn_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", 
                             dos_interpolation_factor = None)
        if fermi is None:
            fermi = parser.efermi
        reciprocal_lattice = parser.reciprocal_lattice
    
        structure = parser.structure
        
        kpoints = parser.kpoints
        kpath = parser.kpath

        ebs = parser.ebs
        
        

    return ebs, kpath, structure, reciprocal_lattice
