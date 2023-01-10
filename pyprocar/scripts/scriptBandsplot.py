__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from typing import List
import os

import numpy as np
import matplotlib.pyplot as plt

from ..utils.info import orbital_names
from .. import io
from ..plotter import EBSPlot
from ..splash import welcome
from ..utils.defaults import settings


# TODO What is the type is for projection mask?
# TODO Needs abinit parsing
# TODO Needs elk parsing

def bandsplot(
    code="vasp",
    procar:str="PROCAR",
    poscar:str="POSCAR",
    outcar:str="OUTCAR",
    abinit_output:str="abinit.out",
    elkin:str="elk.in",
    dirname:str=None,
    kpoints:np.ndarray=None,
    lobster:bool=False,
    mode:str="plain",
    spins:List[int]=None,
    atoms:List[int]=None,
    orbitals:List[int]=None,
    items:dict={},
    fermi:float=None,
    interpolation_factor:int=1,
    interpolation_type:str="cubic",
    projection_mask=None,
    vmax:float=0,
    vmin:float=1,
    kticks=None,
    knames=None,
    kdirect:bool=True,
    elimit: List[float]=None,
    ax:plt.Axes=None,
    title:str=None,
    show:bool=True,
    savefig:str=None,
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


    # Turn interactive plotting off
    # plt.ioff()

    # Verbose section

    settings.modify(kwargs)

    ebs, kpath, structure, reciprocal_lattice = parse(
        code, lobster, dirname ,outcar, poscar, procar, kpoints,
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
            for orb in ["s", "p", "d", "f"]:
                if orb == "f" and not ebs_plot.ebs.norbitals > 9:
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


        weights = ebs_plot.ebs.ebs_sum(atoms=atoms, principal_q_numbers=[-1], orbitals=orbitals, spins=spins)
            
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
                spins=spins
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

    if title:
        ebs_plot.set_title(title=title)
    if settings.ebs.grid:
        ebs_plot.grid()
    if settings.ebs.legend and len(labels) != 0:
        ebs_plot.legend(labels)
    if savefig is not None:
        ebs_plot.save(savefig)
    if show:
        ebs_plot.show()
    return ebs_plot


def parse(code:str='vasp',
          lobster:bool=False,
          dirname:str="",
          outcar:str='OUTCAR',
          poscar:str='PORCAR',
          procar:str='PROCAR',
          kpoints:np.ndarray=None,
          interpolation_factor:int=1,
          fermi:float=None):
    ebs = None
    kpath = None
    structure = None


    if lobster is True:
        parser = io.lobster.LobsterParser(dirname = dirname, 
                        code = code,
                        dos_interpolation_factor = None )

        if fermi is None:
            fermi = parser.efermi
        reciprocal_lattice = parser.reciprocal_lattice
    
        structure = parser.structure
        
        kpoints = parser.kpoints
        kpath = parser.kpath

        ebs = parser.ebs

    elif code == "vasp":
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

        procar = io.vasp.Procar(filename=procar,
                             structure=structure,
                             reciprocal_lattice=reciprocal_lattice,
                             kpath=kpath,
                             efermi=fermi,
                             interpolation_factor=interpolation_factor)
        ebs = procar.ebs
        
        
    elif code == "qe":
        if dirname is None:
            dirname = "bands"
        parser = io.qe.QEParser(dirname = dirname,scf_in_filename = "scf.in", bands_in_filename = "bands.in", 
                             pdos_in_filename = "pdos.in", kpdos_in_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", 
                             dos_interpolation_factor = None)
        if fermi is None:
            fermi = parser.efermi
        reciprocal_lattice = parser.reciprocal_lattice
    
        structure = parser.structure
        
        kpoints = parser.kpoints
        kpath = parser.kpath

        ebs = parser.ebs
        
    elif code == "abinit":
        if dirname is None:
            dirname = "fermi"
        outfile = f"{dirname}{os.sep}abinit.out"
        kpointsfile = f"{dirname}{os.sep}KPOINTS"
        # e_fermi = 0

        output = io.abinit.Output(abinit_output=outfile)
        # e_fermi = 0
        e_fermi = output.fermi
        
        # poscar = io.vasp.Poscar(filename=poscar_file)
        structure = output.structure
        reciprocal_lattice = output.structure.reciprocal_lattice
        ab_kpoints = io.abinit.Kpoints(filename=kpointsfile)

        parser = io.abinit.Procar(
                            filename=dirname,
                            abinit_output=outfile,
                            structure=output.structure,
                            reciprocal_lattice=output.reclat,
                            kpath=ab_kpoints,
                            efermi=output.fermi,
                        )


        structure = parser.structure
        kpoints = parser.kpoints
        kpath = ab_kpoints.kpath
        ebs = parser.ebs

        ebs.bands +=  e_fermi


        

    return ebs, kpath, structure, reciprocal_lattice
