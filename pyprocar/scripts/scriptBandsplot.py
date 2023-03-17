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
from ..utils import welcome
from ..utils.defaults import settings


# TODO What is the type is for projection mask?
# TODO Needs abinit parsing
# TODO Needs elk parsing

def bandsplot(
    code="vasp",
    dirname:str=None,
    mode:str="plain",
    lobster:bool=False,
    spins:List[int]=None,
    atoms:List[int]=None,
    orbitals:List[int]=None,
    items:dict={},
    fermi:float=None,
    interpolation_factor:int=1,
    interpolation_type:str="cubic",
    projection_mask:np.ndarray=None,
    vmax:float=None,
    vmin:float=None,
    kticks=None,
    knames=None,
    elimit: List[float]=None,
    ax:plt.Axes=None,
    title:str=None,
    show:bool=True,
    savefig:str=None,
    **kwargs,
    ):
    """A function to plot the band structutre

    Parameters
    ----------
    code : str, optional
        String to of the code used, by default "vasp"
    dirname : str, optional
        The directory name of the calculation, by default None
    mode : str, optional
        Sting for the mode of the calculation, by default "plain"
    lobster : bool, optional
        Boolean if this is a lobster calculation, by default False
    spins : List[int], optional
        A list of spins, by default None
    atoms : List[int], optional
        A list of atoms, by default None
    orbitals : List[int], optional
        A list of orbitals, by default None
    items : dict, optional
        A dictionary where the keys are the atoms and the values a list of orbitals, by default {}
    fermi : float, optional
        Float for the fermi energy, by default None
    interpolation_factor : int, optional
        The interpolation_factor, by default 1
    interpolation_type : str, optional
        The interpolation type, by default "cubic"
    projection_mask : np.ndarray, optional
        A custom projection mask, by default None
    vmax : float, optional
        Value to normalize the minimum projection value., by default None, by default None
    vmin : float, optional
        Value to normalize the maximum projection value., by default None, by default None
    kticks : _type_, optional
        A list of kticks, by default None
    knames : _type_, optional
        A list of kanems, by default None
    elimit : List[float], optional
        A list of floats to decide the energy window, by default None
    ax : plt.Axes, optional
        A matplotlib axes, by default None
    title : str, optional
        String for the title name, by default None
    show : bool, optional
        Boolean if to show the plot, by default True
    savefig : str, optional
        String to save the plot, by default None
    """

    # Turn interactive plotting off
    # plt.ioff()

    # Verbose section

    settings.modify(kwargs)

    ebs, kpath, structure, reciprocal_lattice = parse(
        code, dirname, lobster, interpolation_factor, fermi)
    
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
            for iorb,orb in enumerate(["s", "p", "d", "f"]):
                if orb == "f" and not ebs_plot.ebs.norbitals > 9:
                    continue
                orbitals = orbital_names[orb]
                labels.append(orb)
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
                                orbitals = np.append(orbitals, orbital_names[iorb]).astype(int)
                            labels.append(ispc + "-" + "".join(it[ispc]))
                        else:
                            orbitals = it[ispc]
                            labels.append(ispc + "-" + "_".join(str(x) for x in it[ispc]))
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
                spins=spins,
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
            dirname:str="",
            lobster:bool=False,
            
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
        if dirname is None:
            dirname = "bands"
        outcar = f"{dirname}{os.sep}OUTCAR"
        poscar = f"{dirname}{os.sep}POSCAR"
        procar = f"{dirname}{os.sep}PROCAR"
        kpoints = f"{dirname}{os.sep}KPOINTS"
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
                             pdos_in_filename = "pdos.in", kpdos_in_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml")
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

        parser = io.abinit.AbinitParser(abinit_output=outfile)

        ebs = parser.abinitprocarobject.ebs
        kpath = parser.abinitprocarobject.ebs.kpath
        e_fermi = parser.fermi
        reciprocal_lattice = parser.reclat
    ebs.bands += e_fermi

    return ebs, kpath, structure, reciprocal_lattice
