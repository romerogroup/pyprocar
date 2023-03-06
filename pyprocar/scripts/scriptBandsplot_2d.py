__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

from typing import List
import os

import numpy as np
import matplotlib.pyplot as plt

from .. import io
from ..plotter import EBSPlot
from ..splash import welcome
from ..utils.defaults import settings


# TODO What is the type is for projection mask?
# TODO Needs abinit parsing
# TODO Needs elk parsing

def bandsplot_2d(
    code="vasp",
    dirname:str=None,
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
    vmax:float=None,
    vmin:float=None,
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
    """A function to plot the 2d bandstructure

    Parameters
    ----------
    code : str, optional
        The code name, by default "vasp"
    dirname : str, optional
        The directory name where the calculation is, by default None
    lobster : bool, optional
        Boolean if this is a lobster calculation, by default False
    mode : str, optional
        String for mode type, by default "plain"
    spins : List[int], optional
        A list of spins, by default None
    atoms : List[int], optional
        A list of atoms, by default None
    orbitals : List[int], optional
        A list of orbitals, by default None
    items : dict, optional
        A dictionary where the keys are the atoms and the values a list of orbitals , by default {}
    fermi : float, optional
        Ther fermi energy, by default None
    interpolation_factor : int, optional
        The interpolation factor, by default 1
    interpolation_type : str, optional
        The interpolation type, by default "cubic"
    projection_mask : np.ndarray, optional
        A custom projection mask, by default None
    vmax : float, optional
        Value to normalize the minimum projection value., by default None, by default None, by default None
    vmin : float, optional
        Value to normalize the maximum projection value., by default None, by default None, by default None
    kticks : _type_, optional
        The kitcks, by default None
    knames : _type_, optional
        The knames, by default None
    kdirect : bool, optional
        _description_, by default True
    elimit : List[float], optional
        The energy window, by default None
    ax : plt.Axes, optional
        A matplotlib axes objext, by default None
    title : str, optional
        String for the title name, by default None
    show : bool, optional
        Boolean to show the plot, by default True
    savefig : str, optional
        String to save the plot, by default None
    """


    # Turn interactive plotting off
    # plt.ioff()

    # Verbose section

    settings.modify(kwargs)

    ebs, kpath, structure, reciprocal_lattice = parse(
        code, dirname, lobster, interpolation_factor, fermi)
    
    return None


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