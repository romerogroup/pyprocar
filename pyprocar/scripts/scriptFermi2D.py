__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"
import os
from typing import List

import numpy as np
import copy
import matplotlib.pyplot as plt

from ..utilsprocar import UtilsProcar
from ..io import ProcarParser
from ..procarselect import ProcarSelect
from ..procarplot import ProcarPlot
from ..procarsymmetry import ProcarSymmetry
from ..core import FermiSurface
from ..io import ElkParser
from ..io import AbinitParser
from ..splash import welcome
from .. import io


def fermi2D(
    file=None,
    code:str="vasp",
    outcar:str='OUTCAR',
    poscar:str='POSCAR',
    procar:str='PROCAR',
    lobster:bool=False,
    dirname:str='',
    abinit_output=None,
    spins:List[int]=[0],
    atoms:List[int]=None,
    orbitals:List[int]=None,
    energy=None,
    fermi:float=None,
    k_z_plane:float=0.0,
    rec_basis=None,
    rot_symm=1,
    translate:List[int]=[0, 0, 0],
    rotation:List[int]=[0, 0, 0, 1],
    human:bool=False,
    mask=None,
    savefig=None,
    spin_texture:bool=False,
    noarrow:bool=False,
    exportplt:bool=False,

    repair:bool=True,
):
    """_summary_

    :param file: _description_, defaults to None
    :type file: _type_, optional
    :param code: _description_, defaults to "vasp"
    :type code: str, optional
    :param outcar: _description_, defaults to 'OUTCAR'
    :type outcar: str, optional
    :param poscar: _description_, defaults to 'POSCAR'
    :type poscar: str, optional
    :param procar: _description_, defaults to 'PROCAR'
    :type procar: str, optional
    :param lobster: _description_, defaults to False
    :type lobster: bool, optional
    :param dirname: _description_, defaults to ''
    :type dirname: str, optional
    :param abinit_output: _description_, defaults to None
    :type abinit_output: _type_, optional
    :param spins: _description_, defaults to [0]
    :type spins: List[int], optional
    :param atoms: _description_, defaults to None
    :type atoms: List[int], optional
    :param orbitals: _description_, defaults to None
    :type orbitals: List[int], optional
    :param energy: _description_, defaults to None
    :type energy: _type_, optional
    :param fermi: _description_, defaults to None
    :type fermi: float, optional
    :param rec_basis: _description_, defaults to None
    :type rec_basis: _type_, optional
    :param rot_symm: _description_, defaults to 1
    :type rot_symm: int, optional
    :param translate: _description_, defaults to [0, 0, 0]
    :type translate: List[int], optional
    :param rotation: _description_, defaults to [0, 0, 0, 1]
    :type rotation: List[int], optional
    :param human: _description_, defaults to False
    :type human: bool, optional
    :param mask: _description_, defaults to None
    :type mask: _type_, optional
    :param savefig: _description_, defaults to None
    :type savefig: _type_, optional
    :param spin_texture: _description_, defaults to False
    :type spin_texture: bool, optional
    :param noarrow: _description_, defaults to False
    :type noarrow: bool, optional
    :param exportplt: _description_, defaults to False
    :type exportplt: bool, optional
    :param repair: _description_, defaults to True
    :type repair: bool, optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: _type_
    """

    welcome()

    # Turn interactive plotting off
    plt.ioff()

    if atoms is None:
        atoms = [-1]
        if human is True:
            print("WARNING: `--human` option given without atoms list!!!!!")

    if orbitals is None:
        orbitals = [-1]

    if rec_basis != None:
        rec_basis = np.array(rec_basis)
        rec_basis.shape = (3, 3)

    if len(translate) != 3 and len(translate) != 1:
        print("Error: --translate option is invalid! (", translate, ")")
        raise RuntimeError("invalid option --translate")

    print("file            : ", file)
    print("outcar          : ", outcar)
    print("Abinit output   : ", abinit_output)
    print("atoms           : ", atoms)
    print("orbitals        : ", orbitals)
    print("spin comp.      : ", spins)
    print("energy          : ", energy)
    print("fermi energy    : ", fermi)
    print("Rec. basis      : ", rec_basis)
    print("rot. symmetry   : ", rot_symm)
    print("origin (trasl.) : ", translate)
    print("rotation        : ", rotation)
    print("masking thres.  : ", mask)
    print("save figure     : ", savefig)
    print("spin_texture    : ", spin_texture)
    print("no_arrows       : ", noarrow)

  

    parser, kpoints, reciprocal_lattice, e_fermi = parse(code=code,
                                            lobster=lobster,
                                            repair=repair,
                                            dirname=dirname,
                                            outcar=outcar,
                                            poscar=poscar,
                                            procar=procar)
    ### End of parsing ###

    # Selecting kpoints in a constant k_z plane
    i_kpoints_near_z_0 = np.where(np.logical_and(kpoints[:,2]< k_z_plane + 0.01, kpoints[:,2] > k_z_plane - 0.01) )
    kpoints = kpoints[i_kpoints_near_z_0,:][0,:,:]
    parser.ebs.bands = parser.ebs.bands[i_kpoints_near_z_0,:][0,:,:]
    parser.ebs.projected = parser.ebs.projected[i_kpoints_near_z_0,:][0,:,:]

    if energy is None:
        energy = 0

    if spin_texture is not True:
        # processing the data
        if orbitals is None and parser.ebs.projected is not None:
            orbitals = np.arange(parser.ebs.norbitals, dtype=int)
        if atoms is None and parser.ebs.projected is not None:
            atoms = np.arange(parser.ebs.natoms, dtype=int)
        projected = parser.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        projected = projected[:,:,spins[0]]
    else:
        # first get the sdp reduced array for all spin components.
        stData = []
        ebsX = copy.deepcopy(parser.ebs)
        ebsY = copy.deepcopy(parser.ebs)
        ebsZ = copy.deepcopy(parser.ebs)

        ebsX.projected = ebsX.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        ebsY.projected = ebsY.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        ebsZ.projected = ebsZ.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)

        ebsX.projected = ebsX.projected[:,:,[0]][:,:,0]
        ebsY.projected = ebsY.projected[:,:,[1]][:,:,0]
        ebsZ.projected = ebsZ.projected[:,:,[2]][:,:,0]


        stData.append(ebsX.projected )
        stData.append(ebsY.projected )
        stData.append(ebsZ.projected )

        projected = parser.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
    # Once the PROCAR is parsed and reduced to 2x2 arrays, we can apply
    # symmetry operations to unfold the Brillouin Zone
    # kpoints = data.kpoints
    # bands = data.bands
    # character = data.spd

    bands = parser.ebs.bands[:,:,0]
    # kpoints = kpoints.dot(reciprocal_lattice  * (parser.alat/(2*np.pi)))
    character = projected
    if spin_texture is True:
        sx, sy, sz = stData[0], stData[1], stData[2]
        symm = ProcarSymmetry(kpoints, bands, sx=sx, sy=sy, sz=sz, character=character)
    else:
        symm = ProcarSymmetry(kpoints, bands, character=character)
        symm.translate(translate)
        symm.general_rotation(rotation[0], rotation[1:])
        # symm.MirrorX()
        symm.rot_symmetry_z(rot_symm)

    fs = FermiSurface(symm.kpoints, symm.bands, symm.character)
    fs.find_energy(energy)

    if not spin_texture:
        fs.plot(mask=mask, interpolation=300)
    else:
        fs.spin_texture(sx=symm.sx, sy=symm.sy, sz=symm.sz, noarrow=noarrow, spin=spins[0])

    if exportplt:
        return plt

    else:
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
            plt.close()  # Added by Nicholas Pike to close memory issue of looping and creating many figures
        else:
            plt.show()
        return


def parse(code:str='vasp',
          lobster:bool=False,
          repair:bool=False,
          dirname:str="",
          outcar:str='OUTCAR',
          poscar:str='PORCAR',
          procar:str='PROCAR',
          apply_symmetry:bool=True):
        if code == "vasp" or code == "abinit":
            if repair:
                repairhandle = UtilsProcar()
                repairhandle.ProcarRepair(procar, procar)
                print("PROCAR repaired. Run with repair=False next time.")

        if code == "vasp":
            outcar = f"{dirname}{os.sep}OUTCAR"
            poscar = f"{dirname}{os.sep}POSCAR"
            procar = f"{dirname}{os.sep}PROCAR"
            kpoints = f"{dirname}{os.sep}KPOINTS"
            filename = f"{dirname}{os.sep}{filename}"
            outcar = io.vasp.Outcar(filename=outcar)
        
            e_fermi = outcar.efermi
        
            poscar = io.vasp.Poscar(filename=poscar)
            structure = poscar.structure
            reciprocal_lattice = poscar.structure.reciprocal_lattice

            parser = io.vasp.Procar(filename=procar,
                                    structure=structure,
                                    reciprocal_lattice=reciprocal_lattice,
                                    efermi=e_fermi,
                                    )

            if apply_symmetry:                       
                parser.ebs.ibz2fbz(parser.rotations)

            bound_ops = -1.0*(parser.ebs.kpoints > 0.5) + 1.0*(parser.ebs.kpoints <= -0.5)
            kpoints_cart = kpoints.dot(reciprocal_lattice)

        elif code == "qe":

            if dirname is None:
                dirname = "bands"
            parser = io.qe.QEParser(dirname = dirname, scf_in_filename = "scf.in", bands_in_filename = "bands.in", 
                                    pdos_in_filename = "pdos.in", kpdos_in_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", 
                                    dos_interpolation_factor = None)
            reciprocal_lattice = parser.reciprocal_lattice

            e_fermi = parser.efermi

            if apply_symmetry:
                parser.ebs.ibz2fbz(parser.rotations)

            bound_ops = -1.0*(parser.ebs.kpoints > 0.5) + 1.0*(parser.ebs.kpoints <= -0.5)
            kpoints = parser.ebs.kpoints  + bound_ops
            kpoints_cart = kpoints.dot(reciprocal_lattice) * (parser.alat/(2*np.pi))

        return parser, kpoints_cart, reciprocal_lattice, e_fermi