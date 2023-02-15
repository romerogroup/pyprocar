__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"
import os
from typing import List

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import colors as mpcolors
from matplotlib import cm

from ..utils import UtilsProcar
from ..core import ProcarSymmetry
from ..core import FermiSurface
from ..io import ElkParser
from ..io import AbinitParser
from ..splash import welcome
from .. import io


def fermi2D(
    code:str,
    dirname:str,
    mode:str='plain',
    band_indices:List[List]=None,
    band_colors:List[List]=None,
    lobster:bool=False,
    spins:List[int]=None,
    atoms:List[int]=None,
    orbitals:List[int]=None,
    energy:float=None,
    k_z_plane:float=0.0,
    rot_symm=1,
    translate:List[int]=[0, 0, 0],
    rotation:List[int]=[0, 0, 0, 1],
    savefig:str=None,
    spin_texture:bool=False,
    arrow_projection:str='z',
    arrow_size:float=None,
    arrow_color:List[int] or str=None,
    arrow_density:float=6,
    no_arrow:bool=False,
    cmap = 'jet',
    color_bar:bool=False,
    add_axes_labels:bool=True,
    add_legend:bool=False,
    exportplt:bool=False,
    
    repair:bool=True,
    ):
    """This function plots the 2d fermi surface in the z = 0 plane

    Parameters
    ----------
    code : str, 
        This parameter sets the code to parse, by default "vasp"
    dirname : str, optional
        This parameter is the directory of the calculation, by default ''
    lobster : bool, optional
        A boolean value to determine to use lobster, by default False
    band_indices : List[List]
        A list of list that contains band indices for a given spin
    band_colors : List[List]
            A list of list that contains colors for the band index 
            corresponding the band_indices for a given spin
    spins : List[int], optional
        List of spins, by default [0]
    atoms : List[int], optional
        List of atoms, by default None
    orbitals : List[int], optional
        List of orbitals, by default None
    energy : float, optional
        The energy to generate the iso surface. 
        When energy is None the 0 is used by default, which is the fermi energy, 
        by default None
    k_z_plane : float, optional
        Which K_z plane to generate 2d fermi surface, by default 0.0
    rot_symm : int, optional
        _description_, by default 1
    translate : List[int], optional
        Matrix to translate the kpoints, by default [0, 0, 0]
    rotation : List[int], optional
         Matrix to rotate the kpoints, by default [0, 0, 0, 1]
    savefig : str, optional
        The filename to save the plot as., by default None
    spin_texture : bool, optional
        Boolean value to determine if spin arrows are plotted, by default False
    arrow_size : float, optional
        Inversely determines the arrow size, by default None
    arrow_color : List[int] or str, optional
        Either a list for the rbg value or a string for the color, by default None
    arrow_density : float, optional
        Inversely determines the arrow density
    no_arrow : bool, optional
        A boolean value to determine if arrows or a heat map is produced for spins, by default False
    add_axes_labels : bool, optional
        Boolean value to add axes labels, by default True
    add_legend : bool, optional
        Boolean value to add legend, by default True
    exportplt : bool, optional
        Boolean value where to return the matplotlib.pyplot state plt, by default False
    color_bar : bool, optional
        Boolean value to plot the color bar, by default False
    cmap : bool, optional
        The colormap to be used, by default False
    repair : bool, optional
        Option for vasp to repair the procar file, by default True

    Returns
    -------
    matplotlib.pyplot
        Returns the matplotlib.pyplot state plt

    Raises
    ------
    RuntimeError
        invalid option --translate
    """
    welcome()

    # Turn interactive plotting off
    plt.ioff()

    if atoms is None:
        atoms = [-1]
        

    if orbitals is None:
        orbitals = [-1]

    



    if len(translate) != 3 and len(translate) != 1:
        print("Error: --translate option is invalid! (", translate, ")")
        raise RuntimeError("invalid option --translate")

    print("dirname         : ", dirname)
    print("bands           : ", band_indices)
    print("atoms           : ", atoms)
    print("orbitals        : ", orbitals)
    print("spin comp.      : ", spins)
    print("energy          : ", energy)
    print("rot. symmetry   : ", rot_symm)
    print("origin (trasl.) : ", translate)
    print("rotation        : ", rotation)
    print("save figure     : ", savefig)
    print("spin_texture    : ", spin_texture)
    print("no_arrows       : ", no_arrow)

    

    parser, kpoints, reciprocal_lattice, e_fermi = parse(code=code,
                                            lobster=lobster,
                                            repair=repair,
                                            dirname=dirname)
    if spins is None:
        spins = np.arange(parser.ebs.bands.shape[-1])
    # if bands is None:
    #     bands = np.arange(parser.ebs.bands.shape[1])
    if energy is None:
        energy = 0
    ### End of parsing ###

    # Selecting kpoints in a constant k_z plane
    i_kpoints_near_z_0 = np.where(np.logical_and(kpoints[:,2]< k_z_plane + 0.01, kpoints[:,2] > k_z_plane - 0.01) )
    kpoints = kpoints[i_kpoints_near_z_0,:][0]
    parser.ebs.bands = parser.ebs.bands[i_kpoints_near_z_0,:][0]
    parser.ebs.projected = parser.ebs.projected[i_kpoints_near_z_0,:][0]
    print('_____________________________________________________')
    for i_spin in spins:
        indices = np.where( np.logical_and(parser.ebs.bands[:,:,i_spin].min(axis=0) < energy, parser.ebs.bands[:,:,i_spin].max(axis=0) > energy))
        if len(indices) != 0:
            print(f"Useful band indices for spin-{i_spin} : {indices[0]}")

    
    
    # parser.ebs.bands = parser.ebs.bands[:,bands,:]
    # parser.ebs.projected = parser.ebs.projected[:,bands,:,:,:,:]

    

    if spin_texture is not True:
        # processing the data
        if orbitals is None and parser.ebs.projected is not None:
            orbitals = np.arange(parser.ebs.norbitals, dtype=int)
        if atoms is None and parser.ebs.projected is not None:
            atoms = np.arange(parser.ebs.natoms, dtype=int)
        projected = parser.ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        projected = projected[:,:,spins]
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

    bands = parser.ebs.bands

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
    fs = FermiSurface(symm.kpoints, symm.bands, symm.character, cmap = cmap,  band_indices=band_indices, band_colors=band_colors)
    fs.find_energy(energy)

    if not spin_texture:
        fs.plot(mode=mode, interpolation=300)
    else:
        fs.spin_texture(sx=symm.sx, 
                        sy=symm.sy, 
                        sz=symm.sz, 
                        arrow_projection=arrow_projection,
                        no_arrow=no_arrow, 
                        spin=spins[0], 
                        arrow_size = arrow_size,
                        arrow_color = arrow_color,
                        arrow_density=arrow_density,
                        color_bar=color_bar
                        )

    if add_axes_labels:
        fs.add_axes_labels()

    if add_legend:
        fs.add_legend()

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
                                    pdos_in_filename = "pdos.in", kpdos_in_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml")
            reciprocal_lattice = parser.reciprocal_lattice

            e_fermi = parser.efermi

            if apply_symmetry:
                parser.ebs.ibz2fbz(parser.rotations)

            bound_ops = -1.0*(parser.ebs.kpoints > 0.5) + 1.0*(parser.ebs.kpoints <= -0.5)
            kpoints = parser.ebs.kpoints  + bound_ops
            kpoints_cart = kpoints.dot(reciprocal_lattice) * (parser.alat/(2*np.pi))

        return parser, kpoints_cart, reciprocal_lattice, e_fermi