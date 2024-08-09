__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "December 01, 2020"

import os
from typing import List
import yaml 
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import colors as mpcolors
from matplotlib import cm

from ..core import ProcarSymmetry, FermiSurface
from ..utils import welcome, ROOT
from .. import io

with open(os.path.join(ROOT,'pyprocar','cfg','fermi_surface_2d.yml'), 'r') as file:
    plot_opt = yaml.safe_load(file)

def fermi2D(
    code:str,
    dirname:str,
    mode:str='plain',
    fermi:float=None,
    fermi_shift:float=0.0,
    band_indices:List[List]=None,
    band_colors:List[List]=None,
    spins:List[int]=None,
    atoms:List[int]=None,
    orbitals:List[int]=None,
    energy:float=None,
    k_z_plane:float=0.0,
    k_z_plane_tol:float=0.01,
    rot_symm=1,
    translate:List[int]=[0, 0, 0],
    rotation:List[int]=[0, 0, 0, 1],
    spin_texture:bool=False,
    exportplt:bool=False,
    savefig:str=None,
    print_plot_opts:bool=False,
    **kwargs
    ):
    """This function plots the 2d fermi surface in the z = 0 plane

    Parameters
    ----------
    code : str, 
        This parameter sets the code to parse, by default "vasp"
    dirname : str, optional
        This parameter is the directory of the calculation, by default ''
    fermi : float, optional
        The fermi energy. If none is given, the fermi energy in the directory will be used, by default None
    fermi_shift : float, optional
        The fermi energy shift, by default 0.0
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
    exportplt : bool, optional
        Boolean value where to return the matplotlib.pyplot state plt, by default False
    print_plot_opts: bool, optional
        Boolean to print the plotting options

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


    modes=["plain","plain_bands","parametric"]
    modes_txt=' , '.join(modes)
    message=f"""
            --------------------------------------------------------
            There are additional plot options that are defined in a configuration file. 
            You can change these configurations by passing the keyword argument to the function
            To print a list of plot options set print_plot_opts=True

            Here is a list modes : {modes_txt}
            --------------------------------------------------------
            """
    print(message)
    if print_plot_opts:
        for key,value in plot_opt.items():
            print(key,':',value)


    parser = io.Parser(code = code, dir = dirname)
    ebs = parser.ebs
    structure = parser.structure

    codes_with_scf_fermi = ['qe', 'elk']
    if code in codes_with_scf_fermi and fermi is None:
        fermi = ebs.efermi
    if fermi is not None:
        ebs.bands -= fermi
        ebs.bands += fermi_shift
        fermi_level = fermi_shift
    else:
        print("""
            WARNING : `fermi` is not set! Set `fermi={value}`. The plot did not shift the bands by the Fermi energy.
            ----------------------------------------------------------------------------------------------------------
            """)
    print(
        f"""
            WARNING : Make sure the kmesh has kz points with kz={k_z_plane} +- {k_z_plane_tol}
            ----------------------------------------------------------------------------------------------------------
            """
    )



    if structure.rotations is not None:
        ebs.ibz2fbz(structure.rotations)
        
    # Shifting all kpoint to first Brillouin zone
    bound_ops = -1.0*(ebs.kpoints > 0.5) + 1.0*(ebs.kpoints <= -0.5)
    ebs.kpoints = ebs.kpoints + bound_ops
    kpoints = ebs.kpoints_cartesian

    if spins is None:
        spins = np.arange(ebs.bands.shape[-1])
    if energy is None:
        energy = 0

    ### End of parsing ###
    # Selecting kpoints in a constant k_z plane
    i_kpoints_near_z_0 = np.where(np.logical_and(kpoints[:,2] < k_z_plane + k_z_plane_tol, 
                                                 kpoints[:,2] > k_z_plane - k_z_plane_tol) )
    kpoints = kpoints[i_kpoints_near_z_0,:][0]
    ebs.bands = ebs.bands[i_kpoints_near_z_0,:][0]
    ebs.projected = ebs.projected[i_kpoints_near_z_0,:][0]
    print('_____________________________________________________')
    for i_spin in spins:
        indices = np.where( np.logical_and(ebs.bands[:,:,i_spin].min(axis=0) < energy, ebs.bands[:,:,i_spin].max(axis=0) > energy))
        if len(indices) != 0:
            print(f"Useful band indices for spin-{i_spin} : {indices[0]}")


    if spin_texture is False:
        # processing the data
        if orbitals is None and ebs.projected is not None:
            orbitals = np.arange(ebs.norbitals, dtype=int)
        if atoms is None and ebs.projected is not None:
            atoms = np.arange(ebs.natoms, dtype=int)
        projected = ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        projected = projected[:,:,spins]
    else:
        # first get the sdp reduced array for all spin components.
        stData = []
        ebsX = copy.deepcopy(ebs)
        ebsY = copy.deepcopy(ebs)
        ebsZ = copy.deepcopy(ebs)

        ebsX.projected = ebsX.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        ebsY.projected = ebsY.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
        ebsZ.projected = ebsZ.ebs_sum(spins=spins, atoms=atoms, orbitals=orbitals, sum_noncolinear=False)

        ebsX.projected = ebsX.projected[:,:,[1]][:,:,0]
        ebsY.projected = ebsY.projected[:,:,[2]][:,:,0]
        ebsZ.projected = ebsZ.projected[:,:,[3]][:,:,0]

        stData.append(ebsX.projected )
        stData.append(ebsY.projected )
        stData.append(ebsZ.projected )

        
        projected = ebs.ebs_sum(spins=spins , atoms=atoms, orbitals=orbitals, sum_noncolinear=False)
    # Once the PROCAR is parsed and reduced to 2x2 arrays, we can apply
    # symmetry operations to unfold the Brillouin Zone
    # kpoints = data.kpoints
    # bands = data.bands
    # character = data.spd

    bands = ebs.bands
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

    fs = FermiSurface(symm.kpoints, symm.bands, symm.character,  
                      band_indices=band_indices, 
                      band_colors=band_colors,
                      **kwargs)
    fs.find_energy(energy)

    if not spin_texture:
        fs.plot(mode=mode, interpolation=300)
    else:
        fs.spin_texture(sx=symm.sx, 
                        sy=symm.sy, 
                        sz=symm.sz, 
                        spin=spins[0])

    fs.add_axes_labels()
    fs.add_legend()
    
    if exportplt:
        return plt

    else:
        if savefig:
            fs.savefig(savefig)  
        else:
            fs.show()
        return
