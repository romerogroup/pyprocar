__author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import sys
import functools
import copy
from typing import List, Tuple


import numpy as np
from matplotlib import colors as mpcolors
from matplotlib import cm
import matplotlib.pyplot as plt
import vtk
import pyvista as pv
from pyvista.utilities import NORMALS, generate_plane, get_array, try_callback

from ..utilsprocar import UtilsProcar
from ..procarselect import ProcarSelect
from .. import io

HBAR_EV = 6.582119 *10**(-16) #eV*s
HBAR_J = 1.0545718 *10**(-34) #eV*s
METER_ANGSTROM = 10**(-10) #m /A
EV_TO_J = 1.602*10**(-19)
FREE_ELECTRON_MASS = 9.11*10**-31 #  kg

# TODO Add integrated properties and fix derivative properties

class PropertyCalculator:
    def __init__(self,
            code:str,
            procar:str="PROCAR",
            outcar:str="OUTCAR",
            poscar:str="POSCAR",
            dirname:str="",
            infile:str="in.bxsf",
            abinit_output:str=None,
            repair:bool=False
        ):
        self.code = code
        self.procar=procar
        self.poscar=poscar
        self.outcar=outcar
        self.dirname=dirname
        self.infile=infile
        self.abinit_output=abinit_output
        self.repair = repair

        self.data, self.reciprocal_lattice, self.procarFile, self.e_fermi = self.__parse_code()

        self.bands= self.data.bands
        self.kpoints= self.data.kpoints
        
        print("This is experimental")

    def calculate_first_and_second_derivative_energy_band(self, 
                                                        iband: int):
        def get_energy(kp_reduced: np.ndarray):
            return kp_reduced_to_energy[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
        def get_cartesian_kp(kp_reduced: np.ndarray):
            return kp_reduced_to_kp_cart[f'({kp_reduced[0]},{kp_reduced[1]},{kp_reduced[2]})']
        def energy_meshgrid_mapping(mesh_list):
            mesh_list_cart = copy.deepcopy(mesh_list)
            F = np.zeros(shape = mesh_list[0].shape)
            for k in range(mesh_list[0].shape[2]):
                for j in range(mesh_list[0].shape[1]):
                    for i in range(mesh_list[0].shape[0]):
                        kx = mesh_list[0][i,j,k]
                        ky = mesh_list[1][i,j,k]
                        kz = mesh_list[2][i,j,k]
        
                        mesh_list_cart[0][i,j,k] = get_cartesian_kp([kx,ky,kz])[0]
                        mesh_list_cart[1][i,j,k] = get_cartesian_kp([kx,ky,kz])[1]
                        mesh_list_cart[2][i,j,k] = get_cartesian_kp([kx,ky,kz])[2]
                        F[i,j,k] = get_energy([kx,ky,kz])
        
            return F

        def fermi_velocity_meshgrid_mapping(mesh_list):
            X = np.zeros(shape = mesh_list[0].shape)
            Y = np.zeros(shape = mesh_list[0].shape)
            Z = np.zeros(shape = mesh_list[0].shape)
            for k in range(mesh_list[0].shape[2]):
                for j in range(mesh_list[0].shape[1]):
                    for i in range(mesh_list[0].shape[0]):
                        kx = mesh_list[0][i,j,k]
                        ky = mesh_list[1][i,j,k]
                        kz = mesh_list[2][i,j,k]
                        
                        X[i,j,k] = get_gradient([kx,ky,kz])[0]
                        Y[i,j,k] = get_gradient([kx,ky,kz])[1]
                        Z[i,j,k] = get_gradient([kx,ky,kz])[2]
        
            return X,Y,Z
        
        kpoints_cart = np.matmul(self.kpoints,  self.reciprocal_lattice)
        
        mesh_list = np.meshgrid(np.unique(self.kpoints[:,0]),
                                np.unique(self.kpoints[:,1]),
                                np.unique(self.kpoints[:,2]), indexing = 'ij')
        
        
        kp_reduced_to_energy = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.kpoints,self.bands[:,iband])}
        kp_reduced_to_kp_cart = {f'({key[0]},{key[1]},{key[2]})':value for (key,value) in zip(self.kpoints,kpoints_cart)}
        kp_reduced_to_mesh_index = {f'({kp[0]},{kp[1]},{kp[2]})': np.argwhere((mesh_list[0]==kp[0]) & 
                                                                              (mesh_list[1]==kp[1]) & 
                                                                              (mesh_list[2]==kp[2]) )[0] for kp in self.kpoints}
        
        
        energy_meshgrid = energy_meshgrid_mapping(mesh_list) 
        # print(type(energy_meshgrid ))
        # print(energy_meshgrid.shape)
        nx = np.unique(self.kpoints[:,0])
        ny = np.unique(self.kpoints[:,1])
        nz = np.unique(self.kpoints[:,2])
        change_in_energy = np.array(np.gradient(energy_meshgrid))
        grad_E_mesh = np.array(np.gradient(energy_meshgrid, nx ,ny ,nz))

        # map mesh format back to kpoint listing
        grad_E = np.array([grad_E_mesh[:,kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]]
                  for kp in self.kpoints])

        # Convert to Cartesian coordinates
        grad_E_cart = np.dot(grad_E, self.reciprocal_lattice) * METER_ANGSTROM

        # Calculate group velocity
        group_velocity = grad_E_cart/(HBAR_EV) #* 2 * np.pi)
        group_velocity_mag = (group_velocity[:,0]**2 + 
                                group_velocity[:,1]**2 + 
                                group_velocity[:,2]**2)**0.5

        
        # grad_E_mesh = np.array(np.gradient(grad_E_mesh, nx ,ny ,nz))
        # gradient_mesh = list(fermi_velocity_meshgrid_mapping(mesh_list))
        grad_grad_E_mesh = np.array(list(map(np.gradient,grad_E_mesh)))

        # map mesh format back to kpoint listing
        grad_grad_E = np.array([grad_grad_E_mesh[:,:,kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
                                        kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]]
                  for kp in self.kpoints])

        # Convert to Cartesian coordinates
        grad_grad_E_cart = np.dot(grad_grad_E, self.reciprocal_lattice) #* METER_ANGSTROM**2
        grad_grad_E_cart_1  = np.array([ [ np.dot(gradient, self.reciprocal_lattice) for gradient in gradient_tensor ] for gradient_tensor in grad_grad_E ])
        # grad_grad_E_cart_1 = np.array([np.dot(grad_grad_E[:,:,igrad], self.reciprocal_lattice) for igrad in range(grad_grad_E.shape[2])] 
                                    # for ) #* METER_ANGSTROM**2
        # grad_grad_E_cart_1 = np.swapaxes(grad_grad_E_cart_1,axis1=0,axis2=1)
        # print(grad_grad_E_cart[0,:,:])
        # print(grad_grad_E_cart[0,:,:])
        print(grad_grad_E[0,:,:])
        print(grad_grad_E[1,:,:])
        print(np.dot(grad_grad_E[0,0,:], self.reciprocal_lattice))
        print(np.dot(grad_grad_E[0,1,:], self.reciprocal_lattice))
        print(np.dot(grad_grad_E[0,2,:], self.reciprocal_lattice))

        print(np.dot( self.reciprocal_lattice,grad_grad_E[0,0,:]))
        print(np.dot( self.reciprocal_lattice,grad_grad_E[0,1,:]))
        print(np.dot( self.reciprocal_lattice,grad_grad_E[0,2,:]))

        # print(np.dot(grad_grad_E[1,:,0], self.reciprocal_lattice))
        # print(np.dot(grad_grad_E[1,:,1], self.reciprocal_lattice))
        # print(np.dot(grad_grad_E[1,:,2], self.reciprocal_lattice))
        # print(grad_grad_E[0,:,:])

        # print(grad_grad_E.shape)
        # print(grad_grad_E_cart.shape)
        # print(grad_grad_E_cart_1.shape)
        # print(grad_grad_E_cart_1[0,:,:])
        
        # print(change_in_energy_gradient.shape)
        # print(change_in_energy_gradient.shape)
        
        
        
        # grad_xx_list = [change_in_energy_gradient[0][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        # grad_xy_list = [change_in_energy_gradient[0][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        # grad_xz_list = [change_in_energy_gradient[0][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        
        # grad_yx_list = [change_in_energy_gradient[1][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        # grad_yy_list = [change_in_energy_gradient[1][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        # grad_yz_list = [change_in_energy_gradient[1][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        
        # grad_zx_list = [change_in_energy_gradient[2][0][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        # grad_zy_list = [change_in_energy_gradient[2][1][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        # grad_zz_list = [change_in_energy_gradient[2][2][kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][0],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][1],
        #                                                 kp_reduced_to_mesh_index[f'({kp[0]},{kp[1]},{kp[2]})'][2]
        #                                                 ]
        #                 for kp in self.kpoints]
        
        # energy_second_derivative_list = np.array([[grad_xx_list,grad_xy_list,grad_xz_list],
        #                                  [grad_yx_list,grad_yy_list,grad_yz_list],
        #                                  [grad_zx_list,grad_zy_list,grad_zz_list],
        #                                  ])
        
        # energy_second_derivative_list = np.swapaxes(energy_second_derivative_list,axis1=0,axis2=2)
        # energy_second_derivative_list = np.swapaxes(energy_second_derivative_list,axis1=1,axis2=2)
        # energy_second_derivative_list = energy_second_derivative_list/(2*np.pi)
        # energy_second_derivative_list = np.multiply(energy_second_derivative_list, np.array([np.linalg.norm(lattice[:,0])*METER_ANGSTROM * len(np.unique(self.kpoints[:,0])),
        #                                                                                                np.linalg.norm(lattice[:,1])*METER_ANGSTROM * len(np.unique(self.kpoints[:,1])),
        #                                                                                                np.linalg.norm(lattice[:,2])*METER_ANGSTROM * len(np.unique(self.kpoints[:,2]))])
        #                                             )
        # energy_second_derivative_list_cart = np.array([ [ np.matmul(lattice,gradient) for gradient in gradient_tensor ] for gradient_tensor in energy_second_derivative_list ])
        
        # inverse_effective_mass_tensor_list = energy_second_derivative_list_cart * EV_TO_J/ HBAR_J**2
        # effective_mass_tensor_list = np.array([ np.linalg.inv(inverse_effective_mass_tensor) for inverse_effective_mass_tensor in inverse_effective_mass_tensor_list])
        # effective_mass_list = np.array([ (3*(1/effective_mass_tensor_list[ikp,0,0] + 
        #                                           1/effective_mass_tensor_list[ikp,1,1] +
        #                                           1/effective_mass_tensor_list[ikp,2,2])**-1)/FREE_ELECTRON_MASS for ikp in range(len(self.XYZ))])
        # return (group_velocity_vector, group_velocity_magnitude, effective_mass_tensor_list,
        #         effective_mass_list, gradient_list, gradient_list_cart)
    
    def __calculate_first_and_second_derivative_energy(self):
        
        self.first_and_second_derivative_energy_property_dict = {}
        for iband in range(len(self.bands[0,:])):
            group_velocity_vector,group_velocity_magnitude,effective_mass_tensor_list,effective_mass_list,gradient_list, gradient_list_cart = self.__calculate_first_and_second_derivative_energy_band(iband)
            self.first_and_second_derivative_energy_property_dict.update({f"band_{iband}" : {"group_velocity_vector" : group_velocity_vector,
                                                                                        "group_velocity_magnitude": group_velocity_magnitude,
                                                                                        "effective_mass_tensor_list":effective_mass_tensor_list,
                                                                                        "effective_mass_list":effective_mass_list}
                                                                })


    def __parse_code(self):
        if self.code == "vasp" or self.code == "abinit":
            if self.repair:
                repairhandle = UtilsProcar()
                repairhandle.ProcarRepair(self.procar, self.procar)
                print("PROCAR repaired. Run with repair=False next time.")

        if self.code == "vasp":
            outcar = io.vasp.Outcar(filename=self.outcar)
            
            e_fermi = outcar.efermi
            
            poscar = io.vasp.Poscar(filename=self.poscar)
            structure = poscar.structure
            reciprocal_lattice = poscar.structure.reciprocal_lattice

            procarFile = io.vasp.Procar(filename=self.procar,
                                    structure=structure,
                                    reciprocal_lattice=reciprocal_lattice,
                                    efermi=e_fermi,
                                    )
            data = ProcarSelect(procarFile, deepCopy=True)

        

        elif self.code == "qe":
            # procarFile = parser
            if dirname is None:
                dirname = "bands"
            procarFile = io.qe.QEParser(scfIn_filename = "scf.in", dirname = dirname, bandsIn_filename = "bands.in", 
                                pdosIn_filename = "pdos.in", kpdosIn_filename = "kpdos.in", atomic_proj_xml = "atomic_proj.xml", 
                                dos_interpolation_factor = None)
            reciprocal_lattice = procarFile.reciprocal_lattice
            data = ProcarSelect(procarFile, deepCopy=True)

            e_fermi = procarFile.efermi

            # procarFile = QEFermiParser()
            # reciprocal_lattice = procarFile.reclat
            # data = ProcarSelect(procarFile, deepCopy=True)
            # if fermi is None:
            #     e_fermi = procarFile.efermi
            # else:
            #     e_fermi = fermi

        return data, reciprocal_lattice, procarFile, e_fermi

    def __format_data(self, 
                    mode:str,
                    bands:List[int]=None,
                    atoms:List[int]=None,
                    orbitals:List[int]=None,
                    spins:List[int]=None, 
                    spin_texture: bool=False,):

        # bands_to_keep = bands
        # if bands_to_keep is None:
        #     bands_to_keep = np.arange(len(self.data.bands[0, :]))

        # self.band_near_fermi = []
        # for iband in range(len(self.data.bands[0,:])):
        #     fermi_tolerance = 0.1
        #     fermi_surface_test = len(np.where(np.logical_and(self.data.bands[:,iband]>=self.e_fermi-fermi_tolerance, self.data.bands[:,iband]<=self.e_fermi+fermi_tolerance))[0])
        #     if fermi_surface_test != 0:
        #         self.band_near_fermi.append(iband)
        # print(f"Bands near the fermi energy : {self.band_near_fermi}")

        spd = []
        if mode == "parametric":
            if orbitals is None:
                orbitals = [-1]
            if atoms is None:
                atoms = [-1]
            if spins is None:
                spins = [0]

            self.data.selectIspin(spins)
            self.data.selectAtoms(atoms, fortran=False)
            self.data.selectOrbital(orbitals)

            for iband, band in enumerate(self.data.bands[0,:]):
                spd.append(self.data.spd[:, iband])
        elif mode == "property_projection":
            for iband in bands_to_keep:
                spd.append(None)
        else:
            for iband in bands_to_keep:
                spd.append(None)
    
        spd_spin = []

        if spin_texture:
            dataX = ProcarSelect(self.procarFile, deepCopy=True)
            dataY = ProcarSelect(self.procarFile, deepCopy=True)
            dataZ = ProcarSelect(self.procarFile, deepCopy=True)

            dataX.kpoints = self.data.kpoints
            dataY.kpoints = self.data.kpoints
            dataZ.kpoints = self.data.kpoints

            dataX.spd = self.data.spd
            dataY.spd = self.data.spd
            dataZ.spd = self.data.spd

            dataX.bands = self.data.bands
            dataY.bands = self.data.bands
            dataZ.bands = self.data.bands

            dataX.selectIspin([1])
            dataY.selectIspin([2])
            dataZ.selectIspin([3])

            dataX.selectAtoms(atoms, fortran=False)
            dataY.selectAtoms(atoms, fortran=False)
            dataZ.selectAtoms(atoms, fortran=False)

            dataX.selectOrbital(orbitals)
            dataY.selectOrbital(orbitals)
            dataZ.selectOrbital(orbitals)
            for iband in bands_to_keep:
                spd_spin.append(
                    [dataX.spd[:, iband], dataY.spd[:, iband], dataZ.spd[:, iband]]
                )
        else:
            for iband in bands_to_keep:
                spd_spin.append(None)

        return spd, spd_spin, bands_to_keep