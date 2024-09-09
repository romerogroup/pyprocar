# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Logan Lang
@author: Pedram Tavadze
@author: Freddy Farah

"""

from typing import List
import itertools
import copy
from scipy.interpolate import CubicSpline

import numpy as np
import networkx as nx
from matplotlib import pylab as plt
import pyvista

from .kpath import KPath
from .brillouin_zone import BrillouinZone
from ..utils import  mathematics
from pyprocar.utils.unfolder import Unfolder
from pyprocar.utils import LOGGER

HBAR_EV = 6.582119 *10**(-16) #eV*s
HBAR_J = 1.0545718 *10**(-34) #eV*s
METER_ANGSTROM = 10**(-10) #m /A
EV_TO_J = 1.602*10**(-19)
FREE_ELECTRON_MASS = 9.11*10**-31 #  kg

# TODO: Check hormonic average effective mass values
# TODO: Check method to calculate the bands integral

class ElectronicBandStructure:
    """This object stores electronic band structure informomration.

        Parameters
        ----------
        kpoints : np.ndarray
            The kpoints array. Will have the shape (n_kpoints, 3)
        bands : np.ndarray
            The bands array. Will have the shape (n_kpoints, n_bands)
        efermi : float
            The fermi energy
        projected : np.ndarray, optional
            The projections array. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
        projected_phase : np.ndarray, optional
            The full projections array that incudes the complex part. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
        kpath : KPath, optional
            The kpath for band structure claculation, defaults to None
        weights : np.ndarray, optional
            The weights of the kpoints. Will have the shape (n_kpoints, 1), defaults to None
        labels : List, optional

        reciprocal_lattice : np.ndarray, optional
            The reciprocal lattice vector matrix. Will have the shape (3, 3), defaults to None
        shifted_to_efermi : bool, optional
             Boolean to determine if the fermi energy is shifted, defaults to False
    """

    def __init__(
        self,
        kpoints:np.ndarray,
        bands:np.ndarray,
        efermi:float,
        n_kx:int=None,
        n_ky:int=None,
        n_kz:int=None,
        projected:np.ndarray = None,
        projected_phase:np.ndarray=None,
        weights:np.ndarray=None,
        kpath:KPath=None,
        labels:List=None,
        reciprocal_lattice:np.ndarray=None,
        ):
        LOGGER.info('Initializing the ElectronicBandStructure object')
        
        self._kpoints = kpoints
        self._kpoints_cartesian = self.reduced_to_cartesian(kpoints,reciprocal_lattice)   
        self._bands = bands - efermi
        self._efermi = efermi
    
        self._projected = projected
        self._projected_phase = projected_phase
        self._reciprocal_lattice= reciprocal_lattice
        self._weights = weights
        self._kpath = kpath
        
        self.is_mesh = True
        if kpath is not None:
            self.is_mesh = False
        self.has_phase = False
        if self.projected_phase is not None:
            self.has_phase = True
        self.labels = labels

        self.ibz_kpoints = None
        self.ibz_bands = None
        self.ibz_projected = None
        self.ibz_projected_phase = None
        self.ibz_weights = None

        self.initial_band_properties = ['bands','projected','projected_phase']
        self.band_derived_properties = ['bands_gradient', 'bands_hessian', 'fermi_velocity', 'harmonic_average_effective_mass', 'fermi_speed']
        self.band_dependent_properties = self.initial_band_properties + self.band_derived_properties 
        self.kpoint_properties =  ['kpoints', 'weights']
        self.initial_properties=self.kpoint_properties + self.initial_band_properties
        self.all_mesh_properties = self.initial_properties + self.band_derived_properties + ['kpoints_cartesian']

        # Initialize mesh properties
        for prop in self.all_mesh_properties:
            setattr(self, "_" + prop+"_mesh", None)
        # Initialize array properties
        for prop in self.band_derived_properties :
            setattr(self, "_" + prop, None)


        self._n_kx=n_kx
        self._n_ky=n_ky
        self._n_kz=n_kz

        self._kx_map = None
        self._ky_map = None
        self._kz_map = None

        # if self.is_mesh:
        #     self._sort_by_kpoints()
        LOGGER.info('Subtracting Fermi Energy from Bands')
        LOGGER.info(f'Is Mesh: {self.is_mesh}')
        LOGGER.info(f'Fermi Energy: {self.efermi}')
        LOGGER.info(f'Kpoints shape: {self.kpoints.shape}')
        LOGGER.info(f'Bands shape: {self.bands.shape}')
        if self.projected is not None:
            LOGGER.info(f'Projected shape: {self.projected.shape}')
        if self.projected_phase is not None:
            LOGGER.info(f'Projected phase shape: {self.projected_phase.shape}')
        if self.kpath is not None:
            LOGGER.info(f"Kpath: {self.kpath}")
        if self.labels is not None:
            LOGGER.info(f"Kpath: {self.labels}")
        if self.reciprocal_lattice is not None:
            LOGGER.info(f"Reciprocal lattice: {self.reciprocal_lattice}")
        if self.weights is not None:
            LOGGER.info(f"Weights: {self.weights}")
        LOGGER.info('Initialized the ElectronicBandStructure object')

    def __str__(self):
        ret = 'Enectronic Band Structure     \n'
        ret += '------------------------     \n'
        ret += 'Total number of kpoints  = {}\n'.format(self.nkpoints)
        ret += 'Total number of bands    = {}\n'.format(self.nbands)
        ret += 'Total number of atoms    = {}\n'.format(self.natoms)
        ret += 'Total number of orbitals = {}\n'.format(self.norbitals)
        return ret
    
    @property
    def nkpoints(self):
        """The number of k points

        Returns
        -------
        int
            The number of k points
        """
        return self.kpoints.shape[0]

    @property
    def n_kx(self):
        """The number of unique kpoints in kx direction in the reduced basis"""

        return self._n_kx
    
    @property
    def n_ky(self):
        """The number of unique kpoints in kx direction in the reduced basis"""
        return self._n_ky
    
    @property
    def n_kz(self):
        """The number of unique kpoints in ky direction in the reduced basis"""
        return self._n_kz
    
    @property
    def nbands(self):
        """The number of bands

        Returns
        -------
        int
            The number of bands
        """
        return self.bands.shape[1]

    @property
    def natoms(self):
        """The number of atoms

        Returns
        -------
        int
            The number of atoms
        """
        return self.projected.shape[2]

    @property
    def nprincipals(self):
        """The number of principal quantum numbers

        Returns
        -------
        int
            The number of principal quantum numbersk points
        """
        return self.projected.shape[3]

    @property
    def norbitals(self):
        """The number of orbitals

        Returns
        -------
        int
            The number of orbitals
        """
        return self.projected.shape[4]

    @property
    def nspins(self):
        """The number of spin channels

        Returns
        -------
        int
            The number of spin channels
        """
        return self.projected.shape[5]

    @property
    def is_non_collinear(self):
        """Boolean to determine if this is a non-colinear calculation

        Returns
        -------
        bool
            Boolean to determine if this is a non-colinear calculation
        """
        if self.nspins == 4:
            return True
        else:
            return False

    @property
    def efermi(self):
         return self._efermi
    @efermi.setter
    def efermi(self, value):
        """This is a setter for the efermi property. 
        If the efermi property gets changed, the bands_gradient and bands_hessian will be recalculated"""
        self._efermi = value

    @property
    def kpoints(self):
        """Returns the kpoints in fractional basis
        """
        return self._kpoints
    @kpoints.setter
    def kpoints(self, value):
        """This is a setter for the kpoints property. 
        If the kpoints property gets changed, the cartesian kpoints will be recalculated"""
        self._kpoints = value
        self._kpoints_cartesian = self.reduced_to_cartesian(self._kpoints,self._reciprocal_lattice)
        self._n_kx=len(np.unique(self.kpoints[:,0]))
        self._n_ky=len(np.unique(self.kpoints[:,1]))
        self._n_kz=len(np.unique(self.kpoints[:,2]))

    @property
    def kpoints_cartesian(self):
        return self._kpoints_cartesian
    @kpoints_cartesian.setter
    def kpoints_cartesian(self, value):
        """This is a setter for the kpoints_cartesian property. 
        If the kpoints_cartesian property gets changed, the reciprocal kpoints will be recalculated"""
        self._kpoints_cartesian = value
    

    @property
    def bands(self):
         return self._bands
    @bands.setter
    def bands(self, value):
        """This is a setter for the bands property. 
        If the bands property gets changed, the bands_gradient and bands_hessian will be recalculated"""
        self._bands = value

        # # If bands are changed, reset all the band derived properties
        # for prop in self.band_derived_properties:
        #     setattr(self, "_" + prop, None)
    
    @property
    def projected(self):
         return self._projected
    @projected.setter
    def projected(self, value):
        """This is a setter for the projected property. 
        If the projected property gets changed, the projected_gradient and projected_hessian will be recalculated"""
        self._projected = value
    
    @property
    def projected_phase(self):
         return self._projected_phase
    @projected_phase.setter
    def projected_phase(self, value):
        """This is a setter for the projected_phase property. 
        If the projected_phase property gets changed, the projected_gradient and projected_hessian will be recalculated"""
        self._projected_phase = value

    @property
    def weights(self):
         return self._weights
    @weights.setter
    def weights(self, value):
        """This is a setter for the weights property. 
        If the weights property gets changed, the projected_gradient and projected_hessian will be recalculated"""
        self._weights = value

    @property
    def kpath(self):
         return self._kpath
    @kpath.setter
    def kpath(self, value):
        """This is a setter for the kpath property. 
        If the kpath property gets changed, the projected_gradient and projected_hessian will be recalculated"""
        self._kpath = value
    
    @property
    def kx_map(self):
         if self._kx_map is None:
             unique_x = np.unique(self.kpoints[:,0])
             self._kx_map = {value: idx for idx, value in enumerate(unique_x)}
         return self._kx_map
    
    @property
    def ky_map(self):
         if self._ky_map is None:
             unique_y = np.unique(self.kpoints[:,1])
             self._ky_map = {value: idx for idx, value in enumerate(unique_y)}
         return self._ky_map
    
    @property
    def kz_map(self):
         if self._kz_map is None:
             unique_z = np.unique(self.kpoints[:,2])
             self._kz_map = {value: idx for idx, value in enumerate(unique_z)}
         return self._kz_map
    
    @property
    def reciprocal_lattice(self):
         return self._reciprocal_lattice

    @property
    def inv_reciprocal_lattice(self):
        """Returns the inverse of the reciprocal lattice"""
        if self.reciprocal_lattice is not None:
            return np.linalg.inv(self.reciprocal_lattice)
        else:
            print(
                "Please provide a reciprocal lattice when initiating the Procar class"
            )
            return None

    @property
    def bands_gradient(self):
        """
        Bands gradient is a numpy array that stores each band gradient a list that corresponds to the self.kpoints
        Shape = [n_kpoints,3,n_bands], where the second dimension represents d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Bands fradient is a numpy array that stores each band gradient in a list that corresponds to the self.kpoints
            Shape = [n_kpoints,3,n_bands], 
            where the second dimension represents d/dx,d/dy,d/dz  
        """

        if self._bands_gradient is None:
            self._bands_gradient = self.mesh_to_array(mesh=self.bands_gradient_mesh)
        return self._bands_gradient
    
    @property
    def bands_hessian(self):
        """
        Bands hessian is a numpy array that stores each band hessian in a list that corresponds to the self.kpoints   
        Shape = [n_kpoints,3,3,n_bands], 
        where the second and third dimension represent d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Bands hessian is a numpy array that stores each band hessian in a list that corresponds to the self.kpoints
            Shape = [n_kpoints,3,3,n_bands], 
            where the second and third dimension represent d/dx,d/dy,d/dz  
        """

        if self._bands_hessian is None:
            self._bands_hessian = self.mesh_to_array(mesh=self.bands_hessian_mesh)
        return self._bands_hessian

    @property
    def fermi_velocity(self):
        """
        fermi_velocity is a numpy array that stores each fermi_velocity a list that corresponds to the self.kpoints
        Shape = [n_kpoints,3,n_bands], where the second dimension represents d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            fermi_velocity is a numpy array that stores each fermi_velocity in a list that corresponds to the self.kpoints
            Shape = [n_kpoints,3,n_bands], 
            where the second dimension represents d/dx,d/dy,d/dz  
        """

        if self._fermi_velocity is None:
            self._fermi_velocity = self.mesh_to_array(mesh=self.fermi_velocity_mesh)
        return self._fermi_velocity
    
    @property
    def fermi_speed(self):
        """
        fermi speed is a numpy array that stores each fermi speed a list that corresponds to the self.kpoints
        Shape = [n_kpoints,n_bands] 
        
        Returns
        -------
        np.ndarray
            fermi speed is a numpy array that stores each fermi speed 
            in a list that corresponds to the self.kpoints
            Shape = [n_kpoints,n_bands], 
        """

        if self._fermi_speed is None:
            self._fermi_speed = self.mesh_to_array(mesh=self.fermi_speed_mesh)
        return self._fermi_speed

    @property
    def harmonic_average_effective_mass(self):
        """
        harmonic average effective mass is a numpy array that stores 
        each harmonic average effective mass in a list that corresponds to the self.kpoints   
        Shape = [n_kpoints,n_bands], 
        
        Returns
        -------
        np.ndarray
            harmonic average effective mass is a numpy array that stores 
            each harmonic average effective mass in a list that corresponds to the self.kpoints
            Shape = [n_kpoints,n_bands],
        """

        if self._harmonic_average_effective_mass is None:
            self._harmonic_average_effective_mass=self.mesh_to_array(mesh=self.harmonic_average_effective_mass_mesh)
        return self._harmonic_average_effective_mass

    @property
    def kpoints_mesh(self):
        """Kpoint mesh representation of the kpoints grid. Shape = [n_kx,n_ky,n_kz,3]
        Returns
        -------
        np.ndarray
            Kpoint mesh representation of the kpoints grid. Shape = [n_kx,n_ky,n_kz,3]
        """

        if self._kpoints_mesh is None:
            self._kpoints_mesh = self.array_to_mesh(self.kpoints,
                                                    nkx=self.n_kx, 
                                                    nky=self.n_ky, 
                                                    nkz=self.n_kz)
        return self._kpoints_mesh
    
    @property
    def kpoints_cartesian_mesh(self):
        """Kpoint cartesian mesh representation of the kpoints grid. Shape = [n_kx,n_ky,n_kz,3]
        Returns
        -------
        np.ndarray
            Kpoint cartesian mesh representation of the kpoints grid. Shape = [n_kx,n_ky,n_kz,3]
        """
        if self._kpoints_cartesian_mesh is None:
            self._kpoints_cartesian_mesh = self.array_to_mesh(self.kpoints_cartesian,
                                                      nkx=self.n_kx, 
                                                      nky=self.n_ky, 
                                                      nkz=self.n_kz)
        return self._kpoints_cartesian_mesh
    
    @property
    def bands_mesh(self,force_update=False):
        """
        Bands mesh is a numpy array that stores each band in a mesh grid.   
        Shape = [n_kx,n_ky,n_kz,n_bands]

        Returns
        -------
        np.ndarray
            Bands mesh is a numpy array that stores each band in a mesh grid. 
            Shape = [n_kx,n_ky,n_kz,n_bands]
        """
        if self._bands_mesh is None or force_update:
            self._bands_mesh = self.array_to_mesh(self.bands,
                                                      nkx=self.n_kx, 
                                                      nky=self.n_ky, 
                                                      nkz=self.n_kz)
        return self._bands_mesh
    
    @property
    def projected_mesh(self):
        """
        projected mesh is a numpy array that stores each projection in a mesh grid.   
        Shape = [n_kx,n_ky,n_kz,n_bands,n_spins,n_atoms,n_orbitals]

        Returns
        -------
        np.ndarray
            Projection mesh is a numpy array that stores each projection in a mesh grid. 
            Shape = [n_kx,n_ky,n_kz,n_bands,n_spins,n_atoms,n_orbitals]
        """

        if self._projected_mesh is None:
            self._projected_mesh = self.array_to_mesh(self.projected, 
                                                      nkx=self.n_kx, 
                                                      nky=self.n_ky, 
                                                      nkz=self.n_kz)
        return self._projected_mesh
    
    @property
    def projected_phase_mesh(self):
        """
        projected phase mesh is a numpy array that stores each projection phases in a mesh grid.   
        Shape = [n_kx,n_ky,n_kz,n_bands,n_spins,n_atoms,n_orbitals]

        Returns
        -------
        np.ndarray
            projected phase mesh is a numpy array that stores each projection phases in a mesh grid.  
            Shape = [n_kx,n_ky,n_kz,n_bands,n_spins,n_atoms,n_orbitals]
        """

        if self._projected_phase_mesh is None:
            self._projected_phase_mesh = self.array_to_mesh(self.projected_phase, 
                                                      nkx=self.n_kx, 
                                                      nky=self.n_ky, 
                                                      nkz=self.n_kz)
        return self._projected_phase_mesh

    @property
    def weights_mesh(self):
        """
        weights mesh is a numpy array that stores each weights in a mesh grid.   
        Shape = [n_kx,n_ky,n_kz,1]

        Returns
        -------
        np.ndarray
            weights mesh is a numpy array that stores each weights in a mesh grid. 
            Shape = [n_kx,n_ky,n_kz,1]
        """

        if self._weights_mesh is None:
            self._weights_mesh = self.array_to_mesh(self.weights,
                                                      nkx=self.n_kx, 
                                                      nky=self.n_ky, 
                                                      nkz=self.n_kz)
        return self._weights_mesh

    @property
    def bands_gradient_mesh(self):
        """
        Bands gradient mesh is a numpy array that stores each band gradient in a mesh grid.   
        Shape = [3,n_bands,n_kx,n_ky,n_kz], where the first dimension represents d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Bands fradient mesh is a numpy array that stores each band gradient in a mesh grid.
            Shape = [n_kx,n_ky,n_kz,3,n_bands], 
            where the first dimension represents d/dx,d/dy,d/dz  
        """

        if self._bands_gradient_mesh is None:
            band_gradients=self.calculate_nd_scalar_derivatives(self.bands_mesh, self.reciprocal_lattice)

            # print(np.array_equal(scalar_diffs,scalar_diffs_2))
            # This is equivalent to the above
            # n_i, n_j, n_k, n_bands, n_spins = self.bands_mesh.shape
            # band_gradients_2 = np.zeros(( n_i, n_j, n_k ,n_bands, n_spins,3))
            # scalar_diffs_2 = np.zeros(( n_i, n_j, n_k ,n_bands, n_spins,3))
            # for i_band in range(n_bands):
            #     for i_spin in range(n_spins):
            #         band_gradients_2[:,:,:,i_band,i_spin,:]=self.calculate_scalar_gradient_2(self.bands_mesh[:,:,:,i_band,i_spin],
            #         reciprocal_lattice=self.reciprocal_lattice)

            #         scalar_diffs_2[:,:,:,i_band,i_spin,:]=self.calculate_scalar_diff_2(self.bands_mesh[:,:,:,i_band,i_spin])


            band_gradients *= METER_ANGSTROM
            self._bands_gradient_mesh = band_gradients
        return self._bands_gradient_mesh
    
    @property
    def bands_hessian_mesh(self):
        """
        Bands hessian mesh is a numpy array that stores each band hessian in a mesh grid.   
        Shape = [n_kx,n_ky,n_kz,n_bands,n_spin,3,3], 
        where the last two dimensions represent d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Bands hessian mesh is a numpy array that stores each band hessian in a mesh grid.
            Shape = [n_kx,n_ky,n_kzn_bands,n_spin,3,3],
            where the first and second dimension represent d/dx,d/dy,d/dz  
        """

        if self._bands_hessian_mesh is None:
            band_hessians=self.calculate_nd_scalar_derivatives(self.bands_gradient_mesh, self.reciprocal_lattice)
            
            # This is equivalent to the previous code
            # n_i, n_j, n_k, n_bands, n_spins, n_dim = self.bands_gradient_mesh.shape
            # band_hessians = np.zeros(( n_i, n_j, n_k, n_bands, n_spins, 3, 3))
            # for i_dim in range(n_dim):
            #     for i_band in range(n_bands):
            #         for i_spin in range(n_spins):
            #             # band_hessians[:,:,:,i_band,i_spin,i_dim,:] = self.calculate_scalar_gradient(scalar_mesh = self.bands_gradient_mesh[:,:,:,i_band,i_spin,i_dim],
            #             #                                                                             mesh_grid=self.kpoints_cartesian_mesh)
            #             band_hessians[:,:,:,i_band,i_spin,i_dim,:] = calculate_scalar_differences_2(scalar_mesh = self.bands_gradient_mesh[:,:,:,i_band,i_spin,i_dim],
            #                                                                                         transform_matrix=self.reciprocal_lattice)
            band_hessians *= METER_ANGSTROM
            self._bands_hessian_mesh = band_hessians
        return self._bands_hessian_mesh
    
    @property
    def fermi_velocity_mesh(self):
        """
        Fermi Velocity mesh is a numpy array that stores each  Fermi Velocity in a mesh grid.   
        Shape = [n_bands,n_kx,n_ky,n_kz,3], where the first dimension represents d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Fermi Velocity mesh is a numpy array that stores each  Fermi Velocity in a mesh grid.
            Shape = [n_bands,n_kx,n_ky,n_kz,3], 
            where the first dimension represents d/dx,d/dy,d/dz  
        """

        if self._fermi_velocity_mesh is None:
            self._fermi_velocity_mesh =  self.bands_gradient_mesh / HBAR_EV
        return self._fermi_velocity_mesh
    
    @property
    def fermi_speed_mesh(self):
        """
        Fermi speed mesh is a numpy array that stores each  Fermi Velocity in a mesh grid.   
        Shape = [n_kx,n_ky,n_kz,n_bands,n_spins],
        
        Returns
        -------
        np.ndarray
            Fermi speed mesh is a numpy array that stores each Fermi Velocity in a mesh grid.
            Shape = [n_kx,n_ky,n_kz,n_bands,n_spins], 
        """

        if self._fermi_speed_mesh is None:
            self._fermi_speed_mesh = np.linalg.norm(self.fermi_velocity_mesh,axis=-1)

        return self._fermi_speed_mesh
    
    @property
    def harmonic_average_effective_mass_mesh(self):
        """
        harmonic average effective mass mesh is a numpy array that stores each 
        harmonic average effective mass mesh in a mesh grid.   
        Shape = [n_bands,n_kx,n_ky,n_kz], 
        
        Returns
        -------
        np.ndarray
            harmonic average effective mass mesh is a numpy array that stores 
            each harmonic average effective mass in a mesh grid.
            Shape = [n_bands,n_kx,n_ky,n_kz], 
        """

        if self._harmonic_average_effective_mass_mesh is None:
            # n_i, n_j, n_k, n_bands, n_spins, n_grad_1, n_grad_2, = self.bands_hessian_mesh.shape
            # self._harmonic_average_effective_mass_mesh = np.zeros(shape=( n_i, n_j, n_k, n_bands, n_spins))
            # print(self.bands_hessian_mesh.shape)
            # for iband in range(n_bands):
            #     for ispin in range(n_spins):
            #         for k in range(n_k):
            #             for j in range(n_j):
            #                 for i in range(n_i):
            #                     hessian = self.bands_hessian_mesh[i,j,k,iband,ispin,...] * EV_TO_J / HBAR_J**2
            #                     self._harmonic_average_effective_mass_mesh[i,j,k,iband,ispin] = harmonic_average_effective_mass(hessian)
            self._harmonic_average_effective_mass_mesh = self.calculate_harmonic_average_effective_mass(self.bands_hessian_mesh* EV_TO_J/ HBAR_J**2 )
            
        return self._harmonic_average_effective_mass_mesh
    
    @property
    def bands_integral(self):
        return self.calculate_nd_scalar_integral(self.bands_mesh,self.reciprocal_lattice)

    @staticmethod
    def reduced_to_cartesian(kpoints,reciprocal_lattice):
        if reciprocal_lattice is not None:
            return np.dot(kpoints, reciprocal_lattice)
        else:
            print(
                "Please provide a reciprocal lattice when initiating the Procar class"
            )
            return
        
    @staticmethod
    def cartesian_to_reduced(cartesian,reciprocal_lattice):
        """Converts cartesian coordinates to fractional coordinates

        Parameters
        ----------
        cartesian : np.ndarray
            The cartesian coordinates. shape = [N,3]
        reciprocal_lattice : np.ndarray
            The reciprocal lattice vector matrix. Will have the shape (3, 3), defaults to None

        Returns
        -------
        np.ndarray
            The fractional coordinates. shape = [N,3]
        """
        if reciprocal_lattice is not None:
            return np.dot(cartesian, np.linalg.inv(reciprocal_lattice))
        else:
            print(
                "Please provide a reciprocal lattice when initiating the Procar class"
            )
            return

    @staticmethod
    def array_to_mesh( array, nkx, nky, nkz):
        """
        Converts a list to a mesh that corresponds to ebs.kpoints
        [n_kx*n_ky*n_kz,...]->[n_kx,n_ky,n_kz,...]. Make sure array is sorted by lexisort

        Parameters
        ----------
        array : np.ndarray
            The array to convert to a mesh
        nkx : int
            The number of kx points
        nky : int
            The number of ky points
        nkz : int
            The number of kz points
            
        Returns
        -------
        np.ndarray
           mesh
        """
        prop_shape = (nkx, nky, nkz,) + array.shape[1:]
        scalar_grid=array.reshape(prop_shape,order='C')
        return scalar_grid
    
    @staticmethod
    def mesh_to_array(mesh):
        """
        Converts a mesh to a list that corresponds to ebs.kpoints
        [n_kx,n_ky,n_kz,...]->[n_kx*n_ky*n_kz,...]
        Parameters
        ----------
        mesh : np.ndarray
            The mesh to convert to a list
        Returns
        -------
        np.ndarray
           lsit
        """
        nkx, nky, nkz = mesh.shape[:3]
        prop_shape = (nkx*nky*nkz,) + mesh.shape[3:]
        array=mesh.reshape(prop_shape)
        return array
    
    @staticmethod
    def calculate_nd_scalar_derivatives(scalar_array, reciprocal_lattice):
        """Transforms the derivatives to cartesian coordinates
            (n,j,k,...)->(n,j,k,...,3)

        Parameters
        ----------
        derivatives : np.ndarray
            The derivatives to transform
        reciprocal_lattice : np.ndarray
            The reciprocal lattice

        Returns
        -------
        np.ndarray
            The transformed derivatives
        """
        letters=['a','b','c','d','e','f','g','h']
        scalar_diffs=calculate_scalar_differences(scalar_array)
        n_dim=len(scalar_diffs.shape[3:])-1
        transform_matrix_einsum_string='ij'
        dim_letters=''.join(letters[0:n_dim])
        scalar_array_einsum_string='uvw' + dim_letters + 'j'
        transformed_scalar_string='uvw' + dim_letters + 'i'
        ein_sum_string=transform_matrix_einsum_string + ',' + scalar_array_einsum_string + '->' + transformed_scalar_string
        print(ein_sum_string)
        scalar_gradients=np.einsum(ein_sum_string, reciprocal_lattice, scalar_diffs)

        return scalar_gradients

    @staticmethod
    def calculate_nd_scalar_integral(scalar_mesh,reciprocal_lattice):
        """Calculate the scalar integral"""
        n1,n2,n3=scalar_mesh.shape[:3]
        volume_reduced_vector=np.array([1,1,1])
        volume_cartesian_vector=np.dot(reciprocal_lattice,volume_reduced_vector)
        volume=np.prod(volume_cartesian_vector)
        dv=volume/(n1*n2*n3)

        scalar_volume_avg=calculate_scalar_volume_averages(scalar_mesh)
        # Compute the integral by summing up the product of scalar values and the volume of each grid cell.
        integral = np.sum(scalar_volume_avg * dv,axis=(0,1,2))
        
        return integral

    @staticmethod
    def calculate_harmonic_average_effective_mass(hessian):
        # letters=['a','b','c','d','e','f','g','h']
        # scalar_diffs=calculate_scalar_differences(self.bands_gradient_mesh)
        # n_dim=len(scalar_diffs.shape[3:])-1
        # transform_matrix_einsum_string='ij'
        # dim_letters=''.join(letters[0:n_dim])
        # scalar_array_einsum_string='uvw' + dim_letters + 'j'
        # transformed_scalar_string='uvw' + dim_letters + 'i'
        # ein_sum_string=transform_matrix_einsum_string + ',' + scalar_array_einsum_string + '->' + transformed_scalar_string
        # band_hessians=np.einsum(ein_sum_string, self.reciprocal_lattice, scalar_diffs)
        # Calculate the trace of each 3x3 matrix along the last two axes
        trace_tensor = np.trace(hessian, axis1=-2, axis2=-1)
        
        # Calculate the harmonic average effective mass for each element
        e_mass = 3 * trace_tensor**-1 / FREE_ELECTRON_MASS
        
        return e_mass
    
    @staticmethod
    def interpolate_mesh_grid(self, mesh_grid,interpolation_factor=2):
        """This function will interpolate an Nd, 3d mesh grid
        [...,nx,ny,nz]

        Parameters
        ----------
        mesh_grid : np.ndarray
            The mesh grid to interpolate
        """
        scalar_dims=mesh_grid.shape[:-3]
        new_mesh=None
        for i, idx in enumerate(np.ndindex(scalar_dims)):
            # Creating slicing list. idx are the scalar dimensions, last 3 are the grid
            mesh_idx = list(idx)
            tmp = [slice(None)]*3
            mesh_idx.extend(tmp)
            new_grid = mathematics.fft_interpolate(mesh_grid[mesh_idx], interpolation_factor=interpolation_factor)
            
            if i==0:
                new_dim = np.append(scalar_dims,new_grid.shape,axis=0)
                new_mesh=np.zeros(shape=(new_dim))

            new_mesh[mesh_idx]=new_grid
        return new_mesh
    
    def update_weights(self, weights):
        self.weights = weights
        return
    
    def ebs_ipr(self):
        """_summary_

        Returns
        -------
        ret : list float
            The IPR projections
        """
        orbitals = np.arange(self.norbitals, dtype=int)
        # sum over orbitals
        proj = np.sum(self.projected[:, :, :, :, orbitals, :], axis=-2)
        # keeping only the last principal quantum number
        proj = proj[:, :, :, -1, :]
        # selecting all atoms:
        atoms = np.arange(self.natoms, dtype=int)
        # the ipr is \frac{\sum_i |c_i|^4}{(\sum_i |c_i^2|)^2}
        # mind, every c_i is c_{i,n,k} with n,k the band and k-point indexes
        num = np.absolute(proj)**2
        num = np.sum(num[:, :, atoms, :], axis=-2)
        den = np.absolute(proj)**1 + 0.0001 # avoiding zero
        den = np.sum(den[:, :, atoms, :], axis=-2)**2
        IPR = num/den
        return IPR

    def ebs_ipr_atom(self):
        """
        It returns the atom-resolved , pIPR:

        pIPR_j =  \\frac{\|c_j\|^4}{(\\sum_i \|c_i^2\|)^2}

        Clearly, \\( \\sum_j pIPR_j = IPR \\).

        Mind: \( c_i \) is the wavefunction \( c(n,k)_i \), in pyprocar we already
        have density projections, \( c_i^2 \).

        *THIS QUANTITY IS NOT READY FOR PLOTTING*, please prefer `self.ebs_ipr()`
        
        Returns
        -------
        ret : list float
            The IPR projections

        """
        orbitals = np.arange(self.norbitals, dtype=int)
        # sum over orbitals
        proj = np.sum(self.projected[:, :, :, :, orbitals, :], axis=-2)
        # keeping only the last principal quantum number
        proj = proj[:, :, :, -1, :]
        # selecting all atoms:
        atoms = np.arange(self.natoms, dtype=int)

        # the partial pIPR is \frac{|c_j|^4}{(\sum_i |c_i^2|)^2}
        # mind, every c_i is c_{i,n,k} with n,k the band and k-point indexes
        num = np.absolute(proj)**2
        den = np.absolute(proj)
        den = np.sum(den[:, :, atoms, :], axis=-2)**2
        pIPR = num/den[:,:,np.newaxis,:]
        # print('pIPR', pIPR.shape)
        return pIPR
        
    def ebs_sum(self, 
                atoms:List[int]=None, 
                principal_q_numbers:List[int]=[-1], 
                orbitals:List[int]=None, 
                spins:List[int]=None, 
                sum_noncolinear:bool=True):
        """_summary_

        Parameters
        ----------
        atoms : List[int], optional
            List of atoms to be summed over, by default None
        principal_q_numbers : List[int], optional
            List of principal quantum numbers to be summed over, by default [-1]
        orbitals : List[int], optional
            List of orbitals to be summed over, by default None
        spins : List[int], optional
            List of spins to be summed over, by default None
        sum_noncolinear : bool, optional
            Determines if the projection should be summed in a non-colinear calculation, by default True

        Returns
        -------
        ret : list float
            The summed projections
        """

        principal_q_numbers = np.array(principal_q_numbers)
        if atoms is None:
            atoms = np.arange(self.natoms, dtype=int)
        if spins is None:
            spins = np.arange(self.nspins, dtype=int)
        if orbitals is None:
            orbitals = np.arange(self.norbitals, dtype=int)
        # sum over orbitals
        ret = np.sum(self.projected[:, :, :, :, orbitals, :], axis=-2)
        # sum over principle quantum number
        ret = np.sum(ret[:, :, :, principal_q_numbers, :], axis=-2)
        # sum over atoms
        ret = np.sum(ret[:, :, atoms, :], axis=-2)
        # sum over spins only in non collinear and reshaping for consistency (nkpoints, nbands, nspins)
        # in non-mag, non-colin nspin=1, in colin nspin=2
        if self.is_non_collinear and sum_noncolinear:
            ret = np.sum(ret[:, :, spins], axis=-1).reshape(
                self.nkpoints, self.nbands, 1
            )
        return ret

    def unfold(self, transformation_matrix=None, structure=None):
        """The method helps unfold the bands. This is done by using the unfolder to find the new kpoint weights.
        The current weights are then updated

        Parameters
        ----------
        transformation_matrix : np.ndarray, optional
            The transformation matrix to transform the basis. Expected size is (3,3), by default None
        structure : pyprocar.core.Structure, optional
            The structure of a material, by default None

        Returns
        -------
        None
            None
        """
        uf = Unfolder(
            ebs=self, transformation_matrix=transformation_matrix, structure=structure,
        )
        self.update_weights(uf.weights)

        return None
    
    def _sort_by_kpoints(self):
        """Sorts the bands and projected arrays by kpoints"""
        sorted_indices = np.lexsort((self.kpoints[:,2], self.kpoints[:,1], self.kpoints[:,0]))

        for prop in self.initial_properties:
            original_value = getattr(self, prop)
            if original_value is not None:
                setattr(self, prop, original_value[sorted_indices,...])
        return None

    def ibz2fbz(self, rotations,decimals=4):
        """Applys symmetry operations to the kpoints, bands, and projections

        Parameters
        ----------
        rotations : np.ndarray
            The point symmetry operations of the lattice
        decimals : int
            The number of decimals to round the kpoints 
            to when checking for uniqueness
        """
        if not self.is_mesh:
            raise ValueError("This function only works for meshes")
        
        n_kpoints = self.kpoints.shape[0]
        n_rotations = rotations.shape[0]

        # Calculate new shape for kpoints and all properties
        total_points = n_kpoints * n_rotations
        new_shape = (total_points, 3)

        properties=self.initial_properties[2:]
        # Initialize new arrays for kpoints and properties
        new_kpoints = np.zeros(new_shape)
    
        self.ibz_kpoints = self.kpoints
        self.ibz_kpoints_cartesian = self.kpoints_cartesian
        new_properties = {}
        for prop in properties:
            
            original_value = getattr(self, prop)
            if original_value is not None:
                setattr(self, f'ibz_{prop}', original_value.copy())

                prop_shape = (total_points,) + original_value.shape[1:]
                new_properties[prop] = np.zeros(prop_shape)

        # Apply rotations and copy properties
        for i, rotation in enumerate(rotations):
            start_idx = i * n_kpoints
            end_idx = start_idx + n_kpoints

            # Rotate kpoints
            rotated_kpoints = self.kpoints.dot(rotation.T)
            
            new_kpoints[start_idx:end_idx] = rotated_kpoints

            # Update properties
            for prop in properties:
                original_value = getattr(self, "ibz_" + prop)
                if original_value is not None:
                    new_properties[prop][start_idx:end_idx] = original_value

        # Apply boundary conditions to kpoints
        new_kpoints = -np.fmod(new_kpoints + 6.5, 1) + 0.5

        # Floating point error can cause the kpoints to be off by 0.000001 or so
        # causing the unique indices to misidentify the kpoints
        new_kpoints = new_kpoints.round(decimals=decimals)
        _, unique_indices = np.unique(new_kpoints, axis=0, return_index=True)

        # Update the object's properties keeping only the unique kpoints
        self.kpoints = new_kpoints[unique_indices]
        self.bz_kpoints = self.kpoints
        self.bz_kpoints_cartesian = self.kpoints_cartesian
        for prop in properties:
            prop_value = getattr(self, prop)
            if prop_value is not None:
                setattr(self, prop, new_properties[prop][unique_indices])
                setattr(self, "bz_" + prop, new_properties[prop][unique_indices])
 
        self._sort_by_kpoints()
        return None

    def ravel_array(self,mesh_grid):
        shape = mesh_grid.shape
        mesh_grid=mesh_grid.reshape(shape[:-3] + (-1,))
        mesh_grid=np.moveaxis(mesh_grid,-1,0)
        return mesh_grid

    def fix_collinear_spin(self):
        """
        Converts data from two spin channels to a single channel, adjusting the spin down values to negatives. This is typically used for plotting the Density of States (DOS).

        Parameters
        ----------
        No parameters are required for this function.

        Returns
        -------
        bool
            Returns True if the function changed the data, False otherwise.
        """

        print('old bands.shape', self.bands.shape)
        if self.bands.shape[2] != 2:
            return False
        shape = list(self.bands.shape)
        shape[1] = shape[1]*2
        shape[-1] = 1
        self.bands.shape = shape
        print('new bands.shape', self.bands.shape)
        
        
        if self.projected is not None:
            print('old projected.shape', self.projected.shape)
            self.projected[... , -1] = -self.projected[... , -1]
            shape = list(self.projected.shape)
            shape[1] = shape[1]*2
            shape[-1] = 1
            self.projected.shape = shape
            print('new projected.shape', self.projected.shape)

        
        return True
    
    def reduce_kpoints_to_plane(self,k_z_plane,k_z_plane_tol):
        """
        Reduces the kpoints to a plane
        """

        i_kpoints_near_z_0 = np.where(np.logical_and(self.kpoints_cartesian[:,2] < k_z_plane + k_z_plane_tol, 
                                                     self.kpoints_cartesian[:,2] > k_z_plane - k_z_plane_tol) )

        for prop in self.initial_properties:
            original_value = getattr(self, prop)
            if original_value is not None:
                setattr(self, prop, original_value[i_kpoints_near_z_0,...][0])
        return None

    def expand_kpoints_to_supercell(self):
        supercell_directions=list(list(itertools.product([1, 0,-1], repeat=2)))
        initial_kpoints=copy.copy(self.kpoints )
        initial_property_values={}
        # Do not use kpoints in intial properties
        for prop in self.initial_properties[1:]:
            initial_property_values[prop]=copy.copy(getattr(self, prop))

        final_kpoints=copy.copy(initial_kpoints)
        # final_bands=copy.copy(self.bands)
        for supercell_direction in supercell_directions:
            if supercell_direction != (0,0):
                new_kpoints=copy.copy(initial_kpoints)
                new_kpoints[:,0] = new_kpoints[:,0] + supercell_direction[0]
                new_kpoints[:,1] = new_kpoints[:,1] + supercell_direction[1]
                
                final_kpoints=np.append(final_kpoints,new_kpoints, axis=0)
                # Do not use kpoints in intial properties
                for prop in self.initial_properties[1:]:
                    original_value = getattr(self, prop)
                    if original_value is not None:
                        initial_value=initial_property_values[prop]
                        new_values=np.append(original_value, initial_value,  axis=0)
                        setattr(self, prop, new_values)
        self.kpoints=final_kpoints

        self._sort_by_kpoints()

    def expand_kpoints_to_supercell_by_axes(self, axes_to_expand=[0, 1, 2]):
        # Validate input
        if not set(axes_to_expand).issubset({0, 1, 2}):
            raise ValueError("axes_to_expand must be a subset of [0, 1, 2]")

        # Create supercell directions based on axes to expand
        supercell_directions = list(itertools.product([1, 0, -1], repeat=len(axes_to_expand)))
        
        initial_kpoints = copy.deepcopy(self.kpoints)
        initial_property_values = {}
        
        # Do not use kpoints in initial properties
        for prop in self.initial_properties[1:]:
            initial_property_values[prop] = copy.deepcopy(getattr(self, prop))

        final_kpoints = copy.deepcopy(initial_kpoints)
        
        for supercell_direction in supercell_directions:
            if supercell_direction != tuple([0] * len(axes_to_expand)):
                new_kpoints = copy.deepcopy(initial_kpoints)
                
                for i, axis in enumerate(axes_to_expand):
                    new_kpoints[:, axis] += supercell_direction[i]
                
                final_kpoints = np.append(final_kpoints, new_kpoints, axis=0)
                
                # Do not use kpoints in initial properties
                for prop in self.initial_properties[1:]:
                    original_value = getattr(self, prop)
                    if original_value is not None:
                        initial_value = initial_property_values[prop]
                        new_values = np.append(original_value, initial_value, axis=0)
                        setattr(self, prop, new_values)
        
        self.kpoints = final_kpoints
        self._sort_by_kpoints()

    def reduce_bands_near_fermi(self, bands=None, tolerance=0.7):
        """
        Reduces the bands to those near the fermi energy
        """
        energy_level = 0
        full_band_index = []
        for iband in range(len(self.bands[0,:,0])):
            fermi_surface_test = len(np.where(np.logical_and(self.bands[:,iband,0]>=energy_level-tolerance, 
                                                             self.bands[:,iband,0]<=energy_level+tolerance))[0])
            if fermi_surface_test != 0:
                full_band_index.append(iband)
        if bands:
            full_band_index=bands
    
        for prop in self.initial_band_properties:
            original_value = getattr(self, prop)
            
            if original_value is not None:
                value = original_value[:,full_band_index,...]
                setattr(self, prop, value)
        return None

    def plot_kpoints(
        self,
        reduced=False,
        show_brillouin_zone=True,
        color="r",
        point_size=4.0,
        render_points_as_spheres=True,
        transformation_matrix=None,
        ):
        """This needs to be moved to core.KPath and updated new implementation of pyvista PolyData

        This method will plot the K points in pyvista

        Parameters
        ----------
        reduced : bool, optional
            Determines wether to plot the kpoints in the reduced or cartesian basis, defaults to False
        show_brillouin_zone : bool, optional
            Boolean to show the Brillouin zone, defaults to True
        color : str, optional
            Color of the points, defaults to "r"
        point_size : float, optional
            Size of points, defaults to 4.0
        render_points_as_spheres : bool, optional
            Boolean for how points are rendered, defaults to True
        transformation_matrix : np.ndarray, optional, optional
            Reciprocal Lattice Matrix, defaults to None

        Returns
        -------
        None
            None
        """
        
        p = pyvista.Plotter()
        if show_brillouin_zone:
            if reduced:
                brillouin_zone = BrillouinZone(
                    np.diag([1, 1, 1]), transformation_matrix,
                )
                brillouin_zone_non = BrillouinZone(np.diag([1, 1, 1]),)
            else:
                brillouin_zone = BrillouinZone(
                    self.reciprocal_lattice, transformation_matrix
                )
                brillouin_zone_non = BrillouinZone(self.reciprocal_lattice,)

            p.add_mesh(
                brillouin_zone.pyvista_obj,
                style="wireframe",
                line_width=3.5,
                color="black",
            )
            p.add_mesh(
                brillouin_zone_non.pyvista_obj,
                style="wireframe",
                line_width=3.5,
                color="white",
            )
        if reduced:
            kpoints = self.kpoints_reduced
        else:
            kpoints = self.kpoints_cartesian
        p.add_mesh(
            kpoints,
            color=color,
            point_size=point_size,
            render_points_as_spheres=render_points_as_spheres,
        )
        if transformation_matrix is not None:
            p.add_mesh(
                np.dot(kpoints, transformation_matrix),
                color="blue",
                point_size=point_size,
                render_points_as_spheres=render_points_as_spheres,
            )
        p.add_axes(
            xlabel="Kx", ylabel="Ky", zlabel="Kz", line_width=6, labels_off=False
        )

        p.show()

    def general_rotation(self, angle, kpoints, sx, sy, sz, rotAxis=[0, 0, 1], store=True):
        """Apply a rotation defined by an angle and an axis.
    
        Returning value: (Kpoints, sx,sy,sz), the rotated Kpoints and spin
                        vectors (if not the case, they will be empty
                        arrays).

        Arguments
        angle: the rotation angle, must be in degrees!

        rotAxis : a fixed Axis when applying the symmetry, usually it is
        from Gamma to another point). It doesn't need to be normalized. 
        The RotAxis can be:
        [x,y,z] : a cartesian vector in k-space.
        'x': [1,0,0], a rotation in the yz plane. 
        'y': [0,1,0], a rotation in the zx plane.
        'z': [0,0,1], a rotation in the xy plane

        """
        sx = np.array([])
        if sx is not None:
            sx = sx
        sy = np.array([])
        if sy is not None:
            sy = sy
        sz = np.array([])
        if sz is not None:
            sz = sz

        if rotAxis == "x" or rotAxis == "X":
            rotAxis = [1, 0, 0]
        if rotAxis == "y" or rotAxis == "Y":
            rotAxis = [0, 1, 0]
        if rotAxis == "z" or rotAxis == "Z":
            rotAxis = [0, 0, 1]
        rotAxis = np.array(rotAxis, dtype=float)
        LOGGER.debug("rotAxis : " + str(rotAxis))
        rotAxis = rotAxis / np.linalg.norm(rotAxis)
        LOGGER.debug("rotAxis Normalized : " + str(rotAxis))
        LOGGER.debug("Angle : " + str(angle))
        angle = angle * np.pi / 180
        # defining a quaternion for rotatoin
        angle = angle / 2
        rotAxis = rotAxis * np.sin(angle)
        qRot = np.array((np.cos(angle), rotAxis[0], rotAxis[1], rotAxis[2]))
        qRotI = np.array((np.cos(angle), -rotAxis[0], -rotAxis[1], -rotAxis[2]))

        LOGGER.debug("Rot. quaternion : " + str(qRot))
        LOGGER.debug("Rot. quaternion conjugate : " + str(qRotI))

        # converting self.kpoints into quaternions
        w = np.zeros((len(kpoints), 1))
        qvectors = np.column_stack((w, kpoints)).transpose()
        LOGGER.debug(
            "Kpoints-> quaternions (transposed):\n" + str(qvectors.transpose())
        )
        qvectors = q_multi(qRot, qvectors)
        qvectors = q_multi(qvectors, qRotI).transpose()
        kpoints = qvectors[:, 1:]
        LOGGER.debug("Rotated kpoints :\n" + str(qvectors))

        # rotating the spin vector (if exist)
        sxShape, syShape, szShape = sx.shape, sy.shape, sz.shape
        LOGGER.debug("Spin vector Shapes : " + str((sxShape, syShape, szShape)))
        # The first entry has to be an array of 0s, w could do the work,
        # but if len(self.sx)==0 qvectors will have a non-defined length
        qvectors = (
            0 * sx.flatten(),
            sx.flatten(),
            sy.flatten(),
            sz.flatten(),
        )
        LOGGER.debug("Spin vector quaternions: \n" + str(qvectors))
        qvectors = q_multi(qRot, qvectors)
        qvectors = q_multi(qvectors, qRotI)
        LOGGER.debug("Spin quaternions after rotation:\n" + str(qvectors))
        sx, sy, sz = qvectors[1], qvectors[2], qvectors[3]
        sx.shape, sy.shape, sz.shape = sxShape, syShape, szShape

        LOGGER.debug("GeneralRotation: ...Done")
        return (kpoints, sx, sy, sz)

    def rot_symmetry_z(self, order, kpoints, bands, projected, sx, sy, sz):
        """Applies the given rotational crystal symmetry to the current
        system. ie: to unfold the irreductible BZ to the full BZ.

        Only rotations along z-axis are performed, you can use
        self.GeneralRotation first. 

        The user is responsible of provide a useful input. The method
        doesn't check the physics.

        """
        character = np.array([])
        if character is not None:
            character = character
        sx = np.array([])
        if sx is not None:
            sx = sx
        sy = np.array([])
        if sy is not None:
            sy = sy
        sz = np.array([])
        if sz is not None:
            sz = sz

        
        LOGGER.debug("RotSymmetryZ:...")
        rotations = [
            self.general_rotation(360 * i / order, store=False) for i in range(order)
        ]
        rotations = list(zip(*rotations))
        LOGGER.debug(
            "self.kpoints.shape (before concat.): " + str(kpoints.shape)
        )
        kpoints = np.concatenate(rotations[0], axis=0)
        LOGGER.debug("self.kpoints.shape (after concat.): " + str(kpoints.shape))
        sx = np.concatenate(rotations[1], axis=0)
        sy = np.concatenate(rotations[2], axis=0)
        sz = np.concatenate(rotations[3], axis=0)
        # the bands and proj. character also need to be enlarged
        bandsChar = [(bands, projected) for i in range(order)]
        bandsChar = list(zip(*bandsChar))
        bands = np.concatenate(bandsChar[0], axis=0)
        projected = np.concatenate(bandsChar[1], axis=0)
        LOGGER.debug("RotSymmZ:...Done")

        return (kpoints, bands, projected, sx, sy, sz)

    def mirror_x(self, kpoints, bands, projected, sx, sy, sz):
        """Applies the given rotational crystal symmetry to the current
        system. ie: to unfold the irreductible BZ to the full BZ.

        """
        character = np.array([])
        if character is not None:
            character = character
        sx = np.array([])
        if sx is not None:
            sx = sx
        sy = np.array([])
        if sy is not None:
            sy = sy
        sz = np.array([])
        if sz is not None:
            sz = sz

        LOGGER.debug("Mirror:...")
        newK = kpoints * np.array([1, -1, 1])
        kpoints = np.concatenate((kpoints, newK), axis=0)
        LOGGER.debug("self.kpoints.shape (after concat.): " + str(kpoints.shape))
        newSx = -1 * sx
        newSy = 1 * sy
        newSz = 1 * sz
        sx = np.concatenate((sx, newSx), axis=0)
        sy = np.concatenate((sy, newSy), axis=0)
        sz = np.concatenate((sz, newSz), axis=0)
        print("self.sx", sx.shape)
        print("self.sy", sy.shape)
        print("self.sz", sz.shape)
        # the bands and proj. character also need to be enlarged
        bands = np.concatenate((bands, bands), axis=0)
        projected = np.concatenate((projected, projected), axis=0)
        print("self.projected", projected.shape)
        print("self.bands", bands.shape)
        LOGGER.debug("Mirror:...Done")

        return (kpoints, bands, projected, sx, sy, sz)

    def translate(self, newOrigin, kpoints):
        """Centers the Kpoints at newOrigin, newOrigin is either and index (of
        some Kpoint) or the cartesian coordinates of one point in the
        reciprocal space.

        """
        LOGGER.debug("Translate():  ...")
        if len(newOrigin) == 1:
            newOrigin = int(newOrigin[0])
            newOrigin = kpoints[newOrigin]
        # Make sure newOrigin is a numpy array
        newOrigin = np.array(newOrigin, dtype=float)
        LOGGER.debug("newOrigin: " + str(newOrigin))
        kpoints = kpoints - newOrigin
        LOGGER.debug("new Kpoints:\n" + str(kpoints))
        LOGGER.debug("Translate(): ...Done")
        return kpoints

def calculate_central_differences_on_meshgrid_axis(scalar_mesh,axis):
    """Calculates the scalar differences over the 
    k mesh grid using central differences

    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz]

    Returns
    -------
    np.ndarray
        scalar_gradient_mesh shape = [n_kx,n_ky,n_kz]
    """
    n=scalar_mesh.shape[axis]
    # Calculate indices with periodic boundary conditions
    plus_one_indices = np.arange(n) + 1
    minus_one_indices = np.arange(n) - 1
    plus_one_indices[-1] = 0
    minus_one_indices[0] = n - 1
    
    if axis==0:
        return (scalar_mesh[plus_one_indices,...] - scalar_mesh[minus_one_indices,...])/2
    elif axis==1:
        return (scalar_mesh[:,plus_one_indices,:,...] - scalar_mesh[:,minus_one_indices,:,...])/2
    elif axis==2:
        return (scalar_mesh[:,:,plus_one_indices,...] - scalar_mesh[:,:,minus_one_indices,...])/2
    
def calculate_forward_averages_on_meshgrid_axis(scalar_mesh,axis):
    """Calculates the scalar differences over the 
    k mesh grid using central differences

    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz]

    Returns
    -------
    np.ndarray
        scalar_gradient_mesh shape = [n_kx,n_ky,n_kz]
    """
    n=scalar_mesh.shape[axis]

    # Calculate indices with periodic boundary conditions
    plus_one_indices = np.arange(n) + 1
    zero_one_indices = np.arange(n)
    plus_one_indices[-1] = 0
    if axis==0:
        return (scalar_mesh[zero_one_indices,...] + scalar_mesh[plus_one_indices,...]) /2
    elif axis==1:
        return (scalar_mesh[:,zero_one_indices,:,...] + scalar_mesh[:,plus_one_indices,:,...]) /2
    elif axis==2:
        return (scalar_mesh[:,:,zero_one_indices,...] + scalar_mesh[:,:,plus_one_indices,...]) /2

def calculate_scalar_volume_averages(scalar_mesh):
    """Calculates the scalar averages over the k mesh grid in cartesian coordinates
    """
    scalar_sums_i=calculate_forward_averages_on_meshgrid_axis(scalar_mesh,axis=0)
    scalar_sums_j=calculate_forward_averages_on_meshgrid_axis(scalar_mesh,axis=1)
    scalar_sums_k=calculate_forward_averages_on_meshgrid_axis(scalar_mesh,axis=2)
    scalar_sums=(scalar_sums_i + scalar_sums_j + scalar_sums_k)/3
    return scalar_sums

def calculate_scalar_differences(scalar_mesh):
    """Calculates the scalar gradient over the k mesh grid in cartesian coordinates

    Uses gradient trnasformation matrix to calculate the gradient
    scalar_differens are calculated by central differences


    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz,...,3]
    """
    scalar_diffs_i=calculate_central_differences_on_meshgrid_axis(scalar_mesh,axis=0)
    scalar_diffs_j=calculate_central_differences_on_meshgrid_axis(scalar_mesh,axis=1)
    scalar_diffs_k=calculate_central_differences_on_meshgrid_axis(scalar_mesh,axis=2)
    scalar_diffs=np.array([scalar_diffs_i,scalar_diffs_j,scalar_diffs_k])
    scalar_diffs=np.moveaxis(scalar_diffs,0,-1)
    return scalar_diffs

def calculate_scalar_differences_2(scalar_mesh,transform_matrix):
    """Calculates the scalar gradient over the k mesh grid in cartesian coordinates

    Uses gradient trnasformation matrix to calculate the gradient
    scalar_differens are calculated by central differences


    Parameters
    ----------
    scalar_mesh : np.ndarray
        The scalar mesh. shape = [n_kx,n_ky,n_kz,...,3]
    """
    scalar_diffs_i=calculate_central_differences_on_meshgrid_axis(scalar_mesh,axis=0)
    scalar_diffs_j=calculate_central_differences_on_meshgrid_axis(scalar_mesh,axis=1)
    scalar_diffs_k=calculate_central_differences_on_meshgrid_axis(scalar_mesh,axis=2)
    scalar_diffs=np.array([scalar_diffs_i,scalar_diffs_j,scalar_diffs_k])
    scalar_diffs=np.moveaxis(scalar_diffs,0,-1)

    scalar_diffs_2=np.einsum('ij,uvwj->uvwi', transform_matrix, scalar_diffs)
    return scalar_diffs_2


def q_multi(q1, q2):
        """
        Multiplication of quaternions, it doesn't fit in any other place
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return np.array((w, x, y, z))


    # def reorder(self, plot=True, cutoff=0.2):
    #     nspins = self.nspins
    #     if self.is_non_collinear:
    #         nspins = 1
    #         # projected = np.sum(self.projected, axis=-1).reshape(self.nkpoints,
    #         #                                                     self.nbands,
    #         #                                                     self.natoms,
    #         #                                                     self.nprincipals,
    #         #                                                     self.norbitals,
    #         #                                                     nspins)
    #         projected = self.projected
    #         projected_phase = self.projected_phase
            
    #     else:
    #         projected = self.projected
    #         projected_phase = self.projected_phase
    #     new_bands = np.zeros_like(self.bands)
    #     new_projected = np.zeros_like(self.projected)
    #     new_projected_phase = np.zeros_like(self.projected_phase)
    #     for ispin in range(nspins):
    #         # DG = nx.Graph()
    #         # X = np.arange(self.nkpoints)
    #         # DG.add_nodes_from(
    #         #     [
    #         #         ((i, j), {"pos": (X[i], self.bands[i, j, ispin])})
    #         #         for i, j in itertools.product(
    #         #             range(self.nkpoints), range(self.nbands)
    #         #         )
    #         #     ]
    #         # )
    #         # pos = nx.get_node_attributes(DG, "pos")
    #         # S = np.zeros(shape=(self.nkpoints, self.nbands, self.nbands))
    #         for ikpoint in range(1, self.nkpoints):
    #             order = []
    #             for iband in range(self.nbands):
    #                 prj = []
    #                 idx = []
    #                 for jband in range(self.nbands):# range(max(0, iband-2), min(iband+2, self.nbands-1)):
    #                     psi_i = projected_phase[ikpoint, iband, :, :, :, ispin].flatten() # |psi(k,iband)>=\sum u(atom, n, ml)
    #                     psi_j = projected_phase[ikpoint-1, jband, :, :, :, ispin].flatten() # |psi(k+1,jband)>
    #                     psi_i /= np.absolute(psi_i).sum()
    #                     psi_j /= np.absolute(psi_j).sum()
    #                     # diff = np.absolute(psi_i-psi_j).sum()
    #                     diff = np.absolute(np.vdot(psi_i, psi_j))
    #                     prj.append(diff)
    #                     if iband == jband and  prj[-1] < cutoff:
    #                         prj[-1] = cutoff
    #                     idx.append(jband)
    #                 jband = idx[np.argmax(prj)]
    #                 self.bands[ikpoint, [iband, jband], ispin] = self.bands[ikpoint, [jband, iband], ispin]
    #         # for ikpoint in range(1, self.nkpoints):
    #         #     order = []
    #         #     for iband in range(self.nbands):
    #         #         prj = []
    #         #         idx = []
    #         #         slope = 
    #         #         for jband in range(max(0, iband-2), min(iband+2, self.nbands-1)):
    #         #             dK = np.linalg.norm(self.kpoints[ikpoint+1] -self.kpoints[ikpoint])
    #         #             dE = self.bands[ikpoint + 1, jband, ispin]
    #         #                 - self.bands[ikpoint, iband, ispin]
    #         #             ) 
    #         #             dEdK = (dE / dK)
                        
    #         #             prj.append(dEdk)
    #         #             if iband == jband and  prj[-1] < cutoff:
    #         #                 prj[-1] = cutoff
    #         #             idx.append(jband)
    #         #         jband = idx[np.argmax(prj)]
    #         #         self.bands[ikpoint, [iband, jband], ispin] = self.bands[ikpoint, [jband, iband], ispin]
    #     #             prj = np.repeat(projected[ikpoint, iband, :, :, :, ispin].reshape(1,
    #     #                                                                               self.natoms, self.nprincipals, self.norbitals), self.nbands, axis=0)
    #     #             dK = np.linalg.norm(self.kpoints[ikpoint+1] -self.kpoints[ikpoint])
    #     #             if dK == 0:
    #     #                 continue
    #     #             prj = np.repeat(
    #     #                 np.sqrt(projected[ikpoint, iband, :, :, :, ispin].astype(np.complex_)).reshape(
    #     #                     1, -1
    #     #                 ),  
    #     #                 self.nbands,
    #     #                 axis=0,
    #     #             )
    #     #             prj /= np.linalg.norm(prj[0])
    #     #             prj_1 = np.sqrt(
    #     #                 projected[ikpoint + 1, :, :, :, :, ispin].astype(np.complex_).reshape(
    #     #                     self.nbands, -1
    #     #                 )
    #     #             )
    #     #             prj_1 = prj_1.T / np.linalg.norm(prj_1, axis=1)
    #     #             prj_1 = prj_1.T
    #     #             # return prj, prj_1
    #     #             # prod = prj*prj_1
    #     #             # prod = prod.sum(axis=-1)
    #     #             # prod = prod.sum(axis=-1)
    #     #             # prod = prod.sum(axis=-1)
    #     #             # return prj, prj_1
    #     #             # # prod = np.exp(-1*prod* 1/(self.bands[ikpoint+1, :, ispin]-self.bands[ikpoint, iband, ispin]))
    #     #             # prod = np.exp(-1/abs(prod))
    #     #             # prod = np.exp(-1*abs(self.bands[ikpoint+1, :, ispin]-self.bands[ikpoint, iband, ispin]+e))
    #     #             dots = np.array([np.vdot(x, y) for x, y in zip(prj, prj_1)])
                    
    #     #             diff = np.linalg.norm(prj - prj_1, axis=1)
    #     #             dE = np.abs(
    #     #                 self.bands[ikpoint + 1, :, ispin]
    #     #                 - self.bands[ikpoint, iband, ispin]
    #     #             ) 
    #     #             dEdK = (dE / dK)
    #     #             jband = np.argsort(diff)
    #     #             counter = 0
    #     #             # while jband[counter] in order:
    #     #             #     counter+=1
    #     #             order.append(jband[counter])  
                    
    #     #             # if iband !=0 and jband== 0:
    #     #             #     print(ikpoint, iband)
                    
    #     #             # print(iband, self.bands[ikpoint, iband, ispin], jband, self.bands[ikpoint, jband, ispin])
                    
    #     #             # diffs.append(diff)
    #     #             # DG.add_weighted_edges_from([((ikpoint, iband),(ikpoint+1, x[0]),x[1]) for x in zip(range(self.nbands), prod)])
    #     #             DG.add_edge((ikpoint, iband), (ikpoint + 1, jband[counter]))
                    
    #     #     if plot:
    #     #         plt.figure(figsize=(16, 9))
    #     #         nodes = nx.draw_networkx_nodes(
    #     #             DG, pos, node_size=5, node_color=["blue", "red"][ispin])
    #     #         edges = nx.draw_networkx_edges(
    #     #             DG,
    #     #             pos,
    #     #             edge_color='red'
    #     #         )
    #     #         plt.show()
    #     # #         if len(order) == 0:
    #     # #             new_bands[ikpoint+1,:,ispin] = self.bands[ikpoint+1,:,ispin]
    #     # #             new_projected[ikpoint+1, :, :, :, :, :] = self.projected[ikpoint+1, :, :, :, :, :]
    #     # #         else :
    #     # #             new_bands[ikpoint+1,:,ispin] = self.bands[ikpoint+1, order,ispin]
    #     # #             new_projected[ikpoint+1, :, :, :, :, :] = self.projected[ikpoint+1, order, :, :, :, :]
                
    #     # # self.bands = new_bands
    #     # # self.projected = new_projected
    #     return
