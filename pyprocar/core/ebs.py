# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Pedram Tavadze
@author: Freddy Farah

"""

# from . import Structure
from typing import List
import itertools

from scipy.interpolate import CubicSpline

import numpy as np
import networkx as nx
from matplotlib import pylab as plt
import pyvista

from .kpath import KPath
from .brillouin_zone import BrillouinZone
from ..utils import Unfolder, mathematics

HBAR_EV = 6.582119 *10**(-16) #eV*s
HBAR_J = 1.0545718 *10**(-34) #eV*s
METER_ANGSTROM = 10**(-10) #m /A
EV_TO_J = 1.602*10**(-19)
FREE_ELECTRON_MASS = 9.11*10**-31 #  kg
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
            A list of orbital names, defaults to None
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
        projected_phase:np.ndarray =None,
        weights:np.ndarray=None,
        kpath:KPath=None,
        labels:List=None,
        reciprocal_lattice:np.ndarray=None,
        ):
        
        
        self.kpoints = kpoints
        self.bands = bands - efermi
        self.efermi = efermi
        
        self.projected = projected
        self.projected_phase = projected_phase
        self.reciprocal_lattice = reciprocal_lattice
        self.kpath = kpath
        if self.projected_phase is not None:
            self.has_phase = True
        else:
            self.has_phase = False
        self.labels = labels
        self.weights = weights
        self.graph = None

        self._n_kx=n_kx
        self._n_ky=n_ky
        self._n_kz=n_kz
        
        self._index_mesh=None
        self._kpoints_mesh=None
        self._kpoints_cartesian_mesh=None
        self._bands_mesh=None
        self._projected_mesh=None
        self._projected_phase_mesh=None
        self._weights_mesh=None
        self._bands_gradient_mesh=None
        self._bands_hessian_mesh=None
        self._fermi_velocity_mesh=None
        self._harmonic_average_effective_mass_mesh=None
        self._fermi_speed_mesh=None

        self._bands_gradient=None
        self._bands_hessian=None
        self._fermi_velocity=None
        self._effective_mass=None
        self._fermi_speed=None
        self._harmonic_average_effective_mass=None
        

            
    @property
    def n_kx(self):
        """The number of unique kpoints in kx direction in the reduced basis

        Returns
        -------
        int
            The number of unique kpoints in kx direction in the reduced basis
        """

        if self._n_kx==None:
            self._n_kx=len( np.unique(self.kpoints[:,0]) )
        return self._n_kx
    
    @property
    def n_ky(self):
        """The number of unique kpoints in kx direction in the reduced basis

        Returns
        -------
        int
            The number of unique kpoints in kx direction in the reduced basis
        """
        if self._n_ky==None:
            self._n_ky=len( np.unique(self.kpoints[:,1]) )
        return self._n_ky
    
    @property
    def n_kz(self):
        """The number of unique kpoints in ky direction in the reduced basis

        Returns
        -------
        int
            The number of unique kpoints in ky direction in the reduced basis
        """
        if self._n_kz==None:
            self._n_kz=len( np.unique(self.kpoints[:,2]) )
        return self._n_kz

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
    def kpoints_cartesian(self):
        """Returns the kpoints in cartesian basis

        Returns
        -------
        np.ndarray
            Returns the kpoints in cartesian basis
        """
        if self.reciprocal_lattice is not None:
            return np.dot(self.kpoints, self.reciprocal_lattice)
        else:
            print(
                "Please provide a reciprocal lattice when initiating the Procar class"
            )
            return

    @property
    def kpoints_reduced(self):
        """Returns the kpoints in fractional basis

        Returns
        -------
        np.ndarray
            Returns the kpoints in fractional basis
        """
        return self.kpoints
    
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
    def index_mesh(self):
        """
        Index mesh stores the the kpoints index in 
        kpoints list at a particular grid point . 
        Shape = [n_kx,n_ky,n_kz]

        Returns
        -------
        np.ndarray
            Index mesh stores the the kpoints index in 
            kpoints list at a particular grid point .  Shape = [n_kx,n_ky,n_kz]

        """
        if self._index_mesh is None:
            kx_unique = np.unique(self.kpoints[:,0])
            ky_unique = np.unique(self.kpoints[:,1])
            kz_unique = np.unique(self.kpoints[:,2])
            n_kx = len(kx_unique)
            n_ky = len(ky_unique)
            n_kz = len(kz_unique)
            self._index_mesh = np.zeros((n_kx,n_ky,n_kz),dtype=int)

            for k in range(n_kz):
                for j in range(n_ky):
                    for i in range(n_kx):

                        kx = kx_unique[i]
                        ky = ky_unique[j]
                        kz = kz_unique[k]

                        where_x_true_indices = np.where(self.kpoints[:,0] == kx)[0]
                        where_x_true_points = self.kpoints[where_x_true_indices]

                        where_xy_true_indices = np.where(where_x_true_points[:,1] == ky)[0]
                        where_xy_true_points = where_x_true_points[where_xy_true_indices]

                        where_xyz_true_indices = np.where(where_xy_true_points[:,2] == kz)[0]
                        where_xyz_true_points = where_xy_true_points[where_xyz_true_indices]

                        original_index = where_x_true_indices[where_xy_true_indices[where_xyz_true_indices]]
                        # print(original_index)
                        self._index_mesh[i,j,k] = original_index
        return self._index_mesh
    
    @property
    def kpoints_mesh(self):
        """Kpoint mesh representation of the kpoints grid. Shape = [3,n_kx,n_ky,n_kz]
        Returns
        -------
        np.ndarray
            Kpoint mesh representation of the kpoints grid. Shape = [3,n_kx,n_ky,n_kz]
        """

        if self._kpoints_mesh is None:
            self._kpoints_mesh = self.create_nd_mesh(nd_list = self.kpoints)
        return self._kpoints_mesh
    
    @property
    def kpoints_cartesian_mesh(self):
        """Kpoint cartesian mesh representation of the kpoints grid. Shape = [3,n_kx,n_ky,n_kz]
        Returns
        -------
        np.ndarray
            Kpoint cartesian mesh representation of the kpoints grid. Shape = [3,n_kx,n_ky,n_kz]
        """
        if self._kpoints_cartesian_mesh is None:
            self._kpoints_cartesian_mesh = self.create_nd_mesh(nd_list = self.kpoints_cartesian)
        return self._kpoints_cartesian_mesh
    
    @property
    def bands_mesh(self,force_update=False):
        """
        Bands mesh is a numpy array that stores each band in a mesh grid.   
        Shape = [n_bands,n_kx,n_ky,n_kz]

        Returns
        -------
        np.ndarray
            Bands mesh is a numpy array that stores each band in a mesh grid. 
            Shape = [n_bands,n_kx,n_ky,n_kz]
        """

        if self._bands_mesh is None or force_update:
            self._bands_mesh = self.create_nd_mesh(nd_list = self.bands)
        return self._bands_mesh
    
    @property
    def projected_mesh(self):
        """
        projected mesh is a numpy array that stores each projection in a mesh grid.   
        Shape = [n_bands,n_spins,n_atoms,n_orbitals,n_kx,n_ky,n_kz]

        Returns
        -------
        np.ndarray
            Projection mesh is a numpy array that stores each projection in a mesh grid. 
            Shape = [n_bands,n_spins,n_atoms,n_orbitals,n_kx,n_ky,n_kz]
        """

        if self._projected_mesh is None:
            self._projected_mesh = self.create_nd_mesh(nd_list = self.projected)
        return self._projected_mesh
    
    @property
    def projected_phase_mesh(self):
        """
        projected phase mesh is a numpy array that stores each projection phases in a mesh grid.   
        Shape = [n_bands,n_spins,n_atoms,n_orbitals,n_kx,n_ky,n_kz]

        Returns
        -------
        np.ndarray
            projected phase mesh is a numpy array that stores each projection phases in a mesh grid.  
            Shape = [n_bands,n_spins,n_atoms,n_orbitals,n_kx,n_ky,n_kz]
        """

        if self._projected_phase_mesh is None:
            self._projected_phase_mesh = self.create_nd_mesh(nd_list = self.projected_phase)
        return self._projected_phase_mesh

    @property
    def weights_mesh(self):
        """
        weights mesh is a numpy array that stores each weights in a mesh grid.   
        Shape = [1,n_kx,n_ky,n_kz]

        Returns
        -------
        np.ndarray
            weights mesh is a numpy array that stores each weights in a mesh grid. 
            Shape = [1,n_kx,n_ky,n_kz]
        """

        if self._weights_mesh is None:
            self._weights_mesh = self.create_nd_mesh(nd_list = self.weights)
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
            Shape = [3,n_bands,n_kx,n_ky,n_kz], 
            where the first dimension represents d/dx,d/dy,d/dz  
        """

        if self._bands_gradient_mesh is None:
            n_bands, n_spins, n_i, n_j, n_k = self.bands_mesh.shape

            band_gradients = np.zeros((3, n_bands, n_spins, n_i, n_j, n_k))

            for i_band in range(n_bands):
                for i_spin in range(n_spins):
                    band_gradients[:,i_band,i_spin,:,:,:] = self.calculate_scalar_gradient(scalar_mesh = self.bands_mesh[i_band,i_spin,:,:,:])
            
            band_gradients *= METER_ANGSTROM
            self._bands_gradient_mesh = band_gradients
        return self._bands_gradient_mesh
    
    @property
    def bands_hessian_mesh(self):
        """
        Bands hessian mesh is a numpy array that stores each band hessian in a mesh grid.   
        Shape = [3,3,n_bands,n_spin,n_kx,n_ky,n_kz], 
        where the first and second dimension represent d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Bands hessian mesh is a numpy array that stores each band hessian in a mesh grid.
            Shape = [3,3,n_bands,n_spin,n_kx,n_ky,n_kz], 
            where the first and second dimension represent d/dx,d/dy,d/dz  
        """

        if self._bands_hessian_mesh is None:
            n_dim, n_bands, n_spins, n_i, n_j, n_k = self.bands_gradient_mesh.shape

            band_hessians = np.zeros((3, 3,n_bands, n_spins, n_i, n_j, n_k))

            for i_dim in range(n_dim):
                for i_band in range(n_bands):
                    for i_spin in range(n_spins):
                        band_hessians[:,i_dim,i_band,i_spin,:,:,:] = self.calculate_scalar_gradient(scalar_mesh = self.bands_gradient_mesh[i_dim,i_band,i_spin,:,:,:])
            
            band_hessians *= METER_ANGSTROM
            self._bands_hessian_mesh = band_hessians
        return self._bands_hessian_mesh
    
    @property
    def fermi_velocity_mesh(self):
        """
        Fermi Velocity mesh is a numpy array that stores each  Fermi Velocity in a mesh grid.   
        Shape = [3,n_bands,n_kx,n_ky,n_kz], where the first dimension represents d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Fermi Velocity mesh is a numpy array that stores each  Fermi Velocity in a mesh grid.
            Shape = [3,n_bands,n_kx,n_ky,n_kz], 
            where the first dimension represents d/dx,d/dy,d/dz  
        """

        if self._fermi_velocity_mesh is None:
            self._fermi_velocity_mesh =  self.bands_gradient_mesh / HBAR_EV
        return self._fermi_velocity_mesh
    
    @property
    def fermi_speed_mesh(self):
        """
        Fermi speed mesh is a numpy array that stores each  Fermi Velocity in a mesh grid.   
        Shape = [n_bands,n_kx,n_ky,n_kz],
        
        Returns
        -------
        np.ndarray
            Fermi speed mesh is a numpy array that stores each  Fermi Velocity in a mesh grid.
            Shape = [n_bands,n_kx,n_ky,n_kz], 
        """

        if self._fermi_speed_mesh is None:
            n_grad_1, n_bands, n_spins, n_i, n_j, n_k = self.fermi_velocity_mesh.shape

            self._fermi_speed_mesh = np.zeros(shape=(n_bands, n_spins, n_i, n_j, n_k))
            for iband in range(n_bands):
                for ispin in range(n_spins):
                    for k in range(n_k):
                        for j in range(n_j):
                            for i in range(n_i):
                                self._fermi_speed_mesh[iband,ispin,i,j,k] =  np.linalg.norm(self.fermi_velocity_mesh[:,iband,ispin,i,j,k])
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
            n_grad_1,n_grad_2, n_bands, n_spins, n_i, n_j, n_k = self.bands_hessian_mesh.shape

            self._harmonic_average_effective_mass_mesh = np.zeros(shape=(n_bands, n_spins, n_i, n_j, n_k))

            for iband in range(n_bands):
                for ispin in range(n_spins):
                    for k in range(n_k):
                        for j in range(n_j):
                            for i in range(n_i):
                                hessian = self.bands_hessian_mesh[...,iband,ispin,i,j,k] * EV_TO_J/ HBAR_J**2
                                self._harmonic_average_effective_mass_mesh[iband,ispin,i,j,k] = harmonic_average_effective_mass(hessian)
        return self._harmonic_average_effective_mass_mesh
    
    def calculate_scalar_gradient(self,scalar_mesh):
        """Calculates the scalar gradient over the k mesh grid in cartesian coordinates

        Parameters
        ----------
        scalar_mesh : np.ndarray
            The scalar mesh. shape = [n_kx,n_ky,n_kz]

        Returns
        -------
        np.ndarray
            scalar_gradient_mesh shape = [3,n_kx,n_ky,n_kz]
        """
        n_kx=len(np.unique(self.kpoints[:,0]))
        n_ky=len(np.unique(self.kpoints[:,1]))
        n_kz=len(np.unique(self.kpoints[:,2]))
        scalar_gradients =  np.zeros((3,n_kx,n_ky,n_kz))

        if n_kx==1 or n_ky==1 or n_kz==1:
            raise ValueError("The mesh cannot be 2 dimensional")
        
        # Calculate cartesian separations along each crystal direction
        sep_vectors_i = np.abs(self.kpoints_cartesian[self.index_mesh[[0]*n_kx, :, :], :] - self.kpoints_cartesian[self.index_mesh[[1]*n_kx, :, :], :])
        sep_vectors_j = np.abs(self.kpoints_cartesian[self.index_mesh[:, [0]*n_ky, :], :] - self.kpoints_cartesian[self.index_mesh[:, [1]*n_ky, :], :])
        sep_vectors_k = np.abs(self.kpoints_cartesian[self.index_mesh[:, :, [0]*n_kz], :] - self.kpoints_cartesian[self.index_mesh[:, :, [1]*n_kz], :])

        # Calculate indices with periodic boundary conditions
        plus_one_indices_x = np.arange(n_kx) + 1
        plus_one_indices_y = np.arange(n_ky) + 1
        plus_one_indices_z = np.arange(n_kz) + 1

        minus_one_indices_x = np.arange(n_kx) - 1
        minus_one_indices_y = np.arange(n_ky) - 1
        minus_one_indices_z = np.arange(n_kz) - 1

        plus_one_indices_x[-1] = 0
        plus_one_indices_y[-1] = 0
        plus_one_indices_z[-1] = 0

        minus_one_indices_x[0] = n_kx - 1
        minus_one_indices_y[0] = n_ky - 1
        minus_one_indices_z[0] = n_kz - 1

        scalar_diffs_i = scalar_mesh[plus_one_indices_x,:,:] - scalar_mesh[minus_one_indices_x,:,:]
        scalar_diffs_j = scalar_mesh[:,plus_one_indices_y,:] - scalar_mesh[:,minus_one_indices_y,:]
        scalar_diffs_k = scalar_mesh[:,:,plus_one_indices_z] - scalar_mesh[:,:,minus_one_indices_z]
        
        # Calculating gradients
        sep_vectors = [sep_vectors_i,sep_vectors_j,sep_vectors_k]
        energy_diffs = [scalar_diffs_i,scalar_diffs_j,scalar_diffs_k]
        for sep_vector,energy_diff in zip(sep_vectors,energy_diffs):
            for i_coord in range(3):
                
                dx = sep_vector[:, :, :, i_coord]
                tmp_grad = energy_diff / (2 * dx)
                # Changing infinities to 0
                tmp_grad = np.nan_to_num(tmp_grad, neginf=0,posinf=0) 
                scalar_gradients[i_coord, :, :, :] += tmp_grad
        return scalar_gradients
    
    def calculate_scalar_integral(self,scalar_mesh):
        """Calculate the scalar integral"""


        edge1 = abs(self.kpoints_cartesian[self.index_mesh[1, 0, 0], :] - self.kpoints_cartesian[self.index_mesh[0, 0, 0], :])
        edge2 = abs(self.kpoints_cartesian[self.index_mesh[0, 1, 0], :] - self.kpoints_cartesian[self.index_mesh[0, 0, 0], :])
        edge3 = abs(self.kpoints_cartesian[self.index_mesh[0, 0, 1], :] - self.kpoints_cartesian[self.index_mesh[0, 0, 0], :])
        
        # Create a matrix with the edge vectors.
        matrix = np.array([edge1, edge2, edge3]).T
        
        # Calculate the volume using the determinant of the matrix.
        dv = abs(np.linalg.det(matrix))

        # Compute the integral by summing up the product of scalar values and the volume of each grid cell.
        integral = np.sum(scalar_mesh * dv)
        
        return integral

    def create_nd_mesh(self, nd_list):

        if nd_list is not  None:

            nd_shape = list(nd_list.shape[1:])


            n_kx=len(np.unique(self.kpoints[:,0]))
            n_ky=len(np.unique(self.kpoints[:,1]))
            n_kz=len(np.unique(self.kpoints[:,2]))
            nd_shape.append(n_kx)
            nd_shape.append(n_ky)
            nd_shape.append(n_kz)
            nd_mesh = np.zeros(nd_shape,dtype=nd_list.dtype)
            non_grid_dims = nd_shape[:-3]
            for i_dim, non_grid_dim in enumerate(non_grid_dims):
                for i in range(non_grid_dim):
                    for x in range(n_kx):
                        for y in range(n_ky):
                            for z in range(n_kz):
                                # Forming slicing tuples for mesh
                                mesh_idx = [ np.s_[:]]*nd_mesh.ndim
                                mesh_idx[i_dim] = i
                                mesh_idx[-3] = x
                                mesh_idx[-2] = y
                                mesh_idx[-1] = z

                                # Forming slicing tuples for list
                                list_idx = [ np.s_[:]]*nd_list.ndim
                                list_idx[0] = self.index_mesh[x,y,z]
                                list_idx[i_dim+1] = i
                                
                                # Assigning mesh values from list
                                nd_mesh[ tuple(mesh_idx) ] = nd_list[ tuple(list_idx) ]
        else:
            nd_mesh = None
        return nd_mesh
    
    def create_vector_mesh(self, vector_list):
        vector_mesh = np.zeros((3,self.n_kx,self.n_ky,self.n_kz),dtype=float)
        for i in range(3):
            vector_mesh[i,:,:,:]= vector_list[ self.index_mesh, i ]
        return vector_mesh
    
    def create_scaler_mesh(self, scalar_list):
        scalar_mesh = np.zeros((1,self.n_kx,self.n_ky,self.n_kz),dtype=float)
        scalar_mesh[0,:,:,:]= scalar_list[ self.index_mesh ]
        return scalar_mesh

    def mesh_to_array(self, mesh):
        """
        Converts a mesh to a list that corresponds to ebs.kpoints  

        Parameters
        ----------
        mesh : np.ndarray
            The mesh to convert to a list
        Returns
        -------
        np.ndarray
           lsit
        """
        nkx, nky, nkz = mesh.shape[-3:]
        tmp_shape=[nkx*nky*nkz]
        tmp_shape.extend(mesh.shape[:-3])
        tmp_array=np.zeros(shape=tmp_shape)
        for k in range(nkz):
            for j in range(nky):
                for i in range(nkx):
                    index=self.index_mesh[i,j,k]
                    tmp_array[index,...]=mesh[...,i,j,k]
        return tmp_array

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

        return None

    def ibz2fbz(self, rotations):
        """Applys symmetry operations to the kpoints, bands, and projections

        Parameters
        ----------
        rotations : np.ndarray
            The point symmetry operations of the lattice
        """
        
        full_kpoints=[]
        full_projected=[]
        full_bands=[]
        full_weights=[]
        full_projected_phases=[]

        # Used for indexing full_kpoints
        ik=0
        # for each symmetry operation
        for i, rotation in enumerate(rotations):
            # for each point
            for j, kpoint in enumerate(self.kpoints):
                # apply symmetry operation to kpoint
                
                new_kp = rotation.dot(kpoint)
                # apply boundary conditions
                new_kp = -np.fmod(new_kp + 6.5, 1 ) + 0.5
                new_kp = np.around(new_kp,decimals=6)
                new_kp=new_kp.tolist()
                if new_kp not in full_kpoints:
                    full_kpoints.append(new_kp)
                    if self.bands is not None:
                        band = self.bands[j].tolist()
                        full_bands.append(band)
                    if self.projected is not None:
                        projection = self.projected[j].tolist()
                        full_projected.append(projection)
                    if self.weights is not None:
                        weight = self.weights[j].tolist()
                        full_weights.append(weight)
                    if self.projected_phase is not None:
                        projected_phase = self.projected_phase[j].tolist()
                        full_projected_phases.append(projected_phase)
        self.kpoints = np.array(full_kpoints)
        self.bands = np.array(full_bands)

        if self.projected is not None:
            self.projected = np.array(full_projected)
        if self.projected_phase is not None:
            self.projected_phase = np.array(full_projected_phases)
        if self.weights is not None:
            self.weights = np.array(full_weights)

        nk=self.kpoints.shape[0]
        if self.n_kx:
            if nk != self.n_kx*self.n_ky*self.n_kz:

                err_text="""
                        nkpoints != n_kx*n_ky*n_kz
                        Error trying to symmetrize the irreducible kmesh. 
                        This is issue is most likely related to 
                        how the DFT code using symmetry operations to reduce the kmesh.
                        Check the recommendations for k-mesh type for the crystal system.
                        If all else fails turn off symmetry.
                        """
                raise ValueError(err_text)

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
    
    def ravel_array(self,mesh_grid):
        shape = mesh_grid.shape
        mesh_grid=mesh_grid.reshape(shape[:-3] + (-1,))
        mesh_grid=np.moveaxis(mesh_grid,-1,0)
        return mesh_grid

    def __str__(self):
        ret = 'Enectronic Band Structure     \n'
        ret += '------------------------     \n'
        ret += 'Total number of kpoints  = {}\n'.format(self.nkpoints)
        ret += 'Total number of bands    = {}\n'.format(self.nbands)
        ret += 'Total number of atoms    = {}\n'.format(self.natoms)
        ret += 'Total number of orbitals = {}\n'.format(self.norbitals)
        return ret

def harmonic_average_effective_mass(tensor):
    inv_effective_mass_tensor = tensor
    e_mass = 3*(inv_effective_mass_tensor[0,0] + inv_effective_mass_tensor[1,1] + inv_effective_mass_tensor[2,2])**-1 /FREE_ELECTRON_MASS
    return e_mass

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
