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

        
        
        self._index_mesh = None
        self._kpoints_mesh = None
        self._kpoints_cartesian_mesh = None
        self._bands_mesh = None
        self._projected_mesh = None
        self._projected_phase_mesh = None
        self._weights_mesh = None
        self._bands_gradient_mesh = None
        self._bands_hessian_mesh = None

            
    @property
    def n_kx(self):
        """The number of unique kpoints in kx direction in the reduced basis

        Returns
        -------
        int
            The number of unique kpoints in kx direction in the reduced basis
        """
        return len( np.unique(self.kpoints[:,0]) )
    
    @property
    def n_ky(self):
        """The number of unique kpoints in kx direction in the reduced basis

        Returns
        -------
        int
            The number of unique kpoints in kx direction in the reduced basis
        """
        return len( np.unique(self.kpoints[:,1]) )
    
    @property
    def n_kz(self):
        """The number of unique kpoints in ky direction in the reduced basis

        Returns
        -------
        int
            The number of unique kpoints in ky direction in the reduced basis
        """
        return len( np.unique(self.kpoints[:,2]) )

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
        if self.nspins == 3:
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
    def index_mesh(self):
        """Index mesh stores the the kpoints index in 
        kpoints list at a particular grid point . 
        Shape = [n_kx,n_ky,n_kz]
        Returns
        -------
        np.ndarray
            Index mesh stores the the kpoints index in 
            kpoints list at a particular grid point .  Shape = [n_kx,n_ky,n_kz]
        """
        if self._index_mesh is None:
            self.update_index_mesh()
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
            self.update_kpoints_mesh()
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
            self.update_kpoints_cartesian_mesh()
        return self._kpoints_cartesian_mesh
    
    @property
    def bands_mesh(self):
        """
        Bands mesh is a numpy array that stores each band in a mesh grid.   
        Shape = [n_bands,n_kx,n_ky,n_kz]

        Returns
        -------
        np.ndarray
            Bands mesh is a numpy array that stores each band in a mesh grid. 
            Shape = [n_bands,n_kx,n_ky,n_kz]
        """

        if self._bands_mesh is None:
            self.update_bands_mesh()
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
            self.update_projected_mesh()
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
            self.update_projected_phase_mesh()
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
            self.update_weights_mesh()
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
            self.update_bands_gradient_mesh()
        return self._bands_gradient_mesh
    
    @property
    def bands_hessian_mesh(self):
        """
        Bands hessian mesh is a numpy array that stores each band hessian in a mesh grid.   
        Shape = [3,3,n_bands,n_kx,n_ky,n_kz], 
        where the first and second dimension represent d/dx,d/dy,d/dz  
        
        Returns
        -------
        np.ndarray
            Bands hessian mesh is a numpy array that stores each band hessian in a mesh grid.
            Shape = [3,3,n_bands,n_kx,n_ky,n_kz], 
            where the first and second dimension represent d/dx,d/dy,d/dz  
        """

        if self._bands_hessian_mesh is None:
            self.update_bands_hessian_mesh()
        return self._bands_hessian_mesh

    def update_kpoints_mesh(self):
        """This method will update the kpoints mesh representation

        Returns
        -------
        np.ndarray
            Updated the kpoints mesh
        """
        self._kpoints_mesh = self.create_nd_mesh(nd_list = self.kpoints)
        return self._kpoints_mesh

    def update_kpoints_cartesian_mesh(self):
        """This method will update the kpoints mesh representation

        Returns
        -------
        np.ndarray
            Updated the kpoints mesh
        """
        self._kpoints_cartesian_mesh = self.create_nd_mesh(nd_list = self.kpoints_cartesian)
        return self._kpoints_cartesian_mesh
    
    def update_bands_mesh(self):
        """This method will update the bands mesh representation

        Returns
        -------
        np.ndarray
            Updated the bands mesh
        """
        self._bands_mesh = self.create_nd_mesh(nd_list = self.bands)
        return self._bands_mesh
    
    def update_projected_mesh(self):
        """This method will update the projected mesh representation

        Returns
        -------
        np.ndarray
            Updated the projected mesh
        """
        self._projected_mesh = self.create_nd_mesh(nd_list = self.projected)
        return self._projected_mesh
    
    def update_projected_phase_mesh(self):
        """This method will update the projected phase mesh representation

        Returns
        -------
        np.ndarray
            Updated the projected_phase mesh
        """
        self._projected_phase_mesh = self.create_nd_mesh(nd_list = self.projected_phase)
        return self._projected_phase_mesh
    
    def update_weights_mesh(self):
        """This method will update the weights mesh representation

        Returns
        -------
        np.ndarray
            Updated the weights mesh
        """
        self._weights_mesh = self.create_nd_mesh(nd_list = self.weights)
        return self._weights_mesh

    def update_bands_gradient_mesh(self):
        """This method will update the bands_gradient mesh representation

        Returns
        -------
        np.ndarray
            Updated the bands_gradient
        """
        n_bands, n_spins, n_i, n_j, n_k = self.bands_mesh.shape

        band_gradients = np.zeros((3, n_bands, n_spins, n_i, n_j, n_k))

        for i_band in range(n_bands):
            for i_spin in range(n_spins):
                band_gradients[:,i_band,i_spin,:,:,:] = self.calculate_scalar_gradient(scalar_mesh = self.bands_mesh[i_band,i_spin,:,:,:])
        
        band_gradients /= HBAR_EV
        band_gradients *= METER_ANGSTROM
        self._bands_gradient_mesh = band_gradients
        return self._bands_gradient_mesh
    
    def update_bands_hessian_mesh(self):
        """This method will update the bands_hessian mesh representation

        Returns
        -------
        np.ndarray
            Updated the bands_hessian
        """
        n_dim, n_bands, n_spins, n_i, n_j, n_k = self.bands_gradient_mesh.shape

        band_hessians = np.zeros((3, 3,n_bands, n_spins, n_i, n_j, n_k))

        for i_dim in range(n_dim):
            for i_band in range(n_bands):
                for i_spin in range(n_spins):
                    band_hessians[:,i_dim,i_band,i_spin,:,:,:] = self.calculate_scalar_gradient(scalar_mesh = self.bands_gradient_mesh[i_dim,i_band,i_spin,:,:,:])
        
        band_hessians *= METER_ANGSTROM
        self._bands_hessian_mesh = band_hessians
        return self._bands_hessian_mesh

    def update_index_mesh(self):
        """This method will update the index mesh

        Returns
        -------
        np.ndarray
            Updated the kpoints mesh
        """

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

                    self._index_mesh[i,j,k] = original_index

        return self._index_mesh
        
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
        nx,ny,nz = self.n_kx,self.n_ky,self.n_kz

        scalar_gradients =  np.zeros((3,nx,ny,nz))

        # Calculate cartesian separations along each crystal direction
        sep_vectors_i = np.abs(self.kpoints_cartesian[self.index_mesh[[0]*nx, :, :], :] - self.kpoints_cartesian[self.index_mesh[[1]*nx, :, :], :])
        sep_vectors_j = np.abs(self.kpoints_cartesian[self.index_mesh[:, [0]*ny, :], :] - self.kpoints_cartesian[self.index_mesh[:, [1]*ny, :], :])
        sep_vectors_k = np.abs(self.kpoints_cartesian[self.index_mesh[:, :, [0]*nz], :] - self.kpoints_cartesian[self.index_mesh[:, :, [1]*nz], :])

        # Calculate indices with periodic boundary conditions
        plus_one_indices = np.arange(20) + 1
        minus_one_indices = np.arange(20) - 1
        plus_one_indices[-1] = 0
        minus_one_indices[0] = 19

        scalar_diffs_i = scalar_mesh[plus_one_indices,:,:] - scalar_mesh[minus_one_indices,:,:]
        scalar_diffs_j = scalar_mesh[:,plus_one_indices,:] - scalar_mesh[:,minus_one_indices,:]
        scalar_diffs_k = scalar_mesh[:,:,plus_one_indices] - scalar_mesh[:,:,minus_one_indices]
        
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
        nd_shape = list(nd_list.shape[1:])
        nd_shape.append(self.n_kx)
        nd_shape.append(self.n_ky)
        nd_shape.append(self.n_kz)

        nd_mesh = np.zeros(nd_shape,dtype=nd_list.dtype)
        non_grid_dims = nd_shape[:-3]
        for i_dim, non_grid_dim in enumerate(non_grid_dims):
            for i in range(non_grid_dim):
                for x in range(self.n_kx):
                    for y in range(self.n_ky):
                        for z in range(self.n_kz):
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

    def update_weights(self, weights):
        """Updates the weights corresponding to the kpoints

        Parameters
        ----------
        weights : List[float]
            A list of weights corresponding to the kpoints

        Returns
        -------
        None
            None
        """
        self.weights = weights
        return None

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
        
        klist = []
        plist = []
        bandslist = []
        weights=[]
        projected_phases=[]

        # for each symmetry operation
        for i, rotation in enumerate(rotations):
            # for each point
            for j, kpoint in enumerate(self.kpoints):
                # apply symmetry operation to kpoint

                
                sympoint_vector = rotation.dot(kpoint)
                # apply boundary conditions
                bound_ops = -1.0*(sympoint_vector > 0.5) + 1.0*(sympoint_vector <= -0.5)
                sympoint_vector += bound_ops

                sympoint_vector=np.around(sympoint_vector,decimals=6)
                sympoint = sympoint_vector.tolist()

                if sympoint not in klist:
                    klist.append(sympoint)
                    if self.bands is not None:
                        band = self.bands[j].tolist()
                        bandslist.append(band)
                    if self.projected is not None:
                        projection = self.projected[j].tolist()
                        plist.append(projection)
                    if self.weights is not None:
                        weight = self.weights[j].tolist()
                        weights.append(weight)
                    if self.weights is not None:
                        weight = self.weights[j].tolist()
                        weights.append(weight)
                    if self.projected_phase is not None:
                        projected_phase = self.projected_phase[j].tolist()
                        projected_phases.append(projected_phase)

        self.kpoints = np.array(klist)
        self.bands = np.array(bandslist)

        if self.projected is not None:
            self.projected = np.array(plist)
        if self.projected_phase is not None:
            self.projected_phase = np.array(projected_phases)
        if self.weights is not None:
            self.weights = np.array(weights)
        # self.sort_bands_and_kpoints()


        

    def __str__(self):
        ret = 'Enectronic Band Structure     \n'
        ret += '------------------------     \n'
        ret += 'Total number of kpoints  = {}\n'.format(self.nkpoints)
        ret += 'Total number of bands    = {}\n'.format(self.nbands)
        ret += 'Total number of atoms    = {}\n'.format(self.natoms)
        ret += 'Total number of orbitals = {}\n'.format(self.norbitals)
        return ret
        
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
