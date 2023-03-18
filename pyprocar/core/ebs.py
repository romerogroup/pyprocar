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
        kpath:KPath=None,
        weights:np.ndarray=None,
        labels:List=None,
        reciprocal_lattice:np.ndarray=None,
        shifted_to_efermi:bool=False,
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

        self.kpoints = np.array(klist)
        self.projected = np.array(plist)
        self.bands = np.array(bandslist)

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
