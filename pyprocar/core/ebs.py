# -*- coding: utf-8 -*-


__author__ = "Pedram Tavadze, Logan Lang, Freddy Farah"
__copyright__ = "Copyright (C) 2007 Free Software Foundation,"
__credits__ = ["Uthpala Herath"]
__license__ = "GNU GENERAL PUBLIC LICENSE"
__version__ = "2.0"
__maintainer__ = "Logan Lang, Pedram Tavadze"
__email__ = "petavazohi@mix.wvu.edu, lllang@mix.wvu.edu"
__status__ = "Production"

import numpy as np
import pyvista
from typing import List

# from ..fermisurface3d import BrillouinZone
from . import Structure
from ..utils import Unfolder, mathematics


class ElectronicBandStructure:
    """Core object Representing the electronic band structure of a crystal 
    system.

    Parameters
    ----------
    bands : np.ndarray
        Eigen-values corresponding to each sampled point from 
        the Brillouin zone.
    kpoints : np.ndarray
        Sampled points from the Brillouin zone in reduced representation.
    efermi : float, optional
        Value of the Fermi energy of the system, by default 0.0
    projected : np.ndarray, optional
        Projection of the wave function onto atomic orbitals modulus 
        squared.
        The order of this array is as follows: 
        ikpoint, iband, iatom, iprincipal, iorbital, ispin
        ``iprincipal`` works like the principal quantum number n. The last
        index should be the total. (iprincipal = -1)
        n = iprincipal => 0, 1, 2, 3, -1 => s, p, d, total
        ``iorbital`` works similar to angular quantum number l, but not the
        same. iorbital follows this order
        (0,1,2,3,4,5,6,7,8) => s,py,pz,px,dxy,dyz,dz2,dxz,dx2-y2.
        ``ispin`` works as magnetic quantum number.
        m = 0,1, for spin up and down
        :math: `|\langle Y_{lm}^{\alpha }|\phi _{n\mathbf {k} }\rangle |^{2}` 
        :math: `Y_{lm}^{\alpha}`, by default None
    projected_phase : np.ndarray, optional
        Projection of the wave function onto atomic orbitals
        The order of this array is similar to the ``projected`` argument
        The elements must be complex numbers, by default None
    weights : np.ndarray, optional
        Weights assigned to each band at a certain point in the Brillouin 
        zone
        (*e.g.,* unfolding weights), by default None
    reciprocal_lattice : np.ndarray, optional
        The reciprocal vectors described with a 3x3 array, by default None
    """

    def __init__(
        self,
        bands: np.ndarray,
        kpoints: np.ndarray,
        efermi: float = 0.0,
        projected: np.ndarray = None,
        projected_phase: np.ndarray = None,
        weights: np.ndarray = None,
        reciprocal_lattice: np.ndarray = None,
    ):
        
        self.kpoints = kpoints
        self.bands = bands - efermi
        self.efermi = efermi
        self.projected = projected
        self.projected_phase = projected_phase
        self.reciprocal_lattice = reciprocal_lattice
        self.weights = weights
        
        
        assert self.bands.shape[0] == self.n_kpoints, (
            "Number of kpoints does not match the number of kpoints in bands"
            )
        assert self.kpoints.shape[1] == 3, (
            "Kpoints must have the shape (n_kpoints, 3)"
            )
        if self.has_projection:
            assert self.projected.shape[0] == self.n_kpoints, (
                'Projection array not consistent with number of kpoints'
                )
            assert self.projected.shape[1] == self.n_bands, (
                'Projection array not consistent with number of bands'
                )
        if self.has_phase:
            assert self.projected_phase.shape[0] == self.n_kpoints, (
                'Projection phase array not consistent with number of kpoints'
                )
            assert self.projected_phase.shape[1] == self.n_bands, (
                'Projection phase array not consistent with number of bands'
            )
            assert self.projected_phase.dtype == np.dtype(np.complex_)
        
        if weights is not None:
            assert weights.shape == bands.shape, (
                "The dimension of the provided weight is not consistent with" 
                "bands"
            )
        if reciprocal_lattice is not None:
            assert reciprocal_lattice.shape == (3, 3)
        
        # self.graph = None
        
    @property
    def has_phase(self) -> bool:
        """If the object contains information about projected wave function 
        onto atomic orbitals in a complex format.
        """        
        return self.projected_phase is not None

    @ property
    def has_projection(self) -> bool:
        """If the object contains information about the projected wave function 
        onto atomic orbitals modulus squared.
        """        
        return self.projected is not None
            
    @property
    def n_kpoints(self) -> int:
        """Total number of sampled points from the Brillouin zone.
        """        
        return self.kpoints.shape[0]

    @property
    def n_bands(self) -> int:
        """Total number of eigen-values at each kpoint.
        """
        return self.bands.shape[1]

    @property
    def n_atoms(self) -> int:
        """Total number of atoms in the crystal structure.
        """
        if self.has_projection:
            return self.projected.shape[2]

    @property
    def n_principals(self) -> int:
        """Total number of quantum principal numbers.
        """
        if self.has_projection:
            return self.projected.shape[3]

    @property
    def n_orbitals(self) -> int:
        """Total number of atomic orbitals projected to.
        """
        if self.has_projection:
            return self.projected.shape[4]

    @property
    def n_spins(self) -> int:
        """Total number of spins in the band structure
        """
        if self.has_projection:
            return self.projected.shape[5]

    @property
    def is_non_collinear(self) -> bool:
        """If the band structure is non collinear.
        """
        if self.has_projection:
            if self.n_spins == 3:
                return True
            else:
                return False
        
    @property
    def kpoints_cartesian(self) -> np.ndarray:
        """list of sampled points in the Brillouin Zone in reduced coordinated
        """
        if self.reciprocal_lattice is not None:
            return np.dot(self.kpoints, self.reciprocal_lattice)
        else:
            print(
                "Please provide a reciprocal lattice when initiating the EBS" 
                "class"
            )
            return

    @property
    def kpoints_reduced(self) -> np.ndarray:
        """list of sampled points in the Brillouin zone in reduced coordinated.
        """
        return self.kpoints

    def ebs_sum(self, 
                atoms: List[int] = None, 
                principal_q_numbers: List[int]=[-1], 
                orbitals: List[int] = None, 
                spins: List[int] = None, 
                sum_non_collinear: bool=False) -> np.ndarray:
        """Sums the projection of specified atoms, principal quantum number, 
        and orbitals. 

        Parameters
        ----------
        atoms : List[int], optional
            List of integers representing atoms to be summed over.
            The order follows the order of atoms in the crystal structure.
            Count starts from 0, by default None
        principal_q_numbers : List[int], optional
            List of integers representing principal quantum numbers to be 
            summed over. Count starts from 0, by default [-1]
        orbitals : List[int], optional
            List of integers representing atomic orbitals to be summed over.
            Count starts from 0, by default None
        spins : List[int], optional
            List of integers representing spins to be summed over
            Only activated in non-collinear and 
            ``sum_non_colinear=True``, by default None
        sum_non_colinear : bool, optional
            _description_, by default False

        Returns
        -------
        np.ndarray
            Array representing the contribution of the defined elements in the
            argument in a specific band in a specific point in the Brillouin 
            zone
        """
        # principal_q_numbers = np.array(principal_q_numbers)
        atoms = atoms or np.arange(self.n_atoms, dtype=int)
        spins = spins or np.arange(self.n_spins, dtype=int)
        orbitals = orbitals or np.arange(self.n_orbitals, dtype=int)
        # sum over orbitals
        ret = np.sum(self.projected[:, :, :, :, orbitals, :], axis=-2)
        # sum over principle quantum number
        ret = np.sum(ret[:, :, :, principal_q_numbers, :], axis=-2)
        # sum over atoms
        ret = np.sum(ret[:, :, atoms, :], axis=-2)
        # sum over spins only in non collinear and reshaping for consistency 
        # (n_kpoints, n_bands, n_spins)
        # in non-mag, non-colin nspin=1, in colin nspin=2
        if self.is_non_collinear and sum_non_collinear:
            ret = np.sum(ret[:, :, spins], axis=-1).reshape(
                self.n_kpoints, self.n_bands, 1
            )
        return ret

    # def update_weights(self, weights: np.ndarray): 
    #     self.weights = weights
    #     return

    def unfold(self, 
               transformation_matrix: np.ndarray, 
               structure : Structure):
        """Given a 3x3 transformation matrix and the corresponding crystal 
        structure, it calculated the unfolding weights and updates the 
        variable self.weights

        Parameters
        ----------
        transformation_matrix : np.ndarray
            3x3 matrix that transforms the original cell to another cell, 
            *e.g.,* from unit cell to a supercell.
            
        structure : Structure
            The object describing the crystal structure
        """
        uf = Unfolder(
            ebs=self, 
            transformation_matrix=transformation_matrix, 
            structure=structure,
        )
        self.weights = uf.weights
        # self.update_weights(uf.weights)
        return

    # def plot_kpoints(
    #     self,
    #     reduced: bool=False,
    #     show_brillouin_zone: bool=True,
    #     color: str="r",
    #     point_size: float=4.0,
    #     render_points_as_spheres: bool=True,
    #     transformation_matrix: np.ndarray=None,
    # ):
    #     """Plots the kpoints using PyVista

    #     Parameters
    #     ----------
    #     reduced : bool, optional
    #         Use reduced coordinates, by default False
    #     show_brillouin_zone : bool, optional
    #         To draw the Brillouin zon, by default True
    #     color : str, optional
    #         Color of the kpoints, by default "r"
    #     point_size : float, optional
    #         Size of each drawn point, by default 4.0
    #     render_points_as_spheres : bool, optional
    #         Show points as small spheres, by default True
    #     transformation_matrix : bool, optional
    #         Transform the kpoints using a 3x3 transformation matrix,
    #         by default None
    #     """        

    #     p = pyvista.Plotter()
    #     if show_brillouin_zone:
    #         if reduced:
    #             brillouin_zone = BrillouinZone(
    #                 np.diag([1, 1, 1]), transformation_matrix,
    #             )
    #             brillouin_zone_non = BrillouinZone(np.diag([1, 1, 1]),)
    #         else:
    #             brillouin_zone = BrillouinZone(
    #                 self.reciprocal_lattice, transformation_matrix
    #             )
    #             brillouin_zone_non = BrillouinZone(self.reciprocal_lattice)

    #         p.add_mesh(
    #             brillouin_zone.pyvista_obj,
    #             style="wireframe",
    #             line_width=3.5,
    #             color="black",
    #         )
    #         p.add_mesh(
    #             brillouin_zone_non.pyvista_obj,
    #             style="wireframe",
    #             line_width=3.5,
    #             color="white",
    #         )
    #     if reduced:
    #         kpoints = self.kpoints_reduced
    #     else:
    #         kpoints = self.kpoints_cartesian
    #     p.add_mesh(
    #         kpoints,
    #         color=color,
    #         point_size=point_size,
    #         render_points_as_spheres=render_points_as_spheres,
    #     )
    #     if transformation_matrix is not None:
    #         p.add_mesh(
    #             np.dot(kpoints, transformation_matrix),
    #             color="blue",
    #             point_size=point_size,
    #             render_points_as_spheres=render_points_as_spheres,
    #         )
    #     p.add_axes(
    #         xlabel="Kx", 
    #         ylabel="Ky", 
    #         zlabel="Kz", 
    #         line_width=6, 
    #         labels_off=False
    #     )
    #     p.show()

    def ibz2fbz(self, rotations):
        """Generates the full Brillouin zone from the irreducible Brillouin
        zone using point symmetries.

        Parameters:
            - self.kpoints: the kpoints used to sample the Brillouin zone
            - self.projected: the projected band structure at each kpoint
            - rotations: the point symmetry operations of the lattice
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
        ret += 'Total number of kpoints  = {}\n'.format(self.n_kpoints)
        ret += 'Total number of bands    = {}\n'.format(self.n_bands)
        ret += 'Total number of atoms    = {}\n'.format(self.n_atoms)
        ret += 'Total number of orbitals = {}\n'.format(self.n_orbitals)
        return ret
