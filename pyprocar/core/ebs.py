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

from ..fermisurface3d import BrillouinZone
from ..utils import Unfolder, mathematics

# TODO add python typing to all functions

class ElectronicBandStructure:
    def __init__(
        self,
        kpoints=None,
        bands=None,
        efermi=None,
        projected=None,
        projected_phase=None,
        spd=None,
        kpath=None,
        weights=None,
        labels=None,
        reciprocal_lattice=None,
        interpolation_factor=None,
        shifted_to_efermi=False,
    ):
        """

        Parameters
        ----------
        kpoints : TYPE, optional
            DESCRIPTION. The default is None.
        bands : TYPE, optional
            bands[ikpoint, iband, ispin]. The default is None.
        projected : list float, optional
            dictionary by the following order
            projected[ikpoint][iband][iatom][iprincipal][iorbital][ispin].
            ``iprincipal`` works like the principal quantum number n. The last
            index should be the total. (iprincipal = -1)
            n = iprincipal => 0, 1, 2, 3, -1 => s, p, d, total
            ``iorbital`` works similar to angular quantum number l, but not the
            same. iorbital follows this order
            (0,1,2,3,4,5,6,7,8) => s,py,pz,px,dxy,dyz,dz2,dxz,dx2-y2.
            ``ispin`` works as magnetic quantum number.
            m = 0,1, for spin up and down
            The default is None.


        structure : TYPE, optional
            DESCRIPTION. The default is None.
        interpolation_factor : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

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
        return self.kpoints.shape[0]

    @property
    def nbands(self):
        return self.bands.shape[1]

    @property
    def natoms(self):
        return self.projected.shape[2]

    @property
    def nprincipals(self):
        return self.projected.shape[3]

    @property
    def norbitals(self):
        return self.projected.shape[4]

    @property
    def nspins(self):
        return self.projected.shape[5]

    @property
    def is_non_collinear(self):
        if self.nspins == 3:
            return True
        else:
            return False

    @property
    def kpoints_cartesian(self):
        if self.reciprocal_lattice is not None:
            return np.dot(self.kpoints, self.reciprocal_lattice)
        else:
            print(
                "Please provide a reciprocal lattice when initiating the Procar class"
            )
            return

    @property
    def kpoints_reduced(self):
        return self.kpoints

    def ebs_sum(self, 
                atoms:List[int]=None, 
                principal_q_numbers:List[int]=[-1], 
                orbitals:List[int]=None, 
                spins:List[int]=None, 
                sum_noncolinear:bool=True):

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

    def apply_symmetries(self, operations=None, structure=None):
        return

    def update_weights(self, weights):
        self.weights = weights
        return

    def unfold(self, transformation_matrix=None, structure=None):

        uf = Unfolder(
            ebs=self, transformation_matrix=transformation_matrix, structure=structure,
        )
        self.update_weights(uf.weights)

        return

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

    def apply_symmetries(self, operations=None, structure=None):
        return

    def update_weights(self, weights):
        self.weights = weights
        return

    def unfold(self, transformation_matrix=None, structure=None):
        uf = Unfolder(
            ebs=self, transformation_matrix=transformation_matrix, structure=structure,
        )
        self.update_weights(uf.weights)

        return

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
        ret += 'Total number of kpoints  = {}\n'.format(self.nkpoints)
        ret += 'Total number of bands    = {}\n'.format(self.nbands)
        ret += 'Total number of atoms    = {}\n'.format(self.natoms)
        ret += 'Total number of orbitals = {}\n'.format(self.norbitals)
        return ret
        
