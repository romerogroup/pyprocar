# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Pedram Tavadze
"""

from scipy.interpolate import CubicSpline
import numpy as np


class ElectronicBandStructure:
    def __init__(
        self,
        kpoints=None,
        eigen_values=None,
        projected=None,
        structure=None,
        spd=None,
        labels=None,
        reciprocal_lattice=None,
        interpolation_factor=None,
    ):
        """


        Parameters
        ----------
        kpoints : TYPE, optional
            DESCRIPTION. The default is None.
        energies : TYPE, optional
            DESCRIPTION. The default is None.
        projected : list float, optional
            dictionary by the following order
            projected[iatom][ikpoint][iband][iprincipal][iorbital][ispin].
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
        self.eigen_values = eigen_values
        self.projected = projected
        self.structure = structure
        self.spd = spd
        self.reciprocal_lattice = reciprocal_lattice
        self.has_phase = False




    @property
    def nkpoints(self):
        return len(self.kpoints)

    @property
    def nbands(self):
        return len(self.eigen_values[0])

    @property
    def natoms(self):
        return len(self.projected)

    @property
    def nprincipals(self):
        return len(self.projected[0][0][0])

    @property
    def norbitals(self):
        return len(self.projected[0][0][0][0])

    @property
    def nspins(self):
        return len(self.projected[0][0][0][0][0])

    def _spd2projected(self, nprinciples=1):
        # This function is for VASP
        # non-pol and colinear
        # spd is formed as (nkpoints,nbands, nspin, natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # non-colinear
        # spd is formed as (nkpoints,nbands, nspin +1 , natom+1, norbital+2)
        # natom+1 > last column is total
        # norbital+2 > 1st column is the number of atom last is total
        # nspin +1 > last column is total
        natoms = self.spd.shape[3] - 1
        nkpoints = self.spd.shape[0]
        
        nbands = self.spd.shape[1]
        norbitals = self.spd.shape[4] - 2
        if self.spd.shape[2] == 4:
            nspins = 3
        else:
            nspins = self.spd.shape[2]
        if nspins == 2 :
            nbands = int(self.spd.shape[1]/2)
        else : 
            nbands = self.spd.shape[1]
        self.projected = np.zeros(
            shape=(natoms, nkpoints, nbands, nprinciples, norbitals, nspins)
        )
        temp_spd = self.spd.copy()
        # (nkpoints,nbands, nspin, natom, norbital)
        temp_spd = np.swapaxes(temp_spd, 2, 4)
        # (nkpoints,nbands, norbital , natom , nspin)
        temp_spd = np.swapaxes(temp_spd, 2, 3)
        # (nkpoints,nbands, natom, norbital, nspin)
        temp_spd = np.swapaxes(temp_spd, 2, 1)
        # (nkpoints, natom, nbands, norbital, nspin)
        temp_spd = np.swapaxes(temp_spd, 1, 0)
        # (natom, nkpoints, nbands, norbital, nspin)
        # projected[iatom][ikpoint][iband][iprincipal][iorbital][ispin]
        if nspins == 3:
            self.projected[:, :, :, 0, :, :] = temp_spd[:-1, :, :, 1:-1, :-1]
        elif nspins == 2:
            self.projected[:, :, :, 0, :, 0] = temp_spd[:-1, :, :nbands, 1:-1, 0]
            self.projected[:, :, :, 0, :, 1] = temp_spd[:-1, :, nbands:, 1:-1, 0]
        else:
            self.projected[:, :, :, 0, :, :] = temp_spd[:-1, :, :, 1:-1, :]
