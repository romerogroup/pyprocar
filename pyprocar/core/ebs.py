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
        projected_phase=None,
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
        self.projected_phase = projected_phase
        self.reciprocal_lattice = reciprocal_lattice
        if self.projected_phase is not None:
            self.has_phase = None




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

