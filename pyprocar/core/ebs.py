# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Pedram Tavadze
"""

from scipy.interpolate import CubicSpline
from matplotlib import pylab as plt
from . import Structure
from ..utils import Unfolder
import numpy as np


class ElectronicBandStructure:
    def __init__(
        self,
        kpoints=None,
        eigenvalues=None,
        efermi=None,
        projected=None,
        projected_phase=None,
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
        energies : TYPE, optional
            DESCRIPTION. The default is None.
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
        if not shifted_to_efermi:
            self.eigenvalues = eigenvalues - efermi
            self.shifted_to_efermi = True
        else:
            self.shifted_to_efermi = True
        self.efermi = efermi
        self.projected = projected
        self.projected_phase = projected_phase
        self.reciprocal_lattice = reciprocal_lattice
        if self.projected_phase is not None:
            self.has_phase = True
        else:
            self.has_phase = False
        self.labels = labels
        self.weights = weights

    @property
    def nkpoints(self):
        return self.projected.shape[0]

    @property
    def nbands(self):
        return self.projected.shape[1]

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

    def extend_BZ(self, transformation_matrix=np.diag([1, 1, 1]), time_reversal=True):
        trans_mat = transformation_matrix
        kmaxs = np.dot(self.kpoints, trans_mat).max(axis=0)
        kmins = np.dot(self.kpoints, trans_mat).min(axis=0)
        KX = np.unique(self.kpoints[:, 0])
        KY = np.unique(self.kpoints[:, 1])
        KZ = np.unique(self.kpoints[:, 2])
        kdx = np.abs(self.KX[-1] - self.KX[-2])
        kdy = np.abs(self.KY[-1] - self.KY[-2])
        kdz = np.abs(self.KZ[-1] - self.KZ[-2])

        return

    def plot(self, elimit=[-5, 5]):

        self.weights /= self.weights.max()
        plt.figure(figsize=(16, 9))
        for iband in range(self.nbands):
            plt.scatter(
                np.arange(self.nkpoints),
                self.eigenvalues[:, iband],
                c=self.weights[:, iband].round(2),
                cmap="Blues",
                s=self.weights[:, iband] * 75,
            )
            plt.plot(
                np.arange(self.nkpoints),
                self.eigenvalues[:, iband],
                color="gray",
                alpha=0.1,
            )

        plt.xlim(0, self.nkpoints)
        plt.axhline(y=0, color="red", linestyle="--")
        plt.ylim(elimit)
        plt.tight_layout()
        plt.show()

    def update_weights(self, weights):
        self.weights = weights
        return

    def unfold(self, transformation_matrix=None, structure=None, ispin=0):
        uf = Unfolder(
            ebs=self,
            transformation_matrix=transformation_matrix,
            structure=structure,
            ispin=0,
        )
        self.update_weights(uf.weights)
        return
