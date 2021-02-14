# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Pedram Tavadze
"""

from scipy.interpolate import CubicSpline
from ..fermisurface3d import BrillouinZone
from . import Structure
from ..utils import Unfolder
import numpy as np
from matplotlib import pylab as plt
import pyvista


class ElectronicBandStructure:
    def __init__(
        self,
        kpoints=None,
        eigenvalues=None,
        efermi=None,
        projected=None,
        projected_phase=None,
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
        if not shifted_to_efermi and efermi is not None:
            self.eigenvalues = eigenvalues - efermi
            self.shifted_to_efermi = True
        else:
            self.shifted_to_efermi = True
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

    @property
    def nkpoints(self):
        return self.kpoints.shape[0]

    @property
    def nbands(self):
        return self.eigenvalues.shape[1]

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

    def extend_BZ(
        self,
        transformation_matrix=np.diag([1, 1, 1]),
        extention_scales=None,
        time_reversal=True,
    ):
        if extention_scales is None:
            print(
                "Please extention_scales as an array with its elements corresponding to each kpath segment"
            )
            return
        trans_mat = transformation_matrix
        iend = 0
        extended_eigen_values = []
        for isegment in range(self.kpath.nsegments):
            kstart = self.kpath.special_kpoints[isegment][0]
            kend = self.kpath.special_kpoints[isegment][1]
            distance = np.linalg.norm(kend - kstart)
            istart = iend
            iend = istart + self.kpath.ngrids[isegment]
            kpoints = self.kpoints[istart:iend]
            eigen_values = self.eigenvalues[istart:iend]
            projected = self.projected[istart:iend]
            if self.has_phase:
                projected_phase = self.projected_phase[istart:iend]
            dkx, dky, dkz = kpoints[1] - kpoints[0]
            dk = np.linalg.norm([dkx, dky, dkz])

            if time_reversal:
                eigen_values_flip = np.flip(eigen_values)
                eigen_values_period = np.append(eigen_values_flip, eigen_values, axis=0)
                if len(extended_eigen_values) == 0:
                    extended_eigen_values = eigen_values_period
                else:
                    extended_eigen_values = np.append(
                        extended_eigen_values, eigen_values_period, axis=0
                    )
                projected_flip = np.flip(projected)
                projected_period = np.append(projected_flip, projected, axis=0)

                if self.has_phase:
                    projected_phase_flip = np.flip(projected_phase)
                    projected_phase_period = np.append(
                        projected_phase_flip, projected_phase, axis=0
                    )
        self.eigenvalues = extended_eigen_values
        self.kpoints = np.append(self.kpoints, self.kpoints, axis=0)
        self.kpath.ngrids = [x * 2 for x in self.kpath.ngrids]

        return

    def apply_symmetries(self, operations=None, structure=None):
        return

    def plot(self, elimit=[-5, 5]):

        if self.weights is not None:
            self.weights /= self.weights.max()
        plt.figure(figsize=(16, 9))
        x = []
        pos = 0
        for isegment in range(self.kpath.nsegments):
            kstart, kend = self.kpath.special_kpoints[isegment]
            distance = np.linalg.norm(kend - kstart)
            x.append(np.linspace(pos, pos + distance, self.kpath.ngrids[isegment]))
            dk = x[-1][-1] - x[-1][-2]
            pos += distance
        x = np.array(x).reshape(-1,)

        for iband in range(self.nbands):
            if self.weights is not None:
                plt.scatter(
                    x,
                    self.eigenvalues[:, iband],
                    c=self.weights[:, iband].round(2),
                    cmap="Blues",
                    s=self.weights[:, iband] * 75,
                )
            plt.plot(
                x, self.eigenvalues[:, iband], color="gray", alpha=0.5,
            )
        for ipos in self.kpath.tick_positions:
            plt.axvline(x[ipos], color="black")
        plt.xticks(x[self.kpath.tick_positions], self.kpath.tick_names)
        plt.xlim(0, x[-1])
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

    def plot_kpoints(
        self,
        reduced=False,
        show_brillouin_zone=True,
        color="r",
        point_size=4.0,
        render_points_as_spheres=True,
        transformation_matrix=None,
    ):
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
