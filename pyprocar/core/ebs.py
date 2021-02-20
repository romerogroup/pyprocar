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
        if self.efermi is None:
            self.eigenvalues = eigenvalues
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
        self, new_kpath, time_reversal=True,
    ):

        iend = 0

        for isegment in range(self.kpath.nsegments):
            extended_kstart = new_kpath.special_kpoints[isegment][0]
            extended_kend = new_kpath.special_kpoints[isegment][1]

            kstart = self.kpath.special_kpoints[isegment][0]
            kend = self.kpath.special_kpoints[isegment][1]

            istart = iend
            iend = istart + self.kpath.ngrids[isegment]
            kpoints = self.kpoints[istart:iend]
            eigen_values = self.eigenvalues[istart:iend]
            projected = self.projected[istart:iend]
            if self.has_phase:
                projected_phase = self.projected_phase[istart:iend]
            dk_vec = kpoints[1] - kpoints[0]
            dkx, dky, dkz = dk_vec
            dk = np.linalg.norm(dk_vec)

            nkpoints_pre_extention = (
                int((np.linalg.norm(kstart - extended_kstart) / dk).round(4)) + 1
            )
            nkpoints_post_extention = (
                int((np.linalg.norm(kend - extended_kend) / dk).round(4)) + 1
            )

            if nkpoints_pre_extention == 1:
                nkpoints_pre_extention = 0
            if nkpoints_post_extention == 1:
                nkpoints_post_extention = 0

            kpoints_pre = []
            for ipoint in range(nkpoints_pre_extention, 0, -1):
                kpoints_pre.append(kstart - dk_vec * ipoint)

            if len(kpoints_pre) != 0:
                kpoints_pre = np.array(kpoints_pre)
            else:
                kpoints_pre = []
            kpoints_post = []
            for ipoint in range(1, nkpoints_post_extention):
                kpoints_post.append(kend + dk_vec * ipoint)
            if len(kpoints_post) != 0:
                kpoints_post = np.array(kpoints_post)
            else:
                kpoints_post = []

            if time_reversal:
                eigen_values_flip = np.flip(eigen_values, axis=0)
                eigen_values_period = np.append(
                    eigen_values_flip[:-1], eigen_values, axis=0
                )
                projected_flip = np.flip(projected, axis=0)
                projected_period = np.append(projected_flip[:-1], projected, axis=0)

                if self.has_phase:
                    projected_phase_flip = np.flip(projected_phase, axis=0).conjugate()# * (
                    #     np.exp(-0.5j * np.pi * np.linspace(-0.01, 0.01, len(projected_ph


                    projected_phase_period = np.append(
                        projected_phase_flip[:-1], projected_phase, axis=0
                    )
                scale_factor = int(
                    np.ceil(
                        (nkpoints_post_extention + nkpoints_pre_extention)
                        / self.kpath.ngrids[isegment]
                    )
                )
                # if scale_factor == 0:
                #     scale_factor = 1
                eigen_values_period = np.pad(
                    eigen_values_flip,
                    ((0, scale_factor * len(eigen_values_period)), (0, 0)),
                    mode="wrap",
                )
                projected_period = np.pad(
                    projected_period,
                    (
                        (0, scale_factor * len(projected_period)),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                        (0, 0),
                    ),
                    mode="wrap",
                )
                if self.has_phase:
                    projected_phase_period = np.pad(
                        projected_phase_period,
                        (
                            (0, scale_factor * len(projected_phase_period)),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                        ),
                        mode="wrap",
                    )
                if nkpoints_post_extention + nkpoints_pre_extention != 0:
                    if nkpoints_pre_extention != 0:
                        extended_kpoints = np.append(kpoints_pre, kpoints, axis=0)
                        extended_eigen_values = np.append(
                            eigen_values_period[-nkpoints_pre_extention:],
                            eigen_values,
                            axis=0,
                        )
                        extended_projected = np.append(
                            projected_period[-nkpoints_pre_extention:],
                            projected,
                            axis=0,
                        )
                        if self.has_phase:
                            extended_projected_phase = np.append(
                                projected_phase_period[-nkpoints_pre_extention:],
                                projected_phase,
                                axis=0,
                            )

                    if nkpoints_post_extention != 0:
                        if nkpoints_pre_extention != 0:
                            extended_kpoints = np.append(
                                extended_kpoints, kpoints_post, axis=0
                            )
                            extended_eigen_values = np.append(
                                extended_eigen_values,
                                eigen_values_period[1:nkpoints_post_extention],
                                axis=0,
                            )
                            extended_projected = np.append(
                                extended_projected,
                                projected_period[1:nkpoints_post_extention],
                                axis=0,
                            )
                            if self.has_phase:
                                extended_projected_phase = np.append(
                                    extended_projected_phase,
                                    projected_phase_period[1:nkpoints_post_extention],
                                    axis=0,
                                )
                        else:
                            extended_kpoints = np.append(kpoints, kpoints_post, axis=0)
                            extended_eigen_values = np.append(
                                eigen_values,
                                eigen_values_period[1:nkpoints_post_extention],
                                axis=0,
                            )
                            extended_projected = np.append(
                                projected,
                                projected_period[1:nkpoints_post_extention],
                                axis=0,
                            )
                            if self.has_phase:
                                extended_projected_phase = np.append(
                                    projected_phase,
                                    projected_phase_period[1:nkpoints_post_extention],
                                    axis=0,
                                )

                else:
                    extended_kpoints = kpoints
                    extended_eigen_values = eigen_values
                    extended_projected = projected
                    if self.has_phase:
                        extended_projected_phase = projected_phase
                if isegment == 0:
                    overall_kpoints = extended_kpoints
                    overall_eigen_values = extended_eigen_values
                    overall_projected = extended_projected
                    if self.has_phase:
                        overall_projected_phase = extended_projected_phase

                else:
                    overall_kpoints = np.append(
                        overall_kpoints, extended_kpoints, axis=0
                    )
                    overall_eigen_values = np.append(
                        overall_eigen_values, extended_eigen_values, axis=0
                    )
                    overall_projected = np.append(
                        overall_projected, extended_projected, axis=0
                    )
                    if self.has_phase:
                        overall_projected_phase = np.append(
                            overall_projected_phase, extended_projected_phase, axis=0
                        )

                self.kpath.ngrids[isegment] = len(extended_kpoints)
                self.kpath.special_kpoints[isegment] = new_kpath.special_kpoints[
                    isegment
                ]

        self.kpoints = overall_kpoints
        self.eigenvalues = overall_eigen_values
        self.projected = overall_projected
        if self.has_phase:
            self.projected_phase = overall_projected_phase
        return

    def apply_symmetries(self, operations=None, structure=None):
        return

    def plot(self, elimit=[-5, 5]):

        # if self.weights is not None:
        #     self.weights /= self.weights.max()
        plt.figure(figsize=(16, 9))

        pos = 0
        for isegment in range(self.kpath.nsegments):
            kstart, kend = self.kpath.special_kpoints[isegment]
            distance = np.linalg.norm(kend - kstart)
            if isegment == 0:
                x = np.linspace(pos, pos + distance, self.kpath.ngrids[isegment])
            else:
                x = np.append(
                    x,
                    np.linspace(pos, pos + distance, self.kpath.ngrids[isegment]),
                    axis=0,
                )
            pos += distance
        x = np.array(x).reshape(-1,)

        for iband in range(self.nbands):
            if self.weights is not None:
                plt.scatter(
                    x,
                    self.eigenvalues[:, iband],
                    c=self.weights[:, iband].round(2),
                    cmap="jet",
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
        # plt.ylim(elimit)
        plt.colorbar()
        plt.tight_layout()

        ####
        plt.figure(figsize=(16, 9))

        pos = 0
        for isegment in range(self.kpath.nsegments):
            kstart, kend = self.kpath.special_kpoints[isegment]
            distance = np.linalg.norm(kend - kstart)
            if isegment == 0:
                x = np.linspace(pos, pos + distance, self.kpath.ngrids[isegment])
            else:
                x = np.append(
                    x,
                    np.linspace(pos, pos + distance, self.kpath.ngrids[isegment]),
                    axis=0,
                )
            pos += distance
        x = np.array(x).reshape(-1,)

        r = np.absolute(self.projected_phase).sum(axis=(2,3,4,5))
        phi = np.angle(self.projected_phase).sum(axis=(2,3,4,5))
        for iband in range(self.nbands):


            plt.scatter(
                x,
                self.eigenvalues[:, iband],
                c=r[:,iband],
                cmap="seismic",
            )
            plt.plot(
                x, self.eigenvalues[:, iband], color="gray", alpha=0.5,
            )

        for ipos in self.kpath.tick_positions:
            plt.axvline(x[ipos], color="black")
        plt.xticks(x[self.kpath.tick_positions], self.kpath.tick_names)
        plt.xlim(0, x[-1])
        plt.axhline(y=0, color="red", linestyle="--")
        # plt.ylim(elimit)
        plt.colorbar()
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
