# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 2021

@author: Pedram Tavadze
@author: Freddy Farah
"""

from . import Structure
from ..fermisurface3d import BrillouinZone
from ..utils import Unfolder, mathematics
from scipy.interpolate import CubicSpline
import itertools
import numpy as np
import networkx as nx
from matplotlib import pylab as plt
import pyvista


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
            bands[ikpoin, iband, ispin]. The default is None.
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
            self.bands = bands - efermi
            self.shifted_to_efermi = True
        else:
            self.shifted_to_efermi = True
        self.efermi = efermi
        if self.efermi is None:
            self.bands = bands
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

    def reorder(self, plot=True):
        e = 1e-3
        nspins = self.nspins
        if self.is_non_collinear:
            nspins = 1
            # projected = np.sum(self.projected, axis=-1).reshape(self.nkpoints,
            #                                                     self.nbands,
            #                                                     self.natoms,
            #                                                     self.nprincipals,
            #                                                     self.norbitals,
            #                                                     nspins)
            projected = self.projected

        else:
            projected = self.projected
        new_bands = np.zeros_like(self.bands)
        new_projected = np.zeros_like(self.projected)
        for ispin in range(nspins):

            DG = nx.Graph()
            X = np.arange(self.nkpoints)
            DG.add_nodes_from(
                [
                    ((i, j), {"pos": (X[i], self.bands[i, j, ispin])})
                    for i, j in itertools.product(
                        range(self.nkpoints), range(self.nbands)
                    )
                ]
            )

            pos = nx.get_node_attributes(DG, "pos")
            new_bands[0,:,ispin] = self.bands[0,:,ispin]
            new_projected[0, :, :, :, :, :] = self.projected[0, :, :, :, :, :]
            for ikpoint in range(self.nkpoints - 1):
                order = []
                
                for iband in range(self.nbands):
                    prj = np.repeat(projected[ikpoint, iband, :, :, :, ispin].reshape(1,
                                                                                      self.natoms, self.nprincipals, self.norbitals), self.nbands, axis=0)
                    dK = np.linalg.norm(self.kpoints[ikpoint+1] -self.kpoints[ikpoint])
                    if dK == 0:
                        continue
                    prj = np.repeat(
                        np.sqrt(projected[ikpoint, iband, :, :, :, ispin].astype(np.complex_)).reshape(
                            1, -1
                        ),  
                        self.nbands,
                        axis=0,
                    )
                    prj /= np.linalg.norm(prj[0])
                    prj_1 = np.sqrt(
                        projected[ikpoint + 1, :, :, :, :, ispin].astype(np.complex_).reshape(
                            self.nbands, -1
                        )
                    )
                    prj_1 = prj_1.T / np.linalg.norm(prj_1, axis=1)
                    prj_1 = prj_1.T
                    # return prj, prj_1
                    # prod = prj*prj_1
                    # prod = prod.sum(axis=-1)
                    # prod = prod.sum(axis=-1)
                    # prod = prod.sum(axis=-1)
                    # return prj, prj_1
                    # # prod = np.exp(-1*prod* 1/(self.bands[ikpoint+1, :, ispin]-self.bands[ikpoint, iband, ispin]))
                    # prod = np.exp(-1/abs(prod))
                    # prod = np.exp(-1*abs(self.bands[ikpoint+1, :, ispin]-self.bands[ikpoint, iband, ispin]+e))
                    dots = np.array([np.vdot(x, y) for x, y in zip(prj, prj_1)])
                    
                    diff = np.linalg.norm(prj - prj_1, axis=1)
                    dE = np.abs(
                        self.bands[ikpoint + 1, :, ispin]
                        - self.bands[ikpoint, iband, ispin]
                    ) 
                    dEdK = (dE / dK)
                    jband = np.argsort(diff)
                    counter = 0
                    # while jband[counter] in order:
                    #     counter+=1
                    order.append(jband[counter])  
                    
                    # if iband !=0 and jband== 0:
                    #     print(ikpoint, iband)
                    
                    # print(iband, self.bands[ikpoint, iband, ispin], jband, self.bands[ikpoint, jband, ispin])
                    
                    # diffs.append(diff)
                    # DG.add_weighted_edges_from([((ikpoint, iband),(ikpoint+1, x[0]),x[1]) for x in zip(range(self.nbands), prod)])
                    DG.add_edge((ikpoint, iband), (ikpoint + 1, jband[counter]))
                    
            if plot:
                plt.figure(figsize=(16, 9))
                nodes = nx.draw_networkx_nodes(
                    DG, pos, node_size=5, node_color=["blue", "red"][ispin])
                edges = nx.draw_networkx_edges(
                    DG,
                    pos,
                    edge_color='red'
                )
                plt.show()

        #         if len(order) == 0:
        #             new_bands[ikpoint+1,:,ispin] = self.bands[ikpoint+1,:,ispin]
        #             new_projected[ikpoint+1, :, :, :, :, :] = self.projected[ikpoint+1, :, :, :, :, :]
        #         else :
        #             new_bands[ikpoint+1,:,ispin] = self.bands[ikpoint+1, order,ispin]
        #             new_projected[ikpoint+1, :, :, :, :, :] = self.projected[ikpoint+1, order, :, :, :, :]
                
        # self.bands = new_bands
        # self.projected = new_projected
        return 

    def ebs_sum(self, atoms=None, principal_q_numbers=[-1], orbitals=None, spins=None):

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
        if self.is_non_collinear:
            ret = np.sum(ret[:, :, spins], axis=-1).reshape(
                self.nkpoints, self.nbands, 1
            )
        return ret

    def interpolate(self, interpolation_factor=2):
        if self.kpath is not None:
            iend = 0
            interpolated_kpoints = []
            interpolated_bands = []
            interpolated_projected = []
            interpolated_projected_phase = []
            for isegment in range(self.kpath.nsegments):
                kstart = self.kpath.special_kpoints[isegment][0]
                kend = self.kpath.special_kpoints[isegment][1]

                istart = iend
                iend = istart + self.kpath.ngrids[isegment]
                bands = self.bands[istart:iend]
                projected = self.projected[istart:iend]

                interpolated_bands.append(
                    mathematics.fft_interpolate(
                        bands, interpolation_factor=interpolation_factor, axis=0,
                    )
                )

                interpolated_projected.append(
                    mathematics.fft_interpolate(
                        projected, interpolation_factor=interpolation_factor, axis=0,
                    )
                )
                if self.has_phase:
                    projected_phase = self.projected_phase[istart:iend]
                    interpolated_projected_phase.append(
                        mathematics.fft_interpolate(
                            projected_phase,
                            interpolation_factor=interpolation_factor,
                            axis=0,
                        )
                    )
                self.kpath.ngrids[isegment] = interpolated_bands[-1].shape[0]
                interpolated_kpoints.append(
                    np.linspace(kstart, kend, self.kpath.ngrids[isegment])
                )
        self.kpoints = np.array(interpolated_kpoints).reshape(-1, 3)
        self.bands = np.array(interpolated_bands).reshape(-1, self.nbands)
        self.projected = np.array(interpolated_projected).reshape(
            -1, self.nbands, self.natoms, self.nprincipals, self.norbitals, self.nspins
        )
        if self.has_phase:
            self.projected_phase = np.array(interpolated_projected_phase).reshape(
                -1,
                self.nbands,
                self.natoms,
                self.nprincipals,
                self.norbitals,
                self.nspins,
            )
        return interpolated_bands

    def _extend_BZ(
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
            bands = self.bands[istart:iend]
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
                bands_flip = np.flip(bands, axis=0)
                bands_period = np.append(bands_flip[:-1], bands, axis=0)
                projected_flip = np.flip(projected, axis=0)
                projected_period = np.append(projected_flip[:-1], projected, axis=0)

                if self.has_phase:
                    projected_phase_flip = np.flip(
                        projected_phase, axis=0
                    ).conjugate()  # * (
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
                bands_period = np.pad(
                    bands_flip,
                    ((0, scale_factor * len(bands_period)), (0, 0)),
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
                        extended_bands = np.append(
                            bands_period[-nkpoints_pre_extention:], bands, axis=0,
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
                            extended_bands = np.append(
                                extended_bands,
                                bands_period[1:nkpoints_post_extention],
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
                            extended_bands = np.append(
                                bands, bands_period[1:nkpoints_post_extention], axis=0,
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
                    extended_bands = bands
                    extended_projected = projected
                    if self.has_phase:
                        extended_projected_phase = projected_phase
                if isegment == 0:
                    overall_kpoints = extended_kpoints
                    overall_bands = extended_bands
                    overall_projected = extended_projected
                    if self.has_phase:
                        overall_projected_phase = extended_projected_phase

                else:
                    overall_kpoints = np.append(
                        overall_kpoints, extended_kpoints, axis=0
                    )
                    overall_bands = np.append(overall_bands, extended_bands, axis=0)
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
        self.bands = overall_bands
        self.projected = overall_projected
        if self.has_phase:
            self.projected_phase = overall_projected_phase
        return

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
        # for each symmetry operation

        for i, _ in enumerate(rotations):
            # for each point
            for j, _ in enumerate(self.kpoints):
                # apply symmetry operation to kpoint
                sympoint_vector = np.dot(rotations[i], self.kpoints[j])
                # apply boundary conditions
                # bound_ops = -1.0*(sympoint_vector > 0.5) + 1.0*(sympoint_vector < -0.5)
                # sympoint_vector += bound_ops

                sympoint = sympoint_vector.tolist()

                if sympoint not in klist:
                    klist.append(sympoint)

                    projection = self.projected[j].tolist()
                    plist.append(projection)

        self.kpoints = np.array(klist)
        self.projected = np.array(plist)
