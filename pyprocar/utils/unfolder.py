import numpy as np

from pyprocar.core.structure import Structure
from pyprocar.utils import np_utils


class Unfolder:
    def __init__(
        self,
        ebs=None,
        transformation_matrix=np.diag([1, 1, 1]),
        structure=None,
        tol_radius=0.1,
    ):
        self.ebs = ebs
        self.trans_mat = transformation_matrix
        self.structure = structure
        self.eigenvectors = None
        self.basis = None
        self.positions = None
        self.cell = structure.lattice
        self.qpoints = ebs.kpoints
        self.tol_radius = tol_radius
        self.trans_rs = None
        self.trans_indices = None

        self._prepare_unfold_basis()
        self._make_translate_maps()

    @property
    def nfold(self):
        N = np.linalg.det(self.trans_mat).round(2)
        if not N.is_integer() or (1 / N).is_integer():
            raise ValueError("This transfare with is not proper.")
            return None
        else:
            return int(N)

    def _prepare_unfold_basis(self):
        # basis, which are the name of the bands e.g. 'Ti|dxy|0'
        # self.eigenvectors = np.zeros(
        #    (self.procar.kpointsCount, self.procar.bandsCount,
        #     (self.procar.orbitalCount - 1) * (self.procar.ionsCount - 1) *
        #     self.procar.ispin), dtype='complex')
        # if self.ispin is None:
        #     iispin = 0
        # else:
        #     self.ispin -= 1
        self.basis = []
        self.positions = []
        self.eigenvectors = np.zeros(
            shape=(self.ebs.nkpoints,
                   self.ebs.nbands,
                   self.ebs.natoms * self.ebs.nprincipals * self.ebs.norbitals, self.ebs.nspins),
            dtype=np_utils.COMPLEX_DTYPE)
        for ispin in range(self.ebs.nspins):
            self.eigenvectors[:, :, :, ispin] = np.reshape(
                self.ebs.projected_phase[:, :, :, :, :, ispin],
                (
                    self.ebs.nkpoints,
                    self.ebs.nbands,
                    self.ebs.natoms * self.ebs.nprincipals * self.ebs.norbitals,
                ),
            )
        norm = np.linalg.norm(self.eigenvectors, ord=2, axis=2)
        self.eigenvectors /= norm[:, :, None]

        for iatom, chem in enumerate(self.structure.atoms):
            for iorb, orb in enumerate(self.ebs.labels):
                # for spin in range(self.ebs.nspins):
                for spin in range(1):
                    # todo: what about spin?
                    self.basis.append("%s|%s|%s" % (None, orb, spin))
                    self.positions.append(
                        self.structure.fractional_coordinates[iatom])

    def _make_translate_maps(self):
        """
        find the mapping between supercell and translated cell.
        Returns:
        ===============
        A N * nbasis array.
        index[i] is the mapping from supercell to translated supercell so that
        T(r_i) psi = psi[indices[i]].

        TODO: vacancies/add_atoms not supported. How to do it? For
        vacancies, a ghost atom can be added. For add_atom, maybe we
        can just ignore them? Will it change the energy spectrum?

        """
        a1 = Structure(
            atoms=["H"], fractional_coordinates=[[0, 0, 0]], lattice=np.diag([1, 1, 1])
        )
        sc = a1.transform(self.trans_mat)
        rs = sc.fractional_coordinates

        # a1 = Atoms(symbols="H", positions=[(0, 0, 0)], cell=[1, 1, 1])
        # sc = make_supercell(a1, self.trans_mat)
        # rs = sc.get_scaled_positions()

        positions = self.positions
        indices = np.zeros([len(rs), len(positions)], dtype="int32")
        for i, ri in enumerate(rs):
            Tpositions = positions + np.array(ri)
            def close_to_int(x): return np.all(
                np.abs(x - np.round(x)) < self.tol_radius)
            for i_basis, pos in enumerate(positions):
                for j_basis, Tpos in enumerate(Tpositions):
                    dpos = Tpos - pos

                    if close_to_int(dpos) and (
                        self.basis[i_basis] == self.basis[j_basis]
                    ):
                        indices[i, j_basis] = i_basis
        self.trans_rs = rs
        self.trans_indices = indices

    def _get_weight(self, evec, qpt, G=None):
        """
        get the weight of a mode which has the wave vector of qpt and
        eigenvector of evec.

        W= sum_1^N < evec| T(r_i)exp(-I (K+G) * r_i| evec>, here
        G=0. T(r_i)exp(-I K r_i)| evec> = evec[indices[i]]

                    N
                1  ---
         W_KJ = -  \                   -j(K+G).r_i
                N  /   <KJ|T(r_i)|KJ> e
                   ---
                   i=1
        """

        if G is None:
            G = np.zeros_like(qpt)
        weight = 0j
        N = self.nfold
        _phase = False
        for r_i, ind in zip(self.trans_rs, self.trans_indices):
            if _phase:
                weight += (
                    np.vdot(evec, evec[ind])
                    * np.exp(1j * 2 * np.pi * np.dot(qpt + G, r_i))
                    / N
                )
            else:
                weight += (
                    np.vdot(evec, evec[ind])
                    * np.exp(-1j * 2 * np.pi * np.dot(G, r_i))
                    / N
                )

        return weight.real

    @property
    def weights(self):
        """
        Get the weight for all the modes.
        """
        nqpts, nfreqs = self.eigenvectors.shape[0], self.eigenvectors.shape[1]
        weights = np.zeros([nqpts, nfreqs, self.ebs.nspins])
        for ispin in range(self.ebs.nspins):
            for iqpt in range(nqpts):
                for ifreq in range(nfreqs):
                    weights[iqpt, ifreq, ispin] = self._get_weight(
                        self.eigenvectors[iqpt, ifreq,
                                          :, ispin], self.qpoints[iqpt]
                    )
            return weights
