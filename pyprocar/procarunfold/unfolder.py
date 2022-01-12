"""
Phonon unfolding: Reciprocal space method. The method is described in
P. B. Allen et al. Phys Rev B 87, 085322 (2013).
This method should be also applicable to other bloch waves on discrete grid, eg. electrons wave function in wannier basis set, magnons, etc. Now only phonon istested.
"""
from ase.build import make_supercell
from ase.atoms import Atoms
import numpy as np


class Unfolder:
    """ phonon unfolding class"""
    def __init__(
            self,
            cell,
            basis,
            positions,
            supercell_matrix,
            eigenvectors,
            qpoints,
            tol_r=0.1,
            compare=None,
            phase=True,
    ):
        """
        Params:
        ===================
        cell: cell matrix. [a,b,c]
        basis: name of the basis. It's used to decide if two basis can
        be identical by translation. eg. for phonons, the basis can be
        ['x','y','z']*natoms, for electrons, it can be
        ['Ni|dxy','Mn|dxy'] if the two dxy are seen as different, or
        ['dxy','dxy'] if they are seen as the same.

        positions: positions(->basis).
        supercell matrix: The matrix that convert the primitive cell
        to supercell.

        eigenvectors: The phonon eigenvectors. format np.array()
        index=[ikpts, ifreq, 3*iatoms+j]. j=0..2

        qpoints: list of q-points.
        tol_r: tolerance. If abs(a-b) <r, they are seen as the same atom.
        """
        self._cell = cell
        self._basis = basis
        self._positions = positions
        self._scmat = supercell_matrix
        self._evecs = eigenvectors
        self._qpts = qpoints
        self._tol_r = tol_r
        self._trans_rs = None
        self._trans_indices = None
        self._make_translate_maps()
        self._phase = phase
        return

    def _translate(self, evec, r):
        """
        T(r) psi: r is integer numbers of primitive cell lattice matrix.
        Params:
        =================
        evec: an eigen vector of supercell
        r: The translate vector
        
        Returns:
        ================
         tevec: translated vector.
        """
        pass

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
        a1 = Atoms(symbols="H", positions=[(0, 0, 0)], cell=[1, 1, 1])
        sc = make_supercell(a1, self._scmat)
        rs = sc.get_scaled_positions()

        positions = self._positions
        indices = np.zeros([len(rs), len(positions)], dtype="int32")
        for i, ri in enumerate(rs):
            inds = []
            Tpositions = positions + np.array(ri)
            close_to_int = lambda x: np.all(
                np.abs(x - np.round(x)) < self._tol_r)
            for i_basis, pos in enumerate(positions):
                for j_basis, Tpos in enumerate(Tpositions):
                    dpos = Tpos - pos
                    if close_to_int(dpos) and (
                            self._basis[i_basis] == self._basis[j_basis]):
                        # indices[i, j_atom * self._ndim:j_atom * self._ndim + self._ndim] = np.arange(i_atom * self._ndim, i_atom * self._ndim + self._ndim)
                        indices[i, j_basis] = i_basis

        self._trans_rs = rs
        self._trans_indices = indices
        # print(indices)

    def get_weight(self, evec, qpt, G=None):
        """
        get the weight of a mode which has the wave vector of qpt and
        eigenvector of evec.

        W= sum_1^N < evec| T(r_i)exp(-I (K+G) * r_i| evec>, here
        G=0. T(r_i)exp(-I K r_i)| evec> = evec[indices[i]]
        """
        if G is None:
            G = np.zeros_like(qpt)
        weight = 0j
        N = len(self._trans_rs)
        for r_i, ind in zip(self._trans_rs, self._trans_indices):
            if self._phase:
                weight += (np.vdot(evec, evec[ind]) *
                           np.exp(1j * 2 * np.pi * np.dot(qpt + G, r_i)) / N)
            else:
                weight += (np.vdot(evec, evec[ind]) / N *
                           np.exp(-1j * 2 * np.pi * np.dot(G, r_i)))
        return weight.real

    def get_weights(self):
        """
        Get the weight for all the modes.
        """
        nqpts, nfreqs = self._evecs.shape[0], self._evecs.shape[1]
        weights = np.zeros([nqpts, nfreqs])
        for iqpt in range(nqpts):
            for ifreq in range(nfreqs):
                weights[iqpt, ifreq] = self.get_weight(
                    self._evecs[iqpt, ifreq, :], self._qpts[iqpt])

        self._weights = weights
        return self._weights
