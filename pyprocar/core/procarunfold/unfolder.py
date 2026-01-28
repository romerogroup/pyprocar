"""
Phonon unfolding: Reciprocal space method. The method is described in
P. B. Allen et al. Phys Rev B 87, 085322 (2013).
This method should be also applicable to other bloch waves on discrete grid, eg. electrons wave function in wannier basis set, magnons, etc. Now only phonon istested.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from ase.atoms import Atoms
from ase.build import make_supercell

from pyprocar.utils.typing_helpers import get_shape_dim, vdot_to_complex


class Unfolder:
    """phonon unfolding class"""

    _cell: npt.NDArray[np.float64]
    _basis: list[str]
    _positions: npt.NDArray[np.float64]
    _scmat: npt.NDArray[np.int64]
    _evecs: npt.NDArray[np.complex128]
    _qpts: npt.NDArray[np.float64]
    _tol_r: float
    _trans_rs: npt.NDArray[np.float64] | None
    _trans_indices: npt.NDArray[np.int32] | None
    _phase: bool
    _weights: npt.NDArray[np.float64]

    def __init__(
        self,
        cell: npt.NDArray[np.float64],
        basis: list[str],
        positions: npt.NDArray[np.float64],
        supercell_matrix: npt.NDArray[np.int64],
        eigenvectors: npt.NDArray[np.complex128],
        qpoints: npt.NDArray[np.float64],
        tol_r: float = 0.1,
        compare: object = None,
        phase: bool = True,
    ) -> None:
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
        del compare  # unused parameter
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
        # Initialize weights - dimensions derived from eigenvectors
        # Note: shape indexing returns Any in strict mode; using len() for first dim
        ev_shape = eigenvectors.shape
        self._weights = np.zeros(ev_shape[:2], dtype=np.float64)
        return

    def _translate(
        self, evec: npt.NDArray[np.complex128], r: npt.NDArray[np.float64]
    ) -> None:
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
        del evec, r  # unused parameters - placeholder method
        pass

    def _make_translate_maps(self) -> None:
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
        rs: npt.NDArray[np.float64] = sc.get_scaled_positions()

        positions = self._positions
        n_rs = get_shape_dim(rs, 0)
        n_pos = get_shape_dim(positions, 0)
        indices: npt.NDArray[np.int32] = np.zeros([n_rs, n_pos], dtype="int32")

        def close_to_int(x: npt.NDArray[np.float64]) -> np.bool_:
            return np.all(np.abs(x - np.round(x)) < self._tol_r)

        for i in range(n_rs):
            ri: npt.NDArray[np.float64] = rs[i, :]
            Tpositions: npt.NDArray[np.float64] = positions + ri

            for i_basis in range(n_pos):
                pos: npt.NDArray[np.float64] = positions[i_basis, :]
                for j_basis in range(n_pos):
                    Tpos: npt.NDArray[np.float64] = Tpositions[j_basis, :]
                    dpos: npt.NDArray[np.float64] = Tpos - pos
                    if close_to_int(dpos) and (self._basis[i_basis] == self._basis[j_basis]):
                        indices[i, j_basis] = i_basis

        self._trans_rs = rs
        self._trans_indices = indices
        # print(indices)

    def get_weight(
        self,
        evec: npt.NDArray[np.complex128],
        qpt: npt.NDArray[np.float64],
        G: npt.NDArray[np.float64] | None = None,  # noqa: N803 - uppercase for physics convention
    ) -> float:
        """
        get the weight of a mode which has the wave vector of qpt and
        eigenvector of evec.

        W= sum_1^N < evec| T(r_i)exp(-I (K+G) * r_i| evec>, here
        G=0. T(r_i)exp(-I K r_i)| evec> = evec[indices[i]]
        """
        g_vector: npt.NDArray[np.float64]
        if G is None:
            g_vector = np.zeros_like(qpt)
        else:
            g_vector = G
        weight: complex = 0j
        if self._trans_rs is None or self._trans_indices is None:
            return 0.0
        n_trans = get_shape_dim(self._trans_rs, 0)
        for i in range(n_trans):
            r_i: npt.NDArray[np.float64] = self._trans_rs[i, :]
            ind: npt.NDArray[np.int32] = self._trans_indices[i, :]
            vdot_result = vdot_to_complex(evec, evec[ind])
            # Cast: np.dot and np.exp return floating[Any]/complexfloating[Any]
            if self._phase:
                dot_val = cast(float, np.dot(qpt + g_vector, r_i))
                exp_val = cast(complex, np.exp(1j * 2 * np.pi * dot_val))
                weight += vdot_result * exp_val / n_trans
            else:
                dot_val = cast(float, np.dot(g_vector, r_i))
                exp_val = cast(complex, np.exp(-1j * 2 * np.pi * dot_val))
                weight += vdot_result / n_trans * exp_val
        return float(weight.real)

    def get_weights(self) -> npt.NDArray[np.float64]:
        """
        Get the weight for all the modes.
        """
        nqpts = get_shape_dim(self._evecs, 0)
        nfreqs = get_shape_dim(self._evecs, 1)
        weights: npt.NDArray[np.float64] = np.zeros([nqpts, nfreqs])
        for iqpt in range(nqpts):
            qpt: npt.NDArray[np.float64] = self._qpts[iqpt, :]
            for ifreq in range(nfreqs):
                evec: npt.NDArray[np.complex128] = self._evecs[iqpt, ifreq, :]
                weights[iqpt, ifreq] = self.get_weight(evec, qpt)

        self._weights = weights
        return self._weights
