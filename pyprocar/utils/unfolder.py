from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyprocar.core.structure import Structure

if TYPE_CHECKING:
    from pyprocar.core.ebs import ElectronicBandStructure


class Unfolder:
    ebs: ElectronicBandStructure
    trans_mat: npt.NDArray[np.int_] | npt.NDArray[np.float64]
    structure: Structure
    eigenvectors: npt.NDArray[np.complex128] | None
    basis: list[str]
    positions: list[npt.NDArray[np.float64]]
    cell: npt.NDArray[np.float64]
    qpoints: npt.NDArray[np.float64]
    tol_radius: float
    trans_rs: npt.NDArray[np.float64] | None
    trans_indices: npt.NDArray[np.int32] | None

    def __init__(
        self,
        ebs: ElectronicBandStructure,
        transformation_matrix: npt.NDArray[np.int_] | npt.NDArray[np.float64] | None = None,
        structure: Structure | None = None,
        tol_radius: float = 0.1,
    ) -> None:
        self.ebs = ebs
        self.trans_mat = transformation_matrix if transformation_matrix is not None else np.diag([1, 1, 1])
        assert structure is not None
        self.structure = structure
        self.eigenvectors = None
        self.basis = []
        self.positions = []
        lattice = structure.lattice
        assert lattice is not None
        self.cell = lattice
        self.qpoints = ebs.kpoints
        self.tol_radius = tol_radius
        self.trans_rs = None
        self.trans_indices = None

        self._prepare_unfold_basis()
        self._make_translate_maps()

    @property
    def nfold(self) -> int:
        n_val = np.linalg.det(self.trans_mat).round(2)
        if not n_val.is_integer() or (1 / n_val).is_integer():
            raise ValueError("This transfare with is not proper.")
        return int(n_val)

    def _prepare_unfold_basis(self) -> None:
        # basis, which are the name of the bands e.g. 'Ti|dxy|0'
        self.basis = []
        self.positions = []
        self.eigenvectors = np.zeros(
            shape=(
                self.ebs.n_kpoints,
                self.ebs.n_bands,
                self.ebs.n_spins,
                self.ebs.n_atoms * self.ebs.n_orbitals,
            ),
            dtype=np.complex128,
        )
        projected_phase = self.ebs.projected_phase
        assert projected_phase is not None
        projected_phase_value = projected_phase.value
        for ispin in range(self.ebs.n_spins):
            self.eigenvectors[:, :, ispin, :] = np.reshape(
                projected_phase_value[:, :, ispin, :, :],
                (
                    self.ebs.n_kpoints,
                    self.ebs.n_bands,
                    self.ebs.n_atoms * self.ebs.n_orbitals,
                ),
            )

        # norm the atomic-orbital axis
        norm = np.linalg.norm(self.eigenvectors, ord=2, axis=-1)
        self.eigenvectors /= norm[:, :, None]

        orbital_names = self.ebs.orbital_names
        assert orbital_names is not None
        atoms = self.structure.atoms
        assert atoms is not None
        fractional_coordinates = self.structure.fractional_coordinates
        assert fractional_coordinates is not None
        for iatom, _chem in enumerate(atoms):
            for _iorb, orb in enumerate(orbital_names):
                for spin in range(1):
                    # TODO: what about spin?
                    self.basis.append("%s|%s|%s" % (None, orb, spin))
                    self.positions.append(fractional_coordinates[iatom])

    def _make_translate_maps(self) -> None:
        r"""
        Find the mapping between supercell and translated cell.

        Returns
        -------
        A N \* nbasis array.
        index[i] is the mapping from supercell to translated supercell so that
        T(r_i) psi = psi[indices[i]].

        TODO: vacancies/add_atoms not supported. How to do it? For
        vacancies, a ghost atom can be added. For add_atom, maybe we
        can just ignore them? Will it change the energy spectrum?

        """
        a1 = Structure(atoms=["H"], fractional_coordinates=[[0, 0, 0]], lattice=np.diag([1, 1, 1]))
        sc = a1.transform(np.asarray(self.trans_mat, dtype=np.float64))
        rs = sc.fractional_coordinates
        assert rs is not None

        positions = self.positions
        indices = np.zeros([len(rs), len(positions)], dtype="int32")
        for i, ri in enumerate(rs):
            Tpositions_arr: Any = positions + np.array(ri)

            def close_to_int(x: npt.NDArray[np.float64]) -> bool:
                return bool(np.all(np.abs(x - np.round(x)) < self.tol_radius))

            for i_basis, pos in enumerate(positions):
                for j_basis, Tpos in enumerate(Tpositions_arr):
                    dpos = Tpos - pos

                    if close_to_int(dpos) and (self.basis[i_basis] == self.basis[j_basis]):
                        indices[i, j_basis] = i_basis
        self.trans_rs = rs
        self.trans_indices = indices

    def _get_weight(
        self,
        evec: npt.NDArray[np.complex128],
        qpt: npt.NDArray[np.float64],
        g_vec: npt.NDArray[np.float64] | None = None,
    ) -> float:
        r"""
        Get the weight of a mode which has the wave vector of qpt and
        eigenvector of evec.

        W= sum_1^N < evec| T(r_i)exp(-I (K+G) \* r_i| evec>, here
        G=0. T(r_i)exp(-I K r_i)| evec> = evec[indices[i]]

                    N
                1  ---
         W_KJ = -  \                   -j(K+G).r_i
                N  /   <KJ|T(r_i)|KJ> e
                   ---
                   i=1
        """
        if g_vec is None:
            g_vec = np.zeros_like(qpt)
        weight = 0j
        n_val = self.nfold
        _phase = False
        assert self.trans_rs is not None
        assert self.trans_indices is not None
        for r_i, ind in zip(self.trans_rs, self.trans_indices):
            if _phase:
                weight += (
                    np.vdot(evec, evec[ind]) * np.exp(1j * 2 * np.pi * np.dot(qpt + g_vec, r_i)) / n_val
                )
            else:
                weight += np.vdot(evec, evec[ind]) * np.exp(-1j * 2 * np.pi * np.dot(g_vec, r_i)) / n_val

        return float(weight.real)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """
        Get the weight for all the modes.
        """
        assert self.eigenvectors is not None
        nqpts, nfreqs = self.eigenvectors.shape[0], self.eigenvectors.shape[1]
        weights = np.zeros([nqpts, nfreqs, self.ebs.n_spins])
        for ispin in range(self.ebs.n_spins):
            for iqpt in range(nqpts):
                for ifreq in range(nfreqs):
                    weights[iqpt, ifreq, ispin] = self._get_weight(
                        self.eigenvectors[iqpt, ifreq, ispin, :], self.qpoints[iqpt]
                    )

        return weights
