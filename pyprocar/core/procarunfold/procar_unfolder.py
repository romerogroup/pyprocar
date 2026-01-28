from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from ase.atoms import Atoms
from ase.io import read
from matplotlib.axes import Axes

from pyprocar.io.procarparser import ProcarParser
from pyprocar.utils.typing_helpers import get_shape_dim

from .fatband import plot_band_weight
from .unfolder import Unfolder


class _ProcarParserWithPhase(Protocol):
    """Protocol for ProcarParser methods used in unfolding."""

    def readFile2(
        self, fname: str, phase: bool = ..., ispin: int | None = ...
    ) -> None: ...

    @property
    def carray(self) -> npt.NDArray[np.complex128]: ...


class ProcarUnfolder:
    fname: str
    supercell_matrix: npt.NDArray[np.int64]
    procar: ProcarParser
    atoms: Atoms
    basis: list[str]
    positions: list[npt.NDArray[np.float64]]
    eigenvectors: npt.NDArray[np.complex128] | None
    unfolder: Unfolder | None

    def __init__(
        self,
        procar: str,
        poscar: str,
        supercell_matrix: npt.NDArray[np.int64],
        ispin: int | None = None,
    ) -> None:
        self.fname = procar
        self.supercell_matrix = supercell_matrix
        self._parse_procar(ispin=ispin)
        atoms_result = read(poscar)
        # read() can return Atoms or list[Atoms], we expect single Atoms
        if isinstance(atoms_result, list):
            self.atoms = atoms_result[0]
        else:
            self.atoms = atoms_result
        self.basis = []
        self.positions = []
        self.eigenvectors = None
        self.unfolder = None

    def _parse_procar(self, ispin: int | None = None) -> None:
        self.procar = ProcarParser()
        # ProcarParser.readFile2 has partially unknown types - cast to protocol via object
        procar_with_phase = cast(_ProcarParserWithPhase, cast(object, self.procar))
        procar_with_phase.readFile2(self.fname, phase=True, ispin=ispin)

    def _prepare_unfold_basis(self, ispin: int | None = None) -> None:
        # basis, which are the name of the bands e.g. 'Ti|dxy|0'
        # self.eigenvectors = np.zeros(
        #    (self.procar.kpointsCount, self.procar.bandsCount,
        #     (self.procar.orbitalCount - 1) * (self.procar.ionsCount - 1) *
        #     self.procar.ispin), dtype='complex')
        ispin_idx: int
        if ispin is None:
            ispin_idx = 0
        else:
            ispin_idx = ispin - 1

        # ProcarParser attributes have int | None types (some may have Unknown due to incomplete typing)
        kpoints_count: int | None = self.procar.kpointsCount
        # bandsCount has partially unknown type in ProcarParser, need explicit cast
        bands_count: int | None = cast(int | None, self.procar.bandsCount)
        ions_count: int | None = self.procar.ionsCount
        orbital_count: int | None = self.procar.orbitalCount

        if kpoints_count is None or bands_count is None or ions_count is None or orbital_count is None:
            raise ValueError("Procar file not properly parsed - missing count values")

        # Type narrowing after None checks
        kpoints_count_int: int = kpoints_count
        bands_count_int: int = bands_count
        ions_count_int: int = ions_count
        orbital_count_int: int = orbital_count

        procar_with_phase = cast(_ProcarParserWithPhase, cast(object, self.procar))
        carray: npt.NDArray[np.complex128] = procar_with_phase.carray
        self.eigenvectors = np.reshape(
            carray[:, :, ispin_idx, :, :],
            (
                kpoints_count_int,
                bands_count_int,
                ions_count_int * orbital_count_int,
            ),
        )
        norm: npt.NDArray[np.float64] = cast(
            npt.NDArray[np.float64], np.linalg.norm(self.eigenvectors, ord=2, axis=2)
        )
        self.eigenvectors /= norm[:, :, None]

        orbital_names: list[str] = self.procar.orbitalName
        nspin: int = self.procar.nspin
        chemical_symbols: list[str] = self.atoms.get_chemical_symbols()
        scaled_positions: npt.NDArray[np.float64] = self.atoms.get_scaled_positions()
        for iatom in range(len(chemical_symbols)):
            for orb in orbital_names:
                for spin in range(nspin):
                    # todo: what about spin?
                    self.basis.append("%s|%s|%s" % (None, orb, spin))
                    self.positions.append(cast(npt.NDArray[np.float64], scaled_positions[iatom]))

    def unfold(self, ispin: int | None = None) -> npt.NDArray[np.float64]:
        # spd: spd[kpoint][band][ispin][atom][orbital]
        # bands[kpt][iband]
        # to unfold,
        # unfolder:
        # def __init__(self, cell, basis, positions , supercell_matrix, eigenvectors, qpoints, tol_r=0.1, compare=None):
        self._prepare_unfold_basis(ispin=ispin)
        if self.eigenvectors is None:
            raise ValueError("Eigenvectors not prepared. Call _prepare_unfold_basis first.")
        cell: npt.NDArray[np.float64] = np.array(self.atoms.cell)
        positions_arr: npt.NDArray[np.float64] = np.array(self.positions)
        kpoints: npt.NDArray[np.float64] = np.array(self.procar.kpoints)
        self.unfolder = Unfolder(
            cell,
            self.basis,
            positions_arr,
            self.supercell_matrix,
            self.eigenvectors,
            kpoints,
            phase=False,
        )
        w = self.unfolder.get_weights()
        return w  # , self.unfolder

    def plot(
        self,
        efermi: float = 5.46,
        ispin: int | None = None,
        ylim: tuple[float, float] = (-5, 10),
        ktick: list[int] | None = None,
        kname: list[str] | None = None,
        show_band: bool = True,
        shift_efermi: bool = True,
        width: float = 4.0,
        color: str = "blue",
        axis: Axes | None = None,
        savetab: str | bool | None = None,
    ) -> Axes:
        if ktick is None:
            ktick = [0, 41, 83, 125, 200]
        if kname is None:
            kname = [r"$\Gamma$", "X", "M", "R", r"$\Gamma$"]

        iispin: int = 0
        if ispin is not None:
            iispin = ispin - 1

        kpoints_count_opt: int | None = self.procar.kpointsCount
        # bandsCount has partially unknown type in ProcarParser, need explicit cast
        bands_count_opt: int | None = cast(int | None, self.procar.bandsCount)
        if kpoints_count_opt is None or bands_count_opt is None:
            raise ValueError("Procar file not properly parsed - missing count values")
        kpoints_count: int = kpoints_count_opt
        bands_count: int = bands_count_opt

        xlist_single: npt.NDArray[np.float64] = np.array(
            list(range(kpoints_count)), dtype=np.float64
        )
        xlist: list[npt.NDArray[np.float64]] = [xlist_single for _ in range(bands_count)]
        uf = self.unfold(ispin=ispin)
        bands: npt.NDArray[np.float64] = np.array(self.procar.bands)
        if savetab is not None and savetab is not False and isinstance(savetab, str):
            nk: int = get_shape_dim(uf, 0)
            nb: int = get_shape_dim(uf, 1)
            tab: npt.NDArray[np.float64] = np.zeros((nb, nk * 2), dtype=float)
            bands_slice: npt.NDArray[np.float64] = cast(npt.NDArray[np.float64], bands[iispin])
            tab[:, ::2] = bands_slice.T
            tab[:, 1::2] = uf.T
            np.savetxt(
                savetab,
                tab,
                delimiter=",",
                fmt="%10.4f",
                header="# nkpoints: %s   nbands:%s \n#E(k1) w(k1) E(k2) w(k2) E(k3) w(k3)..."
                % (nk, nb),
            )
        bands_for_plot: npt.NDArray[np.float64] = cast(npt.NDArray[np.float64], bands[iispin])
        ekslist: npt.NDArray[np.float64] = bands_for_plot.T
        wkslist: npt.NDArray[np.float64] = np.abs(uf.T)
        xticks: list[list[str] | list[int]] = [kname, ktick]
        axes = plot_band_weight(
            xlist,
            ekslist,
            wkslist,
            xticks=xticks,
            efermi=efermi,
            shift_efermi=shift_efermi,
            fatness=int(width),
            color=color,
            axis=axis,
        )
        _ = axes.set_ylim(ylim[0], ylim[1])
        _ = axes.set_xlim(0, kpoints_count - 1)
        shift: float
        if shift_efermi:
            shift = -efermi
        else:
            shift = 0.0
        if show_band:
            bands_for_plot_arr: npt.NDArray[np.float64] = cast(
                npt.NDArray[np.float64], bands[iispin]
            )
            for i in range(bands_count):
                band_line: npt.NDArray[np.float64] = bands_for_plot_arr[:, i]
                _ = axes.plot(
                    band_line + shift,
                    color="gray",
                    linewidth=1,
                    alpha=0.3,
                )
        return axes
