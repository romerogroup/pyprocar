"""BAND_S*_A*.OUT band-character parser for Elk calculations."""

from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

_L_NAMES = ["s", "p", "d", "f", "g", "h"]


def _lm_names(n_columns: int) -> list[str]:
    lmax = round(n_columns**0.5) - 1
    return [f"Y{ang}{m}" for ang in range(lmax + 1) for m in range(-ang, ang + 1)]


class ElkProjections:
    """Parser for Elk BAND_S{species:02d}_A{atom:04d}.OUT band-character files, one per atom.

    Elk's bandstr.f90 writes each file state by state, with one line per
    k-point and a blank line after each state. Each line starts with the path
    distance and the energy. The columns after them depend on the task:

    - 21: the sum over l, then the l = 0..lmaxdb characters
    - 22: the (l,m) characters, l = 0..lmaxdb and m = -l..l
    - 23: the spin-up and spin-down characters (spin-polarized runs only)
    - 24: the moment character, m_z for a collinear run

    In a collinear spin-polarized run the first nbands states are spin up and
    the next nbands spin down. A state's character goes to its own spin channel.

    Parameters
    ----------
    filepaths : list[Path]
        List of paths to BAND_S*_A*.OUT files (one per atom)
    file_strs : list[str]
        List of file contents (alternative to filepaths)
    nkpoints : int
        Number of k-points
    nbands : int
        Number of bands per spin channel
    nspin : int
        Number of spin channels
    natoms : int
        Number of atoms
    task : int
        The Elk task, 21 to 24, that wrote the files
    """

    def __init__(
        self,
        filepaths: list[Path] | None = None,
        file_strs: list[str] | None = None,
        nkpoints: int = 0,
        nbands: int = 0,
        nspin: int = 1,
        natoms: int = 0,
        task: int = 22,
    ):
        self._filepaths: list[Path] = filepaths or []
        self._file_strs: list[str] = file_strs or []
        self._nkpoints: int = nkpoints
        self._nbands: int = nbands
        self._nspin: int = nspin
        self._natoms: int = natoms
        self.task: int = task

    @classmethod
    def from_str(
        cls,
        file_contents: list[str],
        nkpoints: int,
        nbands: int,
        nspin: int = 1,
        task: int = 22,
    ) -> Self:
        """Create parser from file content strings."""
        return cls(
            file_strs=file_contents,
            nkpoints=nkpoints,
            nbands=nbands,
            nspin=nspin,
            natoms=len(file_contents),
            task=task,
        )

    @cached_property
    def file_strs(self) -> list[str]:
        """Lazily load all projection file contents."""
        if not self._file_strs and self._filepaths:
            return [Path(fp).read_text() for fp in self._filepaths]
        return self._file_strs

    @cached_property
    def nkpoints(self) -> int:
        return self._nkpoints

    @cached_property
    def nbands(self) -> int:
        return self._nbands

    @cached_property
    def nspin(self) -> int:
        return self._nspin

    @cached_property
    def natoms(self) -> int:
        return self._natoms

    @cached_property
    def _characters(self) -> npt.NDArray[np.float64]:
        """Columns after distance and energy, as (natoms, nspin * nbands states, nkpoints, ncols)."""
        n_states = self.nspin * self.nbands
        return np.array(
            [
                np.loadtxt(content.splitlines(), ndmin=2)[: n_states * self.nkpoints, 2:].reshape(
                    n_states, self.nkpoints, -1
                )
                for content in self.file_strs
            ]
        )

    @cached_property
    def orbital_names(self) -> list[str]:
        """Names of the orbital axis of ``projected`` for this task."""
        n_columns = self._characters.shape[-1]
        if self.task == 21:
            return _L_NAMES[: n_columns - 1]
        if self.task == 23:
            return ["spin"]
        if self.task == 24:
            return ["moment"]
        return _lm_names(n_columns)

    @cached_property
    def projected(self) -> npt.NDArray[np.float64] | None:
        """Projected array in canonical format.

        Shape: (nkpoints, nbands, natoms, nprincipals, norbitals, nspin)
        """
        if not self.file_strs:
            return None

        characters = self._characters
        if self.task == 21:
            characters = characters[..., 1:]
        elif self.task == 24:
            characters = characters[..., -1:]
        # (natoms, nspin, nbands, nkpoints, ncols)
        by_spin = characters.reshape(self.natoms, self.nspin, self.nbands, self.nkpoints, -1)
        if self.task == 23:
            # each state keeps the character of its own spin
            by_spin = np.stack([by_spin[:, s, ..., s : s + 1] for s in range(self.nspin)], axis=1)
        # -> (nkpoints, nbands, natoms, ncols, nspin), then add the principal axis
        return np.transpose(by_spin, (3, 2, 0, 4, 1))[:, :, :, np.newaxis]
