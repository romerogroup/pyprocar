"""BAND_S*_A*.OUT orbital projections parser for Elk calculations."""

from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt


class ElkProjections:
    """Parser for Elk BAND_S*_A*.OUT orbital projection files.

    Each file contains projections onto spherical harmonics for one atom.
    Files are named BAND_S{species:02d}_A{atom:04d}.OUT

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
    """

    N_ORBITALS: int = 16  # Y00 through Y3-3

    def __init__(
        self,
        filepaths: list[Path] | None = None,
        file_strs: list[str] | None = None,
        nkpoints: int = 0,
        nbands: int = 0,
        nspin: int = 1,
        natoms: int = 0,
    ):
        self._filepaths: list[Path] = filepaths or []
        self._file_strs: list[str] = file_strs or []
        self._nkpoints: int = nkpoints
        self._nbands: int = nbands
        self._nspin: int = nspin
        self._natoms: int = natoms

    @classmethod
    def from_str(
        cls,
        file_contents: list[str],
        nkpoints: int,
        nbands: int,
        nspin: int = 1,
    ) -> Self:
        """Create parser from file content strings."""
        return cls(
            file_strs=file_contents,
            nkpoints=nkpoints,
            nbands=nbands,
            nspin=nspin,
            natoms=len(file_contents),
        )

    @cached_property
    def file_strs(self) -> list[str]:
        """Lazily load all projection file contents."""
        if not self._file_strs and self._filepaths:
            return [Path(fp).read_text() for fp in self._filepaths]
        elif not self._file_strs and not self._filepaths:
            return []
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
    def spd(self) -> npt.NDArray[np.float64]:
        """Raw SPD array in Elk format.

        Shape: (nkpoints, nbands, nspin, natoms+1, norbitals+2)
        - natoms+1: last column is total over atoms
        - norbitals+2: first column is atom index, last is total over orbitals
        """
        if not self.file_strs:
            return np.array([])

        spd = np.zeros(
            (
                self.nkpoints,
                self.nbands,
                self.nspin,
                self.natoms + 1,
                self.N_ORBITALS + 2,
            )
        )

        for iatom, content in enumerate(self.file_strs):
            lines = content.splitlines()
            iline = 0

            for iband in range(self.nbands):
                for ikpoint in range(self.nkpoints):
                    temp = np.array([float(x) for x in lines[iline].split()])
                    spd[ikpoint, iband, 0, iatom, 0] = iatom + 1  # Atom index
                    spd[ikpoint, iband, 0, iatom, 1:-1] = temp[2:]  # Orbital projections
                    iline += 1
                # Skip blank line between bands (only if we have k-points)
                if self.nkpoints > 0:
                    iline += 1

        # Sum over orbitals for each atom
        spd[:, :, :, :, -1] = np.sum(spd[:, :, :, :, 1:-1], axis=4)
        # Sum over atoms
        spd[:, :, :, -1, :] = spd.sum(axis=3)
        spd[:, :, 0, -1, 0] = 0

        # Handle spin polarized case
        if self.nspin == 2:
            # Copy spin up to second spin channel
            spd[:, : self.nbands // 2, 1, :, :] = spd[:, : self.nbands // 2, 0, :, :]
            # Negate spin down projections
            spd[:, self.nbands // 2 :, 1, :, :] = -1 * spd[:, self.nbands // 2 :, 0, :, :]

        return spd

    @cached_property
    def projected(self) -> npt.NDArray[np.float64] | None:
        """Projected array in canonical format.

        Shape: (nkpoints, nbands, natoms, nprincipals, norbitals, nspin)
        """
        if self.spd.size == 0:
            return None

        nprincipals = 1
        projected = np.zeros(
            (
                self.nkpoints,
                self.nbands,
                self.natoms,
                nprincipals,
                self.N_ORBITALS,
                self.nspin,
            ),
            dtype=self.spd.dtype,
        )

        temp_spd = self.spd.copy()
        # Reorder axes: (nkpoints, nbands, nspin, natom, norbital)
        # -> (nkpoints, nbands, natom, norbital, nspin)
        temp_spd = np.swapaxes(temp_spd, 2, 4)
        temp_spd = np.swapaxes(temp_spd, 2, 3)

        if self.nspin == 2:
            projected[:, :, :, 0, :, 0] = temp_spd[:, :, :-1, 1:-1, 0]
            projected[:, :, :, 0, :, 1] = temp_spd[:, :, :-1, 1:-1, 1]
        else:
            projected[:, :, :, 0, :, :] = temp_spd[:, :, :-1, 1:-1, :]

        return projected
