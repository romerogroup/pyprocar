"""DOSCAR.lobster file extractor."""

import logging
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Standard Lobster orbital order
LOBSTER_ORBITALS = [
    "s",
    "p_y",
    "p_z",
    "p_x",
    "d_xy",
    "d_yz",
    "d_z^2",
    "d_xz",
    "d_x^2-y^2",
]

ORBITAL_INDEX = {orb: i for i, orb in enumerate(LOBSTER_ORBITALS)}


class DoscarLobster(Mapping[str, Any]):
    """Extractor for DOSCAR.lobster files.

    Extracts total and projected density of states from Lobster output.

    Parameters
    ----------
    filepath : str | Path | None
        Path to DOSCAR.lobster file.
    file_str : str
        Content of file as string.
    lobsterout_str : str
        Content of lobsterout file (needed for projected DOS orbital mapping).
    """

    _filepath: str | Path | None
    _file_str: str
    _lobsterout_str: str

    def __init__(
        self,
        filepath: str | Path | None = None,
        file_str: str = "",
        lobsterout_str: str = "",
    ):
        logger.info(f"Initializing DoscarLobster extractor for {filepath}")
        self._filepath = filepath
        self._file_str = file_str
        self._lobsterout_str = lobsterout_str

    @classmethod
    def from_str(cls, file_str: str, lobsterout_str: str = "") -> "DoscarLobster":
        """Create extractor from file content strings."""
        return cls(file_str=file_str, lobsterout_str=lobsterout_str)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(self.filepath) as f:
                self._file_str = f.read()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    @cached_property
    def _lines(self) -> list[str]:
        """File content as lines."""
        return self.file_str.splitlines()

    @cached_property
    def nedos(self) -> int:
        """Number of DOS energy points."""
        if len(self._lines) < 6:
            raise ValueError("DOSCAR.lobster seems truncated")
        header = self._lines[5].split()
        return int(float(header[2]))

    @cached_property
    def is_spin_polarized(self) -> bool:
        """Check if DOS is spin-polarized based on column count."""
        # Total DOS line: E, DOS_up, DOS_down, int_up, int_down (5 cols) for spin-pol
        # or: E, DOS, int_DOS (3 cols) for non-spin-pol
        if len(self._lines) < 7:
            return False
        first_data_line = self._lines[6].split()
        return len(first_data_line) == 5

    @cached_property
    def n_spins(self) -> int:
        """Number of spin channels."""
        return 2 if self.is_spin_polarized else 1

    @cached_property
    def _raw_total_dos(self) -> np.ndarray:
        """Raw total DOS block."""
        start = 6
        end = 6 + self.nedos
        block = self._lines[start:end]
        return np.array([[float(x) for x in line.split()] for line in block])

    @cached_property
    def energies(self) -> np.ndarray:
        """Energy grid. Shape: (nedos,)"""
        return self._raw_total_dos[:, 0]

    @cached_property
    def total_dos(self) -> np.ndarray:
        """Total DOS. Shape: (nedos, n_spins)"""
        if self.is_spin_polarized:
            return self._raw_total_dos[:, 1:3]
        return self._raw_total_dos[:, 1:2]

    @cached_property
    def _projected_blocks(self) -> list[tuple[str, np.ndarray]] | None:
        """Parse projected DOS blocks.

        Returns list of (orbital_info_string, data_array) tuples.
        Returns None if no projected DOS present.
        """
        start = 6 + self.nedos
        if start >= len(self._lines):
            return None

        blocks = []
        iline = start
        while iline < len(self._lines):
            # Header line contains ";" separators with orbital info
            header = self._lines[iline]
            if ";" not in header:
                break

            orbital_info = header.split(";")[2] if len(header.split(";")) > 2 else ""
            iline += 1

            # Parse data block
            block_data = []
            for _ in range(self.nedos):
                if iline >= len(self._lines):
                    break
                block_data.append([float(x) for x in self._lines[iline].split()])
                iline += 1

            if block_data:
                blocks.append((orbital_info.strip(), np.array(block_data)))

        return blocks if blocks else None

    @cached_property
    def projected_dos(self) -> np.ndarray | None:
        """Projected DOS. Shape: (nedos, n_spins, n_atoms, n_orbitals).

        Orbitals are mapped to standard Lobster order:
        [s, p_y, p_z, p_x, d_xy, d_yz, d_z^2, d_xz, d_x^2-y^2]
        """
        if self._projected_blocks is None:
            return None

        n_atoms = len(self._projected_blocks)
        n_orbitals = len(LOBSTER_ORBITALS)

        projected = np.zeros((self.nedos, self.n_spins, n_atoms, n_orbitals))

        for iatom, (orbital_info, data) in enumerate(self._projected_blocks):
            # Parse orbital labels from info string
            orbital_labels = orbital_info.split()

            for ilabel, label in enumerate(orbital_labels):
                # Find the standard orbital index
                if label in ORBITAL_INDEX:
                    iorb = ORBITAL_INDEX[label]
                    col_offset = 1  # skip energy column

                    if self.is_spin_polarized:
                        # Columns: E, orb1_up, orb1_down, orb2_up, orb2_down, ...
                        projected[:, 0, iatom, iorb] += data[:, col_offset + 2 * ilabel]
                        projected[:, 1, iatom, iorb] += data[:, col_offset + 2 * ilabel + 1]
                    else:
                        projected[:, 0, iatom, iorb] += data[:, col_offset + ilabel]

        return projected

    @cached_property
    def orbital_labels(self) -> list[str]:
        """List of orbital labels in standard order."""
        return LOBSTER_ORBITALS.copy()

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(["energies", "total_dos", "projected_dos", "orbital_labels", "n_spins"])

    def __len__(self) -> int:
        return 5
