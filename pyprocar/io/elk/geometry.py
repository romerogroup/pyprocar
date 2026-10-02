"""GEOMETRY.OUT parser for Elk calculations."""

import re
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?"
float_re = re.compile(FLOAT)


class ElkGeometry:
    """Parser for Elk GEOMETRY.OUT file.

    GEOMETRY.OUT contains the crystal structure after optimization,
    including lattice vectors and atomic positions.

    Parameters
    ----------
    filepath : Path | None
        Path to GEOMETRY.OUT file
    file_str : str
        Content of GEOMETRY.OUT file (alternative to filepath)
    """

    def __init__(self, filepath: Path | None = None, file_str: str = ""):
        self._filepath: Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, content: str) -> Self:
        """Create parser from file content string."""
        return cls(file_str=content)

    @property
    def filepath(self) -> Path | None:
        """Return filepath if set."""
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        """Lazily load file content."""
        if self._file_str == "" and self._filepath is not None:
            return Path(self._filepath).read_text()
        elif self._file_str == "" and self._filepath is None:
            raise ValueError("No filepath or file_str provided")
        return self._file_str

    @cached_property
    def lattice(self) -> npt.NDArray[np.float64]:
        """Lattice vectors as 3x3 array (rows are vectors)."""
        pattern_matrix = re.compile(
            r"avec[\s\S]*?\n" + rf"((?:[ \t]*{FLOAT}\s+){{8}}" + rf"{FLOAT}\s*\n)"
        )
        match = pattern_matrix.search(self.file_str)
        if match is None:
            raise ValueError("No lattice vectors found in GEOMETRY.OUT")
        matrix_block_str = match.group(1)
        return np.fromstring(matrix_block_str, sep=" ").reshape(3, 3)

    @cached_property
    def nspecies(self) -> int:
        """Number of atomic species."""
        pattern = re.search(r"(?m)^atoms\s*\r?\n\s*(\d+)", self.file_str, re.IGNORECASE)
        if pattern is None:
            raise ValueError("No species count found in GEOMETRY.OUT")
        return int(pattern.group(1))

    @cached_property
    def _atoms_and_coords(self) -> tuple[list[str], npt.NDArray[np.float64]]:
        """Parse atoms and fractional coordinates."""
        pattern_spc = re.compile(r"(?mi)^\s*'([A-Za-z]+\.in)'[\s\S]*?^\s*(\d+).*\s")

        atoms: list[str] = []
        fractional_coords: list[list[float]] = []

        for m in pattern_spc.finditer(self.file_str):
            atom_count = int(m.group(2))
            atom_symbols = [m.group(1).replace(".in", "")] * atom_count
            atoms += atom_symbols
            start = m.end()
            tail = self.file_str[start:].splitlines()
            pos_lines = tail[:atom_count]
            for line in pos_lines:
                fractional_coords += [[float(x) for x in float_re.findall(line)]]

        coords_array = np.array(fractional_coords)
        # Handle optional magnetic field columns
        if coords_array.shape[1] == 6:
            coords_array = coords_array[:, :3]

        return atoms, coords_array

    @cached_property
    def atoms(self) -> list[str]:
        """List of atom symbols."""
        return self._atoms_and_coords[0]

    @cached_property
    def fractional_coordinates(self) -> npt.NDArray[np.float64]:
        """Fractional coordinates as (natom, 3) array."""
        return self._atoms_and_coords[1]

    @cached_property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)
