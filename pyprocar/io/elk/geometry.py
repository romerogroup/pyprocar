"""GEOMETRY.OUT parser for Elk calculations."""

import re
from collections.abc import Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Self

import numpy as np
import numpy.typing as npt

from pyprocar.utils.units import AU_TO_ANG


def bool_fortran(string: str) -> bool:
    """Convert Fortran boolean string to Python bool."""
    return string.strip().lower() in (".true.", "true", "t", ".t.")


class ElkCell(NamedTuple):
    lattice: npt.NDArray[np.float64]
    atoms: list[str]
    fractional_coordinates: npt.NDArray[np.float64]


def block_lines(lines: Sequence[str], keyword: str) -> Sequence[str] | None:
    """Lines after the last line whose first token is ``keyword``.

    Elk reads the first token of a block line, so ``scale : global`` opens
    ``scale``, and a later copy of a block overrides an earlier one.
    """
    for i in range(len(lines) - 1, -1, -1):
        tokens = lines[i].split()
        if tokens and tokens[0] == keyword:
            return lines[i + 1 :]
    return None


def _rows_after(lines: Sequence[str], keyword: str) -> Iterator[list[str]] | None:
    block = block_lines(lines, keyword)
    if block is None:
        return None
    return (row.split() for row in block if row.strip())


def _fortran_floats(tokens: Sequence[str]) -> list[float]:
    return [float(token.lower().replace("d", "e")) for token in tokens]


def parse_elk_cell(text: str) -> ElkCell:
    """Read the cell from the elk.in blocks that GEOMETRY.OUT also uses.

    Follows Elk 11.2.3 ``readinput.f90``: ``avec`` rows are lattice vectors in
    Bohr, scaled by ``scale``, then row i by ``scale<i>``, then Cartesian
    column x/y/z by ``scalex/y/z``. With ``molecule .true.`` the atom
    positions are Cartesian Bohr and convert to lattice coordinates through
    the scaled lattice.
    """
    lines = text.splitlines()

    def scalar(keyword: str) -> float:
        rows = _rows_after(lines, keyword)
        return 1.0 if rows is None else _fortran_floats(next(rows)[:1])[0]

    rows = _rows_after(lines, "avec")
    if rows is None:
        raise ValueError("No avec block found")
    avec = np.array([_fortran_floats(next(rows)[:3]) for _ in range(3)])
    avec *= scalar("scale")
    avec *= np.array([[scalar("scale1")], [scalar("scale2")], [scalar("scale3")]])
    avec *= np.array([scalar("scalex"), scalar("scaley"), scalar("scalez")])

    rows = _rows_after(lines, "atoms")
    if rows is None:
        raise ValueError("No atoms block found")
    atoms: list[str] = []
    positions: list[list[float]] = []
    for _ in range(int(next(rows)[0])):
        species = next(rows)[0].strip("'\"").removesuffix(".in")
        for _ in range(int(next(rows)[0])):
            atoms.append(species)
            positions.append(_fortran_floats(next(rows)[:3]))
    fractional = np.array(positions)

    molecule = _rows_after(lines, "molecule")
    if molecule is not None and bool_fortran(next(molecule)[0]):
        fractional = fractional @ np.linalg.inv(avec)

    return ElkCell(avec * AU_TO_ANG, atoms, fractional)


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
    def cell(self) -> ElkCell:
        """Lattice in Angstrom, atom symbols and fractional coordinates."""
        return parse_elk_cell(self.file_str)

    @property
    def lattice(self) -> npt.NDArray[np.float64]:
        """Lattice vectors in Angstrom as 3x3 array (rows are vectors); Elk writes Bohr."""
        return self.cell.lattice

    @cached_property
    def nspecies(self) -> int:
        """Number of atomic species."""
        pattern = re.search(r"(?m)^atoms\s*\r?\n\s*(\d+)", self.file_str, re.IGNORECASE)
        if pattern is None:
            raise ValueError("No species count found in GEOMETRY.OUT")
        return int(pattern.group(1))

    @property
    def atoms(self) -> list[str]:
        """List of atom symbols."""
        return self.cell.atoms

    @property
    def fractional_coordinates(self) -> npt.NDArray[np.float64]:
        """Fractional coordinates as (natom, 3) array."""
        return self.cell.fractional_coordinates

    @cached_property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)
