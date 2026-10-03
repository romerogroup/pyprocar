"""elk.in input file parser for Elk calculations."""

import logging
from dataclasses import dataclass
from functools import cached_property
from itertools import takewhile
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from pyprocar.core.kpoints import normalize_kpoint_name
from pyprocar.io.elk.geometry import ElkCell, block_lines, bool_fortran, parse_elk_cell

user_logger = logging.getLogger("user")


@dataclass(frozen=True, slots=True)
class Plot1D:
    vertices: npt.NDArray[np.float64]
    npoints: int
    labels: list[str]


# Elk 11.2.3 readinput.f90 defaults: nvp1d=2 vertices (0,0,0) and (1,1,1), npp1d=200.
ELK_DEFAULT_PLOT1D = Plot1D(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), 200, ["0", "1"])


def _vertex_label(line: str) -> str:
    """Text after the last ':' of a plot1d vertex line.

    KPath maps "Gamma" to its own symbol but not "\\Gamma", so a backslash is
    dropped only when that turns the label into a KPath alias. Other LaTeX,
    such as "\\Sigma_1", is kept for KPath to render.
    """
    if ":" not in line:
        return ""
    label = line.rpartition(":")[2].replace(",", "").replace("vlvp1d", "").replace(" ", "")
    plain = label.lstrip("\\")
    return plain if normalize_kpoint_name(plain) != plain else label


class ElkIn:
    """Parser for Elk elk.in input file.

    The elk.in file contains calculation parameters including:
    - tasks: calculation type identifiers
    - spinpol: spin polarization flag
    - lattice vectors and atomic positions
    - k-path for band structure (plot1d block)

    Parameters
    ----------
    filepath : Path | None
        Path to elk.in file
    file_str : str
        Content of elk.in file (alternative to filepath)
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
    def _lines(self) -> list[str]:
        return self.file_str.splitlines()

    @cached_property
    def tasks(self) -> list[int]:
        """List of task numbers from tasks block, which ends at a blank line."""
        block = block_lines(self._lines, "tasks")
        if block is None:
            raise ValueError("No 'tasks' block found in elk.in")
        return [int(line.split()[0]) for line in takewhile(str.strip, block)]

    @cached_property
    def is_bands_calculation(self) -> bool:
        """Check if this is a band structure calculation (tasks 20 to 24)."""
        return any(t in self.tasks for t in [20, 21, 22, 23, 24])

    @cached_property
    def spinpol(self) -> bool:
        """Spin polarization flag."""
        block = block_lines(self._lines, "spinpol")
        rows = [line.split() for line in block or [] if line.strip()]
        return bool(rows) and bool_fortran(rows[0][0])

    @cached_property
    def nspin(self) -> int:
        """Number of spin channels (1 or 2)."""
        return 2 if self.spinpol else 1

    @cached_property
    def nspecies(self) -> int:
        """Number of atomic species."""
        return len(self.composition)

    @cached_property
    def composition(self) -> dict[str, int]:
        """Dictionary of species -> atom count, in elk.in order."""
        result: dict[str, int] = {}
        for atom in self.atoms:
            result[atom] = result.get(atom, 0) + 1
        return result

    @cached_property
    def cell(self) -> ElkCell:
        """Lattice in Angstrom, atom symbols and fractional coordinates."""
        return parse_elk_cell(self.file_str)

    @property
    def lattice(self) -> npt.NDArray[np.float64]:
        """Lattice vectors in Angstrom as 3x3 array (rows are vectors); Elk writes Bohr."""
        return self.cell.lattice

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

    # K-path properties (from plot1d block)

    @cached_property
    def plot1d(self) -> Plot1D:
        """Band path vertices, total point count and vertex labels."""
        block = block_lines(self._lines, "plot1d")
        if block is None:
            user_logger.warning(
                "elk.in has no plot1d block; using the Elk default path (0,0,0) to (1,1,1)"
                + " with 200 points"
            )
            return ELK_DEFAULT_PLOT1D
        rows = [line for line in block if line.strip()]
        nvertices, npoints = (int(token) for token in rows[0].split()[:2])
        vertex_lines = rows[1 : 1 + nvertices]
        vertices = np.array([[float(x) for x in line.split()[:3]] for line in vertex_lines])
        labels = [_vertex_label(line) for line in vertex_lines]
        if not all(labels):
            labels = [str(x) for x in range(nvertices)]
        return Plot1D(vertices, npoints, labels)

    @cached_property
    def has_kpath(self) -> bool:
        """Check if elk.in has a plot1d block."""
        return block_lines(self._lines, "plot1d") is not None

    @cached_property
    def n_high_sym(self) -> int:
        """Number of high-symmetry points."""
        return len(self.plot1d.vertices)

    @cached_property
    def nkpoints(self) -> int:
        """Total number of k-points along path."""
        return self.plot1d.npoints

    @cached_property
    def n_segments(self) -> int:
        """Number of path segments."""
        return max(0, self.n_high_sym - 1)

    @cached_property
    def high_symmetry_points(self) -> npt.NDArray[np.float64]:
        """High-symmetry point coordinates as (n_high_sym, 3) array."""
        return self.plot1d.vertices

    @cached_property
    def knames(self) -> list[list[str]]:
        """K-point labels as list of [start, end] pairs per segment."""
        labels = self.plot1d.labels
        return [[labels[i], labels[i + 1]] for i in range(self.n_segments)]
