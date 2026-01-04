"""elk.in input file parser for Elk calculations."""

import re
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt


def bool_fortran(string: str) -> bool:
    """Convert Fortran boolean string to Python bool."""
    return string.strip().lower() in (".true.", "true", "t", ".t.")


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
    def tasks(self) -> list[int]:
        """List of task numbers from tasks block."""
        pattern = re.compile(r"(?m)^[ \t]*tasks[ \t]*\n" + r"((?:[ \t]*\d+[ \t]*\n)+)")
        match = pattern.search(self.file_str)
        if not match:
            raise ValueError("No 'tasks' block found in elk.in")
        return [int(n) for n in match.group(1).split()]

    @cached_property
    def is_bands_calculation(self) -> bool:
        """Check if this is a band structure calculation (task 20, 21, or 22)."""
        return any(t in self.tasks for t in [20, 21, 22])

    @cached_property
    def spinpol(self) -> bool:
        """Spin polarization flag."""
        match = re.findall(r"spinpol\s*([.a-zA-Z]*)", self.file_str)
        if len(match) != 0:
            return bool_fortran(match[0])
        return False

    @cached_property
    def nspin(self) -> int:
        """Number of spin channels (1 or 2)."""
        return 2 if self.spinpol else 1

    @cached_property
    def nspecies(self) -> int:
        """Number of atomic species."""
        match = re.findall(r"atoms\n\s*([0-9]*)", self.file_str)
        if match:
            return int(match[0])
        return 0

    @cached_property
    def composition(self) -> dict[str, int]:
        """Dictionary of species -> atom count."""
        result: dict[str, int] = {}
        for match in re.findall(r"'([A-Za-z]*).in'.*\n\s*([0-9]*)", self.file_str):
            result[match[0]] = int(match[1])
        return result

    @cached_property
    def lattice(self) -> npt.NDArray[np.float64]:
        """Lattice vectors as 3x3 array (rows are vectors)."""
        raw_lattice = re.findall(r"avec\s*\n(.*\n.*\n.*)", self.file_str)
        if not raw_lattice:
            raise ValueError("No lattice vectors found in elk.in")

        lattice = np.zeros((3, 3))
        for i, vec in enumerate(raw_lattice[0].split("\n")):
            lattice[i, :] = [float(coord) for coord in vec.strip().split()]

        # Apply scale factor if present
        scale_match = re.findall(r"scale\s*\n(.*)", self.file_str)
        if scale_match:
            scale = float(scale_match[0])
            lattice *= scale

        return lattice

    @cached_property
    def atoms(self) -> list[str]:
        """List of atom symbols."""
        atoms: list[str] = []
        raw_species = re.findall(r"'([A-Za-z]*).in'.*\n.*\n\s*([0-9.\s]*)", self.file_str)
        for specie_name, coords_block in raw_species:
            n_atoms = len(coords_block.strip().split("\n"))
            atoms.extend([specie_name] * n_atoms)
        return atoms

    @cached_property
    def fractional_coordinates(self) -> npt.NDArray[np.float64]:
        """Fractional coordinates as (natom, 3) array."""
        coords: list[list[float]] = []
        raw_species = re.findall(r"'([A-Za-z]*).in'.*\n.*\n\s*([0-9.\s]*)", self.file_str)
        for _, coords_block in raw_species:
            for line in coords_block.strip().split("\n"):
                coords.append([float(c) for c in line.split()[:3]])
        return np.array(coords)

    @cached_property
    def natoms(self) -> int:
        """Number of atoms."""
        return len(self.atoms)

    # K-path properties (from plot1d block)

    @cached_property
    def _plot1d_info(self) -> tuple[int, int] | None:
        """Parse plot1d block header: (n_high_sym, n_kpoints)."""
        match = re.findall(r"plot1d\n\s*([0-9]*)\s*([0-9]*)", self.file_str)
        if not match:
            return None
        return int(match[0][0]), int(match[0][1])

    @cached_property
    def has_kpath(self) -> bool:
        """Check if k-path information is present."""
        return self._plot1d_info is not None

    @cached_property
    def n_high_sym(self) -> int:
        """Number of high-symmetry points."""
        if self._plot1d_info is None:
            return 0
        return self._plot1d_info[0]

    @cached_property
    def nkpoints(self) -> int:
        """Total number of k-points along path."""
        if self._plot1d_info is None:
            return 0
        return self._plot1d_info[1]

    @cached_property
    def n_segments(self) -> int:
        """Number of path segments."""
        return max(0, self.n_high_sym - 1)

    @cached_property
    def ngrids(self) -> list[int]:
        """Number of k-points per segment."""
        if self.n_segments == 0:
            return []
        points_per_segment = self.nkpoints // self.n_segments
        return [points_per_segment] * self.n_segments

    @cached_property
    def high_symmetry_points(self) -> npt.NDArray[np.float64]:
        """High-symmetry point coordinates as (n_high_sym, 3) array."""
        if self.n_high_sym == 0:
            return np.array([])

        pattern = r"plot1d.*\n.*\n\s* " + self.n_high_sym * r"(.*)\n*"
        match = re.findall(pattern, self.file_str)
        if not match:
            return np.array([])

        points = np.zeros((self.n_high_sym, 3))
        for i, raw_kpoint in enumerate(match[0]):
            points[i, :] = [float(k) for k in raw_kpoint.split()[:3]]
        return points

    @cached_property
    def knames(self) -> list[list[str]]:
        """K-point labels as list of [start, end] pairs per segment."""
        if self.n_high_sym == 0:
            return []

        pattern = r"plot1d\n\s*[0-9]*\s*[0-9]*.*\n" + self.n_high_sym * r".*:(.*)\n"
        match = re.findall(pattern, self.file_str)

        if len(match) == 0 or len(match[0]) != self.n_high_sym:
            # Use numeric labels as fallback
            labels = [str(x) for x in range(self.n_high_sym)]
        else:
            labels = [
                "$%s$" % x.replace(",", "").replace("vlvp1d", "").replace(" ", "")
                for x in match[0]
            ]

        # Convert to segment pairs
        knames: list[list[str]] = []
        for i in range(self.n_segments):
            knames.append([labels[i], labels[i + 1]])
        return knames

    @cached_property
    def special_kpoints(self) -> npt.NDArray[np.float64]:
        """Special k-points as (n_segments, 2, 3) array of [start, end] pairs."""
        if self.n_segments == 0:
            return np.array([])

        special = np.zeros((self.n_segments, 2, 3))
        for i in range(self.n_segments):
            special[i, 0, :] = self.high_symmetry_points[i, :]
            special[i, 1, :] = self.high_symmetry_points[i + 1, :]
        return special
