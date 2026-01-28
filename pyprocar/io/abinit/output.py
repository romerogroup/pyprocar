"""Abinit output file parser."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import override

from pyprocar.core import Structure
from pyprocar.utils import elements

logger = logging.getLogger(__name__)


class AbinitOutput(Mapping[str, Any]):
    """Parse the fermi energy, reciprocal lattice vectors and structure
    from the Abinit output file.
    """

    _filepath: Path | None
    _file_str: str

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        self._filepath = Path(filepath) if filepath else None
        self._file_str = file_str

    @classmethod
    def from_str(cls, input_str: str) -> "AbinitOutput":
        """Create parser from string content (for testing)."""
        return cls(file_str=input_str)

    @classmethod
    def is_file_of_type(cls, filepath: str | Path) -> bool:
        """Check if file is an ABINIT output file by content."""
        try:
            with open(filepath, errors="ignore") as f:
                for _ in range(10):
                    line = f.readline()
                    if "Version" in line and "ABINIT" in line:
                        return True
        except Exception:
            pass
        return False

    @cached_property
    def file_str(self) -> str:
        """Lazy file reading."""
        if self._file_str == "" and self._filepath is not None:
            with open(self._filepath) as f:
                self._file_str = f.read()
        elif self._file_str == "" and self._filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    @cached_property
    def fermi(self) -> float:
        """Fermi energy in eV."""
        # Try to find fermi in eV first
        match_ev = re.findall(
            r"Fermi\w*.\(\w*.HOMO\)\s*energy\s*\(eV\)\s*\=\s*([0-9.+-]*)", 
            self.file_str
        )
        if match_ev:
            return float(match_ev[0])
        
        # Otherwise look for hartree format
        match_ha = re.findall(
            r"Fermi\w*.\(\w*.HOMO\)\s*energy\s*\(\w*\)\s*\=\s*([0-9.+-]*)", 
            self.file_str
        )
        if match_ha:
            fermi_ha = float(match_ha[0])
            return 27.211396641308 * fermi_ha  # Hartree to eV
        
        # Fallback to old regex without units
        match = re.findall(
            r"Fermi\w*.\(\w*.HOMO\)\s*\w*\s*\(\w*\)\s*\=\s*([0-9.+-]*)", 
            self.file_str
        )
        fermi_ha = float(match[0])
        return 27.211396641308 * fermi_ha  # Hartree to eV

    @cached_property
    def nspin(self) -> int:
        """Number of spin channels (nsppol)."""
        match = re.findall(r"nsppol\s*=\s*([1-9]*)", self.file_str)
        return int(match[0])

    @cached_property
    def reclat(self) -> np.ndarray:
        """Reciprocal lattice vectors."""
        lattice_block = re.findall(r"G\([1,2,3]\)=\s*([0-9.\s-]*)", self.file_str)
        if len(lattice_block) < 3:
            return np.array([])
        lattice_block = lattice_block[-3:]  # Take last 3 matches
        return np.array(
            [lattice_block[i].split() for i in range(len(lattice_block))],
            dtype=float,
        )

    @cached_property
    def coordinates(self) -> np.ndarray:
        """Reduced atomic coordinates."""
        coordinate_block = re.findall(r"xred\s*([+-.0-9E\s]*)", self.file_str)[-1].split()
        if not coordinate_block:
            coordinate_block = re.findall(
                r"reduced\scoordinates\s\(array\sxred\)\sfor\s*[1-9]\satoms\n([+-.0-9E\s]*)\n",
                self.file_str,
            )[-1].split()
        coordinate_list = np.array([float(x) for x in coordinate_block])
        return coordinate_list.reshape(len(coordinate_list) // 3, 3)

    @cached_property
    def lattice(self) -> np.ndarray:
        """Direct lattice vectors in Angstrom."""
        # acell in Bohr
        acell = re.findall(r"acell\s*([+-.0-9E\s]*)", self.file_str)[-1].split()
        acell = np.array([float(x) for x in acell]) * 0.529177  # Bohr to Angstrom

        # rprim
        rprim_block = re.findall(r"rprim\s*([+-.0-9E\s]*)", self.file_str)[-1].split()
        rprim_list = np.array([float(x) for x in rprim_block])
        rprim = rprim_list.reshape(len(rprim_list) // 3, 3)

        lattice = np.zeros(shape=(3, 3))
        for i in range(len(acell)):
            lattice[i, :] = acell[i] * rprim[i, :]
        return lattice

    @cached_property
    def atoms(self) -> list[str]:
        """Atomic elements list."""
        typat = re.findall(r"typat\s*([+-.0-9E\s]*)", self.file_str)[-1].split()
        typat = [int(x) for x in typat]
        znucl = re.findall(r"znucl\s*([+-.0-9E\s]*)", self.file_str)[-1].split()
        znucl = [int(float(x)) for x in znucl]
        return [elements.atomic_symbol(znucl[x - 1]) for x in typat]

    @cached_property
    def structure(self) -> Structure:
        """Structure object."""
        return Structure(
            atoms=self.atoms,
            fractional_coordinates=self.coordinates,
            lattice=self.lattice,
        )

    @cached_property
    def version(self) -> str | None:
        """Abinit version string."""
        match = re.findall(r"\.Version\s+(\d+\.\d+\.\d+)", self.file_str)
        return match[0] if match else None

    # Mapping protocol implementation
    @override
    def __contains__(self, key: object) -> bool:
        return key in self.__dict__

    @override
    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    @override
    def __iter__(self) -> Iterator[str]:
        return self.__dict__.__iter__()

    @override
    def __len__(self) -> int:
        return self.__dict__.__len__()
