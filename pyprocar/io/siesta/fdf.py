"""SIESTA FDF input file extractor."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import override

logger = logging.getLogger(__name__)


def _extract_block(text: str, block_name: str) -> str | None:
    """Extract content from a %block...%endblock section.

    Parameters
    ----------
    text : str
        The full FDF file content
    block_name : str
        Name of the block (case insensitive)

    Returns
    -------
    str | None
        Block content or None if not found
    """
    pattern = rf"%block\s+{block_name}\s*\n([\s\S]*?)%endblock\s+{block_name}"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


class FDF(Mapping[str, Any]):
    """Extractor for SIESTA .fdf input files.

    Parses system label, lattice vectors, atomic positions, and k-path information
    from SIESTA input files.
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing FDF parser for {filepath}")
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, input: str) -> "FDF":
        """Create FDF instance from file content string."""
        return cls(file_str=input)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(file=self.filepath) as rf:
                self._file_str = rf.read()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    @cached_property
    def system_label(self) -> str:
        """Extract SystemLabel from FDF file."""
        match = re.findall(r"SystemLabel\s+([0-9A-Za-z_-]+)", self.file_str, re.IGNORECASE)
        if not match:
            raise ValueError("No SystemLabel found in FDF file")
        return match[0]

    @cached_property
    def lattice_constant(self) -> float:
        """Extract LatticeConstant from FDF file."""
        match = re.findall(r"LatticeConstant\s+([0-9.]+)", self.file_str, re.IGNORECASE)
        if not match:
            return 1.0  # Default if not specified
        return float(match[0])

    @cached_property
    def lattice_vectors(self) -> np.ndarray:
        """Extract lattice vectors from LatticeVectors block."""
        raw_lattice = _extract_block(self.file_str, "LatticeVectors")
        if raw_lattice is None:
            raise ValueError("No LatticeVectors block found in FDF file")

        lines = raw_lattice.split("\n")
        lattice = np.zeros(shape=(3, 3))
        for i, line in enumerate(lines[:3]):
            coords = line.split()
            for j, coord in enumerate(coords[:3]):
                lattice[i, j] = float(coord)
        return lattice

    @cached_property
    def atomic_coords_format(self) -> str:
        """Extract AtomicCoordinatesFormat from FDF file."""
        match = re.findall(r"AtomicCoordinatesFormat\s+(\w+)", self.file_str, re.IGNORECASE)
        if not match:
            return "Fractional"  # Default
        return match[0]

    @cached_property
    def species_labels(self) -> dict[str, str]:
        """Extract species index to label mapping from ChemicalSpeciesLabel block."""
        raw_species = _extract_block(self.file_str, "ChemicalSpeciesLabel")
        if raw_species is None:
            raise ValueError("No ChemicalSpeciesLabel block found in FDF file")

        mapping: dict[str, str] = {}
        for line in raw_species.split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                index = parts[0]
                label = parts[2]
                mapping[index] = label
        return mapping

    @cached_property
    def atomic_positions(self) -> np.ndarray:
        """Extract atomic positions from AtomicCoordinatesAndAtomicSpecies block."""
        raw_positions = _extract_block(self.file_str, "AtomicCoordinatesAndAtomicSpecies")
        if raw_positions is None:
            raise ValueError("No AtomicCoordinatesAndAtomicSpecies block found")

        lines = raw_positions.split("\n")
        n_atoms = len(lines)
        positions = np.zeros(shape=(n_atoms, 3))
        for i, line in enumerate(lines):
            parts = line.split()
            for j in range(3):
                positions[i, j] = float(parts[j])
        return positions

    @cached_property
    def atoms(self) -> list[str]:
        """Extract atom list with species labels."""
        raw_positions = _extract_block(self.file_str, "AtomicCoordinatesAndAtomicSpecies")
        if raw_positions is None:
            return []

        atoms: list[str] = []
        for line in raw_positions.split("\n"):
            parts = line.split()
            if len(parts) >= 4:
                species_index = parts[3]
                atoms.append(self.species_labels.get(species_index, species_index))
        return atoms

    @cached_property
    def has_band_lines(self) -> bool:
        """Check if BandLines block exists."""
        return _extract_block(self.file_str, "BandLines") is not None

    @cached_property
    def band_lines(self) -> list[dict[str, int | list[float] | str]] | None:
        """Parse BandLines block for k-path specification.

        Returns list of dicts with keys: npoints, kpoint, label
        """
        raw_kpath = _extract_block(self.file_str, "BandLines")
        if raw_kpath is None:
            return None

        result: list[dict[str, int | list[float] | str]] = []
        for line in raw_kpath.split("\n"):
            parts = line.split()
            if len(parts) >= 5:
                result.append({
                    "npoints": int(parts[0]),
                    "kpoint": [float(parts[1]), float(parts[2]), float(parts[3])],
                    "label": parts[4] if len(parts) > 4 else "",
                })
        return result

    # Mapping interface
    @override
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.atoms)

    @override
    def __len__(self) -> int:
        return len(self.atoms)
