"""SIESTA .bands output file extractor."""

import logging
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, override

import numpy as np

logger = logging.getLogger(__name__)


class Bands(Mapping[str, Any]):
    """Extractor for SIESTA .bands output files.

    File format:
    - Line 1: Fermi energy (eV)
    - Line 2: min_k, max_k (path length bounds)
    - Line 3: min_E, max_E (energy bounds)
    - Line 4: n_bands, n_spins, n_kpoints
    - Following: k_distance followed by eigenvalues for each band/spin
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing Bands parser for {filepath}")
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, input: str) -> "Bands":
        """Create Bands instance from file content string."""
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
    def _lines(self) -> list[str]:
        """Split file into lines."""
        return self.file_str.strip().split("\n")

    @cached_property
    def fermi_energy(self) -> float:
        """Extract Fermi energy from first line (eV)."""
        return float(self._lines[0].strip())

    @cached_property
    def k_path_bounds(self) -> tuple[float, float]:
        """Extract min/max k-path length from line 2."""
        parts = self._lines[1].split()
        return float(parts[0]), float(parts[1])

    @cached_property
    def energy_bounds(self) -> tuple[float, float]:
        """Extract min/max energy from line 3."""
        parts = self._lines[2].split()
        return float(parts[0]), float(parts[1])

    @cached_property
    def n_bands(self) -> int:
        """Number of bands."""
        return int(self._lines[3].split()[0])

    @cached_property
    def n_spins(self) -> int:
        """Number of spin components."""
        return int(self._lines[3].split()[1])

    @cached_property
    def n_kpoints(self) -> int:
        """Number of k-points."""
        return int(self._lines[3].split()[2])

    @cached_property
    def _parsed_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Parse k-distances and band energies from file.

        Returns:
            k_distances: shape (n_kpoints,)
            bands: shape (n_kpoints, n_bands, n_spins)
        """
        # Flatten all data after header (lines 0-3)
        raw_data = " ".join(self._lines[4:]).split()

        k_distances: list[float] = []
        bands = np.zeros((self.n_kpoints, self.n_bands, self.n_spins))

        idx = 0
        for ik in range(self.n_kpoints):
            # First value is k-distance
            k_distances.append(float(raw_data[idx]))
            idx += 1

            # Following values are eigenvalues for each spin and band
            for ispin in range(self.n_spins):
                for iband in range(self.n_bands):
                    bands[ik, iband, ispin] = float(raw_data[idx])
                    idx += 1

        return np.array(k_distances), bands

    @cached_property
    def k_distances(self) -> np.ndarray:
        """K-point distances along path. Shape: (n_kpoints,)"""
        return self._parsed_data[0]

    @cached_property
    def bands(self) -> np.ndarray:
        """Band eigenvalues. Shape: (n_kpoints, n_bands, n_spins)"""
        return self._parsed_data[1]

    # Mapping interface
    @override
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(["fermi_energy", "n_bands", "n_spins", "n_kpoints", "bands"])

    @override
    def __len__(self) -> int:
        return self.n_kpoints
