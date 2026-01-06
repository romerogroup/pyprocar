"""FATBAND file extractor."""

import logging
import re
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Fatband(Mapping[str, Any]):
    """Extractor for FATBAND_*.lobster files.

    Each FATBAND file contains band energies and orbital projections
    for a specific element/orbital combination.

    Parameters
    ----------
    filepath : str | Path | None
        Path to FATBAND file.
    file_str : str
        Content of file as string.
    """

    _filepath: str | Path | None
    _file_str: str

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing Fatband extractor for {filepath}")
        self._filepath = filepath
        self._file_str = file_str

    @classmethod
    def from_str(cls, file_str: str) -> "Fatband":
        """Create extractor from file content string."""
        return cls(file_str=file_str)

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
    def element(self) -> str:
        """Element name for this FATBAND file."""
        match = re.findall(r"#\s*FATBAND\s+for\s+(\S+)", self.file_str)
        if not match:
            raise ValueError("Could not extract element from FATBAND file")
        return match[0]

    @cached_property
    def orbital(self) -> str:
        """Orbital name for this FATBAND file."""
        match = re.findall(r"#\s*FATBAND\s+for\s+\S+\s+(\S+)", self.file_str)
        if not match:
            raise ValueError("Could not extract orbital from FATBAND file")
        # Remove leading parenthesis if present (e.g., "(s" -> "s")
        return match[0].lstrip("(").rstrip(")")

    @cached_property
    def n_bands(self) -> int:
        """Number of bands."""
        match = re.findall(r"NBANDS\s+(\d+)", self.file_str)
        if not match:
            raise ValueError("Could not extract NBANDS from FATBAND file")
        return int(match[0])

    @cached_property
    def kpoints(self) -> np.ndarray:
        """K-point coordinates. Shape: (n_kpoints, 3)"""
        pattern = r"#\s*K-Point\s+\d+\s*:\s*([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)"
        matches = re.findall(pattern, self.file_str)
        if not matches:
            raise ValueError("Could not extract k-points from FATBAND file")
        return np.array([[float(x) for x in m] for m in matches])

    @cached_property
    def n_kpoints(self) -> int:
        """Number of k-points."""
        return len(self.kpoints)

    @cached_property
    def is_spin_polarized(self) -> bool:
        """Check if calculation is spin-polarized.

        Spin-polarized files have 2*n_bands data lines per k-point block.
        """
        # Split by k-point markers and check first block
        blocks = re.split(r"#\s*K-Point", self.file_str)[1:]
        if not blocks:
            return False
        first_block = blocks[0]
        data_lines = [line for line in first_block.split("\n")[1:] if line.strip() and not line.startswith("#")]
        return len(data_lines) == 2 * self.n_bands

    @cached_property
    def n_spins(self) -> int:
        """Number of spin channels."""
        return 2 if self.is_spin_polarized else 1

    @cached_property
    def _parsed_data(self) -> dict[str, np.ndarray]:
        """Parse band energies and projections.

        Returns
        -------
        dict with:
            'bands': shape (n_kpoints, n_bands, n_spins)
            'projections': shape (n_kpoints, n_bands, n_spins)
        """
        bands = np.zeros((self.n_kpoints, self.n_bands, self.n_spins))
        projections = np.zeros((self.n_kpoints, self.n_bands, self.n_spins))

        # Split by k-point markers
        blocks = re.split(r"#\s*K-Point", self.file_str)[1:]

        for ik, block in enumerate(blocks):
            lines = block.split("\n")[1:]  # Skip the coordinate line
            data_lines = [line for line in lines if line.strip() and not line.startswith("#")]

            if self.is_spin_polarized:
                # First n_bands lines are spin-up, next n_bands are spin-down
                for iband in range(self.n_bands):
                    if iband < len(data_lines):
                        parts = data_lines[iband].split()
                        bands[ik, iband, 0] = float(parts[1])
                        projections[ik, iband, 0] = float(parts[2])

                    if iband + self.n_bands < len(data_lines):
                        parts = data_lines[iband + self.n_bands].split()
                        bands[ik, iband, 1] = float(parts[1])
                        projections[ik, iband, 1] = float(parts[2])
            else:
                for iband, line in enumerate(data_lines[: self.n_bands]):
                    parts = line.split()
                    bands[ik, iband, 0] = float(parts[1])
                    projections[ik, iband, 0] = float(parts[2])

        return {"bands": bands, "projections": projections}

    @cached_property
    def bands(self) -> np.ndarray:
        """Band energies. Shape: (n_kpoints, n_bands, n_spins)"""
        return self._parsed_data["bands"]

    @cached_property
    def projections(self) -> np.ndarray:
        """Orbital projections. Shape: (n_kpoints, n_bands, n_spins)"""
        return self._parsed_data["projections"]

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(["element", "orbital", "kpoints", "bands", "projections", "n_spins"])

    def __len__(self) -> int:
        return 6
