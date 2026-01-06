"""BXSF file extractor."""

import logging
import re
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Bxsf(Mapping[str, Any]):
    """Extractor for BXSF (XCrySDen Band Structure Format) files.

    BXSF files contain band energies on a uniform k-grid for Fermi surface visualization.
    The format includes redundant boundary points (+1 in each dimension) which are
    excluded in the actual k-grid.

    Parameters
    ----------
    filepath : str | Path | None
        Path to .bxsf file.
    file_str : str
        Content of .bxsf file as string.
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing Bxsf extractor for {filepath}")
        self._filepath = filepath
        self._file_str = file_str

    @classmethod
    def from_str(cls, file_str: str) -> "Bxsf":
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
    def fermi_energy(self) -> float:
        """Fermi energy in eV."""
        match = re.findall(r"Fermi\s+Energy:\s*([\d.eE+-]+)", self.file_str)
        if not match:
            raise ValueError("No Fermi energy found in BXSF file")
        return float(match[0])

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray:
        """Reciprocal lattice vectors (3x3 array)."""
        pattern = r"BEGIN_BLOCK_BANDGRID_3D\n.*\n.*\n.*\n.*\n.*\n" + 3 * r"\s*(.*)\s*\n"
        match = re.findall(pattern, self.file_str)
        if not match:
            raise ValueError("No reciprocal lattice found in BXSF file")
        return np.array([[float(y) for y in x.split()] for x in match[0]])

    @cached_property
    def origin(self) -> np.ndarray:
        """Origin of k-grid in reciprocal space."""
        pattern = r"BEGIN_BLOCK_BANDGRID_3D\n.*\n.*\n.*\n.*\n(.*)"
        match = re.findall(pattern, self.file_str)
        if not match:
            raise ValueError("No origin found in BXSF file")
        return np.array([float(x) for x in match[0].split()])

    @cached_property
    def nkfs_dim(self) -> np.ndarray:
        """Grid dimensions including redundant boundary points."""
        pattern = r"BEGIN_BLOCK_BANDGRID_3D\n.*\n.*\n.*\n(.*)"
        match = re.findall(pattern, self.file_str)
        if not match:
            raise ValueError("No grid dimensions found in BXSF file")
        return np.array([int(x) for x in match[0].split()])

    @cached_property
    def nk_dim(self) -> tuple[int, int, int]:
        """Grid dimensions excluding redundant boundary (actual k-grid)."""
        return (int(self.nkfs_dim[0]) - 1, int(self.nkfs_dim[1]) - 1, int(self.nkfs_dim[2]) - 1)

    @cached_property
    def n_bands(self) -> int:
        """Number of bands."""
        pattern = r"BEGIN_BLOCK_BANDGRID_3D\n.*\n.*\n\s*(\d+)"
        match = re.findall(pattern, self.file_str)
        if not match:
            raise ValueError("No n_bands found in BXSF file")
        return int(match[0])

    @cached_property
    def _parsed_data(self) -> dict[str, np.ndarray]:
        """Parse band energies and k-points from BXSF file.

        Returns dict with 'bands' and 'kpoints' arrays.
        Bands shape: (n_kpoints, n_bands, n_spins)
        Kpoints shape: (n_kpoints, 3)
        """
        # Find all band blocks
        band_blocks = re.findall(r"(?<=BAND:).*\n([\s\S]*?)(?=[A-Za-z])", self.file_str)

        # Find band labels
        band_labels = re.findall(r"BAND:\s*(\d+)", self.file_str)
        band_labels = [int(label) for label in band_labels]

        # Total k-points with boundary
        nkfs_total = int(np.prod(self.nkfs_dim))

        # Maximum band index determines array size
        max_band = max(band_labels) if band_labels else self.n_bands

        # Pre-allocate arrays (2 spins for compatibility)
        bands_full = np.zeros(shape=[nkfs_total, max_band, 2])
        kpoints_full = np.zeros(shape=[nkfs_total, 3])

        # Track which k-point indices are boundary (to be removed)
        boundary_indices = []

        # Parse each band block
        for band_label, band_block in zip(band_labels, band_blocks, strict=True):
            band_energies = [float(e) for e in band_block.split()]
            iband = band_label - 1
            i_kpoint = 0

            for i in range(self.nkfs_dim[0]):
                for j in range(self.nkfs_dim[1]):
                    for k in range(self.nkfs_dim[2]):
                        bands_full[i_kpoint, iband, 0] = band_energies[i_kpoint]

                        kpoints_full[i_kpoint, :] = np.array(
                            [
                                i / (self.nkfs_dim[0] - 1),
                                j / (self.nkfs_dim[1] - 1),
                                k / (self.nkfs_dim[2] - 1),
                            ]
                        )

                        # Track boundary points (redundant in BXSF format)
                        is_boundary = (
                            i == self.nkfs_dim[0] - 1
                            or j == self.nkfs_dim[1] - 1
                            or k == self.nkfs_dim[2] - 1
                        )
                        if is_boundary and i_kpoint not in boundary_indices:
                            boundary_indices.append(i_kpoint)

                        i_kpoint += 1

        # Remove boundary points
        kpoints = np.delete(kpoints_full, boundary_indices, axis=0)
        bands = np.delete(bands_full, boundary_indices, axis=0)

        logger.debug(f"Parsed BXSF: bands shape={bands.shape}, kpoints shape={kpoints.shape}")
        return {"bands": bands, "kpoints": kpoints}

    @cached_property
    def bands(self) -> np.ndarray:
        """Band energies. Shape: (n_kpoints, n_bands, n_spins)."""
        return self._parsed_data["bands"]

    @cached_property
    def kpoints(self) -> np.ndarray:
        """K-point coordinates in fractional reciprocal space. Shape: (n_kpoints, 3)."""
        return self._parsed_data["kpoints"]

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(["fermi_energy", "reciprocal_lattice", "bands", "kpoints", "nk_dim"])

    def __len__(self) -> int:
        return 5
