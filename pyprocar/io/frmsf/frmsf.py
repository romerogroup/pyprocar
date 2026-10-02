"""FrmSrf file extractor."""

import logging
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Frmsf(Mapping[str, Any]):
    """Extractor for FrmSrf (FermiSurfer) files.

    FrmSrf files contain band energies and projections on a uniform k-grid.

    Parameters
    ----------
    filepath : str | Path | None
        Path to .frmsf file.
    file_str : str
        Content of .frmsf file as string.
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing Frmsf extractor for {filepath}")
        self._filepath = filepath
        self._file_str = file_str

    @classmethod
    def from_str(cls, file_str: str) -> "Frmsf":
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
    def _lines(self) -> list[str]:
        """Split file into lines for parsing."""
        return self.file_str.strip().split("\n")

    @cached_property
    def nk_dim(self) -> tuple[int, int, int]:
        """K-grid dimensions."""
        dims = [int(x) for x in self._lines[0].split()]
        return (dims[0], dims[1], dims[2])

    @cached_property
    def kpoint_generation_method(self) -> int:
        """K-point generation method (0, 1, or 2)."""
        return int(self._lines[1])

    @cached_property
    def n_bands(self) -> int:
        """Number of bands."""
        return int(self._lines[2])

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray:
        """Reciprocal lattice vectors (3x3 array)."""
        return np.array([[float(y) for y in x.split()] for x in self._lines[3:6]])

    @cached_property
    def n_kpoints(self) -> int:
        """Total number of k-points."""
        return self.nk_dim[0] * self.nk_dim[1] * self.nk_dim[2]

    def _kpoint_from_indices(self, n1: int, n2: int, n3: int) -> np.ndarray:
        """Generate k-point coordinates based on generation method.

        Parameters
        ----------
        n1, n2, n3 : int
            1-based indices in the k-grid (range 1 to N).

        Returns
        -------
        np.ndarray
            K-point coordinates in fractional reciprocal space.
        """
        N1, N2, N3 = self.nk_dim
        method = self.kpoint_generation_method

        if method == 0:
            # Monkhorst-Pack style
            return np.array([
                (2 * n1 - 1 - N1) / N1,
                (2 * n2 - 1 - N2) / N2,
                (2 * n3 - 1 - N3) / N3,
            ])
        elif method == 1:
            # Gamma-centered style
            return np.array([(n1 - 1) / N1, (n2 - 1) / N2, (n3 - 1) / N3])
        elif method == 2:
            # Shifted gamma-centered
            return np.array([
                (2 * n1 - 1) / (2 * N1),
                (2 * n2 - 1) / (2 * N2),
                (2 * n3 - 1) / (2 * N3),
            ])
        else:
            raise ValueError(f"Unknown kpoint generation method: {method}")

    @cached_property
    def _parsed_data(self) -> dict[str, np.ndarray]:
        """Parse bands, kpoints, and projections from file.

        Returns dict with 'bands', 'kpoints', and optionally 'projections'.
        """
        # All values after line 6 (0-indexed)
        values = np.array([float(x) for x in " ".join(self._lines[6:]).split()])

        # Calculate number of projections
        n_values = len(values)
        values_for_bands = self.n_kpoints * self.n_bands
        n_projections = int((n_values - values_for_bands) / self.n_kpoints)

        # Pre-allocate arrays
        bands = np.zeros(shape=(self.n_kpoints, self.n_bands))
        kpoints = np.zeros(shape=(self.n_kpoints, 3))

        if n_projections > 0:
            projections = np.zeros(shape=(self.n_kpoints, n_projections))
        else:
            projections = None

        # Parse data
        counter = 0
        n_properties = 2 if n_projections > 0 else 1

        for iproperty in range(1, n_properties + 1):
            for iband in range(self.n_bands):
                kpoint_counter = 0
                for i in range(1, self.nk_dim[0] + 1):
                    for j in range(1, self.nk_dim[1] + 1):
                        for k in range(1, self.nk_dim[2] + 1):
                            if iproperty == 1:
                                bands[kpoint_counter, iband] = values[counter]
                                kpoints[kpoint_counter, :] = self._kpoint_from_indices(i, j, k)
                            elif iproperty == 2 and projections is not None:
                                projections[kpoint_counter, iband] = values[counter]

                            kpoint_counter += 1
                            counter += 1

        logger.debug(f"Parsed FrmSrf: bands shape={bands.shape}, kpoints shape={kpoints.shape}")
        result = {"bands": bands, "kpoints": kpoints}
        if projections is not None:
            result["projections"] = projections
        return result

    @cached_property
    def bands(self) -> np.ndarray:
        """Band energies. Shape: (n_kpoints, n_bands)."""
        return self._parsed_data["bands"]

    @cached_property
    def kpoints(self) -> np.ndarray:
        """K-point coordinates. Shape: (n_kpoints, 3)."""
        return self._parsed_data["kpoints"]

    @cached_property
    def projections(self) -> np.ndarray | None:
        """Projections if available. Shape: (n_kpoints, n_projections)."""
        return self._parsed_data.get("projections")

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(["reciprocal_lattice", "bands", "kpoints", "nk_dim"])

    def __len__(self) -> int:
        return 4
