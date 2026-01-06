"""SIESTA parser adapter."""

import logging
from functools import cached_property
from pathlib import Path

import numpy as np

from pyprocar.core import KPath, Structure
from pyprocar.core.kpoints import normalize_kpoint_name
from pyprocar.core.dos import DensityOfStates
from pyprocar.core.ebs import ElectronicBandStructure, get_ebs_from_data
from pyprocar.io.base import BaseParser
from pyprocar.io.siesta.bands import Bands
from pyprocar.io.siesta.fdf import FDF

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


class SiestaParser(BaseParser):
    """Parser adapter for SIESTA calculations.

    Combines FDF and Bands extractors to produce canonical data objects.
    Auto-detects files based on directory contents and SystemLabel.

    Parameters
    ----------
    dirpath : str | Path
        Directory containing SIESTA calculation files.
    fdf : str | Path | FDF | None, optional
        Path to .fdf file (relative to dirpath) or pre-instantiated FDF.
        If None, auto-detects first .fdf file in directory.
    bands : str | Path | Bands | None, optional
        Path to .bands file or pre-instantiated Bands.
        If None, derives from SystemLabel.
    """

    def __init__(
        self,
        dirpath: str | Path,
        fdf: str | Path | FDF | None = None,
        bands: str | Path | Bands | None = None,
    ):
        super().__init__(dirpath)

        # Initialize FDF extractor
        self._fdf: FDF | None = self._initialize_fdf(fdf)

        # Initialize Bands extractor (depends on FDF for SystemLabel)
        self._bands: Bands | None = self._initialize_bands(bands)

    def _initialize_fdf(self, param: str | Path | FDF | None) -> FDF | None:
        """Initialize FDF extractor from path or instance."""
        if param is None:
            # Auto-detect .fdf file
            fdf_files = list(self.dirpath.glob("*.fdf"))
            if not fdf_files:
                user_logger.warning(f"No .fdf file found in {self.dirpath}")
                return None
            if len(fdf_files) > 1:
                user_logger.warning(
                    f"Multiple .fdf files found in {self.dirpath}, using {fdf_files[0].name}"
                )
            return FDF(fdf_files[0])

        if isinstance(param, FDF):
            return param

        filepath = self.dirpath / Path(param)
        if filepath.exists():
            return FDF(filepath)

        user_logger.warning(f"FDF file not found: {filepath}")
        return None

    def _initialize_bands(self, param: str | Path | Bands | None) -> Bands | None:
        """Initialize Bands extractor from path or instance."""
        if isinstance(param, Bands):
            return param

        if param is not None:
            filepath = self.dirpath / Path(param)
            if filepath.exists():
                return Bands(filepath)
            user_logger.warning(f"Bands file not found: {filepath}")
            return None

        # Auto-detect using SystemLabel from FDF
        if self._fdf is None:
            return None

        try:
            bands_path = self.dirpath / f"{self._fdf.system_label}.bands"
            if bands_path.exists():
                return Bands(bands_path)
            user_logger.warning(f"Bands file not found: {bands_path}")
        except Exception as e:
            user_logger.warning(f"Error detecting bands file: {e}")

        return None

    @cached_property
    def fermi(self) -> float | None:
        """Fermi energy in eV."""
        if self._bands is None:
            return None
        return self._bands.fermi_energy

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray | None:
        """Reciprocal lattice vectors."""
        if self._fdf is None:
            return None
        try:
            direct = self._fdf.lattice_vectors
            return 2 * np.pi * np.linalg.inv(direct).T
        except Exception as e:
            logger.warning(f"Error computing reciprocal lattice: {e}")
            return None

    @property
    def structure(self) -> Structure | None:
        """Crystal structure."""
        if self._fdf is None:
            return None

        try:
            coord_format = self._fdf.atomic_coords_format.lower()

            if coord_format == "fractional":
                return Structure(
                    atoms=self._fdf.atoms,
                    lattice=self._fdf.lattice_vectors,
                    fractional_coordinates=self._fdf.atomic_positions,
                )
            else:
                return Structure(
                    atoms=self._fdf.atoms,
                    lattice=self._fdf.lattice_vectors,
                    cartesian_coordinates=self._fdf.atomic_positions,
                )
        except Exception as e:
            user_logger.warning(f"Error creating structure: {e}")
            return None

    @property
    def kpath(self) -> KPath | None:
        """K-point path for band structure."""
        if self._fdf is None or not self._fdf.has_band_lines:
            return None

        try:
            band_lines = self._fdf.band_lines
            if band_lines is None:
                return None

            # Build segment names as list of tuples (start_name, end_name)
            segment_names: list[tuple[str, str]] = []
            # Build special kpoint map: name -> coordinates (use normalized names)
            special_kpoint_map: dict[str, np.ndarray] = {}
            # Build n_grids: number of points per segment
            n_grids: list[int] = []

            for i in range(len(band_lines) - 1):
                # Normalize labels to match KPath's internal normalization
                start_name = normalize_kpoint_name(band_lines[i]["label"])
                end_name = normalize_kpoint_name(band_lines[i + 1]["label"])
                segment_names.append((start_name, end_name))

                # Add to special kpoint map with normalized names
                special_kpoint_map[start_name] = np.array(band_lines[i]["kpoint"])
                special_kpoint_map[end_name] = np.array(band_lines[i + 1]["kpoint"])

                # n_grids for this segment
                n_grids.append(band_lines[i + 1]["npoints"])

            return KPath(
                n_grids=n_grids,
                segment_names=segment_names,
                special_kpoint_map=special_kpoint_map,
            )
        except Exception as e:
            user_logger.warning(f"Error creating kpath: {e}")
            return None

    @property
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure."""
        if self._bands is None:
            user_logger.warning("No bands file available for EBS")
            return None

        if self.fermi is None:
            user_logger.warning("No Fermi energy available for EBS")
            return None

        try:
            # SIESTA .bands doesn't provide k-point coordinates, only k-distances
            # We need to provide placeholder kpoints for EBS to work
            n_kpoints = self._bands.n_kpoints
            kpoints = np.zeros((n_kpoints, 3))  # Placeholder k-points

            ebs_kwargs: dict = {
                "kpoints": kpoints,
                "bands": self._bands.bands,
                "projected": None,  # Future work
                "projected_phase": None,
                "fermi": self.fermi,
                "reciprocal_lattice": self.reciprocal_lattice,
                "orbital_names": None,
                "structure": self.structure,
            }

            return get_ebs_from_data(**ebs_kwargs)
        except Exception as e:
            user_logger.warning(f"Error creating EBS: {e}")
            return None

    @property
    def dos(self) -> DensityOfStates | None:
        """Density of states. Not yet implemented for SIESTA."""
        # Future work: Parse .PDOS.xml or .PDOS files
        return None
