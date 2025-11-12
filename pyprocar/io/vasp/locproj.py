import collections
import gzip
import logging
import re
from pathlib import Path
from typing import Dict, List, Union
from functools import cached_property

import numpy as np

from pyprocar.utils import np_utils


logger = logging.getLogger(__name__)


class Locproj(collections.abc.Mapping):
    """
    A class to parse the LOCPROJ file from VASP.

    The LOCPROJ file contains information about the projections of the
    Kohn-Sham orbitals onto localized orbitals specified with the LOCPROJ tag.

    Format differs from PROJCAR:
    - First line contains dimensions: n_spins n_k n_bands n_proj
    - Each ISITE line represents one projection (orbital at a position)
    - Data is organized by (spin, k-point, band) blocks

    Parameters
    ----------
    filepath : Union[str, Path]
        The LOCPROJ filepath

    Returns
    -------
    dict
        Dictionary with keys:
        - "frac_coords": np.ndarray of shape [n_proj, 3]
        - "angular_types": list[str] of orbital names
        - "radial_specs": list[dict] with keys "type" and "params"
        - "projections": np.ndarray of shape [n_k, n_bands, n_spins, n_proj]
          with complex dtype
    """

    def __init__(self, filepath: Union[str, Path] = None, file_str: str = None):
        logger.info(f"Initializing Locproj parser for {filepath}")
        self._filepath = filepath
        self._file_str = file_str
        
    @classmethod
    def from_str(cls, input: str):
        return cls(file_str=input)

    @property
    def filepath(self):
        if self._filepath is None:
            raise ValueError("filepath not found. Likely, Locproj provided as string.")
        return Path(self._filepath)

    @cached_property
    def file_str(self):
        if self._file_str is None:
            file_stream = self._open_file(self.filepath)
            self._file_str = file_stream.read()
            file_stream.close()
        return self._file_str

    @cached_property
    def _dimensions(self) -> tuple:
        """
        Parse dimensions from the first line.

        Format: n_spins  n_k  n_bands  n_proj  # optional comment
        
        Returns
        -------
        tuple
            (n_spins, n_k, n_bands, n_proj)
        """
        logger.debug("Parsing LOCPROJ dimensions")

        lines = self.file_str.split("\n")
        
        # Find first non-empty line
        first_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break

        if not first_line:
            raise ValueError("LOCPROJ file appears to be empty")

        # Extract numbers from first line (ignore comments after #)
        if "#" in first_line:
            first_line = first_line.split("#")[0].strip()

        numbers = first_line.split()

        if len(numbers) < 4:
            raise ValueError(
                f"First line should contain 4 numbers (n_spins n_k n_bands n_proj), "
                f"got: {first_line}"
            )

        n_spins = int(numbers[0])
        n_k = int(numbers[1])
        n_bands = int(numbers[2])
        n_proj = int(numbers[3])

        logger.debug(
            f"Dimensions: n_spins={n_spins}, n_k={n_k}, "
            f"n_bands={n_bands}, n_proj={n_proj}"
        )
        
        return (n_spins, n_k, n_bands, n_proj)

    @cached_property
    def n_spins(self) -> int:
        """Number of spin channels."""
        return self._dimensions[0]

    @cached_property
    def n_k(self) -> int:
        """Number of k-points."""
        return self._dimensions[1]

    @cached_property
    def n_bands(self) -> int:
        """Number of bands."""
        return self._dimensions[2]

    @cached_property
    def n_proj(self) -> int:
        """Number of projections."""
        return self._dimensions[3]

    @cached_property
    def _header_data(self) -> tuple:
        """
        Parse the header section containing localized orbital specifications.

        Format for each projection:
        ISITE: <index> R= <x> <y> <z> <type> : <orbital>

        Example:
        ISITE:     1    R=      0.0000000     0.0000000     0.0000000  Hydrogen-like    :     py
        
        Returns
        -------
        tuple
            (frac_coords, angular_types, radial_specs)
        """
        logger.debug("Parsing LOCPROJ header section")

        # Find all localized orbital specifications
        # Pattern: ISITE: <num> R= <coords> <type> : <orbital>
        orbital_pattern = re.compile(
            r"ISITE:\s*(\d+)\s+R=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([^:\n]+):\s*([^\n]*)"
        )

        matches = orbital_pattern.findall(self.file_str)

        if not matches:
            raise ValueError("No localized orbital specifications found in LOCPROJ file")

        if len(matches) != self.n_proj:
            logger.warning(
                f"Number of ISITE lines ({len(matches)}) does not match "
                f"n_proj from header ({self.n_proj})"
            )

        logger.debug(f"Found {len(matches)} localized orbital specifications")

        # Extract fractional coordinates, angular types, and radial specs
        frac_coords_list = []
        angular_types_list = []
        radial_specs_list = []

        for match in matches:
            isite = int(match[0])
            x, y, z = float(match[1]), float(match[2]), float(match[3])
            radial_type_str = match[4].strip()
            orbital_str = match[5].strip()

            # Store fractional coordinates
            frac_coords_list.append([x, y, z])

            # Store angular type (orbital name)
            angular_types_list.append(orbital_str)

            # Parse radial type and parameters
            radial_spec = self._parse_radial_spec(radial_type_str, "")
            radial_specs_list.append(radial_spec)

        # Convert to numpy array
        frac_coords = np.array(frac_coords_list, dtype=np.float64)
        angular_types = angular_types_list
        radial_specs = radial_specs_list

        logger.debug(f"Parsed {len(frac_coords_list)} projections")
        logger.debug(f"Fractional coordinates shape: {frac_coords.shape}")
        logger.debug(f"Angular types: {angular_types}")
        
        return (frac_coords, angular_types, radial_specs)

    @cached_property
    def frac_coords(self) -> np.ndarray:
        """Fractional coordinates of projections [n_proj, 3]."""
        return self._header_data[0]

    @cached_property
    def angular_types(self) -> List[str]:
        """Angular types (orbital names) for each projection."""
        return self._header_data[1]

    @cached_property
    def radial_specs(self) -> List[Dict]:
        """Radial specifications for each projection."""
        return self._header_data[2]

    def _parse_radial_spec(
        self, radial_type_str: str, params_str: str = ""
    ) -> Dict[str, Union[str, Dict]]:
        """
        Parse radial specification string into dictionary.

        Examples:
        - "Hydrogen-like" -> {"type": "Hydrogen-like", "params": {}}
        - "PAW projector" -> {"type": "PAW projector", "params": {}}
        - "PS partial wave" -> {"type": "PS partial wave", "params": {}}
        """
        radial_type_str = radial_type_str.strip()
        params_str = params_str.strip()

        # Check for Hydrogen-like
        if "Hydrogen-like" in radial_type_str or "Hy" in radial_type_str:
            params = {}
            # Parse parameters from params_str if provided
            n_match = re.search(r"n=\s*(\d+)", params_str)
            if n_match:
                params["N"] = int(n_match.group(1))

            za_match = re.search(r"za=\s*([\d.]+)", params_str)
            if za_match:
                params["za"] = float(za_match.group(1))

            return {"type": "Hydrogen-like", "params": params}

        # Check for PAW projector
        if "PAW projector" in radial_type_str or radial_type_str.startswith("Pr"):
            params = {}
            n_match = re.search(r"N=\s*(\d+)", params_str)
            if n_match:
                params["N"] = int(n_match.group(1))
            return {"type": "PAW projector", "params": params}

        # Check for PS partial wave
        if "PS partial wave" in radial_type_str or radial_type_str.startswith("Ps"):
            params = {}
            n_match = re.search(r"N=\s*(\d+)", params_str)
            if n_match:
                params["N"] = int(n_match.group(1))
            return {"type": "PS partial wave", "params": params}

        # Default: return as-is
        return {"type": radial_type_str, "params": {}}

    @cached_property
    def projections(self) -> np.ndarray:
        """
        Parse projection data section.

        Format:
        orbital <spin> <k> <band> <energy> <weight>
            <proj_idx> <real> <imag>
            <proj_idx> <real> <imag>
            ...
            
        Returns
        -------
        np.ndarray
            Projections array of shape [n_k, n_bands, n_spins, n_proj]
        """
        logger.debug("Parsing LOCPROJ projection data section")

        # Initialize projections array
        projections_array = np.zeros(
            (self.n_k, self.n_bands, self.n_spins, self.n_proj),
            dtype=np_utils.COMPLEX_DTYPE,
        )

        # Find all "orbital" blocks
        # Pattern: orbital <spin> <k> <band> <energy> <weight>
        orbital_block_pattern = re.compile(
            r"orbital\s+(\d+)\s+(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)"
        )

        matches = list(orbital_block_pattern.finditer(self.file_str))

        if not matches:
            raise ValueError("No orbital blocks found in LOCPROJ file")

        logger.debug(f"Found {len(matches)} orbital blocks")

        # Process each orbital block
        for i, match in enumerate(matches):
            spin = int(match.group(1))
            k = int(match.group(2))
            band = int(match.group(3))
            energy = float(match.group(4))
            weight = float(match.group(5))

            # Convert to 0-indexed
            spin_idx = spin - 1
            k_idx = k - 1
            band_idx = band - 1

            # Validate indices
            if not (
                0 <= spin_idx < self.n_spins
                and 0 <= k_idx < self.n_k
                and 0 <= band_idx < self.n_bands
            ):
                logger.warning(
                    f"Index out of bounds: spin={spin}, k={k}, band={band}, skipping"
                )
                continue

            # Extract projection values after this match
            # Find the start of the projection values (after the "orbital" line)
            start_pos = match.end()

            # Find the end position (start of next "orbital" line or end of file)
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(self.file_str)

            block_str = self.file_str[start_pos:end_pos]

            # Parse projection values
            self._parse_orbital_block(block_str, k_idx, band_idx, spin_idx, projections_array)

        logger.debug(f"Projections array shape: {projections_array.shape}")
        
        return projections_array

    def _parse_orbital_block(
        self, block_str: str, k_idx: int, band_idx: int, spin_idx: int, projections_array: np.ndarray
    ) -> None:
        """
        Parse a single orbital block containing projection values.

        Format:
            <proj_idx> <real> <imag>
            <proj_idx> <real> <imag>
            ...
        """
        # Find all projection value lines
        # Pattern: <number> <real> <imag>
        proj_pattern = re.compile(r"^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)", re.MULTILINE)

        for match in proj_pattern.finditer(block_str):
            proj_idx = int(match.group(1)) - 1  # Convert to 0-indexed
            real_val = float(match.group(2))
            imag_val = float(match.group(3))

            # Validate projection index
            if not (0 <= proj_idx < self.n_proj):
                logger.warning(f"Projection index {proj_idx + 1} out of bounds, skipping")
                continue

            # Store complex value
            projections_array[k_idx, band_idx, spin_idx, proj_idx] = complex(
                real_val, imag_val
            )

    def _validate_file(self, filepath: Union[str, Path]) -> Path:
        """
        Validate and resolve the LOCPROJ file path.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to LOCPROJ file

        Returns
        -------
        Path
            Validated file path
        """
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / "LOCPROJ"
        elif not filepath.is_file():
            # Try with .gz extension
            gz_path = filepath.with_suffix(".gz")
            if gz_path.is_file():
                return gz_path
            raise IOError(f"LOCPROJ file not found: {filepath}")

        return filepath

    def _open_file(self, filepath: Union[str, Path]):
        """
        Open LOCPROJ file, handling gzipped files.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to file

        Returns
        -------
        file-like object
            Opened file stream
        """
        filepath = Path(filepath)
        if filepath.suffix == ".gz" or filepath.with_suffix(".gz").is_file():
            gz_path = filepath.with_suffix(".gz") if filepath.suffix != ".gz" else filepath
            if gz_path.is_file():
                return gzip.open(gz_path, mode="rt")

        if filepath.is_file():
            return open(filepath, "r")

        raise IOError(f"LOCPROJ file not found: {filepath}")

    def to_dict(self) -> Dict:
        """
        Return parsed data as dictionary.

        Returns
        -------
        dict
            Dictionary with keys: frac_coords, angular_types, radial_specs, projections
        """
        return {
            "frac_coords": self.frac_coords,
            "angular_types": self.angular_types,
            "radial_specs": self.radial_specs,
            "projections": self.projections,
        }

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

