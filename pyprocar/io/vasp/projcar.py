import collections
import gzip
import logging
import re
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from pyprocar.utils import np_utils

logger = logging.getLogger(__name__)


class Projcar(collections.abc.Mapping):
    """
    A class to parse the PROJCAR file from VASP.

    The PROJCAR file contains information about the projections of the
    Kohn-Sham orbitals onto localized orbitals specified with the LOCPROJ tag.

    Parameters
    ----------
    filepath : Union[str, Path]
        The PROJCAR filepath

    Returns
    -------
    dict
        Dictionary with keys:
        - "frac_coords": np.ndarray of shape [n_atoms, 3]
        - "radial_specs": list[dict] with keys "type" and "params"
        - "projections": np.ndarray of shape [n_k, n_bands, n_spins, n_atoms, n_orbitals]
          with complex dtype
    """

    def __init__(self, filepath: Union[str, Path]):
        logger.info(f"Initializing Projcar parser for {filepath}")
        self.filepath = self._validate_file(filepath)
        self.filename = self.filepath.name

        self.file_str = None
        self.meta_lines = []

        # Parsed data
        self.frac_coords = None
        self.radial_specs = None
        self.projections = None
        self.angular_types = None
        self.n_k = None
        self.n_bands = None
        self.n_spins = None
        self.n_atoms = None
        self.n_orbitals = None

        self._read()
        logger.info("Projcar parsing completed")

    def _read(self) -> None:
        """
        Main entry point for parsing the PROJCAR file.
        """
        file_stream = self._open_file(self.filepath)

        # Read first line (header comment)
        self.meta_lines.append(file_stream.readline())

        # Read the rest of the file
        self.file_str = file_stream.read()
        file_stream.close()

        # Parse header section (localized orbital specifications)
        self._read_header()

        # Parse projection data
        self._read_projections()

        return

    def _read_header(self) -> None:
        """
        Parse the header section containing localized orbital specifications.

        Format for each orbital:
        ISITE: <index> R: <x> <y> <z> Radial type: <type> Angular type: <type>
        """
        logger.debug("Parsing PROJCAR header section")

        # Find all localized orbital specifications
        # Pattern: ISITE: <num> R: <coords> Radial type: <type> Angular type: <type>
        orbital_pattern = re.compile(
            r"ISITE:\s*(\d+)\s+R:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
            r"Radial type:\s*([^\n]+?)\s+Angular type:\s*([^\n]+)"
        )

        matches = orbital_pattern.findall(self.file_str)

        if not matches:
            raise ValueError("No localized orbital specifications found in PROJCAR file")

        logger.debug(f"Found {len(matches)} localized orbital specifications")

        # Extract fractional coordinates and radial specs
        frac_coords_list = []
        radial_specs_list = []
        angular_types_list = []

        for match in matches:
            isite = int(match[0])
            x, y, z = float(match[1]), float(match[2]), float(match[3])
            radial_type_str = match[4].strip()
            angular_type_str = match[5].strip()

            # Store fractional coordinates
            frac_coords_list.append([x, y, z])

            # Parse radial type and parameters
            radial_spec = self._parse_radial_spec(radial_type_str)
            radial_specs_list.append(radial_spec)

            # Parse angular types (split by spaces, as they can be multiple)
            angular_types = angular_type_str.split()
            angular_types_list.append(angular_types)

        # Convert to numpy array
        self.frac_coords = np.array(frac_coords_list, dtype=np.float64)
        self.n_atoms = len(frac_coords_list)

        # Store radial specs
        self.radial_specs = radial_specs_list

        # Determine number of orbitals from angular types
        # Use the first set of angular types to determine n_orbitals
        if angular_types_list:
            self.angular_types = angular_types_list[0]
            self.n_orbitals = len(self.angular_types)
        else:
            raise ValueError("No angular types found")

        logger.debug(f"Parsed {self.n_atoms} atoms with {self.n_orbitals} orbitals each")
        logger.debug(f"Fractional coordinates shape: {self.frac_coords.shape}")
        logger.debug(f"Angular types: {self.angular_types}")

    def _parse_radial_spec(self, radial_type_str: str) -> Dict[str, Union[str, Dict]]:
        """
        Parse radial specification string into dictionary.

        Examples:
        - "PAW projector" -> {"type": "PAW projector", "params": {}}
        - "PS partial wave" -> {"type": "PS partial wave", "params": {}}
        - "Hydrogen-like" -> {"type": "Hydrogen-like", "params": {"N": 1, "za": 1.0}}
        - "Hy 2 1.5" -> {"type": "Hy", "params": {"N": 2, "za": 1.5}}
        """
        radial_type_str = radial_type_str.strip()

        # Check for Hydrogen-like with parameters
        hy_pattern = re.compile(r"Hy(?:drogen-like)?(?:\s+(\d+))?(?:\s+([\d.]+))?")
        hy_match = hy_pattern.match(radial_type_str)

        if hy_match:
            n_str = hy_match.group(1)
            za_str = hy_match.group(2)

            params = {}
            if n_str:
                params["N"] = int(n_str)
            else:
                params["N"] = 1  # Default

            if za_str:
                params["za"] = float(za_str)
            else:
                params["za"] = 1.0  # Default

            return {"type": "Hydrogen-like", "params": params}

        # Check for PAW projector with optional parameters
        if "PAW projector" in radial_type_str or radial_type_str.startswith("Pr"):
            # Extract parameters if present (e.g., "Pr 2 3")
            pr_pattern = re.compile(r"Pr(?:\s+(\d+))?(?:\s+(\d+))?")
            pr_match = pr_pattern.match(radial_type_str)
            params = {}
            if pr_match and pr_match.group(1):
                params["N"] = int(pr_match.group(1))
            if pr_match and pr_match.group(2):
                params["species"] = int(pr_match.group(2))
            return {"type": "PAW projector", "params": params}

        # Check for PS partial wave with optional parameters
        if "PS partial wave" in radial_type_str or radial_type_str.startswith("Ps"):
            # Extract parameters if present (e.g., "Ps 2 3")
            ps_pattern = re.compile(r"Ps(?:\s+(\d+))?(?:\s+(\d+))?")
            ps_match = ps_pattern.match(radial_type_str)
            params = {}
            if ps_match and ps_match.group(1):
                params["N"] = int(ps_match.group(1))
            if ps_match and ps_match.group(2):
                params["species"] = int(ps_match.group(2))
            return {"type": "PS partial wave", "params": params}

        # Default: return as-is
        return {"type": radial_type_str, "params": {}}

    def _read_projections(self) -> None:
        """
        Parse projection data section.

        Format:
        spin <n> k-point <n> : <coords> band <n> # energy <val> # occ. <val>
        <projection data rows with real and imaginary parts>
        """
        logger.debug("Parsing PROJCAR projection data section")

        # Find all band headers with their context (spin, k-point)
        # Pattern: spin <n> ... k-point <n> ... band <n> ...
        band_pattern = re.compile(
            r"spin\s+(\d+).*?k-point\s+(\d+)\s*:.*?band\s+(\d+)\s*#\s*energy\s+([-\d.]+)",
            re.DOTALL,
        )
        band_matches = band_pattern.findall(self.file_str)

        if not band_matches:
            # Try simpler pattern without spin (non-spin-polarized)
            band_pattern = re.compile(
                r"k-point\s+(\d+)\s*:.*?band\s+(\d+)\s*#\s*energy\s+([-\d.]+)",
                re.DOTALL,
            )
            band_matches = band_pattern.findall(self.file_str)
            # Add spin index of 1 for non-spin-polarized
            band_matches = [(1, m[0], m[1], m[2]) for m in band_matches]

        if not band_matches:
            raise ValueError("No projection data found in PROJCAR file")

        # Determine dimensions
        unique_spins = sorted(set(int(m[0]) for m in band_matches))
        self.n_spins = len(unique_spins)

        unique_kpoints = sorted(set(int(m[1]) for m in band_matches))
        self.n_k = len(unique_kpoints)

        # Count bands per k-point/spin combination
        bands_per_k_spin = {}
        for spin, kpoint, band, _ in band_matches:
            key = (int(spin), int(kpoint))
            if key not in bands_per_k_spin:
                bands_per_k_spin[key] = []
            bands_per_k_spin[key].append(int(band))

        # Get maximum number of bands
        if bands_per_k_spin:
            self.n_bands = max(len(bands) for bands in bands_per_k_spin.values())
        else:
            self.n_bands = 1

        logger.debug(
            f"Dimensions: n_k={self.n_k}, n_bands={self.n_bands}, "
            f"n_spins={self.n_spins}, n_atoms={self.n_atoms}, "
            f"n_orbitals={self.n_orbitals}"
        )

        # Initialize projections array
        self.projections = np.zeros(
            (self.n_k, self.n_bands, self.n_spins, self.n_atoms, self.n_orbitals),
            dtype=np_utils.COMPLEX_DTYPE,
        )

        # Parse projection values
        self._parse_projection_values()

        logger.debug(f"Projections array shape: {self.projections.shape}")


    def _parse_projection_values(self) -> None:
        """
        Parse actual projection values (real and imaginary parts).

        The data follows this structure:
        spin <n>
        k-point <n> : <coords>
        band <n> # energy <val> # occ. <val>
        <projection data rows>
        """
        lines = self.file_str.split("\n")
        current_k = -1
        current_spin = 0  # Default to 0 (non-spin-polarized)
        current_band = -1
        in_data_section = False
        data_lines = []

        for line in lines:
            # Check for spin header
            spin_match = re.search(r"spin\s+(\d+)", line)
            if spin_match:
                # Process previous band's data if any
                if in_data_section and data_lines:
                    self._process_band_data(
                        current_k, current_band, current_spin, data_lines
                    )
                    data_lines = []

                current_spin = int(spin_match.group(1)) - 1  # 0-indexed
                in_data_section = False
                continue

            # Check for k-point header
            kpoint_match = re.search(
                r"k-point\s+(\d+)\s*:\s+([-\d.\s]+)", line
            )
            if kpoint_match:
                # Process previous band's data if any
                if in_data_section and data_lines:
                    self._process_band_data(
                        current_k, current_band, current_spin, data_lines
                    )
                    data_lines = []

                current_k = int(kpoint_match.group(1)) - 1  # 0-indexed
                in_data_section = False
                continue

            # Check for band header
            band_match = re.search(
                r"band\s+(\d+)\s*#\s*energy\s+([-\d.]+)", line
            )
            if band_match:
                # Process previous band's data if any
                if in_data_section and data_lines:
                    self._process_band_data(
                        current_k, current_band, current_spin, data_lines
                    )
                    data_lines = []

                current_band = int(band_match.group(1)) - 1  # 0-indexed
                in_data_section = True
                continue

            # Collect data lines (projection values)
            if in_data_section and line.strip():
                # Skip lines that are headers or comments
                if re.match(r"^\s*(spin|k-point|band|#|ISITE|Radial|Angular)", line):
                    continue

                # Check if line contains numbers (projection data)
                if re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line):
                    data_lines.append(line)

        # Process last band's data
        if in_data_section and data_lines:
            self._process_band_data(current_k, current_band, current_spin, data_lines)

    def _process_band_data(
        self, k_idx: int, band_idx: int, spin_idx: int, data_lines: List[str]
    ) -> None:
        """
        Process projection data for a single band.

        Data format: lines with numbers representing real and imaginary parts
        of projections for each atom and orbital.

        The data may be organized as:
        - One line per atom with all orbital values
        - Or multiple lines with orbital labels and values
        """
        if k_idx < 0 or band_idx < 0 or spin_idx < 0:
            return

        if k_idx >= self.n_k or band_idx >= self.n_bands or spin_idx >= self.n_spins:
            logger.warning(
                f"Index out of bounds: k={k_idx}, band={band_idx}, spin={spin_idx}"
            )
            return

        # Parse data lines
        # Try to identify the format
        # Check if first line contains orbital labels or just numbers
        atom_idx = 0
        for line in data_lines:
            if atom_idx >= self.n_atoms:
                break

            line = line.strip()
            if not line:
                continue

            # Skip header lines with orbital names
            if any(orb in line.lower() for orb in ["s", "p", "d", "f", "tot"]):
                # Check if this is a data line starting with orbital name
                # or a header line
                if re.match(r"^\s*\d+\s", line) or re.match(
                    r"^\s*[-+]?\d*\.?\d+", line
                ):
                    # Data line that happens to contain orbital name
                    pass
                else:
                    # Header line, skip
                    continue

            # Extract all numbers from the line
            numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)

            if not numbers:
                continue

            # Determine format: check if we have enough values
            n_values = len(numbers)

            # Try to match orbital count
            # If we have exactly n_orbitals values, assume real-only
            # If we have n_orbitals * 2 values, assume complex (real, imag pairs)
            # If we have more, might include atom index or other metadata

            if n_values >= self.n_orbitals * 2:
                # Complex values (real, imag pairs)
                # Skip first number if it looks like an atom index
                start_idx = 0
                if n_values > self.n_orbitals * 2:
                    # Might have atom index at start
                    start_idx = 1

                for orb_idx in range(self.n_orbitals):
                    idx = start_idx + orb_idx * 2
                    if idx + 1 < len(numbers):
                        real_val = float(numbers[idx])
                        imag_val = float(numbers[idx + 1])
                        self.projections[
                            k_idx, band_idx, spin_idx, atom_idx, orb_idx
                        ] = complex(real_val, imag_val)
            elif n_values >= self.n_orbitals:
                # Real values only
                # Skip first number if it looks like an atom index
                start_idx = 0
                if n_values > self.n_orbitals:
                    start_idx = 1

                for orb_idx in range(self.n_orbitals):
                    idx = start_idx + orb_idx
                    if idx < len(numbers):
                        real_val = float(numbers[idx])
                        self.projections[
                            k_idx, band_idx, spin_idx, atom_idx, orb_idx
                        ] = complex(real_val, 0.0)

            atom_idx += 1

    def _validate_file(self, filepath: Union[str, Path]) -> Path:
        """
        Validate and resolve the PROJCAR file path.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to PROJCAR file

        Returns
        -------
        Path
            Validated file path
        """
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / "PROJCAR"
        elif not filepath.is_file():
            # Try with .gz extension
            gz_path = filepath.with_suffix(".gz")
            if gz_path.is_file():
                return gz_path
            raise IOError(f"PROJCAR file not found: {filepath}")

        return filepath

    def _open_file(self, filepath: Union[str, Path]):
        """
        Open PROJCAR file, handling gzipped files.

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

        raise IOError(f"PROJCAR file not found: {filepath}")

    def to_dict(self) -> Dict:
        """
        Return parsed data as dictionary.

        Returns
        -------
        dict
            Dictionary with keys: frac_coords, radial_specs, projections
        """
        return {
            "frac_coords": self.frac_coords,
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

