import collections
import gzip
import logging
import re
from functools import cached_property
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

    def __init__(self, filepath: Union[str, Path] = None, file_str: str = None):
        """
        Initialize Projcar parser.

        Parameters
        ----------
        filepath : Union[str, Path], optional
            Path to PROJCAR file
        file_str : str, optional
            PROJCAR file content as string
        """
        self._file_str = file_str
        self._filepath = filepath

    @classmethod
    def from_str(cls, input: str):
        """Create Projcar instance from string content."""
        return cls(file_str=input)

    @property
    def filepath(self):
        """Return validated filepath."""
        if self._filepath is None:
            raise ValueError("filepath not found. Likely, Projcar provided as string.")
        return self._validate_file(self._filepath)

    @cached_property
    def file_str(self):
        """Read and cache file content."""
        if self._file_str is None:
            logger.info(f"Reading PROJCAR file: {self.filepath}")
            file_stream = self._open_file(self.filepath)
            # Skip first line (header comment)
            _ = file_stream.readline()
            # Read the rest of the file
            self._file_str = file_stream.read()
            file_stream.close()
            logger.info("PROJCAR file read completed")
        return self._file_str

    @cached_property
    def frac_coords(self):
        """Parse and return fractional coordinates."""
        logger.debug("Parsing fractional coordinates from PROJCAR header")
        
        # Find all localized orbital specifications
        orbital_pattern = re.compile(
            r"ISITE:\s*(\d+)\s+R=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([^:\n]+):\s*([^\n]*)"
        )
        matches = orbital_pattern.findall(self.file_str)
        
        if not matches:
            raise ValueError("No localized orbital specifications found in PROJCAR file")
        
        # Extract fractional coordinates
        frac_coords_list = []
        for match in matches:
            x, y, z = float(match[1]), float(match[2]), float(match[3])
            frac_coords_list.append([x, y, z])
        
        frac_coords = np.array(frac_coords_list, dtype=np.float64)
        logger.debug(f"Fractional coordinates shape: {frac_coords.shape}")
        return frac_coords
    
    @cached_property
    def radial_specs(self):
        """Parse and return radial specifications."""
        logger.debug("Parsing radial specifications from PROJCAR header")
        
        # Find all localized orbital specifications
        orbital_pattern = re.compile(
            r"ISITE:\s*(\d+)\s+R=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([^:\n]+):\s*([^\n]*)"
        )
        matches = orbital_pattern.findall(self.file_str)
        
        if not matches:
            raise ValueError("No localized orbital specifications found in PROJCAR file")
        
        # Parse radial type and parameters
        radial_specs_list = []
        for match in matches:
            radial_type_str = match[4].strip()
            params_str = match[5].strip()
            radial_spec = self._parse_radial_spec(radial_type_str, params_str)
            radial_specs_list.append(radial_spec)
        
        logger.debug(f"Parsed {len(radial_specs_list)} radial specifications")
        return radial_specs_list
    
    @cached_property
    def n_atoms(self):
        """Return number of atoms."""
        return len(self.frac_coords)

    def _parse_radial_spec(self, radial_type_str: str, params_str: str = "") -> Dict[str, Union[str, Dict]]:
        """
        Parse radial specification string into dictionary.

        Examples:
        - "Hydrogen-like" with "n= 1 za= 1.0000" -> {"type": "Hydrogen-like", "params": {"N": 1, "za": 1.0}}
        - "PAW projector" -> {"type": "PAW projector", "params": {}}
        - "PS partial wave" -> {"type": "PS partial wave", "params": {}}
        """
        radial_type_str = radial_type_str.strip()
        params_str = params_str.strip()

        # Check for Hydrogen-like with parameters
        if "Hydrogen-like" in radial_type_str or "Hy" in radial_type_str:
            params = {}
            
            # Parse parameters from params_str (e.g., "n= 1  za=  1.0000")
            n_match = re.search(r"n=\s*(\d+)", params_str)
            if n_match:
                params["N"] = int(n_match.group(1))
            else:
                params["N"] = 1  # Default

            za_match = re.search(r"za=\s*([\d.]+)", params_str)
            if za_match:
                params["za"] = float(za_match.group(1))
            else:
                params["za"] = 1.0  # Default

            return {"type": "Hydrogen-like", "params": params}

        # Check for PAW projector with optional parameters
        if "PAW projector" in radial_type_str or radial_type_str.startswith("Pr"):
            params = {}
            # Extract parameters if present in params_str
            n_match = re.search(r"N=\s*(\d+)", params_str)
            if n_match:
                params["N"] = int(n_match.group(1))
            return {"type": "PAW projector", "params": params}

        # Check for PS partial wave with optional parameters
        if "PS partial wave" in radial_type_str or radial_type_str.startswith("Ps"):
            params = {}
            # Extract parameters if present in params_str
            n_match = re.search(r"N=\s*(\d+)", params_str)
            if n_match:
                params["N"] = int(n_match.group(1))
            return {"type": "PS partial wave", "params": params}

        # Default: return as-is
        return {"type": radial_type_str, "params": {}}

    @cached_property
    def angular_types(self):
        """Extract and return orbital types from the band header line."""
        logger.debug("Extracting angular types from PROJCAR")
        
        # Find the first band header line (contains orbital names)
        header_pattern = re.compile(r"^\s*band\s+(.+)$", re.MULTILINE)
        header_match = header_pattern.search(self.file_str)
        
        if not header_match:
            raise ValueError("No band header line with orbital types found")
        
        # Extract orbital names from the header
        orbitals_str = header_match.group(1).strip()
        orbital_names = orbitals_str.split()
        
        logger.debug(f"Extracted {len(orbital_names)} orbital types: {orbital_names}")
        return orbital_names
    
    @cached_property
    def n_orbitals(self):
        """Return number of orbitals."""
        return len(self.angular_types)

    @cached_property
    def n_k(self):
        """Return number of k-points."""
        logger.debug("Determining number of k-points")
        
        kpoint_pattern = re.compile(r"k-point:\s*(\d+)\s+spin:\s*(\d+)", re.MULTILINE)
        kpoint_matches = kpoint_pattern.findall(self.file_str)
        
        if not kpoint_matches:
            raise ValueError("No k-point/spin headers found in PROJCAR file")
        
        unique_kpoints = sorted(set(int(m[0]) for m in kpoint_matches))
        n_k = len(unique_kpoints)
        logger.debug(f"Number of k-points: {n_k}")
        return n_k
    
    @cached_property
    def n_spins(self):
        """Return number of spins."""
        logger.debug("Determining number of spins")
        
        kpoint_pattern = re.compile(r"k-point:\s*(\d+)\s+spin:\s*(\d+)", re.MULTILINE)
        kpoint_matches = kpoint_pattern.findall(self.file_str)
        
        if not kpoint_matches:
            raise ValueError("No k-point/spin headers found in PROJCAR file")
        
        unique_spins = sorted(set(int(m[1]) for m in kpoint_matches))
        n_spins = len(unique_spins)
        logger.debug(f"Number of spins: {n_spins}")
        return n_spins
    
    @cached_property
    def n_bands(self):
        """Return number of bands."""
        logger.debug("Determining number of bands")
        
        # Find band numbers within k-point sections only
        # Split by k-point headers and look for band data lines
        kpoint_sections = re.split(r"k-point:\s*\d+\s+spin:\s*\d+", self.file_str)
        
        band_numbers = set()
        for section in kpoint_sections[1:]:  # Skip first section (before any k-point)
            # Find lines that start with whitespace followed by a number
            # and contain numerical data (projections)
            lines = section.split("\n")
            for line in lines:
                # Skip header lines
                if "band" in line.lower() or "ISITE" in line or not line.strip():
                    continue
                
                # Check if line starts with a band number
                match = re.match(r"^\s*(\d+)\s+([-\d.\s]+)$", line)
                if match:
                    band_num = int(match.group(1))
                    band_numbers.add(band_num)
        
        if not band_numbers:
            raise ValueError("No band data found")
        
        n_bands = max(band_numbers)
        logger.debug(f"Number of bands: {n_bands}")
        return n_bands
    
    @cached_property
    def projections(self):
        """Parse and return projection data."""
        logger.debug("Parsing PROJCAR projections")
        logger.debug(
            f"Dimensions: n_k={self.n_k}, n_bands={self.n_bands}, "
            f"n_spins={self.n_spins}, n_atoms={self.n_atoms}, "
            f"n_orbitals={self.n_orbitals}"
        )
        
        # Initialize projections array
        projections = np.zeros(
            (self.n_k, self.n_bands, self.n_spins, self.n_atoms, self.n_orbitals),
            dtype=np_utils.COMPLEX_DTYPE,
        )
        
        # Parse projection values
        self._parse_projection_values(projections)
        
        logger.debug(f"Projections array shape: {projections.shape}")
        return projections


    def _parse_projection_values(self, projections: np.ndarray) -> None:
        """
        Parse actual projection values (real and imaginary parts).

        The data follows this structure:
        ISITE: <atom_num> ...
        k-point: <k> spin: <s>
        band    <orbitals>
        <band_num> <values>
        
        Parameters
        ----------
        projections : np.ndarray
            Array to fill with projection values
        """
        # Split file into sections by ISITE
        isite_pattern = re.compile(r"ISITE:\s*(\d+)", re.MULTILINE)
        isite_positions = [(m.start(), int(m.group(1))) for m in isite_pattern.finditer(self.file_str)]
        
        # Process each ISITE section (atom)
        for i, (start_pos, isite_num) in enumerate(isite_positions):
            # Determine end position (start of next ISITE or end of file)
            if i < len(isite_positions) - 1:
                end_pos = isite_positions[i + 1][0]
            else:
                end_pos = len(self.file_str)
            
            atom_section = self.file_str[start_pos:end_pos]
            atom_idx = isite_num - 1  # 0-indexed
            
            if atom_idx >= self.n_atoms:
                logger.warning(f"ISITE {isite_num} exceeds n_atoms={self.n_atoms}, skipping")
                continue
            
            # Find all k-point sections within this atom section
            kpoint_pattern = re.compile(r"k-point:\s*(\d+)\s+spin:\s*(\d+)", re.MULTILINE)
            kpoint_positions = [(m.start(), int(m.group(1)), int(m.group(2))) 
                                 for m in kpoint_pattern.finditer(atom_section)]
            
            for j, (kp_start, kpoint_num, spin_num) in enumerate(kpoint_positions):
                # Determine end of this k-point section
                if j < len(kpoint_positions) - 1:
                    kp_end = kpoint_positions[j + 1][0]
                else:
                    kp_end = len(atom_section)
                
                kpoint_section = atom_section[kp_start:kp_end]
                k_idx = kpoint_num - 1  # 0-indexed
                spin_idx = spin_num - 1  # 0-indexed
                
                # Parse band data lines
                self._parse_kpoint_section(kpoint_section, k_idx, spin_idx, atom_idx, projections)

    def _parse_kpoint_section(
        self, kpoint_section: str, k_idx: int, spin_idx: int, atom_idx: int, projections: np.ndarray
    ) -> None:
        """
        Parse a k-point section for a single atom.

        Format:
        k-point: <n> spin: <n>
        band    <orbitals>
        <band_num> <values>
        ...
        
        Parameters
        ----------
        kpoint_section : str
            Section string to parse
        k_idx : int
            k-point index
        spin_idx : int
            Spin index
        atom_idx : int
            Atom index
        projections : np.ndarray
            Array to fill with projection values
        """
        if k_idx >= self.n_k or spin_idx >= self.n_spins or atom_idx >= self.n_atoms:
            logger.warning(
                f"Index out of bounds: k={k_idx}, spin={spin_idx}, atom={atom_idx}"
            )
            return

        # Find band data lines (lines starting with a number)
        lines = kpoint_section.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip header lines
            if "k-point" in line or "band" in line.lower():
                continue

            # Check if line starts with a band number
            match = re.match(r"^\s*(\d+)\s+([-\d.\s]+)$", line)
            if match:
                band_num = int(match.group(1))
                values_str = match.group(2).strip()
                
                band_idx = band_num - 1  # 0-indexed
                
                if band_idx >= self.n_bands:
                    continue
                
                # Extract all numbers from values string
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", values_str)
                
                if not numbers:
                    continue
                
                # Determine if we have complex (real, imag pairs) or real only
                n_values = len(numbers)
                
                if n_values >= self.n_orbitals * 2:
                    # Complex values (real, imag pairs)
                    for orb_idx in range(self.n_orbitals):
                        idx = orb_idx * 2
                        if idx + 1 < len(numbers):
                            real_val = float(numbers[idx])
                            imag_val = float(numbers[idx + 1])
                            projections[
                                k_idx, band_idx, spin_idx, atom_idx, orb_idx
                            ] = complex(real_val, imag_val)
                elif n_values >= self.n_orbitals:
                    # Real values only
                    for orb_idx in range(self.n_orbitals):
                        if orb_idx < len(numbers):
                            real_val = float(numbers[orb_idx])
                            projections[
                                k_idx, band_idx, spin_idx, atom_idx, orb_idx
                            ] = complex(real_val, 0.0)

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
        """Check if key exists as an attribute."""
        return hasattr(self, key)

    def __getitem__(self, key):
        """Get attribute value."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        """Iterate over public attributes."""
        # Return all public attributes (not starting with _)
        for key in dir(self):
            if not key.startswith("_") and not callable(getattr(self, key)):
                yield key

    def __len__(self):
        """Return number of public attributes."""
        return sum(1 for _ in self)

