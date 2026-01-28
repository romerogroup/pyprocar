"""Lobster parser adapter."""

from __future__ import annotations

import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from typing_extensions import override

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.io.base import BaseParser
from pyprocar.io.lobster.doscar_lobster import LOBSTER_ORBITALS, DoscarLobster
from pyprocar.io.lobster.fatband import Fatband
from pyprocar.io.lobster.lobsterout import LobsterOut

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

# Conversion constant
HARTREE_TO_EV = 27.211386245988


class LobsterParser(BaseParser):
    """Parser for Lobster calculations.

    Parameters
    ----------
    dirpath : str | Path
        Directory containing Lobster output files.
    structure_parser : BaseParser | None
        External parser (VASP or QE) for structure information.
        Required for creating ElectronicBandStructure.
    scfin_filepath : str | Path | None
        Path to QE scf.in file for k-path extraction (QE mode only).
    lobsterout : str | Path | LobsterOut | None
        Path to lobsterout file or pre-created extractor.
    doscar : str | Path | DoscarLobster | None
        Path to DOSCAR.lobster file or pre-created extractor.
    """

    _structure_parser: BaseParser | None
    _scfin_filepath: str | Path | None
    _lobsterout: LobsterOut | None
    _doscar: DoscarLobster | None
    _fatbands: list[Fatband]

    def __init__(
        self,
        dirpath: str | Path,
        structure_parser: BaseParser | None = None,
        scfin_filepath: str | Path | None = None,
        lobsterout: str | Path | LobsterOut | None = "lobsterout",
        doscar: str | Path | DoscarLobster | None = "DOSCAR.lobster",
    ):
        super().__init__(dirpath)
        self._structure_parser = structure_parser
        self._scfin_filepath = scfin_filepath

        # Initialize extractors
        self._lobsterout = self._init_extractor(lobsterout, LobsterOut)
        self._doscar = self._init_extractor(doscar, DoscarLobster)
        self._fatbands = []

        # Auto-detect and load FATBAND files
        if self._lobsterout is not None:
            self._load_fatbands()

    def _init_extractor(
        self, value: str | Path | Any | None, extractor_class: type
    ) -> Any | None:
        """Initialize extractor from path, string, or existing instance."""
        if value is None:
            return None
        if isinstance(value, extractor_class):
            return value
        if isinstance(value, (str, Path)):
            filepath = self.dirpath / Path(value).name
            if filepath.exists():
                return extractor_class(filepath)
            else:
                user_logger.warning(f"{extractor_class.__name__} file not found: {filepath}")
                return None
        return None

    def _load_fatbands(self) -> None:
        """Load FATBAND files based on lobsterout info."""
        if self._lobsterout is None:
            return

        for filename in self._lobsterout.fatband_filenames:
            filepath = self.dirpath / filename
            if filepath.exists():
                self._fatbands.append(Fatband(filepath))
            else:
                logger.debug(f"FATBAND file not found: {filepath}")

    @cached_property
    def _scfin_content(self) -> str | None:
        """Content of QE scf.in file for k-path extraction."""
        if self._scfin_filepath is None:
            # Try default location
            scfin_path = self.dirpath / "scf.in"
            if not scfin_path.exists():
                return None
            self._scfin_filepath = scfin_path

        filepath = self.dirpath / Path(self._scfin_filepath).name
        if not filepath.exists():
            return None

        with open(filepath) as f:
            return f.read()

    @cached_property
    def _kpath_from_qe(self) -> tuple[KPath, list[int], list[str]] | None:
        """Extract k-path information from QE scf.in file.

        Returns (kpath, kticks, knames) or None if not available.
        """
        if self._scfin_content is None:
            return None

        try:
            # Parse K_POINTS block
            num_k_match = re.findall(r"K_POINTS.*\n([0-9]*)", self._scfin_content)
            if not num_k_match:
                return None

            num_k = int(num_k_match[0])
            pattern = r"K_POINTS.*\n\s*[0-9]*.*\n" + num_k * r"(.*)\n"
            raw_kpoints = re.findall(pattern, self._scfin_content)

            if not raw_kpoints:
                return None

            raw_kpoints = raw_kpoints[0]
            knames: list[str] = []
            kticks: list[int] = []
            ngrids: list[int] = []
            tick_index = 0

            for line in raw_kpoints:
                parts = line.split()
                if len(parts) >= 5:
                    # High-symmetry point with label
                    knames.append(parts[4].replace("!", ""))
                    kticks.append(tick_index)
                if len(parts) >= 4:
                    weight = float(parts[3])
                    ngrids.append(int(weight))
                    if weight == 0:
                        tick_index += 1

            if not knames:
                return None

            # We need kpoints from fatband to get actual coordinates
            if not self._fatbands:
                return None

            kpoints = self._fatbands[0].kpoints

            # Build segment_names as list of tuples
            n_segments = len(knames) - 1
            segment_names: list[tuple[str, str]] = []
            special_kpoint_map: dict[str, np.ndarray] = {}

            for i in range(n_segments):
                start_name = knames[i]
                end_name = knames[i + 1]
                segment_names.append((start_name, end_name))
                special_kpoint_map[start_name] = kpoints[kticks[i]]
                special_kpoint_map[end_name] = kpoints[kticks[i + 1]]

            # Only include n_grids if available
            kpath_kwargs: dict[str, Any] = {
                "kpoints": kpoints,
                "segment_names": segment_names,
                "special_kpoint_map": special_kpoint_map,
            }
            if ngrids and len(ngrids) >= n_segments:
                kpath_kwargs["n_grids"] = ngrids[:n_segments]

            kpath = KPath(**kpath_kwargs)

            return kpath, kticks, knames

        except Exception as e:
            logger.warning(f"Failed to parse k-path from scf.in: {e}")
            return None

    @property
    @override
    def structure(self) -> Structure | None:
        """Structure from external parser."""
        if self._structure_parser is not None:
            return self._structure_parser.structure
        return None

    @property
    @override
    def kpath(self) -> KPath | None:
        """K-path information (QE mode only)."""
        result = self._kpath_from_qe
        if result is not None:
            return result[0]
        return None

    @cached_property
    def _aggregated_bands(self) -> dict[str, np.ndarray] | None:
        """Aggregate band data from all FATBAND files.

        Returns dict with 'kpoints', 'bands', 'projected' arrays.
        """
        if not self._fatbands:
            return None

        # Use first fatband for dimensions
        fb0 = self._fatbands[0]
        n_kpoints = fb0.n_kpoints
        n_bands = fb0.n_bands
        n_spins = fb0.n_spins

        # Get ion list from lobsterout
        ions_list = self._lobsterout.ions_list if self._lobsterout else []
        n_ions = len(set(ions_list)) if ions_list else 1
        n_orbitals = len(LOBSTER_ORBITALS)

        # Initialize arrays
        kpoints = fb0.kpoints.copy()
        bands = fb0.bands.copy()

        # projected shape: (n_kpoints, n_bands, n_atoms, n_principals, n_orbitals, n_spins)
        projected = np.zeros((n_kpoints, n_bands, n_ions, 1, n_orbitals, n_spins))

        # Map element names to ion indices
        unique_ions = list(dict.fromkeys(ions_list))  # preserve order
        ion_index_map = {ion: i for i, ion in enumerate(unique_ions)}

        # Aggregate projections from all FATBAND files
        for fb in self._fatbands:
            element = fb.element
            orbital = fb.orbital

            # Find ion index
            if element in ion_index_map:
                iion = ion_index_map[element]
            else:
                logger.warning(f"Element {element} not found in ions list")
                continue

            # Find orbital index
            iorb = None
            for i, orb in enumerate(LOBSTER_ORBITALS):
                if orb == orbital:
                    iorb = i
                    break

            if iorb is None:
                logger.warning(f"Orbital {orbital} not in standard list")
                continue

            # Add projections
            projected[:, :, iion, 0, iorb, :] += fb.projections

        return {
            "kpoints": kpoints,
            "bands": bands,
            "projected": projected,
        }

    @property
    def fermi(self) -> float | None:
        """Fermi energy from structure parser."""
        if self._structure_parser is not None and hasattr(self._structure_parser, "fermi"):
            fermi_val = getattr(self._structure_parser, "fermi", None)
            if isinstance(fermi_val, (int, float)):
                return float(fermi_val)
        return None

    @property
    @override
    def reciprocal_lattice(self) -> np.ndarray | None:
        """Reciprocal lattice from structure parser."""
        if self._structure_parser is not None:
            struct = self._structure_parser.structure
            if struct is not None:
                return struct.reciprocal_lattice
        return None

    @property
    @override
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure from FATBAND files."""
        if self._aggregated_bands is None:
            return None

        data = self._aggregated_bands
        fermi = self.fermi if self.fermi is not None else 0.0

        # Shift bands to Fermi level
        bands_shifted = data["bands"] + fermi

        return ElectronicBandStructure(
            kpoints=data["kpoints"],
            bands=bands_shifted,
            projected=data["projected"],
            fermi=fermi,
            projected_phase=None,
            orbital_names=LOBSTER_ORBITALS[:-1],  # Exclude last orbital for compatibility
            reciprocal_lattice=self.reciprocal_lattice,
        )

    @property
    @override
    def dos(self) -> DensityOfStates | None:
        """Density of states from DOSCAR.lobster."""
        if self._doscar is None:
            return None

        try:
            total: list[np.ndarray] = []
            for ispin in range(self._doscar.n_spins):
                total.append(self._doscar.total_dos[:, ispin])

            # Format projected DOS for DensityOfStates
            projected: list[list[list[list[np.ndarray]]]] | None = None
            if self._doscar.projected_dos is not None:
                # Convert from (nedos, n_spins, n_atoms, n_orbitals) to expected format
                pdos = self._doscar.projected_dos
                n_atoms = pdos.shape[2]
                n_orbitals = pdos.shape[3]

                projected = []
                for iatom in range(n_atoms):
                    atom_data: list[list[np.ndarray]] = []
                    for iorb in range(n_orbitals):
                        spin_data: list[np.ndarray] = []
                        for ispin in range(self._doscar.n_spins):
                            spin_data.append(pdos[:, ispin, iatom, iorb])
                        atom_data.append(spin_data)
                    projected.append([atom_data])

            return DensityOfStates(
                energies=self._doscar.energies,
                total=total,
                projected=projected,
            )

        except Exception as e:
            user_logger.warning(f"Error creating DOS from DOSCAR.lobster: {e}")
            return None
