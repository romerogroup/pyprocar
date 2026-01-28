"""Elk DFT code parser orchestrator."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from pyprocar.core import DensityOfStates, Structure
from pyprocar.core.ebs import ElectronicBandStructure, get_ebs_from_data
from pyprocar.core.kpoints import KPath
from pyprocar.io.base import BaseParser
from pyprocar.io.elk.bands import ElkBands
from pyprocar.io.elk.dos import ElkDOS
from pyprocar.io.elk.elkin import ElkIn
from pyprocar.io.elk.fermi import ElkFermi
from pyprocar.io.elk.geometry import ElkGeometry
from pyprocar.io.elk.projections import ElkProjections

logger = logging.getLogger(__name__)

ORBITAL_NAMES = [
    "Y00",
    "Y1-1",
    "Y10",
    "Y11",
    "Y2-2",
    "Y2-1",
    "Y20",
    "Y21",
    "Y22",
    "Y3-3",
    "Y3-2",
    "Y3-1",
    "Y30",
    "Y3-1",
    "Y3-2",
]


class ElkParser(BaseParser):
    """Parser for Elk DFT calculation outputs.

    Combines individual file extractors to produce canonical pyprocar objects
    (ElectronicBandStructure, DensityOfStates, Structure, KPath).

    Parameters
    ----------
    dirpath : str | Path
        Directory containing Elk calculation outputs
    elkin : str | ElkIn | None
        Path to elk.in file or pre-initialized ElkIn instance
    fermi : str | ElkFermi | None
        Path to FERMI.OUT file or pre-initialized ElkFermi instance
    geometry : str | ElkGeometry | None
        Path to GEOMETRY.OUT file or pre-initialized ElkGeometry instance
    kdirect : bool
        If True, return k-points in direct coordinates; if False, in Cartesian

    Examples
    --------
    >>> parser = ElkParser("/path/to/elk/calc")
    >>> ebs = parser.ebs
    >>> dos = parser.dos
    """

    def __init__(
        self,
        dirpath: str | Path,
        elkin: str | ElkIn | None = "elk.in",
        fermi: str | ElkFermi | None = "FERMI.OUT",
        geometry: str | ElkGeometry | None = "GEOMETRY.OUT",
        kdirect: bool = True,
    ):
        super().__init__(dirpath)
        self._kdirect: bool = kdirect

        # Initialize individual parsers
        self._elkin: ElkIn | None = self._init_elkin(elkin)
        self._fermi_parser: ElkFermi | None = self._init_fermi(fermi)
        self._geometry: ElkGeometry | None = self._init_geometry(geometry)

    def _init_elkin(self, param: str | ElkIn | None) -> ElkIn | None:
        """Initialize ElkIn parser."""
        if param is None:
            return None
        if isinstance(param, ElkIn):
            return param
        filepath = self.dirpath / Path(param)
        if filepath.exists():
            return ElkIn(filepath)
        return None

    def _init_fermi(self, param: str | ElkFermi | None) -> ElkFermi | None:
        """Initialize ElkFermi parser."""
        if param is None:
            return None
        if isinstance(param, ElkFermi):
            return param
        filepath = self.dirpath / Path(param)
        if filepath.exists():
            return ElkFermi(filepath)
        # Try lowercase (original code uses "fermi.OUT")
        filepath_lower = self.dirpath / "fermi.OUT"
        if filepath_lower.exists():
            return ElkFermi(filepath_lower)
        return None

    def _init_geometry(self, param: str | ElkGeometry | None) -> ElkGeometry | None:
        """Initialize ElkGeometry parser."""
        if param is None:
            return None
        if isinstance(param, ElkGeometry):
            return param
        filepath = self.dirpath / Path(param)
        if filepath.exists():
            return ElkGeometry(filepath)
        return None

    # Convenience accessors for extractors

    @property
    def elkin(self) -> ElkIn | None:
        """Elk input file parser."""
        return self._elkin

    @property
    def fermi_parser(self) -> ElkFermi | None:
        """Fermi energy parser."""
        return self._fermi_parser

    @property
    def geometry_parser(self) -> ElkGeometry | None:
        """Geometry output parser."""
        return self._geometry

    # Derived properties

    @cached_property
    def fermi(self) -> float:
        """Fermi energy in eV."""
        if self._fermi_parser is None:
            raise ValueError("No FERMI.OUT file found")
        return self._fermi_parser.fermi_ev

    @cached_property
    def nspin(self) -> int:
        """Number of spin channels."""
        if self._elkin is None:
            return 1
        return self._elkin.nspin

    @cached_property
    def is_bands_calculation(self) -> bool:
        """Check if this is a band structure calculation."""
        if self._elkin is None:
            return False
        return self._elkin.is_bands_calculation

    @cached_property
    def composition(self) -> dict[str, int]:
        """Species composition dictionary."""
        if self._elkin is None:
            return {}
        return self._elkin.composition

    @cached_property
    def natoms(self) -> int:
        """Number of atoms."""
        if self._geometry is not None:
            return self._geometry.natoms
        if self._elkin is not None:
            return self._elkin.natoms
        return 0

    @property
    @override
    def reciprocal_lattice(self) -> npt.NDArray[np.float64] | None:
        """Reciprocal lattice vectors."""
        try:
            lattice = self._get_lattice()
            result: npt.NDArray[np.float64] = 2 * np.pi * np.linalg.inv(lattice).T
            return result
        except ValueError:
            return None

    @property
    def reclat(self) -> np.ndarray | None:
        """Alias for reciprocal_lattice (for compatibility)."""
        return self.reciprocal_lattice

    def _get_lattice(self) -> np.ndarray:
        """Get lattice from geometry or elkin."""
        if self._geometry is not None:
            return self._geometry.lattice
        if self._elkin is not None:
            return self._elkin.lattice
        raise ValueError("No lattice information available")

    # Bands parser (lazy initialization)

    @cached_property
    def _bands_parser(self) -> ElkBands | None:
        """Bands file parser."""
        if not self.is_bands_calculation or self._elkin is None:
            return None

        bands_path = self.dirpath / "BANDS.OUT"
        bandlines_path = self.dirpath / "BANDLINES.OUT"

        if not bands_path.exists() or not bandlines_path.exists():
            return None

        return ElkBands(
            bands_filepath=bands_path,
            bandlines_filepath=bandlines_path,
            nkpoints=self._elkin.nkpoints,
            nspin=self.nspin,
            high_symmetry_points=self._elkin.high_symmetry_points,
        )

    # Projections parser (lazy initialization)

    @cached_property
    def _projections_parser(self) -> ElkProjections | None:
        """Projections file parser."""
        if not self.is_bands_calculation or self._elkin is None:
            return None

        if self._bands_parser is None:
            return None

        # Find all BAND_S*_A*.OUT files
        filepaths: list[Path] = []
        ispc = 1
        for _spc_name, count in self.composition.items():
            for iatom in range(count):
                filepath = self.dirpath / f"BAND_S{ispc:02d}_A{iatom + 1:04d}.OUT"
                if filepath.exists():
                    filepaths.append(filepath)
            ispc += 1

        if not filepaths:
            return None

        return ElkProjections(
            filepaths=filepaths,
            nkpoints=self._bands_parser.nkpoints,
            nbands=self._bands_parser.nbands,
            nspin=self.nspin,
            natoms=len(filepaths),
        )

    # DOS parser (lazy initialization)

    @cached_property
    def _dos_parser(self) -> ElkDOS | None:
        """DOS file parser."""
        tdos_path = self.dirpath / "TDOS.OUT"
        if not tdos_path.exists():
            return None
        return ElkDOS(dirpath=self.dirpath)

    # Structure property

    @cached_property
    def _structure(self) -> Structure | None:
        """Crystal structure (cached)."""
        if self._geometry is not None:
            return Structure(
                atoms=self._geometry.atoms,
                lattice=self._geometry.lattice,
                fractional_coordinates=self._geometry.fractional_coordinates,
            )
        if self._elkin is not None:
            return Structure(
                atoms=self._elkin.atoms,
                lattice=self._elkin.lattice,
                fractional_coordinates=self._elkin.fractional_coordinates,
            )
        return None

    @property
    @override
    def structure(self) -> Structure | None:
        """Crystal structure."""
        return self._structure

    # KPath property

    @cached_property
    def _kpath(self) -> KPath | None:
        """K-point path for band structure (cached)."""
        if self._bands_parser is None or self._elkin is None:
            return None

        # Convert ngrids to list[int]
        n_grids = self._bands_parser.ngrids.tolist()

        # Convert knames from list[list[str]] to list[tuple[str, str]]
        segment_names = [(pair[0], pair[1]) for pair in self._elkin.knames]

        return KPath(
            kpoints=self._bands_parser.kpoints,
            n_grids=n_grids,
            segment_names=segment_names,
            reciprocal_lattice=self.reciprocal_lattice,
        )

    @property
    @override
    def kpath(self) -> KPath | None:
        """K-point path for band structure."""
        return self._kpath

    # EBS property

    @cached_property
    def _ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure (cached)."""
        if self._bands_parser is None:
            return None

        # kpath should be available when bands are available
        kpath = self._kpath
        if kpath is None:
            return None

        # Apply Fermi energy shift to bands
        bands = self._bands_parser.bands + self.fermi

        # Get projections if available
        projected = None
        if self._projections_parser is not None:
            projected = self._projections_parser.projected

        # Transform k-points to Cartesian if requested
        kpoints = self._bands_parser.kpoints
        reclat = self.reciprocal_lattice
        if not self._kdirect and reclat is not None:
            kpoints = np.dot(kpoints, reclat)

        return get_ebs_from_data(
            kpoints=kpoints,
            bands=bands,
            projected=projected,
            projected_phase=None,
            fermi=self.fermi,
            reciprocal_lattice=self.reciprocal_lattice,
            orbital_names=ORBITAL_NAMES,
            structure=self._structure,
            kpath=kpath,
        )

    @property
    @override
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure."""
        return self._ebs

    # DOS property

    @cached_property
    def _dos(self) -> DensityOfStates | None:
        """Density of states (cached)."""
        if self._dos_parser is None or not self._dos_parser.has_dos:
            return None

        energies = self._dos_parser.energies
        total = self._dos_parser.total
        if energies is None or total is None:
            return None

        # Apply Fermi energy shift
        energies = energies + self.fermi

        return DensityOfStates(
            energies=energies,
            total=total,
            fermi=self.fermi,
            projected=self._dos_parser.projected,
        )

    @property
    @override
    def dos(self) -> DensityOfStates | None:
        """Density of states."""
        return self._dos
