"""Elk DFT code parser orchestrator."""

import logging
from functools import cached_property
from itertools import pairwise
from pathlib import Path

import numpy as np

from pyprocar.core import DensityOfStates, Structure
from pyprocar.core.ebs import ElectronicBandStructure, get_ebs_from_data
from pyprocar.core.kpoints import KPath, insert_continuous_points
from pyprocar.io.base import BaseParser
from pyprocar.io.elk.bands import ElkBands
from pyprocar.io.elk.dos import ElkDOS
from pyprocar.io.elk.elkin import ElkIn
from pyprocar.io.elk.fermi import ElkFermi
from pyprocar.io.elk.geometry import ElkGeometry
from pyprocar.io.elk.projections import ElkProjections

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

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
    ):
        super().__init__(dirpath)

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
        for name in (param, "EFERMI.OUT", "fermi.OUT"):
            filepath = self.dirpath / name
            if filepath.exists():
                return ElkFermi(filepath)
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
    def fermi(self) -> float | None:
        """Fermi energy in eV."""
        if self._fermi_parser is None:
            return None
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

    @cached_property
    def reciprocal_lattice(self) -> np.ndarray | None:
        """Reciprocal lattice vectors in 1/Angstrom, without the 2*pi."""
        lattice = self._get_lattice()
        if lattice is None:
            return None
        return np.linalg.inv(lattice).T

    @property
    def reclat(self) -> np.ndarray | None:
        """Alias for reciprocal_lattice (for compatibility)."""
        return self.reciprocal_lattice

    def _get_lattice(self) -> np.ndarray | None:
        """Get lattice from geometry or elkin."""
        if self._geometry is not None:
            return self._geometry.lattice
        if self._elkin is not None:
            return self._elkin.lattice
        return None

    # Bands parser (lazy initialization)

    @cached_property
    def _bands_parser(self) -> ElkBands | None:
        """Bands file parser."""
        if not self.is_bands_calculation or self._elkin is None:
            return None

        # Elk writes BAND.OUT for task 20 and BAND_Sss_Aaaaa.OUT, with the same
        # first two columns, for tasks 21 and 22 (bandstr.f90).
        bandlines_path = self.dirpath / "BANDLINES.OUT"
        bands_path = next(
            (
                path
                for path in (self.dirpath / "BAND.OUT", self.dirpath / "BAND_S01_A0001.OUT")
                if path.exists()
            ),
            None,
        )
        if bands_path is None or not bandlines_path.exists():
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
            user_logger.warning(
                f"No GEOMETRY.OUT in {self.dirpath}; reading the structure from elk.in"
            )
            return Structure(
                atoms=self._elkin.atoms,
                lattice=self._elkin.lattice,
                fractional_coordinates=self._elkin.fractional_coordinates,
            )
        return None

    @property
    def structure(self) -> Structure | None:
        """Crystal structure."""
        return self._structure

    # KPath property

    @cached_property
    def _kpath(self) -> KPath | None:
        """K-point path for band structure (cached)."""
        if self._bands_parser is None or self._elkin is None:
            return None

        kticks = self._bands_parser.kticks
        n_grids = [end - start + 1 for start, end in pairwise(kticks)]
        segment_names = [(pair[0], pair[1]) for pair in self._elkin.knames]

        return KPath(
            kpoints=self._along_path(self._bands_parser.kpoints),
            n_grids=n_grids,
            segment_names=segment_names,
            reciprocal_lattice=self.reciprocal_lattice,
        )

    @property
    def kpath(self) -> KPath | None:
        """K-point path for band structure."""
        return self._kpath

    def _along_path(self, per_kpoint: np.ndarray) -> np.ndarray:
        """Repeat the rows at inner vertices; Elk lists each vertex once, KPath needs it twice."""
        assert self._bands_parser is not None
        return insert_continuous_points(per_kpoint, self._bands_parser.kticks)

    # EBS property

    @cached_property
    def _ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure (cached)."""
        if self._bands_parser is None or self.fermi is None:
            return None

        # kpath should be available when bands are available
        kpath = self._kpath
        if kpath is None:
            return None

        bands = self._along_path(self._bands_parser.bands + self.fermi)

        projected = None
        if self._projections_parser is not None:
            raw_projected = self._projections_parser.projected
            if raw_projected is not None:
                projected = self._along_path(raw_projected)

        return get_ebs_from_data(
            kpoints=self._along_path(self._bands_parser.kpoints),
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
    def ebs(self) -> ElectronicBandStructure | None:
        """Electronic band structure."""
        return self._ebs

    # DOS property

    @cached_property
    def _dos(self) -> DensityOfStates | None:
        """Density of states (cached)."""
        if (
            self._dos_parser is None
            or not self._dos_parser.has_dos
            or self.fermi is None
        ):
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
    def dos(self) -> DensityOfStates | None:
        """Density of states."""
        return self._dos
