"""Elk DFT code parser orchestrator."""

import logging
import re
from functools import cached_property
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
from pyprocar.utils.log_utils import warn_user

# Elk 6 writes "Elk version 6.3.02 started"; later versions write "Elk code version 11.2.3".
_ELK_VERSION = re.compile(r"Elk (?:code )?version (\d+)\.(\d+)\.(\d+)")

logger = logging.getLogger(__name__)


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
        if self._elkin is None:
            return None
        tasks = set(self._elkin.tasks)
        if not self.is_bands_calculation:
            return None

        # bandstr.f90 writes BAND.OUT for task 20 and BAND_Sss_Aaaaa.OUT for tasks
        # 21-24; the first two columns (distance, energy) are the same in all of them.
        band_out = self.dirpath / "BAND.OUT"
        band_s = self.dirpath / "BAND_S01_A0001.OUT"
        preferred, other = (band_out, band_s) if 20 in tasks else (band_s, band_out)
        bands_path = preferred if preferred.exists() else other
        if band_out.exists() and band_s.exists() and not (20 in tasks and tasks & {21, 22}):
            warn_user(
                f"Both BAND.OUT and BAND_S01_A0001.OUT are in {self.dirpath};"
                + f" reading {bands_path.name}, which the elk.in tasks write"
            )
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

        if self._bands_parser is None or self._bands_parser.bands_filepath is None:
            return None
        if not self._bands_parser.bands_filepath.name.startswith("BAND_S"):
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

        # bandstr.f90 gives tasks 21-24 the same file names, so the last one in elk.in wrote them.
        character_tasks = [t for t in self._elkin.tasks if t in (21, 22, 23, 24)]
        if not character_tasks:
            warn_user(
                "elk.in lists no task 21 to 24, so the band characters in the BAND_S files"
                + " are not read"
            )
            return None
        task = character_tasks[-1]
        return ElkProjections(
            filepaths=filepaths,
            nkpoints=self._bands_parser.nkpoints,
            nbands=self._bands_parser.nbands,
            nspin=self.nspin,
            natoms=len(filepaths),
            task=task,
            irrep_basis=task == 22 and self._task_22_irrep_basis(),
        )

    def _task_22_irrep_basis(self) -> bool:
        """Whether task 22 wrote its (l,m) characters in the irreducible-representation basis.

        Elk 10.7.8 and later do so unless elk.in sets lmirep to .false. Without the version from
        INFO.OUT the basis is unknown: task 22 of those versions writes ELMIREP.OUT, but task 10
        of every version does too, so the characters keep their Ylm slot names.
        """
        if self._elkin is None or not self._elkin.lmirep:
            return False
        info = self.dirpath / "INFO.OUT"
        version = None
        if info.exists():
            with info.open(encoding="utf-8", errors="replace") as lines:
                version = next(
                    (m for line in lines if (m := re.search(_ELK_VERSION, line))), None
                )
        if version is not None:
            return tuple(int(part) for part in version.groups()) >= (10, 7, 8)
        if (self.dirpath / "ELMIREP.OUT").exists():
            missing = "INFO.OUT names no Elk version" if info.exists() else "there is no INFO.OUT"
            warn_user(
                f"{self.dirpath} has ELMIREP.OUT, which task 10 of any Elk version or task 22 of"
                + f" Elk 10.7.8 and later writes, and {missing}, so pyprocar cannot tell whether"
                + " the task-22 band characters are in the irreducible-representation basis. They"
                + " are named by their Ylm slots; add the INFO.OUT of the run to name them by the"
                + " basis Elk used."
            )
        return False

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
            warn_user(
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

        n_grids = self._bands_parser.ngrids.tolist()
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
        orbital_names = None
        if self._projections_parser is not None:
            raw_projected = self._projections_parser.projected
            if raw_projected is not None:
                projected = self._along_path(raw_projected)
                orbital_names = self._projections_parser.orbital_names

        return get_ebs_from_data(
            kpoints=self._along_path(self._bands_parser.kpoints),
            bands=bands,
            projected=projected,
            projected_phase=None,
            fermi=self.fermi,
            reciprocal_lattice=self.reciprocal_lattice,
            orbital_names=orbital_names,
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
            orbital_names=self._dos_parser.orbital_names,
            structure=self.structure,
        )

    @property
    def dos(self) -> DensityOfStates | None:
        """Density of states."""
        return self._dos
