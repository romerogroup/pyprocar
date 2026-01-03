import logging
from functools import cached_property
from pathlib import Path
from typing import overload

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, Structure, get_ebs_from_data
from pyprocar.core import kpoints as kpoints_core
from pyprocar.core.atomic_orbital_index import OrbitalIndexer
from pyprocar.io.base import BaseParser
from pyprocar.io.vasp.doscar import Doscar
from pyprocar.io.vasp.kpoints import Kpoints
from pyprocar.io.vasp.outcar import Outcar
from pyprocar.io.vasp.poscar import Poscar
from pyprocar.io.vasp.procar import Procar
from pyprocar.io.vasp.vasprun import VaspXML

logger = logging.getLogger(__name__)

ORBITAL_ORDERING = OrbitalIndexer()


class VaspParser(BaseParser):
    def __init__(
        self,
        dirpath: str | Path = "",
        outcar: str | Path | Outcar | None = "OUTCAR",
        procar: str | Path | Procar | None = "PROCAR",
        kpoints: str | Path | Kpoints | None = "KPOINTS",
        poscar: str | Path | Poscar | None = "POSCAR",
        doscar: str | Path | Doscar | None = "DOSCAR",
        vasprun: str | Path | VaspXML | None = "vasprun.xml",
    ):
        super().__init__(dirpath)

        # Initialize parser objects by checking if they are already parser instances or paths
        self.outcar: Outcar | None = self._initialize_parser(outcar, Outcar)
        self.procar: Procar | None = self._initialize_parser(procar, Procar)
        self.kpoints: Kpoints | None = self._initialize_parser(kpoints, Kpoints)
        self.poscar: Poscar | None = self._initialize_parser(poscar, Poscar)
        self.vasprun: VaspXML | None = self._initialize_parser(vasprun, VaspXML)
        self.doscar: Doscar | None = self._initialize_parser(doscar, Doscar)

    @overload
    def _initialize_parser(
        self, param: str | Path | Outcar | None, parser_class: type[Outcar]
    ) -> Outcar | None: ...

    @overload
    def _initialize_parser(
        self, param: str | Path | Procar | None, parser_class: type[Procar]
    ) -> Procar | None: ...

    @overload
    def _initialize_parser(
        self, param: str | Path | Kpoints | None, parser_class: type[Kpoints]
    ) -> Kpoints | None: ...

    @overload
    def _initialize_parser(
        self, param: str | Path | Poscar | None, parser_class: type[Poscar]
    ) -> Poscar | None: ...

    @overload
    def _initialize_parser(
        self, param: str | Path | VaspXML | None, parser_class: type[VaspXML]
    ) -> VaspXML | None: ...

    @overload
    def _initialize_parser(
        self, param: str | Path | Doscar | None, parser_class: type[Doscar]
    ) -> Doscar | None: ...

    def _initialize_parser(
        self,
        param: (str | Path | Outcar | Procar | Kpoints | Poscar | VaspXML | Doscar | None),
        parser_class: (
            type[Outcar]
            | type[Procar]
            | type[Kpoints]
            | type[Poscar]
            | type[VaspXML]
            | type[Doscar]
        ),
    ) -> Outcar | Procar | Kpoints | Poscar | VaspXML | Doscar | None:
        """
        Initialize a parser object from either a path or an existing parser instance.

        Parameters
        ----------
        param : str | Path | ParserType | None
            Either a file path or an already instantiated parser object
        parser_class : type
            The parser class to instantiate if param is a path

        Returns
        -------
        ParserType | None
            The parser object or None if param is None or file doesn't exist
        """
        if param is None:
            return None

        # Check if it's already a parser instance
        if isinstance(param, parser_class):
            return param

        # It's a path (str or Path), so we need to create the parser
        if not isinstance(param, (str, Path)):
            return None

        filepath = self.dirpath / Path(param) if self.dirpath else Path(param)

        if filepath.exists():
            return parser_class(filepath)

        return None

    @classmethod
    def from_str(
        cls,
        outcar: str | None = None,
        procar: str | None = None,
        kpoints: str | None = None,
        poscar: str | None = None,
        vasprun: str | None = None,
        doscar: str | None = None,
    ) -> "VaspParser":
        """
        Create a VaspParser from file content strings.

        Parameters
        ----------
        outcar : str | None
            Content of OUTCAR file
        procar : str | None
            Content of PROCAR file
        kpoints : str | None
            Content of KPOINTS file
        poscar : str | None
            Content of POSCAR file
        vasprun : str | None
            Content of vasprun.xml file
        doscar : str | None
            Content of DOSCAR file

        Returns
        -------
        VaspParser
            Parser instance with data loaded from strings
        """
        outcar_obj = Outcar.from_str(outcar) if outcar else None
        procar_obj = Procar.from_str(procar) if procar else None
        kpoints_obj = Kpoints.from_str(kpoints) if kpoints else None
        poscar_obj = Poscar.from_str(poscar) if poscar else None
        vasprun_obj = VaspXML.from_str(vasprun) if vasprun else None
        doscar_obj = Doscar.from_str(doscar) if doscar else None

        return cls(
            dirpath="",
            outcar=outcar_obj,
            procar=procar_obj,
            kpoints=kpoints_obj,
            poscar=poscar_obj,
            vasprun=vasprun_obj,
            doscar=doscar_obj,
        )

    @cached_property
    def version(self) -> str | None:
        if self.outcar:
            version = self.outcar.version
        elif self.vasprun:
            version = self.vasprun.version
        else:
            version = None
        return version

    @cached_property
    def version_tuple(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.version.split("."))

    @cached_property
    def is_spin_polarized(self) -> bool:
        if self.vasprun:
            return self.vasprun.is_spin_polarized
        return False

    @property
    def kpath(self) -> kpoints_core.KPath | None:
        if self.kpoints is None:
            return None

        kpoints = self.procar.kpoints if self.procar else None

        if self.kpoints.knames is None:
            return None

        return kpoints_core.KPath(
            kpoints=kpoints,
            segment_names=self.kpoints.knames,
            n_grids=self.kpoints.ngrids,
            reciprocal_lattice=self.outcar.reciprocal_lattice,
        )

    @property
    def kgrid_info(self) -> kpoints_core.KGridInfo | None:
        if self.kpoints is None:
            return None

        kgrid = self.kpoints.get("kgrid", None)
        kgrid_mode = self.kpoints.get("mode", None)
        k_shift = self.kpoints.get("kshift", None)

        if kgrid is None or kgrid_mode is None or k_shift is None:
            return None

        return kpoints_core.KGridInfo(kgrid=kgrid, kgrid_mode=kgrid_mode, kshift=k_shift)

    @cached_property
    def fermi(self) -> float | None:
        if self.outcar is not None:
            fermi = self.outcar.fermi
        elif self.vasprun is not None:
            fermi = self.vasprun.fermi
        else:
            fermi = None

        return fermi

    @property
    def ebs(self) -> ElectronicBandStructure | None:
        if self.procar is None:
            logger.warning(
                "Issue with procar file. Either it was not found or there is an issue with the parser"
            )
            return None
        if self.outcar is None:
            logger.warning(
                "Issue with outcar file. Either it was not found or there is an issue with the parser"
            )
            return None

        return get_ebs_from_data(
            kpoints=self.procar.kpoints,
            bands=self.procar.bands,
            projected=self.procar.projected,
            projected_phase=self.procar.projected_phase,
            fermi=self.outcar.fermi,
            reciprocal_lattice=self.outcar.reciprocal_lattice,
            orbital_names=self.orbitals,
            structure=self.structure,
            kpath=self.kpath,
            kgrid_info=self.kgrid_info,
        )

    @cached_property
    def energies(self) -> np.ndarray | None:
        if self.vasprun is not None and self.vasprun.has_dos:
            energies = self.vasprun.dos_total["energies"]

        elif self.doscar is not None and self.doscar.has_dos:
            energies = self.doscar.energies
        else:
            return None

        # if self.is_spin_polarized:
        #     energies = np.repeat(energies, 2, axis=0)
        return energies

    @cached_property
    def total_dos(self) -> np.ndarray | None:
        if self.vasprun is not None and self.vasprun.has_dos:
            total_dos = np.moveaxis(self.vasprun.total, (0), (-1))
        elif self.doscar is not None and self.doscar.has_dos:
            total_dos = self.doscar.total
        else:
            return None

        total_dos = total_dos[..., np.newaxis] if len(total_dos.shape) == 1 else total_dos
        return total_dos

    @cached_property
    def projected_dos(self) -> np.ndarray | None:
        if self.vasprun is not None and self.vasprun.has_dos:
            logger.info("Using vasprun projected dos")
            return np.moveaxis(self.vasprun.projected, (0, 1, 2, 3), (2, 3, 1, 0))
        elif self.doscar is not None and self.doscar.has_dos:
            logger.info("Using doscar projected dos")
            return self.doscar.projected_dos
        else:
            return None

    @property
    def dos(self) -> DensityOfStates | None:
        try:
            dos = DensityOfStates(
                energies=self.energies,
                total=self.total_dos,
                fermi=self.fermi,
                projected=self.projected_dos,
                orbital_names=self.orbitals,
                structure=self.structure,
            )
        except Exception:
            msg = (
                "Issue with parsing the DOS. "
                "Either it was not found or there is an issue with the parser"
            )
            logger.warning(msg)
            dos = None
        return dos

    @property
    def structure(self) -> Structure | None:
        if self.poscar is not None:
            logger.info("Using poscar structure")
            atoms = self.poscar.atoms
            fractional_coordinates = self.poscar.coordinates
            lattice = self.poscar.lattice
        elif self.vasprun is not None:
            logger.info("Using vasprun structure")
            atoms = self.vasprun.atoms
            fractional_coordinates = self.vasprun.initial_structure.positions
            lattice = self.vasprun.initial_structure.crystal.basis

            return self.vasprun
        else:
            logger.warning(
                "Issue with poscar file. Either it was not found or there is an issue with the parser"
            )
            return None

        rotations = self.outcar.rotations if self.outcar else None

        return Structure(
            atoms=atoms,
            fractional_coordinates=fractional_coordinates,
            lattice=lattice,
            rotations=rotations,
        )

    @cached_property
    def orbitals(self) -> list[str]:
        return ORBITAL_ORDERING.flat_conventional
