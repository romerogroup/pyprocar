import logging
from functools import cached_property
from pathlib import Path

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
        dirpath: str | Path,
        incar: str | Path = "INCAR",
        outcar: str | Path = "OUTCAR",
        procar: str | Path = "PROCAR",
        kpoints: str | Path = "KPOINTS",
        poscar: str | Path = "POSCAR",
        doscar: str | Path = "DOSCAR",
        vasprun: str | Path = "vasprun.xml",
    ):
        super().__init__(dirpath)
        
        outcar_filepath = Path(outcar)
        incar_filepath = Path(incar)
        procar_filepath = Path(procar)
        kpoints_filepath = Path(kpoints)
        poscar_filepath = Path(poscar)
        vasprun_filepath = Path(vasprun)
        doscar_filepath = Path(doscar)
        
        self.incar_filepath: Path = self.dirpath / incar_filepath.name
        self.outcar_filepath: Path = self.dirpath / outcar_filepath.name
        self.procar_filepath: Path = self.dirpath / procar_filepath.name
        self.kpoints_filepath: Path = self.dirpath / kpoints_filepath.name
        self.poscar_filepath: Path = self.dirpath / poscar_filepath.name
        self.vasprun_filepath: Path = self.dirpath / vasprun_filepath.name
        self.doscar_filepath: Path = self.dirpath / doscar_filepath.name
        
        self.procar: Procar | None = None
        self.outcar: Outcar | None = None
        self.kpoints: Kpoints | None = None
        self.poscar: Poscar | None = None
        self.vasprun: VaspXML | None = None
        self.doscar: Doscar | None = None
        
        if self.outcar_filepath.exists():
            self.outcar = Outcar(self.outcar_filepath)
        if self.procar_filepath.exists():
            self.procar = Procar(self.procar_filepath)
        if self.kpoints_filepath.exists():
            self.kpoints = Kpoints(self.kpoints_filepath)
        if self.poscar_filepath.exists():
            self.poscar = Poscar(self.poscar_filepath)
        if self.vasprun_filepath.exists():
            self.vasprun = VaspXML(self.vasprun_filepath)
        if self.doscar_filepath.exists():
            self.doscar = Doscar(self.doscar_filepath)
            
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
        
        return kpoints_core.KGridInfo(
            kgrid=kgrid,
            kgrid_mode=kgrid_mode,
            kshift=k_shift
            )
        
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
        
        total_dos = total_dos[...,np.newaxis] if len(total_dos.shape) == 1 else total_dos
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
        except Exception as e:
            logger.warning("Issue with parsing the DOS. "+
                           "Either it was not found or there is an issue with the parser: {e}")
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
