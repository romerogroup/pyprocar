import collections
import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.core import DensityOfStates, Structure, get_ebs_from_data
from pyprocar.core import kpoints as kpoints_core
from pyprocar.io.base import BaseParser
from pyprocar.io.vasp.doscar import Doscar
from pyprocar.io.vasp.kpoints import Kpoints
from pyprocar.io.vasp.outcar import Outcar
from pyprocar.io.vasp.poscar import Poscar
from pyprocar.io.vasp.procar import Procar
from pyprocar.io.vasp.vasprun import VaspXML
from pyprocar.utils.info import OrbitalOrdering

logger = logging.getLogger(__name__)

ORBITAL_ORDERING = OrbitalOrdering()

class VaspParser(BaseParser):
    def __init__(
        self,
        dirpath: Union[str, Path],
        incar: Union[str, Path] = "INCAR",
        outcar: Union[str, Path] = "OUTCAR",
        procar: Union[str, Path] = "PROCAR",
        kpoints: Union[str, Path] = "KPOINTS",
        poscar: Union[str, Path] = "POSCAR",
        doscar: Union[str, Path] = "DOSCAR",
        vasprun: Union[str, Path] = "vasprun.xml",
    ):
        super().__init__(dirpath)
        
        outcar_filepath = Path(outcar)
        incar_filepath = Path(incar)
        procar_filepath = Path(procar)
        kpoints_filepath = Path(kpoints)
        poscar_filepath = Path(poscar)
        vasprun_filepath = Path(vasprun)
        doscar_filepath = Path(doscar)
        
        self.incar_filepath = self.dirpath / incar_filepath.name
        self.outcar_filepath = self.dirpath / outcar_filepath.name
        self.procar_filepath = self.dirpath / procar_filepath.name
        self.kpoints_filepath = self.dirpath / kpoints_filepath.name
        self.poscar_filepath = self.dirpath / poscar_filepath.name
        self.vasprun_filepath = self.dirpath / vasprun_filepath.name
        self.doscar_filepath = self.dirpath / doscar_filepath.name
        
        self.procar = None
        self.outcar = None
        self.kpoints = None
        self.poscar = None
        self.vasprun = None
        self.doscar = None
        
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
    def version(self):
        version = None
        if self.outcar:
            version = self.outcar.version
        elif self.vasprun:
            version = self.vasprun.version
        return version

    @cached_property
    def version_tuple(self):
        return tuple(int(x) for x in self.version.split("."))

    @property
    def kpath(self):
        if self.kpoints is None:
            return None
        
        
        kpoints = None
        if self.procar:
            kpoints = self.procar.kpoints
            
        if self.kpoints.knames is None:
            return None

        return kpoints_core.KPath(
            kpoints=kpoints,
            segment_names=self.kpoints.knames,
            n_grids=self.kpoints.ngrids,
            reciprocal_lattice=self.outcar.reciprocal_lattice,
        )
        
    @property
    def kgrid_info(self):
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
    def fermi(self):
        if self.outcar is not None:
            logger.info(f"Parsing fermi energy from outcar: {self.outcar.fermi}")
            return self.outcar.fermi
        elif self.vasprun is not None:
            logger.info(f"Parsing fermi energy from vasprun.xml: {self.vasprun.fermi}")
            return self.vasprun.fermi
        else:
            return None

    @property
    def ebs(self):
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
            projected=self.procar._spd2projected(self.procar.spd),
            projected_phase=self.procar._spd2projected(self.procar.spd_phase),
            fermi=self.outcar.fermi,
            reciprocal_lattice=self.outcar.reciprocal_lattice,
            orbital_names=self.procar.orbitalNames[:-1],
            structure=self.structure,
            kpath=self.kpath,
            kgrid_info=self.kgrid_info,
        )

    @cached_property
    def energies(self):
        if self.vasprun is not None and self.vasprun.has_dos:
            energies = self.vasprun.dos_total["energies"]
            return energies
        elif self.doscar is not None and self.doscar.has_dos:
            return self.doscar.energies
        else:
            return None
    
    @cached_property
    def total_dos(self):
        if self.vasprun is not None and self.vasprun.has_dos:
            total_dos = np.moveaxis(self.vasprun.total_dos, (0), (-1))
        elif self.doscar is not None and self.doscar.has_dos:
            total_dos = self.doscar.total_dos
        else:
            return None
        
        total_dos = total_dos[...,np.newaxis] if len(total_dos.shape) == 1 else total_dos
        return total_dos
    
    @cached_property
    def projected_dos(self):
        if self.vasprun is not None and self.vasprun.has_dos:
            return np.moveaxis(self.vasprun.dos_projected, (0, 1, 2, 3), (2, 3, 1, 0))
        elif self.doscar is not None and self.doscar.has_dos:
            return self.doscar.projected_dos
        else:
            return None
    
    @property
    def dos(self):
        if self.vasprun is not None and self.vasprun.has_dos:
            dos = DensityOfStates(
                energies=self.energies,
                total=self.total_dos,
                fermi=self.fermi,
                projected=self.projected_dos,
                orbital_names=self.orbitals,
            )
            return dos
        else:
            logger.warning(
                "Issue with parsing the DOS. Either it was not found or there is an issue with the parser"
            )
            return None
        

    @property
    def structure(self):
        if self.poscar is not None:
            logger.info("Using poscar structure")
            return Structure(
                atoms=self.poscar.atoms,
                fractional_coordinates=self.poscar.coordinates,
                lattice=self.poscar.lattice,
                rotations=self.outcar.rotations,
            )

        elif self.vasprun is not None and self.vasprun.structure is not None:
            logger.info("Using vasprun structure")
            return self.vasprun.structure
        else:
            logger.warning(
                "Issue with poscar file. Either it was not found or there is an issue with the parser"
            )
            return None
        
    @cached_property
    def orbitals(self):
        return ORBITAL_ORDERING.flat_conventional