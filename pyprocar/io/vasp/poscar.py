import logging
import re
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, override

import numpy as np

from pyprocar.utils.strings import remove_comment

logger = logging.getLogger(__name__)


class Poscar(Mapping[str, Any]):
    """
    A class to parse the POSCAR file

    Parameters
    ----------
    filepath : str, optional
        The POSCAR filepath, by default "POSCAR"
    file_str : str, optional
        The POSCAR file content as string, by default None
    """

    def __init__(self, 
                 filepath: str | Path | None = None, 
                 file_str: str = ""):
        logger.info(f"Initializing Poscar parser for {filepath}")
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str
        
    @classmethod
    def from_str(cls, input: str) -> "Poscar":
        return cls(file_str=input)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(file=self.filepath) as rf:
                self._file_str = rf.read()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str
    
    @cached_property
    def lines(self) -> list[str]:
        if self.filepath:
            with open(file=self.filepath) as rf:
                return rf.readlines()
        elif self.file_str:
            return self.file_str.splitlines()
        else:
            raise ValueError("No file path or file string provided")

    @cached_property
    def has_selective_dynamics_line(self) -> bool:
        return "Selective Dynamics" in self.file_str
    
    @cached_property
    def has_species_names_line(self) -> bool:
        return any([char.isalpha() for char in self.lines[5]])
    
    @cached_property
    def comment(self) -> str:
        return self.lines[0]
    
    @cached_property
    def scale(self) -> float:
        return float(remove_comment(self.lines[1]))
    
    @cached_property
    def lattice(self) -> np.ndarray:
        lattice = np.zeros(shape=(3, 3))
        for i in range(3):
            lattice[i, :] = [float(x) for x in remove_comment(self.lines[i + 2]).split()[:3]]
        lattice *= self.scale
        return lattice

    @cached_property
    def species(self) -> list[str]:
        if self.has_species_names_line:
            species = [x for x in self.lines[5].split()]
        elif self.filepath:
            base_dir = self.filepath.parent
            potcar_path = base_dir / "POTCAR"
            if not potcar_path.exists():
                raise FileNotFoundError(f"POTCAR file not found at {potcar_path}")
            with open(potcar_path) as rf:
                potcar = rf.read()
            species = re.findall(
                r"\s*PAW[PBE_\s]*([A-Z][a-z]*)[_a-z]*[0-9]*[a-zA-Z]*[0-9]*.*\s[0-9.]*",
                potcar,
            )[::2]
        else:
            raise ValueError("No species detected.")
        return species
    
    @property
    def composition_line_number(self) -> int:
        return 6 if self.has_species_names_line else 5
    
    @cached_property
    def composition(self) -> list[int]:
        composition:list[int] = []
        
        composition_line = self.lines[self.composition_line_number]
        for n_specie in composition_line.split():
            composition.append(int(n_specie))
        return composition
      
    @cached_property
    def atoms(self) -> list[str]:
        atoms:list[str] = []
        for i in range(len(self.composition)):
            for x in self.composition[i] * [self.species[i]]:
                atoms.append(x)
        return atoms
    
    @cached_property
    def n_atoms(self) -> int:
        return sum(self.composition)
    
    @cached_property
    def coord_system(self) -> str:
        line = remove_comment(self.lines[self.coord_system_line_number]).strip().lower()
        if line == "direct":
            return "direct"
        if line == "cartesian":
            return "cartesian"
        raise ValueError("Invalid coordinate system line")
    
    @cached_property
    def coord_system_line_number(self) -> int:
        coord_system_line_number = None
        for i, line in enumerate(self.lines):
            tmp_str = remove_comment(line).strip().lower()
            if tmp_str in ("direct", "cartesian"):
                coord_system_line_number = i
                break
        if coord_system_line_number is None:
            raise ValueError("Coordinate system line not found")
        return coord_system_line_number
    
    @cached_property
    def ion_positions(self) -> np.ndarray:
        # Use regular expression to parse ion position lines following the coord_system line
        coordinates = np.zeros(shape=(self.n_atoms, 3))
        for i in range(self.n_atoms):
            ion_position_line = self.lines[i + self.coord_system_line_number + 1]
            coordinates[i, :] = [float(x) for x in ion_position_line.split()[:3]]
        return coordinates
    
    @property
    def coordinates(self) -> np.ndarray:
        return self.ion_positions
        
    @override
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)
    
    @override
    def __iter__(self) -> Iterator[str]: 
        return iter(self.atoms)
    
    @override
    def __len__(self) -> int: 
        return len(self.atoms)

