import collections
import logging
import re
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.utils.strings import remove_comment

logger = logging.getLogger(__name__)

class Poscar(collections.abc.Mapping):
    """
    A class to parse the POSCAR file

    Parameters
    ----------
    filepath : str, optional
        The POSCAR filepath, by default "POSCAR"
    rotations : list, optional
        The rotations of the POSCAR file, by default None
    """

    def __init__(self, filepath: Union[str, Path], rotations=None):
        self.filepath = Path(filepath)
        self.atoms, self.coordinates, self.lattice = self._parse_poscar()

    def _parse_poscar(self):
        """
        Reads VASP POSCAR file-type and returns the pyprocar structure

        Parameters
        ----------
        filename : str, optional
            Path to POSCAR file. The default is 'CONTCAR'.

        Returns
        -------
        None.

        """
        with open(self.filepath, "r") as rf:
            lines = rf.readlines()

        comment = lines[0]
        self.comment = comment
        scale = float(remove_comment(lines[1]))

        lattice = np.zeros(shape=(3, 3))
        for i in range(3):
            lattice[i, :] = [float(x) for x in remove_comment(lines[i + 2]).split()[:3]]
        lattice *= scale
        if any([char.isalpha() for char in lines[5]]):
            species = [x for x in lines[5].split()]
            shift = 1
        else:
            shift = 0

            base_dir = self.filename.parent
            potcar_path = base_dir / "POTCAR"
            if potcar_path.exists():
                with open(potcar_path, "r") as rf:
                    potcar = rf.read()

                species = re.findall(
                    r"\s*PAW[PBE_\s]*([A-Z][a-z]*)[_a-z]*[0-9]*[a-zA-Z]*[0-9]*.*\s[0-9.]*",
                    potcar,
                )[::2]

        composition = [int(x) for x in remove_comment(lines[5 + shift].strip()).split()]
        atoms = []
        for i in range(len(composition)):
            for x in composition[i] * [species[i]]:
                atoms.append(x)
        natom = sum(composition)
        # if lines[6 + shift][0].lower() == "s":
        line = lines[6 + shift]
        if re.findall(r"\w+|$", line)[0].lower()[0] == "s":
            # shift = 2
            shift += 1
        match = re.findall(r"\w+|$", lines[6 + shift])[0].lower()
        if match[0] == "d":
            direct = True
        elif match[0] == "c":
            warnings.warn("Warning the POSCAR is not in Direct coordinates.")
            direct = False
        else:
            raise RuntimeError("The POSCAR is not in Direct or Cartesian coordinates.")
        coordinates = np.zeros(shape=(natom, 3))
        for i in range(natom):
            coordinates[i, :] = [float(x) for x in lines[i + 7 + shift].split()[:3]]
        return atoms, coordinates, lattice

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)
