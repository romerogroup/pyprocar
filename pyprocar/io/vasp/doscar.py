import logging
from functools import cached_property
from pathlib import Path
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class HeaderLine1(TypedDict):
    natoms: int
    natoms_check: int
    lorbit_flag: int
    ncdij: int


class HeaderLine6(TypedDict):
    emax: float
    emin: float
    nedos: int
    efermi: float
    weight: float


class Doscar:
    """
    A class to parse the DOSCAR file

    Parameters
    ----------
    filepath : str | Path, optional
        The DOSCAR filepath, by default "DOSCAR"
    file_str : str, optional
        The DOSCAR file content as a string, by default None
    """

    def __init__(
        self,
        filepath: str | Path | None = None,
        file_str: str = "",
    ) -> None:
        logger.info("Initializing DOSCAR parser for %s", filepath)
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, input_str: str) -> "Doscar":
        return cls(file_str=input_str)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(file=self.filepath) as rf:
                file_str = rf.read()
        elif self._file_str != "":
            file_str = self._file_str
        else:
            raise ValueError("No file path or file string provided")
        return file_str

    @cached_property
    def lines(self) -> list[str]:
        return self.file_str.splitlines()

    # ---------------------------
    # === HEADER SECTION ===
    # ---------------------------

    @cached_property
    def header_line1(self) -> HeaderLine1:
        nions, nions2, lorbit_flag, ncdij = map(int, self.lines[0].split())
        return {
            "natoms": nions,
            "natoms_check": nions2,
            "lorbit_flag": lorbit_flag,
            "ncdij": ncdij,
        }

    @cached_property
    def header_line2(self):
        vals = list(map(float, self.lines[1].split()))
        return {
            "volume": vals[0],
            "a": vals[1],
            "b": vals[2],
            "c": vals[3],
            "potim": vals[4],
        }

    @cached_property
    def tebeg(self):
        return float(self.lines[2].split()[0])

    @cached_property
    def car_label(self):
        return self.lines[3].strip()

    @cached_property
    def system_name(self):
        return self.lines[4].strip()

    @cached_property
    def header_line6(self) -> HeaderLine6:
        emax, emin, nedos, efermi, weight = self.lines[5].split()
        return {
            "emax": float(emax),
            "emin": float(emin),
            "nedos": int(nedos),
            "efermi": float(efermi),
            "weight": float(weight),
        }

    # convenience properties
    @cached_property
    def natoms(self) -> int:
        return self.header_line1["natoms"]

    @cached_property
    def nedos(self) -> int:
        return self.header_line6["nedos"]

    @cached_property
    def efermi(self) -> float:
        return self.header_line6["efermi"]

    # ---------------------------
    # === TOTAL DOS ===
    # ---------------------------

    @cached_property
    def is_spin_pol(self) -> bool:
        sample_line = self.lines[6].split()
        return len(sample_line) == 5

    @cached_property
    def _raw_total_dos(self):
        """
        For ISPIN=1 → shape (NEDOS, 3) [E, DOS, IntDOS]
        For ISPIN=2 → shape (NEDOS, 5) [E, DOSup, DOSdown, IntDOSup, IntDOSdown]
        """
        start = 6
        end = 6 + self.nedos
        block = self.lines[start:end]
        return np.array([list(map(float, line.split())) for line in block])

    @cached_property
    def total_dos(self) -> np.ndarray:
        return self._raw_total_dos[:, 1:3] if self.is_spin_pol else self._raw_total_dos[:, 1:2]
    
    @cached_property
    def integrated_dos(self) -> np.ndarray:
        return self._raw_total_dos[:, 3:] if self.is_spin_pol else self._raw_total_dos[:, 2:]
    
    @cached_property
    def energies(self) -> np.ndarray:
        return self._raw_total_dos[:, 0]    

    # ---------------------------
    # === PROJECTED DOS (PDOS) ===
    # ---------------------------

    @cached_property
    def projected_dos(self) -> np.ndarray | None:
        """
        Return PDOS in shape (n_energies, n_spins, n_atoms, n_orbitals).
        Returns None if PDOS not present.
        """
        # projected DOS starts after total_dos
        nedos = self.nedos
        natoms = self.natoms
        start = 6 + nedos
        if start >= len(self.lines):
            return None

        # determine n_spins and n_orbitals by inspecting one atom block
        sample_idx = start + 1
        sample_line = self.lines[sample_idx].split()
        ncols = len(sample_line)  # includes energy column

        if self.is_spin_pol:
            n_spins = 2
            n_orbitals = (ncols - 1) // n_spins
        else:
            n_spins = 1
            n_orbitals = ncols - 1

        # Note: non-collinear case would yield (s_total, s_x, s_y, s_z, …)
        # In that case n_spins=4 and orbitals adjusted accordingly
        if ncols > 10 and (ncols - 1) % 4 == 0:  # heuristic for LNONCOLLINEAR
            n_spins = 4
            n_orbitals = (ncols - 1) // 4

        # initialize array
        pdos = np.zeros((nedos, n_spins, natoms, n_orbitals))

        idx = start
        for atom in range(natoms):
            idx += 1  # skip atom header line
            block = self.lines[idx : idx + nedos]
            arr = np.array([list(map(float, line.split())) for line in block])

            cols = arr[:, 1:]     # rest are projections

            if n_spins == 1:
                pdos[:, 0, atom, :] = cols
            else:
                # reshape (NEDOS, n_spins, n_orbitals)
                reshaped = cols.reshape(nedos, n_orbitals, n_spins)
                reshaped = np.transpose(reshaped, (0, 2, 1))
                pdos[:, :, atom, :] = reshaped

            idx += nedos

        return pdos