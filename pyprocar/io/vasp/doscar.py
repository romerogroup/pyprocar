import logging
from functools import cached_property
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Doscar:
    """
    Parser for the VASP DOSCAR file.
    Extracts:
      - Number of atoms
      - Total DOS (single or spin-pol.)
      - Projected DOS (if present)
      - Metadata (Efermi, volume, etc.)
    """

    def __init__(self, filepath: Path | str):
        self._filepath = Path(filepath)
        self._lines = self._readlines()

    def _readlines(self):
        with open(self._filepath, "r") as f:
            return f.readlines()

    @cached_property
    def lines(self):
        return self._lines

    # ---------------------------
    # === HEADER SECTION ===
    # ---------------------------

    @cached_property
    def header_line1(self):
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
    def header_line6(self):
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
    def natoms(self): return self.header_line1["natoms"]

    @cached_property
    def nedos(self): return self.header_line6["nedos"]

    @cached_property
    def efermi(self): return self.header_line6["efermi"]

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
        total_dos = None
        if self.is_spin_pol:
            total_dos = self._raw_total_dos[:, 1:3]
        else:
            total_dos = self._raw_total_dos[:, 1:2]
        return total_dos
    
    @cached_property
    def integrated_dos(self) -> np.ndarray:
        if self.is_spin_pol:
            return self._raw_total_dos[:, 3:]
        else:
            return self._raw_total_dos[:, 2:]
    
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
        start = 6 + self.nedos
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
        pdos = np.zeros((self.nedos, n_spins, self.natoms, n_orbitals))

        idx = start
        for atom in range(self.natoms):
            idx += 1  # skip atom header line
            block = self.lines[idx : idx + self.nedos]
            arr = np.array([list(map(float, line.split())) for line in block])

            energies = arr[:, 0]  # first column is energy
            cols = arr[:, 1:]     # rest are projections

            if n_spins == 1:
                pdos[:, 0, atom, :] = cols
            else:
                # reshape (NEDOS, n_spins, n_orbitals)
                reshaped = cols.reshape(self.nedos, n_orbitals, n_spins)
                reshaped = np.transpose(reshaped, (0, 2, 1))
                pdos[:, :, atom, :] = reshaped

            idx += self.nedos

        return pdos