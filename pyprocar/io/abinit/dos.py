"""Abinit DOS file parser."""

import logging
import re
from functools import cached_property
from pathlib import Path

import numpy as np

from pyprocar.utils.units import HARTREE_TO_EV

logger = logging.getLogger(__name__)

# DOS_AT columns: energy, l=0..4 DOS, l=0..4 integrated DOS, then lm-resolved DOS from lm=0 0.
_LM_COLUMNS = slice(11, 20)
# Those columns are lm = 0 0, 1 -1, ..., 2 2 in Abinit's real spherical harmonics, named
# as Abinit's own PROCAR header names them.
_LM_NAMES = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]


def _read_dos_file(text: str) -> tuple[np.ndarray, float]:
    nsppol = int(re.findall(r"nsppol\s*=\s*(\d)", text)[0])
    fermi = float(re.findall(r"Fermi energy\s*:\s*(\S+)", text)[0])
    rows = np.loadtxt(text.splitlines(), comments="#", ndmin=2)
    rows = rows.reshape(nsppol, -1, rows.shape[1])
    rows[..., 0] *= HARTREE_TO_EV
    return rows, fermi * HARTREE_TO_EV


class AbinitDOS:
    """Parse Abinit DOS files (abinito_DOS_TOTAL* and abinito_DOS_AT*).

    Energies and the Fermi energy are absolute, in eV. DOS values are in
    electrons/eV.
    """

    def __init__(self, dirpath: str | Path):
        """Initialize DOS parser from directory.

        Parameters
        ----------
        dirpath : str | Path
            Directory containing DOS files
        """
        self._dirpath = Path(dirpath)

    @classmethod
    def is_file_of_type(cls, filepath: str | Path) -> bool:
        """Check if file is an Abinit DOS file."""
        filepath = Path(filepath)
        if "DOS_TOTAL" in filepath.name or "DOS_AT" in filepath.name:
            try:
                with open(filepath) as f:
                    header = f.read(100)
                    return "ABINIT" in header or "nsppol" in header
            except Exception:
                pass
        return False

    @cached_property
    def total_dos_filepath(self) -> Path | None:
        """Path to total DOS file."""
        files = list(self._dirpath.glob("abinito_DOS_TOTAL*"))
        return files[0] if files else None

    @cached_property
    def projected_dos_filepaths(self) -> list[Path]:
        """Paths to projected DOS files."""
        return list(self._dirpath.glob("abinito_DOS_AT*"))

    @cached_property
    def _total_dos_data(self) -> tuple[np.ndarray, float]:
        if self.total_dos_filepath is None:
            raise FileNotFoundError("No total DOS file found")
        return _read_dos_file(self.total_dos_filepath.read_text())

    @cached_property
    def dos_total(self) -> np.ndarray:
        """Total DOS in electrons/eV with shape (n_energies, n_spin)."""
        return self._total_dos_data[0][:, :, 1].T / HARTREE_TO_EV

    @cached_property
    def energies(self) -> np.ndarray:
        """Absolute energy grid in eV."""
        return self._total_dos_data[0][0, :, 0]

    @cached_property
    def fermi(self) -> float:
        """Fermi energy in eV."""
        return self._total_dos_data[1]

    @cached_property
    def orbital_names(self) -> list[str] | None:
        """Names of the lm columns that ``projected`` keeps."""
        return list(_LM_NAMES) if self.projected_dos_filepaths else None

    @cached_property
    def projected(self) -> np.ndarray | None:
        """Projected DOS in electrons/eV with shape (n_energies, n_spins, n_atoms, n_orbitals)."""
        if not self.projected_dos_filepaths:
            return None

        atoms: dict[int, np.ndarray] = {}
        for filepath in self.projected_dos_filepaths:
            text = filepath.read_text()
            atom_index = int(re.findall(r"iatom=\s*(\d+)", text)[0])
            atoms[atom_index] = _read_dos_file(text)[0][:, :, _LM_COLUMNS] / HARTREE_TO_EV

        return np.transpose(np.array([atoms[i] for i in sorted(atoms)]), (2, 1, 0, 3))
