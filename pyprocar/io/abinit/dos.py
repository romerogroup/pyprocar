"""Abinit DOS file parser."""

from __future__ import annotations

import logging
import re
from functools import cached_property
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AbinitDOS:
    """Parse Abinit DOS files (abinito_DOS_TOTAL* and abinito_DOS_AT*)."""

    _dirpath: Path

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
    def _total_dos_data(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Parse total DOS file and return (dos, energies, fermi)."""
        if self.total_dos_filepath is None:
            raise FileNotFoundError("No total DOS file found")

        with open(self.total_dos_filepath) as f:
            text_lines = f.readlines()
            header_text = "".join(text_lines[:13])
            dos_text = text_lines[13:]

        nsppol = int(re.findall(r"nsppol\s=\s(\d)", header_text)[0])
        fermi = float(re.findall(r"Fermi energy\s:\s*([-\d*.]*)", header_text)[0])

        energy_details = re.findall(
            r"between\s*([-\d*.]*)\s*and\s*([-\d*.]*)\s*Hartree\s*by\s*steps\s*of\s*([-\d*.]*)",
            header_text,
        )[0]
        e_min, e_max, e_step = [float(x) for x in energy_details]
        energies = np.arange(e_min, e_max + e_step, e_step)
        n_energies = energies.shape[0]

        if nsppol == 2:
            # Spin-polarized
            n_spin_header = 3
            n_up_start = n_spin_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]

            n_block_spacing = 3
            n_down_start = n_spin_header + n_energies + n_block_spacing
            n_down_end = n_down_start + n_energies
            dos_down = dos_text[n_down_start:n_down_end]

            dos_down = np.array([[float(v) for v in line.split()] for line in dos_down])[:, 1]
            dos_up = np.array([[float(v) for v in line.split()] for line in dos_up])[:, 1]
            dos_total = np.vstack([dos_up, dos_down])
        else:
            # Non-spin-polarized
            n_header = 2
            n_up_start = n_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]
            dos_up = np.array([[float(v) for v in line.split()] for line in dos_up])[:, 1]
            dos_total = dos_up[None, :]

        # Shift energies by fermi
        energies -= fermi
        return dos_total, energies, fermi

    @cached_property
    def dos_total(self) -> np.ndarray:
        """Total DOS array with shape (n_energies, n_spin)."""
        dos = self._total_dos_data[0]
        # Transpose from (n_spin, n_energies) to (n_energies, n_spin)
        return dos.T

    @cached_property
    def energies(self) -> np.ndarray:
        """Energy grid (shifted by Fermi energy)."""
        return self._total_dos_data[1]

    @cached_property
    def fermi(self) -> float:
        """Fermi energy in Hartree."""
        return self._total_dos_data[2]

    def _parse_projected_dos_file(self, filepath: Path) -> tuple[np.ndarray, int]:
        """Parse a single projected DOS file."""
        with open(filepath) as f:
            text_lines = f.readlines()
            header_text = "".join(text_lines[:13])
            dos_text = text_lines[13:]

        nsppol = int(re.findall(r"nsppol\s=\s(\d)", header_text)[0])
        energy_details = re.findall(
            r"between\s*([-\d*.]*)\s*and\s*([-\d*.]*)\s*Hartree\s*by\s*steps\s*of\s*([-\d*.]*)",
            header_text,
        )[0]
        e_min, e_max, e_step = [float(x) for x in energy_details]
        energies = np.arange(e_min, e_max + e_step, e_step)
        n_energies = energies.shape[0]

        atom_detail_text = "".join(dos_text[:4])
        atom_index = int(re.findall(r"iatom=\s*(\d)", atom_detail_text)[0])

        if nsppol == 2:
            n_spin_header = 7
            n_up_start = n_spin_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]

            n_block_spacing = 7
            n_down_start = n_up_end + n_block_spacing
            n_down_end = n_down_start + n_energies
            dos_down = dos_text[n_down_start:n_down_end]

            dos_down = np.array([[float(v) for v in line.split()] for line in dos_down])[:, 11:20]
            dos_up = np.array([[float(v) for v in line.split()] for line in dos_up])[:, 11:20]
            dos_atom = np.dstack([dos_up, dos_down])
        else:
            n_spin_header = 6
            n_up_start = n_spin_header
            n_up_end = n_up_start + n_energies
            dos_up = dos_text[n_up_start:n_up_end]
            dos_up = np.array([[float(v) for v in line.split()] for line in dos_up])[:, 11:20]
            dos_atom = dos_up[:, :, None]

        return dos_atom, atom_index

    @cached_property
    def projected(self) -> np.ndarray | None:
        """Projected DOS array with shape (n_energies, n_spins, n_atoms, n_orbitals)."""
        if not self.projected_dos_filepaths:
            return None

        n_atoms = len(self.projected_dos_filepaths)
        projected_list: list[np.ndarray | None] = [None] * n_atoms

        for filepath in self.projected_dos_filepaths:
            dos_atom, atom_index = self._parse_projected_dos_file(filepath)
            projected_list[atom_index - 1] = dos_atom

        projected_arr = np.array(projected_list)
        # Shape: (n_atoms, n_energies, n_orbitals, n_spins)
        # Transpose to: (n_energies, n_spins, n_atoms, n_orbitals)
        projected_arr = np.transpose(projected_arr, (1, 3, 0, 2))
        return projected_arr
