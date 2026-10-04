"""DOS parser for Elk calculations (TDOS.OUT and PDOS_*.OUT)."""

import re
from functools import cached_property
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

from pyprocar.io.elk.projections import lm_names
from pyprocar.utils.units import HARTREE_TO_EV


def parse_dos_block(
    dos_block: str,
) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    """Parse a DOS block from elk output file.

    Returns (energies, dos) arrays or (None, None) if block is empty.
    """
    data = np.array([x.split() for x in dos_block.splitlines() if x.strip()]).astype(
        float
    )
    if len(data) == 0:
        return None, None
    return data[:, 0], data[:, 1]


class ElkDOS:
    """Parser for Elk DOS files (TDOS.OUT and PDOS_*.OUT).

    TDOS.OUT contains total density of states.
    PDOS_S{species}_A{atom}.OUT files contain projected DOS per atom.

    Parameters
    ----------
    dirpath : Path | None
        Directory containing DOS files
    tdos_str : str
        Content of TDOS.OUT (alternative to dirpath)
    pdos_strs : dict
        Dictionary mapping (species, atom) -> content for PDOS files
    """

    N_ORBITALS: int = 16

    def __init__(
        self,
        dirpath: Path | None = None,
        tdos_str: str = "",
        pdos_strs: dict[tuple[str, str], str] | None = None,
    ):
        self._dirpath: Path | None = dirpath
        self._tdos_str: str = tdos_str
        self._pdos_strs: dict[tuple[str, str], str] = pdos_strs or {}

    @classmethod
    def from_str(
        cls,
        tdos_content: str,
        pdos_contents: dict[tuple[str, str], str] | None = None,
    ) -> Self:
        """Create parser from file content strings."""
        return cls(tdos_str=tdos_content, pdos_strs=pdos_contents)

    @cached_property
    def tdos_str(self) -> str:
        """Lazily load TDOS.OUT content."""
        if self._tdos_str == "" and self._dirpath is not None:
            tdos_path = Path(self._dirpath) / "TDOS.OUT"
            if tdos_path.exists():
                return tdos_path.read_text()
        return self._tdos_str

    @cached_property
    def pdos_strs(self) -> dict[tuple[str, str], str]:
        """Lazily load PDOS file contents."""
        if not self._pdos_strs and self._dirpath is not None:
            result: dict[tuple[str, str], str] = {}
            for filepath in Path(self._dirpath).iterdir():
                if "PDOS" in filepath.stem:
                    parts = filepath.stem.split("_")
                    if len(parts) >= 3:
                        spc, atm = parts[1], parts[2]
                        result[(spc, atm)] = filepath.read_text()
            return result
        return self._pdos_strs

    @cached_property
    def has_dos(self) -> bool:
        """Check if DOS data is available."""
        return bool(self.tdos_str)

    @cached_property
    def _tdos_parsed(
        self,
    ) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.float64] | None]:
        """Parse TDOS.OUT into (dos_per_spin, energies)."""
        if not self.tdos_str:
            return [], None

        blocks = re.split(r"\n\s*\n", self.tdos_str)
        tdos: list[npt.NDArray[np.float64]] = []
        energies: npt.NDArray[np.float64] | None = None

        for block in blocks:
            e, dos = parse_dos_block(block)
            if dos is not None and e is not None:
                tdos.append(dos)
                energies = e

        return tdos, energies

    @cached_property
    def energies_hartree(self) -> npt.NDArray[np.float64] | None:
        """Energy grid in Hartree."""
        return self._tdos_parsed[1]

    @cached_property
    def energies(self) -> npt.NDArray[np.float64] | None:
        """Energy grid in eV (without Fermi shift - apply separately)."""
        if self.energies_hartree is None:
            return None
        return self.energies_hartree * HARTREE_TO_EV

    @cached_property
    def nspin(self) -> int:
        """Number of spin channels."""
        return len(self._tdos_parsed[0])

    @cached_property
    def total(self) -> npt.NDArray[np.float64] | None:
        """Total DOS in states/eV as (nenergies, nspin), spin down positive."""
        tdos_list = self._tdos_parsed[0]
        if not tdos_list:
            return None
        return self._to_core_units(np.column_stack(tdos_list))

    def _to_core_units(self, dos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert states/Hartree to states/eV; Elk writes spin down (axis 1) negative."""
        dos = dos / HARTREE_TO_EV
        if self.nspin == 2:
            dos[:, 1] *= -1
        return dos

    @cached_property
    def _pdos_parsed(self) -> dict[tuple[str, str], dict[int, npt.NDArray[np.float64]]]:
        """Parse all PDOS files into {(spc, atm): {orbital_idx: dos}}."""
        result: dict[tuple[str, str], dict[int, npt.NDArray[np.float64]]] = {}

        for (spc, atm), content in self.pdos_strs.items():
            result[(spc, atm)] = {}
            blocks = [b for b in re.split(r"\n\s*\n", content) if b.strip()]

            for i, block in enumerate(blocks):
                _, dos = parse_dos_block(block)
                if dos is not None:
                    result[(spc, atm)][i] = dos

        return result

    @cached_property
    def orbital_names(self) -> list[str]:
        """Each PDOS block's (l, m) slot, in the basis Elk writes to ELMIREP.OUT."""
        return lm_names(self.N_ORBITALS)

    @cached_property
    def natoms(self) -> int:
        """Number of atoms with PDOS data."""
        return len(self.pdos_strs)

    @cached_property
    def projected(self) -> npt.NDArray[np.float64] | None:
        """Projected DOS in states/eV as (nenergies, nspin, natoms, norbitals).

        Each PDOS file holds one block per (spin, lm), spin outermost.
        """
        if not self._pdos_parsed or self.energies is None:
            return None

        projected = np.zeros(
            (len(self.energies), self.nspin, self.natoms, self.N_ORBITALS)
        )
        for iatom, key in enumerate(sorted(self._pdos_parsed)):
            for block_index, dos in self._pdos_parsed[key].items():
                ispin, iorbital = divmod(block_index, self.N_ORBITALS)
                projected[:, ispin, iatom, iorbital] = dos
        return self._to_core_units(projected)
