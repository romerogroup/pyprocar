"""Lobster output file extractor."""

import logging
import re
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LobsterOut(Mapping[str, Any]):
    """Extractor for Lobster output files (lobsterout).

    Extracts:
    - FATBAND filenames and their element/orbital assignments
    - Ion list for projections
    - Spin polarization info

    Parameters
    ----------
    filepath : str | Path | None
        Path to lobsterout file.
    file_str : str
        Content of lobsterout file as string.
    """

    _filepath: str | Path | None
    _file_str: str

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing LobsterOut extractor for {filepath}")
        self._filepath = filepath
        self._file_str = file_str

    @classmethod
    def from_str(cls, file_str: str) -> "LobsterOut":
        """Create extractor from file content string."""
        return cls(file_str=file_str)

    @property
    def filepath(self) -> Path | None:
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        if self._file_str == "" and self.filepath is not None:
            with open(self.filepath) as f:
                self._file_str = f.read()
        elif self._file_str == "" and self.filepath is None:
            raise ValueError("No file path or file string provided")
        return self._file_str

    @cached_property
    def ions_list(self) -> list[str]:
        """List of ion/element names from FatBand calculations."""
        pattern = r"calculating FatBand for Element: (.*) Orbital.*"
        return re.findall(pattern, self.file_str)

    @cached_property
    def fatband_info(self) -> list[tuple[str, list[str]]]:
        """List of (element, [orbitals]) tuples for FATBAND files.

        Returns
        -------
        list[tuple[str, list[str]]]
            Each tuple contains (element_name, [orbital1, orbital2, ...])
        """
        pattern = r"calculating FatBand for Element: (.*) Orbital\(s\):\s*(.*)"
        matches = re.findall(pattern, self.file_str)
        result = []
        for element, orbitals_str in matches:
            orbitals = orbitals_str.split()
            result.append((element, orbitals))
        return result

    @cached_property
    def fatband_filenames(self) -> list[str]:
        """List of expected FATBAND filenames based on lobsterout info."""
        filenames = []
        for element, orbitals in self.fatband_info:
            for orbital in orbitals:
                filenames.append(f"FATBAND_{element}_{orbital}.lobster")
        return filenames

    @cached_property
    def is_spin_polarized(self) -> bool:
        """Check if calculation is spin-polarized."""
        # Look for spin-related keywords in output
        return "ISPIN" in self.file_str or "spinPolarized" in self.file_str.lower()

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(["ions_list", "fatband_info", "fatband_filenames", "is_spin_polarized"])

    def __len__(self) -> int:
        return 4
