"""FERMI.OUT parser for Elk calculations."""

from functools import cached_property
from pathlib import Path
from typing import Self

HARTREE_TO_EV = 27.211386245988


class ElkFermi:
    """Parser for Elk FERMI.OUT file.

    FERMI.OUT contains a single line with the Fermi energy in Hartree.

    Parameters
    ----------
    filepath : Path | None
        Path to FERMI.OUT file
    file_str : str
        Content of FERMI.OUT file (alternative to filepath)

    Examples
    --------
    >>> fermi = ElkFermi(Path("FERMI.OUT"))
    >>> fermi.fermi_ev
    -2.345

    >>> fermi = ElkFermi.from_str("   -0.08625742\\n")
    >>> fermi.fermi_ev
    -2.345...
    """

    def __init__(self, filepath: Path | None = None, file_str: str = ""):
        self._filepath: Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, content: str) -> Self:
        """Create parser from file content string."""
        return cls(file_str=content)

    @property
    def filepath(self) -> Path | None:
        """Return filepath if set."""
        if self._filepath is None:
            return None
        return Path(self._filepath)

    @cached_property
    def file_str(self) -> str:
        """Lazily load file content."""
        if self._file_str == "" and self._filepath is not None:
            return Path(self._filepath).read_text()
        elif self._file_str == "" and self._filepath is None:
            raise ValueError("No filepath or file_str provided")
        return self._file_str

    @cached_property
    def fermi_hartree(self) -> float:
        """Fermi energy in Hartree (native Elk units)."""
        return float(self.file_str.split()[0])

    @cached_property
    def fermi_ev(self) -> float:
        """Fermi energy in eV."""
        return self.fermi_hartree * HARTREE_TO_EV
