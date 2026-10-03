"""SIESTA FDF input file extractor."""

import logging
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, override

import numpy as np

from pyprocar.utils.units import AU_TO_ANG

logger = logging.getLogger(__name__)

_LENGTH_UNITS_IN_ANGSTROM = {
    "m": 1e10,
    "cm": 1e8,
    "nm": 10.0,
    "pm": 0.01,
    "ang": 1.0,
    "angstrom": 1.0,
    "bohr": AU_TO_ANG,
}


def normalize_label(label: str) -> str:
    """FDF label as FDF compares it: case-insensitive, ignoring '.', '_' and '-'."""
    return re.sub(r"[._-]", "", label).lower()


def _token_rows(text: str) -> list[list[str]]:
    """Non-empty lines split into tokens; '#', '!' and ';' start comments."""
    rows = [re.split(r"[#!;]", line, maxsplit=1)[0].split() for line in text.splitlines()]
    return [row for row in rows if row]


@dataclass
class _ParsedFDF:
    """FDF content keyed by normalized label.

    ``errors`` holds the message for each block that could not be read, so
    only a request for that block fails. ``redirects`` names the files that
    ``%block Name < file`` lines point to.
    """

    labels: dict[str, list[str]] = field(default_factory=dict)
    blocks: dict[str, list[list[str]]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    redirects: list[str] = field(default_factory=list)


def _parse_fdf(text: str, directory: Path | None) -> _ParsedFDF:
    """Split FDF text into labels and blocks, keyed by normalized label.

    Each label maps to the tokens after it, and each block to its token rows.
    The first occurrence of a label or block wins. ``%block Name < file``
    reads the block rows from ``file``, relative to ``directory``. An
    unterminated block is recorded as an error, and its lines are read as
    labels again.
    """
    parsed = _ParsedFDF()
    open_block: tuple[str, str] | None = None
    rows: list[list[str]] = []

    def drop_open_block() -> None:
        assert open_block is not None
        parsed.errors.setdefault(open_block[0], f"%block {open_block[1]} has no %endblock")
        for row in rows:
            parsed.labels.setdefault(normalize_label(row[0]), row[1:])

    for tokens in _token_rows(text):
        keyword = tokens[0].lower()
        if open_block is not None:
            if keyword == "%endblock" and (
                len(tokens) == 1 or normalize_label(tokens[1]) == open_block[0]
            ):
                parsed.blocks.setdefault(open_block[0], rows)
                open_block = None
                continue
            if keyword not in ("%block", "%endblock"):
                rows.append(tokens)
                continue
            drop_open_block()
            open_block = None
            if keyword == "%endblock":
                continue
        if keyword == "%block" and len(tokens) > 3 and tokens[2] == "<":
            name, target = normalize_label(tokens[1]), tokens[3]
            parsed.redirects.append(target)
            if directory is None:
                parsed.errors.setdefault(name, f"%block {tokens[1]} < {target} needs a file path")
            elif not (directory / target).is_file():
                parsed.errors.setdefault(name, f"%block {tokens[1]} < {target}: file not found")
            else:
                parsed.blocks.setdefault(name, _token_rows((directory / target).read_text()))
        elif keyword == "%block" and len(tokens) > 1:
            open_block = (normalize_label(tokens[1]), tokens[1])
            rows = []
        else:
            parsed.labels.setdefault(normalize_label(tokens[0]), tokens[1:])
    if open_block is not None:
        drop_open_block()
    return parsed


class FDF(Mapping[str, Any]):
    """Extractor for SIESTA .fdf input files.

    Parses system label, lattice vectors, atomic positions, and k-path information
    from SIESTA input files.
    """

    def __init__(self, filepath: str | Path | None = None, file_str: str = ""):
        logger.info(f"Initializing FDF parser for {filepath}")
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, input: str) -> "FDF":
        """Create FDF instance from file content string."""
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
    def _parsed(self) -> _ParsedFDF:
        directory = None if self.filepath is None else self.filepath.parent
        return _parse_fdf(self.file_str, directory)

    def label(self, name: str) -> list[str] | None:
        """Tokens after an FDF label, or None when the label is absent."""
        return self._parsed.labels.get(normalize_label(name))

    def block(self, name: str) -> list[list[str]] | None:
        """Token rows of an FDF block, or None when the block is absent.

        Raises ValueError when the block is present but could not be read.
        """
        key = normalize_label(name)
        if key in self._parsed.errors:
            raise ValueError(self._parsed.errors[key])
        return self._parsed.blocks.get(key)

    @property
    def redirect_targets(self) -> list[str]:
        """Files named by ``%block Name < file`` lines."""
        return self._parsed.redirects

    @cached_property
    def system_label(self) -> str:
        """Extract SystemLabel from FDF file."""
        tokens = self.label("SystemLabel")
        if not tokens:
            raise ValueError("No SystemLabel found in FDF file")
        return tokens[0]

    @cached_property
    def lattice_constant(self) -> float:
        """LatticeConstant in Angstrom.

        Siesta refuses a value without a unit; this reader falls back to
        Bohr for one. Without the keyword, 1.0 is returned, so LatticeVectors
        are read as Angstrom.
        """
        tokens = self.label("LatticeConstant")
        if not tokens:
            return 1.0
        unit = tokens[1].lower() if len(tokens) > 1 else "bohr"
        if unit not in _LENGTH_UNITS_IN_ANGSTROM:
            raise ValueError(f"Unsupported LatticeConstant unit: {tokens[1]}")
        return float(tokens[0]) * _LENGTH_UNITS_IN_ANGSTROM[unit]

    @cached_property
    def band_lines_scale(self) -> str:
        """BandLinesScale: "pi/a" (the Siesta default) or "ReciprocalLatticeVectors"."""
        tokens = self.label("BandLinesScale")
        return tokens[0] if tokens else "pi/a"

    @cached_property
    def lattice_vectors(self) -> np.ndarray:
        """Lattice vectors in Angstrom: the LatticeVectors block times LatticeConstant."""
        rows = self.block("LatticeVectors")
        if rows is None:
            raise ValueError("No LatticeVectors block found in FDF file")
        lattice = np.array([[float(x) for x in row[:3]] for row in rows[:3]])
        return lattice * self.lattice_constant

    @cached_property
    def atomic_coords_format(self) -> str:
        """AtomicCoordinatesFormat, NotScaledCartesianBohr by default as in Siesta."""
        tokens = self.label("AtomicCoordinatesFormat")
        return tokens[0] if tokens else "NotScaledCartesianBohr"

    @cached_property
    def species_labels(self) -> dict[str, str]:
        """Extract species index to label mapping from ChemicalSpeciesLabel block."""
        rows = self.block("ChemicalSpeciesLabel")
        if rows is None:
            raise ValueError("No ChemicalSpeciesLabel block found in FDF file")
        return {row[0]: row[2] for row in rows if len(row) >= 3}

    @cached_property
    def atomic_positions(self) -> np.ndarray:
        """Positions in the AtomicCoordinatesAndAtomicSpecies block, as written."""
        rows = self.block("AtomicCoordinatesAndAtomicSpecies")
        if rows is None:
            raise ValueError("No AtomicCoordinatesAndAtomicSpecies block found")
        return np.array([[float(x) for x in row[:3]] for row in rows])

    @cached_property
    def cartesian_positions(self) -> np.ndarray:
        """Atomic positions in Cartesian Angstrom, read per AtomicCoordinatesFormat."""
        coord_format = normalize_label(self.atomic_coords_format)
        positions = self.atomic_positions
        if coord_format in ("fractional", "scaledbylatticevectors"):
            return positions @ self.lattice_vectors
        if coord_format == "scaledcartesian":
            return positions * self.lattice_constant
        if coord_format in ("ang", "notscaledcartesianang"):
            return positions
        if coord_format in ("bohr", "notscaledcartesianbohr"):
            return positions * AU_TO_ANG
        raise ValueError(f"Unsupported AtomicCoordinatesFormat: {self.atomic_coords_format}")

    @cached_property
    def atoms(self) -> list[str]:
        """Extract atom list with species labels."""
        rows = self.block("AtomicCoordinatesAndAtomicSpecies")
        if rows is None:
            return []
        return [self.species_labels.get(row[3], row[3]) for row in rows if len(row) >= 4]

    @cached_property
    def has_band_lines(self) -> bool:
        """Check if BandLines block exists."""
        return self.block("BandLines") is not None

    @cached_property
    def band_lines(self) -> list[dict] | None:
        """Parse BandLines block for k-path specification.

        Returns list of dicts with keys: npoints, kpoint, label
        """
        rows = self.block("BandLines")
        if rows is None:
            return None
        return [
            {
                "npoints": int(row[0]),
                "kpoint": [float(row[1]), float(row[2]), float(row[3])],
                "label": row[4] if len(row) > 4 else "",
            }
            for row in rows
            if len(row) >= 4
        ]

    # Mapping interface
    @override
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.atoms)

    @override
    def __len__(self) -> int:
        return len(self.atoms)
