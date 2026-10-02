import logging
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, override

import numpy as np

from pyprocar.utils.strings import remove_comment

logger = logging.getLogger(__name__)


class Kpoints(Mapping[str, Any]):
    """
    A class to parse the KPOINTS file

    Parameters
    ----------
    filepath : str | Path, optional
        The KPOINTS filepath, by default "KPOINTS"
    file_str : str, optional
        The KPOINTS file content as a string, by default None
    """

    _MAPPING_KEYS: tuple[str, ...] = (
        "comment",
        "mode",
        "ngrids",
        "automatic",
        "kgrid",
        "kshift",
        "special_kpoints",
        "knames",
        "cartesian",
    )

    def __init__(
        self,
        filepath: str | Path | None = None,
        file_str: str = "",
    ) -> None:
        logger.info("Initializing Kpoints parser for %s", filepath)
        self._filepath: str | Path | None = filepath
        self._file_str: str = file_str

    @classmethod
    def from_str(cls, input_str: str) -> "Kpoints":
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

    @cached_property
    def comment(self) -> str:
        if not self.lines:
            raise ValueError("KPOINTS file is empty")
        return self.lines[0]

    @cached_property
    def _raw_ngrid_values(self) -> list[int]:
        if len(self.lines) < 2:
            return []
        tokens = self._split_numeric_tokens(self.lines[1])
        return [int(float(token)) for token in tokens]

    @cached_property
    def ngrids(self) -> list[int]:
        ngrids = self._raw_ngrid_values.copy()
        if self.mode == "line" and len(ngrids) == 1 and self.special_kpoints is not None:
            ngrids = [ngrids[0]] * self.special_kpoints.shape[0]
        return ngrids

    @cached_property
    def automatic(self) -> bool:
        return bool(self._raw_ngrid_values and self._raw_ngrid_values[0] == 0)

    @cached_property
    def mode(self) -> str | None:
        if len(self.lines) < 3:
            return None
        raw_mode = self._clean_line(self.lines[2]).lower()
        normalized = raw_mode.replace("-", " ").replace("_", " ")
        if normalized.startswith("monkhorst"):
            return "monkhorst-pack"
        if normalized.startswith("gamma"):
            return "gamma"
        if normalized.startswith("line"):
            return "line"
        return None

    @cached_property
    def kgrid(self) -> list[int] | None:
        if self.mode not in {"gamma", "monkhorst-pack"}:
            return None
        if len(self.lines) < 4:
            raise ValueError("KPOINTS file missing k-grid definition")
        tokens = self._split_numeric_tokens(self.lines[3])
        if len(tokens) < 3:
            raise ValueError("Invalid k-grid line in KPOINTS file")
        return [int(float(token)) for token in tokens[:3]]

    @cached_property
    def kshift(self) -> list[int] | None:
        if self.mode not in {"gamma", "monkhorst-pack"}:
            return None
        if len(self.lines) < 5:
            return [0, 0, 0]
        tokens = self._split_numeric_tokens(self.lines[4])
        if not tokens:
            return [0, 0, 0]
        return [int(float(token)) for token in tokens[:3]]

    @cached_property
    def cartesian(self) -> bool:
        if self.mode != "line":
            return False
        if len(self.lines) < 4:
            raise ValueError("Line-mode KPOINTS missing coordinate type declaration")
        coordinate_line = self._clean_line(self.lines[3]).lower()
        return coordinate_line.startswith("c")

    @cached_property
    def _line_mode_data(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.mode != "line":
            return (None, None)
        if len(self.lines) < 5:
            raise ValueError("Line-mode KPOINTS missing k-path points")

        points: list[list[float]] = []
        names: list[str] = []
        for line in self.lines[4:]:
            stripped = line.strip()
            if not stripped:
                continue
            coords_part, _, label_part = line.partition("!")
            coord_tokens = coords_part.split()
            if len(coord_tokens) < 3:
                continue
            points.append([float(token) for token in coord_tokens[:3]])
            names.append(label_part.replace("!", "").strip())

        if not points:
            return (None, None)
        if len(points) % 2 != 0:
            raise ValueError("Line-mode KPOINTS must have an even number of points")

        n_segments = len(points) // 2
        special_points = np.array(points, dtype=float).reshape(n_segments, 2, 3)
        knames = np.array(names, dtype=object).reshape(n_segments, 2)
        return special_points, knames

    @cached_property
    def special_kpoints(self) -> np.ndarray | None:
        special_points, _ = self._line_mode_data
        return special_points

    @cached_property
    def knames(self) -> np.ndarray | None:
        _, knames = self._line_mode_data
        return knames

    def _clean_line(self, line: str) -> str:
        return remove_comment(line, "!").strip()

    def _split_numeric_tokens(self, line: str) -> list[str]:
        cleaned = self._clean_line(line)
        if not cleaned:
            return []
        return cleaned.split()

    @override
    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._MAPPING_KEYS)

    @override
    def __len__(self) -> int:
        return len(self._MAPPING_KEYS)
