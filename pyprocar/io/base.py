from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pyprocar.core import (
        DensityOfStates,
        ElectronicBandStructure,
        KPath,
        Structure,
    )


class BaseParser(ABC):
    """Base class for all DFT code parsers."""

    dirpath: Path

    def __init__(self, dirpath: str | Path) -> None:
        self.dirpath = Path(dirpath).resolve()

    @property
    @abstractmethod
    def ebs(self) -> ElectronicBandStructure | None:
        """Return the electronic band structure, or None if not available."""
        ...

    @property
    @abstractmethod
    def dos(self) -> DensityOfStates | None:
        """Return the density of states, or None if not available."""
        ...

    @property
    @abstractmethod
    def structure(self) -> Structure | None:
        """Return the crystal structure, or None if not available."""
        ...

    @property
    @abstractmethod
    def kpath(self) -> KPath | None:
        """Return the k-point path, or None if not available."""
        ...

    @property
    def version(self) -> str | None:
        """Return the code version string, or None if not available."""
        return None

    @property
    def version_tuple(self) -> tuple[int, ...] | None:
        """Return the code version as a tuple, or None if not available."""
        return None

    @property
    def reciprocal_lattice(self) -> npt.NDArray[np.float64] | None:
        """Return the reciprocal lattice, or None if not available."""
        return None
