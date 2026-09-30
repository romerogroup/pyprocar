from pathlib import Path

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure


class BaseParser:
    """Interface every DFT code adapter satisfies.

    Each member returns ``None`` when the calculation directory lacks the data
    or the adapter does not support it; unsupported data never raises.
    Energies (EBS bands, DOS energies) are absolute, not shifted by ``fermi``.
    """

    def __init__(self, dirpath: str | Path):
        self.dirpath = Path(dirpath).resolve()

    @property
    def ebs(self) -> ElectronicBandStructure | None:
        return None

    @property
    def dos(self) -> DensityOfStates | None:
        return None

    @property
    def structure(self) -> Structure | None:
        return None

    @property
    def kpath(self) -> KPath | None:
        return None

    @property
    def fermi(self) -> float | None:
        return None

    @property
    def version(self) -> str | None:
        return None

    @property
    def reciprocal_lattice(self) -> np.ndarray | None:
        return None
