from pathlib import Path

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure


class BaseParser:
    """Interface every DFT code adapter satisfies.

    Each member returns ``None`` when the calculation directory lacks the data
    or the adapter does not support it; unsupported data never raises.
    Energies (EBS bands, DOS energies) are not shifted by the Fermi energy the
    object carries: subtracting ``ebs.fermi`` or ``dos.fermi`` puts E_F at zero.
    When a code writes Fermi-relative energies and the Fermi energy is unknown
    (Lobster without a ``structure_parser``), the object carries ``fermi=0.0``.
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
        """Reciprocal lattice vectors as rows, in 1/Angstrom without the 2*pi factor."""
        return None
