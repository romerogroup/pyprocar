from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from pyprocar.io.abinit import AbinitParser
from pyprocar.io.base import BaseParser
from pyprocar.io.bxsf import BxsfParser
from pyprocar.io.dftbplus import DFTBParser
from pyprocar.io.elk import ElkParser
from pyprocar.io.frmsf import FrmsfParser
from pyprocar.io.lobster import LobsterParser
from pyprocar.io.procarparser import ProcarParser
from pyprocar.io.qe import QEParser
from pyprocar.io.siesta import SiestaParser
from pyprocar.io.vasp import VaspParser

__all__ = [
    "AbinitParser",
    "BaseParser",
    "BxsfParser",
    "CodeParser",
    "DFTBParser",
    "ElkParser",
    "FrmsfParser",
    "LobsterParser",
    "Parser",
    "ParserType",
    "ProcarParser",
    "QEParser",
    "SiestaParser",
    "VaspParser",
    "get_parser",
]

if TYPE_CHECKING:
    from pyprocar.core import (
        DensityOfStates,
        ElectronicBandStructure,
        KPath,
        Structure,
    )

# Define return type as union of all parsers
ParserType = (
    AbinitParser
    | BxsfParser
    | DFTBParser
    | ElkParser
    | FrmsfParser
    | LobsterParser
    | QEParser
    | SiestaParser
    | VaspParser
)


class CodeParser(Enum):
    lobster = LobsterParser
    abinit = AbinitParser
    bxsf = BxsfParser
    frmsf = FrmsfParser
    qe = QEParser
    siesta = SiestaParser
    vasp = VaspParser
    elk = ElkParser
    dftbplus = DFTBParser

    @classmethod
    def as_list(cls) -> list[str]:
        return [code.name for code in cls]


def get_parser(
    code: str,
    dirpath: str | Path,
    custom_parser: type[BaseParser] | None = None,
    # Each parser has different optional kwargs (outcar path, procar, etc.)
    # so Any is appropriate here for this factory function
    **kwargs: Any,
) -> ParserType:
    """Handles which DFT parser to use."""
    if code in CodeParser.as_list():
        return CodeParser[code].value(dirpath=dirpath, **kwargs)
    elif custom_parser is not None:
        return custom_parser(dirpath=dirpath, **kwargs)  # pyright: ignore[reportReturnType]
    else:
        msg = f"Invalid code: {code}. Valid codes are: \n"
        for c in CodeParser.as_list():
            msg += f"    {c}\n"
        raise ValueError(msg)


class Parser(BaseParser):
    """
    The parser class will be the main object to be used throughout the code.
    This class will handle getting the main inputs (ebs, dos, structure, kpath, reciprocal_lattice) from the various DFT parsers.
    The bands must not be shifted so that the fermi energy is 0.0
    """

    code: str
    parser: ParserType

    def __init__(self, code: str, dirpath: str | Path, **kwargs: Any) -> None:
        super().__init__(dirpath=dirpath)
        self.code = code
        self.parser = get_parser(code, self.dirpath, **kwargs)

    @property
    @override
    def version(self) -> str | None:
        return self.parser.version

    @property
    @override
    def version_tuple(self) -> tuple[int, ...] | None:
        return self.parser.version_tuple

    @property
    @override
    def ebs(self) -> ElectronicBandStructure | None:
        return self.parser.ebs

    @property
    @override
    def dos(self) -> DensityOfStates | None:
        return self.parser.dos

    @property
    @override
    def structure(self) -> Structure | None:
        return self.parser.structure

    @property
    @override
    def kpath(self) -> KPath | None:
        return self.parser.kpath

    @property
    @override
    def reciprocal_lattice(self) -> npt.NDArray[np.float64] | None:
        return self.parser.reciprocal_lattice
