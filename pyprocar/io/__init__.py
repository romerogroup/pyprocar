from enum import Enum
from pathlib import Path

from pyprocar.io.abinit import AbinitParser
from pyprocar.io.base import BaseParser
from pyprocar.io.bxsf import BxsfParser
from pyprocar.io.elk import ElkParser
from pyprocar.io.frmsf import FrmsfParser
from pyprocar.io.lobster import LobsterParser
from pyprocar.io.procarparser import ProcarParser
from pyprocar.io.qe import QEParser
from pyprocar.io.siesta import SiestaParser
from pyprocar.io.vasp import VaspParser


class CodeParser(Enum):
    lobster = LobsterParser
    abinit = AbinitParser
    bxsf = BxsfParser
    frmsf = FrmsfParser
    qe = QEParser
    siesta = SiestaParser
    vasp = VaspParser
    elk = ElkParser

    @classmethod
    def as_list(cls):
        return [code.name for code in cls]


def get_parser(
    code: str,
    dirpath: str | Path,
    custom_parser: type[BaseParser] | None = None,
    **kwargs,
) -> BaseParser:
    if code in CodeParser.as_list():
        parser = CodeParser[code].value(dirpath=dirpath, **kwargs)
    elif custom_parser is not None:
        parser = custom_parser(dirpath=dirpath, **kwargs)
    else:
        msg = f"Invalid code: {code}. Valid codes are: \n"
        for code in CodeParser.as_list():
            msg += f"    {code}\n"
        raise ValueError(msg)

    return parser
