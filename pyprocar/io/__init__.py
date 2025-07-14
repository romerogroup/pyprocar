from enum import Enum
from pathlib import Path
from typing import Union

# from pyprocar.io import abinit, dftbplus, elk, lobster, qe, siesta, vasp
from pyprocar.io.abinit import AbinitParser
from pyprocar.io.base import BaseParser
from pyprocar.io.bxsf import BxsfParser
from pyprocar.io.dftbplus import DFTBParser
from pyprocar.io.elk import ElkParser

# from . import bsxf
# from . import frmsf
from pyprocar.io.lobster import LobsterParser
from pyprocar.io.procarparser import ProcarParser
from pyprocar.io.qe import QEParser
from pyprocar.io.siesta import SiestaParser
from pyprocar.io.vasp import VaspParser


class CodeParser(Enum):
    lobster = LobsterParser
    abinit = AbinitParser
    bxsf = BxsfParser
    qe = QEParser
    siesta = SiestaParser
    vasp = VaspParser
    elk = ElkParser
    dftbplus = DFTBParser
    
    @classmethod
    def as_list(cls):
        return [code.name for code in cls]


def get_parser(code: str, 
               dirpath: Union[str, Path],
               custom_parser: BaseParser = None,
               **kwargs):
    """Handles which DFT parser to use"""

    is_lobster_calc = code.split("_")[0] == "lobster"
    
    if code in CodeParser.as_list():
        parser = CodeParser[code].value(dirpath=dirpath, **kwargs)
    elif custom_parser is not None:
        parser = custom_parser(dirpath=dirpath, **kwargs)
    else:
        msg=f"Invalid code: {code}. Valid codes are: \n"
        for code in CodeParser.as_list():
            msg += f"    {code}\n"
        raise ValueError(msg)

    return parser

class Parser(BaseParser):
    """
    The parser class will be the main object to be used through out the code.
    This class will handle getting the main inputs (ebs,dos,structure,kpath,reciprocal_lattice) from the various dft parsers.
    The bands must not be shifted so that the fermi energy 0.0
    """

    def __init__(self, code: str, dirpath: Union[str, Path], **kwargs):
        super().__init__(dirpath=dirpath)
        self.code = code
        self.parser=get_parser(code, self.dirpath, **kwargs)

    
    @property
    def version(self):
        return self.parser.version
    
    @property
    def version_tuple(self):
        return self.parser.version_tuple

    @property
    def ebs(self):
        return self.parser.ebs
    
    @property
    def dos(self):
        return self.parser.dos
    
    @property
    def structure(self):
        return self.parser.structure
    
    @property
    def kpath(self):
        return self.parser.kpath
    
    @property
    def reciprocal_lattice(self):
        return self.parser.reciprocal_lattice
    
    