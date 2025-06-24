from pathlib import Path
from typing import Union

from pyprocar.io import abinit, dftbplus, elk, lobster, qe, siesta, vasp
from pyprocar.io.abinit import AbinitParser
from pyprocar.io.base import BaseParser
from pyprocar.io.bxsf import BxsfParser
from pyprocar.io.dftbplus import DFTBParser
from pyprocar.io.elk import ElkParser

# from . import bsxf
# from . import frmsf
from pyprocar.io.lobster import LobsterParser
from pyprocar.io.parser import Parser
from pyprocar.io.procarparser import ProcarParser
from pyprocar.io.qe import QEParser
from pyprocar.io.siesta import SiestaParser
from pyprocar.io.vasp import VaspParser


def get_parser(code: str, dirpath: Union[str, Path]):
    """Handles which DFT parser to use"""

    is_lobster_calc = code.split("_")[0] == "lobster"
    if is_lobster_calc:
        parser = LobsterParser(dirpath=dirpath)

    elif code == "abinit":
        parser = AbinitParser(dirpath=dirpath)

    elif code == "bxsf":
        parser = BxsfParser(dirpath=dirpath)

    elif code == "qe":
        parser = QEParser(dirpath=dirpath)

    elif code == "siesta":
        parser = SiestaParser(dirpath=dirpath)

    elif code == "vasp":
        parser = VaspParser(dirpath=dirpath)

    elif code == "elk":
        parser = ElkParser(dirpath=dirpath)

    elif code == "dftb+":
        parser = DFTBParser(dirpath=dirpath)

    return parser

class Parser(BaseParser):
    """
    The parser class will be the main object to be used through out the code.
    This class will handle getting the main inputs (ebs,dos,structure,kpath,reciprocal_lattice) from the various dft parsers.
    """

    def __init__(self, code: str, dirpath: Union[str, Path], verbose: int = 2):
        super().__init__(dirpath=dirpath)
        self.code = code
        self.parser=get_parser(code, dirpath)

    
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
    
    