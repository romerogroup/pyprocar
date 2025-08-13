import logging
import os
from pathlib import Path
from typing import Union

import numpy as np

from pyprocar.core import DensityOfStates, ElectronicBandStructure, Structure
from pyprocar.io import abinit, bxsf, dftbplus, elk, frmsf, lobster, qe, siesta, vasp
from pyprocar.utils import UtilsProcar
from pyprocar.utils.log_utils import set_verbose_level

logger = logging.getLogger(__name__)


class Parser:
    """
    The parser class will be the main object to be used through out the code.
    This class will handle getting the main inputs (ebs,dos,structure,kpath,reciprocal_lattice) from the various dft parsers.
    """

    def __init__(self, code: str, dirpath: Union[str, Path], verbose: int = 0):
        self.code = code
        self.dirpath = Path(dirpath)

        self.ebs = None
        self.dos = None
        self.structure = None

        self._parse()


    def _parse(self):
        """Handles which DFT parser to use"""

        is_lobster_calc = self.code.split("_")[0] == "lobster"
        if is_lobster_calc:
            self._parse_lobster()

        elif self.code == "abinit":
            self._parse_abinit()

        elif self.code == "bxsf":
            self._parse_bxsf()

        elif self.code == "qe":
            self._parse_qe()

        elif self.code == "siesta":
            self._parse_siesta()

        elif self.code == "vasp":
            self._parse_vasp()

        elif self.code == "elk":
            self._parse_elk()

        elif self.code == "dftb+":
            self._parse_dftbplus()

        if self.ebs:
            # self.ebs.bands = self.ebs.bands - self.ebs.efermi
            self.ebs.bands += self.ebs.efermi
        if self.dos:
            self.dos.energies += self.dos.efermi
        return None

    def _parse_abinit(self):
        """parses abinit files

        Returns
        -------
        None
            None
        """
        abinit_parser = abinit.AbinitParser(dirpath=self.dirpath)

        self.dos = abinit_parser.dos
        self.ebs = abinit_parser.ebs
        self.kpath = abinit_parser.ebs.kpath
        self.structure = abinit_parser.structure

        return None

    def _parse_bxsf(self):
        """parses bxsf files.

        Returns
        -------
        None
            None
        """

        parser = bxsf.BxsfParser(filepaths="in.frmsf")

        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos

        return None

    def _parse_elk(self):
        """parses bxsf files.

        Returns
        -------
        None
            None
        """
        # try:
        #     dos = elk.read_dos(path = self.dirpath)
        #     self.dos = dos
        # except:
        #     self.dos = None

        try:
            parser = elk.ElkParser(dirpath=self.dirpath)
            self.dos = parser.dos
            self.structure = parser.structure

        except Exception as e:
            logger.debug(e)
            self.dos = None
            self.structure = None

        if not self.dos:

            try:
                parser = elk.ElkParser(dirpath=self.dirpath)
                self.ebs = parser.ebs
                self.kpath = parser.kpath
                self.structure = parser.structure
            except Exception as e:
                logger.debug(e)
                self.ebs = None
                self.kpath = None
                self.structure = None
        return None

    def _parse_frmsf(self):
        """parses frmsf files. Needs to be finished

        Returns
        -------
        None
            None
        """
        parser = frmsf.FrmsfParser(filepath="in.frmsf")

        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos

        return None

    def _parse_lobster(self):
        """parses lobster files

        Returns
        -------
        None
            None
        """
        code_type = self.code.split("_")[1]
        parser = lobster.LobsterParser(
            dirpath=self.dirpath, code=code_type, dos_interpolation_factor=None
        )

        self.ebs = parser.ebs
        self.structure = parser.structure
        self.kpath = parser.kpath
        self.dos = parser.dos

        return None

    def _parse_qe(self):
        """parses qe files

        Returns
        -------
        None
            None
        """

        parser = qe.QEParser(dirpath=self.dirpath)

        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos
        return None

    def _parse_siesta(self):
        """parses siesta files. Needs to be finished

        Returns
        -------
        None
            None
        """

        parser = siesta.SiestaParser(
            fdf_filepath=self.dirpath / "SIESTA.fdf",
        )

        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos

        return None

    def _parse_vasp(self):
        """parses vasp files

        Returns
        -------
        None
            None
        """
        logger.info(f"Parsing VASP files in {self.dirpath}")

        outcar = self.dirpath / "OUTCAR"
        poscar = self.dirpath / "POSCAR"
        procar = self.dirpath / "PROCAR"
        kpoints = self.dirpath / "KPOINTS"
        vasprun = self.dirpath / "vasprun.xml"

        parser = vasp.VaspParser(
            outcar=outcar,
            poscar=poscar,
            procar=procar,
            kpoints=kpoints,
            vasprun=vasprun,
        )

        self.ebs = parser.ebs
        self.structure = parser.structure
        self.dos = parser.dos

        return None

    def _parse_dftbplus(self):
        """parses DFTB+ files, these files do not have an array-like
        structure, and the process is *slow* .

        Then, they are converted to VASP-like files and parsed by the
        standard `parse_vasp`

        Returns
        -------
        None
            None

        """
        # This creates the vasp files, if needed
        parser = dftbplus.DFTBParser(
            dirname=self.dirpath,
            eigenvec_filename="eigenvec.out",
            bands_filename="band.out",
            detailed_out="detailed.out",
            detailed_xml="detailed.xml",
        )

        self._parse_vasp()

        return None
