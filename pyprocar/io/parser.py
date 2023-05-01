import os

import numpy as np

from ..core import ElectronicBandStructure
from ..core import DensityOfStates
from ..core import Structure
from ..utils import UtilsProcar

from . import vasp, qe, abinit, lobster, siesta, frmsf, bxsf

class Parser:
    """
    The parser class will be the main object to be used through out the code. 
    This class will handle getting the main inputs (ebs,dos,structure,kpath,reciprocal_lattice) from the various dft parsers.
    """
    code : str = None
    dir : str = None
    ebs : ElectronicBandStructure = None
    dos : DensityOfStates = None
    structure : Structure = None

    def __init__(self,code:str , dir : str):
        self.code = code
        self.dir = dir

        self.parse()

    def parse(self):
        """Handles which DFT parser to use"""

        is_lobster_calc = self.code.split("_")[0] == "lobster"
        if is_lobster_calc:
            self.parse_lobster()

        elif self.code == "abinit":
            self.parse_abinit()

        elif self.code == "bxsf":
            self.parse_bxsf()
            
        elif self.code == "qe":
            self.parse_qe()

        elif self.code == "siesta":
            self.parse_siesta()

        elif self.code == "vasp":
            self.parse_vasp()

        self.ebs.bands += self.ebs.efermi

        return None

    def parse_abinit(self):
        """parses abinit files

        Returns
        -------
        None
            None
        """
        outfile = f"{self.dir}{os.sep}abinit.out"
        kpointsfile = f"{self.dir}{os.sep}KPOINTS"

        parser = abinit.Output(abinit_output=outfile)
        
        self.ebs = parser.abinitprocarobject.ebs
        self.kpath = parser.abinitprocarobject.ebs.kpath
        self.structure = parser.abinitprocarobject.structure

        return None
    
    def parse_bxsf(self):
        """parses bxsf files.

        Returns
        -------
        None
            None
        """
        
        parser = bxsf.BxsfParser(infile = 'in.frmsf')


        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos

        return None
    
    def parse_frmsf(self):
        """parses frmsf files. Needs to be finished

        Returns
        -------
        None
            None
        """
        parser = frmsf.FrmsfParser(infile = 'in.frmsf')

        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos

        return None
    
    def parse_lobster(self):
        """parses lobster files

        Returns
        -------
        None
            None
        """
        code_type = self.code.split("_")[1]
        parser = lobster.LobsterParser(
                            dirname = self.dir, 
                            code = code_type,
                            dos_interpolation_factor = None 
                            )

        self.ebs = parser.ebs
        self.structure = parser.structure
        self.kpath = parser.kpath
        self.dos = parser.dos

        return None
    
    def parse_qe(self):
        """parses qe files

        Returns
        -------
        None
            None
        """

        parser = qe.QEParser(
                            dirname = self.dir,
                            scf_in_filename = "scf.in", 
                            bands_in_filename = "bands.in", 
                            pdos_in_filename = "pdos.in", 
                            kpdos_in_filename = "kpdos.in", 
                            atomic_proj_xml = "atomic_proj.xml"
                            )

        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos
        return None
    
    def parse_siesta(self):
        """parses siesta files. Needs to be finished

        Returns
        -------
        None
            None
        """
        
        parser = siesta.SiestaParser(
                            fdf_filename = f"{self.dir}{os.sep}SIESTA.fdf",
                            )

        self.ebs = parser.ebs
        self.kpath = parser.kpath
        self.structure = parser.structure
        self.dos = parser.dos

        return None

    def parse_vasp(self):
        """parses vasp files

        Returns
        -------
        None
            None
        """
        
        outcar = f"{self.dir}{os.sep}OUTCAR"
        poscar = f"{self.dir}{os.sep}POSCAR"
        procar = f"{self.dir}{os.sep}PROCAR"
        kpoints = f"{self.dir}{os.sep}KPOINTS"
        vasprun = f"{self.dir}{os.sep}vasprun.xml"

        repairhandle = UtilsProcar()
        repairhandle.ProcarRepair(procar, procar)
        
        outcar = vasp.Outcar(outcar)
        poscar = vasp.Poscar(poscar,rotations = outcar.rotations)

        try:
            kpoints = vasp.Kpoints(kpoints)
            self.kpath = kpoints.kpath
        except:
            self.kpath = None

        

        procar = vasp.Procar(
                            filename=procar,
                            structure=poscar.structure,
                            reciprocal_lattice=poscar.structure.reciprocal_lattice,
                            kpath=self.kpath,
                            efermi=outcar.efermi,
                            interpolation_factor=1
                            )
        vasprun = vasp.VaspXML(filename = vasprun)
        
        self.ebs = procar.ebs
        self.structure = poscar.structure

        try:
            self.dos = vasprun.dos
        except Exception as e:
            print(e)
            self.dos = None

        return None
        
