import os

import numpy as np
import logging

from ..core import ElectronicBandStructure
from ..core import DensityOfStates
from ..core import Structure
from ..utils import UtilsProcar
from . import vasp, qe, abinit, lobster, siesta, frmsf, bxsf, elk, dftbplus


logger = logging.getLogger(__name__)

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

        elif self.code == "elk":
            self.parse_elk()

        elif self.code == "dftb+":
            self.parse_dftbplus()
            
        if self.ebs:
            # self.ebs.bands = self.ebs.bands - self.ebs.efermi
            self.ebs.bands += self.ebs.efermi
        if self.dos:
            self.dos.energies += self.dos.efermi
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
        abinit_output = abinit.Output(abinit_output=outfile)
        abinit_kpoints = abinit.AbinitKpoints(filename=kpointsfile)

        parser =  abinit.AbinitProcar(  
                                        dirname=self.dir,
                                        abinit_output=outfile,
                                        kpath=abinit_kpoints.kpath,
                                        reciprocal_lattice=abinit_output.reclat,
                                        efermi=abinit_output.fermi
                                        )
        
        abinit_dos = abinit.AbinitDOSParser(dirname=self.dir)

        self.dos = abinit_dos.dos
        self.ebs = parser.abinitprocarobject.ebs
        self.kpath = parser.abinitprocarobject.ebs.kpath
        self.structure = abinit_output.structure
        

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
    
    def parse_elk(self):
        """parses bxsf files.

        Returns
        -------
        None
            None
        """
        # try:
        #     dos = elk.read_dos(path = self.dir)
        #     self.dos = dos
        # except:
        #     self.dos = None

        try:
            parser=elk.ElkParser(path=self.dir)
            self.dos = parser.dos
            self.structure = parser.structure
            
        except Exception as e:

            self.dos = None
            self.structure = None

        
        if not self.dos:

            try:
                parser=elk.ElkParser(path=self.dir)
                self.ebs = parser.ebs
                self.kpath = parser.kpath
                self.structure = parser.structure
            except Exception as e:
                self.ebs = None
                self.kpath = None
                self.structure = None
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
        logger.info(f"Parsing VASP files in {self.dir}")
        
        outcar = os.path.join(self.dir, "OUTCAR")
        poscar = os.path.join(self.dir, "POSCAR")
        procar = os.path.join(self.dir, "PROCAR")
        kpoints = os.path.join(self.dir, "KPOINTS")
        vasprun = os.path.join(self.dir, "vasprun.xml")

        repairhandle = UtilsProcar()
        repairhandle.ProcarRepair(procar, procar)
        
        outcar = vasp.Outcar(outcar)

        try:
            poscar = vasp.Poscar(poscar,rotations = outcar.rotations)
        except Exception as e:
            logger.exception(f"Error parsing poscar file \n{e}")
            poscar = vasp.Poscar(poscar,rotations = None)

        try:
            kpoints = vasp.Kpoints(kpoints)
            self.kpath = kpoints.kpath
        except Exception as e:
            logger.exception(f"Error parsing kpoints file \n{e}")
            self.kpath=None

        procar = vasp.Procar(
                            filename=procar,
                            structure=poscar.structure,
                            reciprocal_lattice=poscar.structure.reciprocal_lattice,
                            kpath=self.kpath,
                            n_kx=outcar.n_kx,
                            n_ky=outcar.n_ky,
                            n_kz=outcar.n_kz,
                            efermi=outcar.efermi,
                            interpolation_factor=1
                            )
        
        try:
            vasprun = vasp.VaspXML(filename = vasprun)
        except Exception as e:
            logger.exception(f"Error parsing vasprun.xml file \n{e}")
            pass
        
        self.ebs = procar.ebs

        self.structure = poscar.structure

        try:
            self.dos = vasprun.dos
        except Exception as e:
            logger.exception(f"Error extracting dos from vasprun.xml file \n{e}")
            self.dos = None

        return None

    def parse_dftbplus(self):
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
        parser = dftbplus.DFTBParser(dirname = self.dir,
                                     eigenvec_filename = 'eigenvec.out',
                                     bands_filename = 'band.out',
                                     detailed_out = 'detailed.out',
                                     detailed_xml = 'detailed.xml'
                                     )
        
        self.parse_vasp()
        
        return None

