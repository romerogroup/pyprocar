"""Abinit KPOINTS file parser."""

from pyprocar.io.vasp import Kpoints


class AbinitKpoints(Kpoints):
    """Parse VASP-format KPOINTS file for Abinit calculations.
    
    Abinit can use VASP-format KPOINTS files, so this class
    simply inherits from the VASP Kpoints parser.
    """
    pass
