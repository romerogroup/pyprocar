import os
import pyprocar
from pyprocar.core import Structure
import pytest
import numpy as np

code_name = "vasp"

DATA_DIR = f"{pyprocar.PROJECT_DIR}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}bands"

outcar = f"{DATA_DIR}{os.sep}OUTCAR"
poscar = f"{DATA_DIR}{os.sep}POSCAR"
procar = f"{DATA_DIR}{os.sep}PROCAR"
kpoints = f"{DATA_DIR}{os.sep}KPOINTS"
vaspxml = f"{DATA_DIR}{os.sep}vasprun.xml"

poscar = pyprocar.io.vasp.Poscar(poscar)
atoms, coordinates, lattice = poscar.atoms, poscar.coordinates, poscar.lattice 

def test_init_Structure():
    structure = Structure(atoms=atoms, fractional_coordinates=coordinates, lattice=lattice )
    assert structure.atoms == ['Fe']
    assert np.all(structure.fractional_coordinates == coordinates )
    assert np.all(structure.lattice == lattice )
