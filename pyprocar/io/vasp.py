# -*- coding: utf-8 -*-

from ..core import Structure
import numpy as np
import os
import re

def parse_poscar(filename='CONTCAR'):
    """
    Reads VASP POSCAR file-type and returns the pyprocar structure

    Parameters
    ----------
    filename : str, optional
        Path to POSCAR file. The default is 'CONTCAR'.

    Returns
    -------
    None.

    """
    rf = open(filename,'r')
    lines = rf.readlines()
    rf.close()
    comment = lines[0]
    scale = float(lines[1])
    lattice = np.zeros(shape=(3, 3))
    for i in range(3):
        lattice[i, :] = [float(x) for x in lines[i+2].split()[:3]]
    lattice *= scale
    if any([char.isalpha() for char in lines[5]]):
        species = [x for x in lines[5].split()]
        shift = 1
    else :
        shift = 0
        if os.path.exists('POTCAR'):
            base_dir = filename.replace(filename.split(os.sep)[-1], "")
            if base_dir=='':
                base_dir='.'
            rf = open(base_dir+os.sep+'POTCAR','r')
            potcar = rf.read()
            rf.close()
            species = re.findall("\s*PAW[PBE_\s]*([A-Z][a-z]*)[_a-z]*[0-9]*[a-zA-Z]*[0-9]*.*\s[0-9.]*",
                       potcar)[::2]
    composition = [int(x) for x in lines[5+shift].split()]
    atoms = []
    for i in range(len(composition)):
        for x in composition[i]*[species[i]]:
            atoms.append(x)
    natom = sum(composition)
    if lines[6+shift][0].lower() == 's':
        shift = 2
    if lines[6+shift][0].lower() == 'd':
        direct = True
    elif lines[6+shift][0].lower() == 'c':
        print("havn't implemented conversion to cartesian yet")
        direct = False
    coordinates = np.zeros(shape=(natom, 3))
    for i in range(natom):
        coordinates[i,:] = [float(x) for x in lines[i+7+shift].split()[:3]]
    if direct :
        return Structure(atoms=atoms, 
                         fractional_coordinates=coordinates, 
                         lattice=lattice)
    else:
        return Structure(atoms=atoms, 
                         cartesian_coordinates=coordinates, 
                         lattice=lattice)
    
        
    
    
    
    