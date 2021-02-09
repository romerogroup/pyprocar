# -*- coding: utf-8 -*-

import spglib
import numpy as np
from . import elements

N_avogadro = 6.022140857e23



class Structure:
    def __init__(self, 
                 atoms=None, 
                 cartesian_coordinates=None, 
                 fractional_coordinates=None, 
                 lattice=None):
        """
        Class to define a peridic crystal structure.

        Parameters
        ----------
        atoms : list str
            A list of atomic symbols, with the same order as the
            ``fractional_coordinates``.
        fractional_coordinates : list (n,3) float.
            A (natom,3) list of fractional coordinatesd of atoms.
        lattice : list (3,3) float.
            A (3,3) matrix representing the lattice vectors.

        Returns
        -------
        None.

        """
        self.fractional_coordinates = np.array(fractional_coordinates)
        self.atoms = np.array(atoms)
        self.lattice = np.array(lattice)
        self.wyckoff_positions = None
        self.group = None
        if lattice is not None and fractional_coordinates is not None:
            self.get_wyckoff_positions()


    @property
    def volume(self):
        """
        Volume of the unit cell.

        Returns
        -------
        float
            Volume of the unit cell(m).

        """
        return abs(np.linalg.det(self.lattice)) * 1e-30

    @property
    def masses(self):
        """
        list of masses of each atom.

        Returns
        -------
        list float
            Masses of each atom.

        """
        return [elements.atomic_mass(x) * 1.0e-3 for x in self.atoms]

    @property
    def density(self):
        """
        Density of the cell.

        Returns
        -------
        float
            Density of the cell.

        """
        return np.sum(self.masses) / (self.volume * N_avogadro)

    @property
    def species(self):
        """
        list of different species present in the cell.

        Returns
        -------
        list str
            List of different species present in the cell.

        """
        return np.unique(self.atoms)

    @property
    def nspecies(self):
        """
        Number of species present in the cell.

        Returns
        -------
        int
            Number of species present in the cell.

        """
        return len(self.species)

    @property
    def natoms(self):
        """
        Number of atoms

        Returns
        -------
        int
            Number of atoms.

        """
        return len(self.atoms)

    @property
    def atomic_numbers(self):
        """
        List of atomic numbers

        Returns
        -------
        list
            List of atomic numbers.

        """
        return [elements.atomic_number(x) for x in self.atoms]

    @property
    def _spglib_cell(self):
        return (self.lattice, self.fractional_coordinates, self.atomic_numbers)

    def get_space_group_number(self, symprec=1e-5):
        return spglib.get_symmetry_dataset(self._spglib_cell,
                                           symprec)["number"]

    def get_space_group_international(self, symprec=1e-5):
        return spglib.get_symmetry_dataset(self._spglib_cell,
                                           symprec)["international"]

    def get_wyckoff_positions(self, symprec=1e-5):
        wyckoff_positions = np.empty(shape=(self.natoms), dtype='<U4')
        wyckoffs_temp = np.array(
            spglib.get_symmetry_dataset(self._spglib_cell, 
                                        symprec)["wyckoffs"])
        group = np.zeros(shape=(self.natoms), dtype=np.int)
        counter = 0
        for iwyckoff in np.unique(wyckoffs_temp):
            idx = np.where(wyckoffs_temp == iwyckoff)[0]
            for ispc in np.unique(self.atoms[idx]):
                idx2 = np.where(self.atoms[idx] == ispc)[0]
                multiplicity = len(idx2)
                wyckoff_positions[idx][idx2]
                for i in idx[idx2]:
                    wyckoff_positions[i] = str(multiplicity)+iwyckoff
                    group[i] = counter
                counter += 1
        self.wyckoff_positions = wyckoff_positions
        self.group = group
        return wyckoff_positions
    
    
    
    def get_spglib_symmetry_dataset(self, symprec=1e-5):
        return spglib.get_symmetry_dataset(self._spglib_cell,symprec)

