# -*- coding: utf-8 -*-

# import spglib
import numpy as np
from . import elements

N_avogadro = 6.022140857e23


class Structure:
    def __init__(self, atoms=None, fractional_coordinates=None, lattice=None):
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
        self.fractional_coordinates = fractional_coordinates
        self.atoms = atoms
        self.lattice = lattice

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

    # @property
    # def spglib_cell(self):
    #     return (self.lattice, self.fractional_coordinates, self.atomic_numbers)

    # Question for uthpala: do you think we should include spglib?

    # def get_space_group_number(self, symprec=1e-5):
    #     return spglib.get_symmetry_dataset(self.spglib_cell, symprec)["number"]

    # def get_space_group_international(self, symprec=1e-5):
    #     return spglib.get_symmetry_dataset(self.spglib_cell, symprec)["international"]

    # def get_wyckoff_positions(self, symprec=1e-5):
    #     return spglib.get_symmetry_dataset(self.spglib_cell, symprec)["wyckoffs"]
