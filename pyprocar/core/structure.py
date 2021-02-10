# -*- coding: utf-8 -*-

import spglib
import numpy as np
from scipy.spatial import ConvexHull

# from . import Surface
from . import elements


N_avogadro = 6.022140857e23


class Structure:
    def __init__(
        self,
        atoms=None,
        cartesian_coordinates=None,
        fractional_coordinates=None,
        lattice=None,
    ):
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
        if fractional_coordinates is not None:
            self.fractional_coordinates = np.array(fractional_coordinates)
            self.cartesian_coordinates = np.dot(fractional_coordinates, lattice)
        elif cartesian_coordinates is not None:
            self.cartesian_coordinates = cartesian_coordinates
            self.fractional_coordinates = np.dot(
                cartesian_coordinates, np.linalg.inv(lattice)
            )
        else:
            self.cartesian_coordinates = None
            self.fractional_coordinates = None
        self.atoms = np.array(atoms)
        self.lattice = np.array(lattice)

        if (
            self.lattice is not None
            and self.fractional_coordinates is not None
            and atoms is not None
        ):
            self.has_complete_data = True
        else:
            self.has_complete_data = False
        self.wyckoff_positions = None
        self.group = None

        if self.has_complete_data:
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
    def a(self):
        return np.linalg.norm(self.lattice[0, :])

    @property
    def b(self):
        return np.linalg.norm(self.lattice[1, :])

    @property
    def c(self):
        return np.linalg.norm(self.lattice[2, :])

    @property
    def alpha(self):
        return np.rad2deg(
            np.arccos(
                np.dot(self.lattice[1, :], self.lattice[2, :]) / (self.b * self.c)
            )
        )

    @property
    def beta(self):
        return np.rad2deg(
            np.arccos(
                np.dot(self.lattice[0, :], self.lattice[2, :]) / (self.a * self.c)
            )
        )

    @property
    def gamma(self):
        return np.rad2deg(
            np.arccos(
                np.dot(self.lattice[0, :], self.lattice[1, :]) / (self.a * self.b)
            )
        )

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
    def reciprocal_lattice(self):
        reciprocal_lattice = np.zeros_like(self.lattice)
        a = self.lattice[0, :]
        b = self.lattice[1, :]
        c = self.lattice[2, :]
        volume = self.volume * 1e30

        a_star = (2 * np.pi) * np.cross(b, c) / volume
        b_star = (2 * np.pi) * np.cross(c, a) / volume
        c_star = (2 * np.pi) * np.cross(a, b) / volume
        reciprocal_lattice[0, :] = a_star
        reciprocal_lattice[1, :] = b_star
        reciprocal_lattice[2, :] = c_star

        return reciprocal_lattice

    @property
    def _spglib_cell(self):
        return (self.lattice, self.fractional_coordinates, self.atomic_numbers)

    def get_space_group_number(self, symprec=1e-5):
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec)["number"]

    def get_space_group_international(self, symprec=1e-5):
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec)["international"]

    def get_wyckoff_positions(self, symprec=1e-5):
        wyckoff_positions = np.empty(shape=(self.natoms), dtype="<U4")
        wyckoffs_temp = np.array(
            spglib.get_symmetry_dataset(self._spglib_cell, symprec)["wyckoffs"]
        )
        group = np.zeros(shape=(self.natoms), dtype=np.int)
        counter = 0
        for iwyckoff in np.unique(wyckoffs_temp):
            idx = np.where(wyckoffs_temp == iwyckoff)[0]
            for ispc in np.unique(self.atoms[idx]):
                idx2 = np.where(self.atoms[idx] == ispc)[0]
                multiplicity = len(idx2)
                wyckoff_positions[idx][idx2]
                for i in idx[idx2]:
                    wyckoff_positions[i] = str(multiplicity) + iwyckoff
                    group[i] = counter
                counter += 1
        self.wyckoff_positions = wyckoff_positions
        self.group = group
        return wyckoff_positions

    def _get_lattice_corners(self, lattice):
        origin = np.array([0, 0, 0])
        edges = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    new_point = (
                        origin
                        + lattice[0, :] * x
                        + lattice[1, :] * y
                        + lattice[2, :] * z
                    )
                    edges.append(new_point)
        return np.array(edges)

    @property
    def lattice_corners(self):
        return self._get_lattice_corners(self.lattice)

    @property
    def cell_convex_hull(self):
        return ConvexHull(self.lattice_corners)

    def plot_cell_convex_hull(self):
        surface = Surface(
            verts=self.cell_convex_hull.points, faces=self.cell_convex_hull.simplices
        )
        surface.pyvista_obj.plot()

    def get_spglib_symmetry_dataset(self, symprec=1e-5):
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec)

    def transform(
        self, transformation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ):
        scale = np.linalg.det(transformation_matrix).round(2)
        if not scale.is_integer() or (1 / scale).is_integer():
            raise ValueError("This transfare with is not proper.")
            return None
        scale = int(scale)
        new_lattice = np.dot(self.lattice, transformation_matrix)
        temp_structure = Structure(lattice=new_lattice)
        new_atoms = []
        new_fractional = []
        for iatom, atom_coord in enumerate(self.cartesian_coordinates):
            new_atoms_cartesian = []
            p = atom_coord
            for x in range(-1 * scale, scale):
                for y in range(-1 * scale, scale):
                    for z in range(-1 * scale, scale):
                        p = (
                            x * self.lattice[0, :]
                            + y * self.lattice[1, :]
                            + z * self.lattice[2, :]
                            + atom_coord
                        )
                        if temp_structure.is_point_inside(p):
                            new_atoms_cartesian.append(p)
            new_atoms_cartesian = np.array(new_atoms_cartesian)
            new_atoms_fractional = np.dot(
                new_atoms_cartesian, np.linalg.inv(new_lattice)
            )
            new_atoms_fractional[new_atoms_fractional >= 1] -= 1
            new_atoms_fractional = np.unique(new_atoms_fractional, axis=0)
            new_fractional.append(new_atoms_fractional)
            new_atoms.append([self.atoms[iatom]] * len(new_atoms_fractional))
        new_atoms = np.reshape(new_atoms, (-1,))
        new_fractional = np.reshape(new_fractional, (-1, 3))
        return Structure(
            atoms=new_atoms, fractional_coordinates=new_fractional, lattice=new_lattice
        )

    def is_point_inside(self, point, lattice=None):
        if lattice is None:
            lattic = self.lattice
        edges = self._get_lattice_corners(lattic).tolist()
        edges.append(point)
        new_convex_hull = ConvexHull(edges)
        if new_convex_hull.area == self.cell_convex_hull.area:
            return True
        else:
            return False

    def supercell(self, matrix):
        return self.transform(matrix)
