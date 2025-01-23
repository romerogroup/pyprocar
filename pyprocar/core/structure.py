# __author__ = "Pedram Tavadze and Logan Lang"
__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"


import numpy as np
import spglib
from scipy.spatial import ConvexHull

from pyprocar.core.surface import Surface
from pyprocar.utils import elements

# TODO add __str__ method

N_AVOGADRO = 6.022140857e23


class Structure:
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

    def __init__(
        self,
        atoms=None,
        cartesian_coordinates=None,
        fractional_coordinates=None,
        lattice=None,
        rotations=None,
    ):

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

        self.rotations = rotations

        return None

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
        return np.sum(self.masses) / (self.volume * N_AVOGADRO)

    @property
    def a(self):
        """
        The magnitude of the first crystal lattice vector

        Returns
        -------
        float
            The magnitude of the first crystal lattice vector

        """
        return np.linalg.norm(self.lattice[0, :])

    @property
    def b(self):
        """
        The magnitude of the second crystal lattice vector

        Returns
        -------
        float
            The magnitude of the second crystal lattice vector

        """
        return np.linalg.norm(self.lattice[1, :])

    @property
    def c(self):
        """
        The magnitude of the third crystal lattice vector

        Returns
        -------
        float
            The magnitude of the third crystal lattice vector

        """
        return np.linalg.norm(self.lattice[2, :])

    @property
    def alpha(self):
        """
        The angle between the of the second and third crystal lattice vectors

        Returns
        -------
        float
            The angle between the of the second and third crystal lattice vectors

        """
        return np.rad2deg(
            np.arccos(
                np.dot(self.lattice[1, :], self.lattice[2, :]) / (self.b * self.c)
            )
        )

    @property
    def beta(self):
        """
        The angle between the of the first and third crystal lattice vectors

        Returns
        -------
        float
            The angle between the of the first and third crystal lattice vectors

        """
        return np.rad2deg(
            np.arccos(
                np.dot(self.lattice[0, :], self.lattice[2, :]) / (self.a * self.c)
            )
        )

    @property
    def gamma(self):
        """
        The angle between the of the first and second crystal lattice vectors

        Returns
        -------
        float
            The angle between the of the first and second crystal lattice vectors

        """
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
        """The reciprocal lattice matrix corresponding the the crystal lattice

        Returns
        -------
        np.ndarray
            The reciprocal lattice matrix corresponding the the crystal lattice
        """
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
        """Return the structure in spglib format

        Returns
        -------
        Tuple
            Return the structure in spglib format (lattice, frac_coords, atomic_numbers)
        """
        return (self.lattice, self.fractional_coordinates, self.atomic_numbers)

    def get_space_group_number(self, symprec=1e-5):
        """Returns the Space Group Number of the material

        Parameters
        ----------
        symprec : float, optional
            tolerence for symmetry, by default 1e-5

        Returns
        -------
        int
            The Space Group Number
        """
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec).number

    def get_space_group_international(self, symprec=1e-5):
        """Returns the international Space Group Number of the material

        Parameters
        ----------
        symprec : float, optional
            tolerence for symmetry, by default 1e-5

        Returns
        -------
        str
            The international Space Group Number
        """
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec).international

    def get_wyckoff_positions(self, symprec=1e-5):
        """Returns the wyckoff positions

        Parameters
        ----------
        symprec : float, optional
            tolerence for symmetry, by default 1e-5

        Returns
        -------
        np.ndarray
            The wyckoff positions
        """
        wyckoff_positions = np.empty(shape=(self.natoms), dtype="<U4")
        wyckoffs_temp = np.array(
            spglib.get_symmetry_dataset(self._spglib_cell, symprec).wyckoffs
        )
        group = np.zeros(shape=(self.natoms), dtype=int)
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
        """Returns the corners of the crystal lattice

        Parameters
        ----------
        lattice : np.ndarray
            The crystal lattice

        Returns
        -------
        np.ndarray
            Returns the corners of the crystal lattice
        """
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
        """Returns the corners of the crystal lattice

        Returns
        -------
        np.ndarray
            Returns the corners of the crystal lattice
        """
        return self._get_lattice_corners(self.lattice)

    @property
    def cell_convex_hull(self):
        """Returns the cell convex hull

        Returns
        -------
        scipy.spatial.ConvexHull
            Returns the cell convex hull
        """
        return ConvexHull(self.lattice_corners)

    def plot_cell_convex_hull(self):
        """
        A method to plot the the convex hull
        """
        surface = Surface(
            verts=self.cell_convex_hull.points, faces=self.cell_convex_hull.simplices
        )
        surface.pyvista_obj.plot()
        return None

    def get_spglib_symmetry_dataset(self, symprec=1e-5):
        """Returns the spglib symmetry dataset

        Parameters
        ----------
        symprec : float, optional
            tolerence for symmetry, by default 1e-5

        Returns
        -------
        dict
            spglib symmetry dataset
        """
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec)

    def transform(
        self, transformation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ):
        """Transform the crystla lattice by a transformation matrix

        Parameters
        ----------
        transformation_matrix : np.ndarray, optional
            The transformation matrix, by default np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        Returns
        -------
        pyprocar.core.Structure
            The transformed structure

        Raises
        ------
        ValueError
            Raise error if the transform is not proper
        """
        scale = np.linalg.det(transformation_matrix).round(2)
        if not scale.is_integer() or (1 / scale).is_integer():
            raise ValueError("This transform is not proper.")
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
        """A method to determine if a point is inside the unitcell

        Parameters
        ----------
        point : np.ndarray
            The point in question
        lattice : np.ndarray, optional
            The crystal lattice matrix, by default None

        Returns
        -------
        bool
            Boolean if a point is inside the unitcell
        """
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
        """A method to transform the Structure to a supercell

        Parameters
        ----------
        matrix : np.ndarray
            The matrix to transform the Structure

        Returns
        -------
        pyprocar.core.Structure
            The transformed structure
        """
        return self.transform(matrix)
