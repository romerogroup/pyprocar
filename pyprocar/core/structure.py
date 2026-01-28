# __author__ = "Pedram Tavadze and Logan Lang"
from __future__ import annotations

__maintainer__ = "Pedram Tavadze and Logan Lang"
__email__ = "petavazohi@mail.wvu.edu, lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv
import spglib
from scipy.spatial import ConvexHull
from typing_extensions import override

from pyprocar.core.serializer import get_serializer
from pyprocar.utils import elements
from pyprocar.utils.typing_helpers import (
    arccos_deg,
    array_str_to_list,
    arrays_equal,
    det_to_float,
    dot_matrix,
    dot_scalar,
    inv_matrix,
    norm_to_float,
    sum_to_float,
)

# TODO add __str__ method

N_AVOGADRO = 6.022140857e23

logger = logging.getLogger(__name__)


class Structure:
    """Class to define a periodic crystal structure.

    Parameters
    ----------
    atoms : Sequence[str] | None
        A list of atomic symbols, with the same order as the
        ``fractional_coordinates``.
    fractional_coordinates : npt.ArrayLike | None
        A (natom,3) array of fractional coordinates of atoms.
    lattice : npt.ArrayLike | None
        A (3,3) matrix representing the lattice vectors.
    cartesian_coordinates : npt.ArrayLike | None
        A (natom,3) array of cartesian coordinates of atoms.
    rotations : npt.ArrayLike | None
        Rotation matrices for symmetry operations.
    """

    atoms: npt.NDArray[np.str_] | None
    lattice: npt.NDArray[np.float64] | None
    cartesian_coordinates: npt.NDArray[np.float64] | None
    fractional_coordinates: npt.NDArray[np.float64] | None
    _rotations: npt.NDArray[np.float64]
    _wyckoff_positions: npt.NDArray[np.str_] | None
    _group: npt.NDArray[np.intp] | None

    def __init__(
        self,
        atoms: Sequence[str] | npt.ArrayLike | None = None,
        cartesian_coordinates: npt.ArrayLike | None = None,
        fractional_coordinates: npt.ArrayLike | None = None,
        lattice: npt.ArrayLike | None = None,
        rotations: npt.ArrayLike | None = None,
    ) -> None:
        # Validate that we have at least some data
        if (
            atoms is None
            and lattice is None
            and fractional_coordinates is None
            and cartesian_coordinates is None
        ):
            raise ValueError("Structure requires at least atoms or lattice to be provided")

        # Handle atoms - convert to array and validate
        if atoms is not None:
            self.atoms = np.asarray(atoms, dtype=np.str_)
            if self.atoms.ndim == 0 or self.atoms.shape[0] == 0:
                raise ValueError("atoms must be a non-empty list")
        else:
            self.atoms = None

        # Handle lattice - convert to array if provided
        if lattice is not None:
            self.lattice = np.asarray(lattice, dtype=np.float64)
            if self.lattice.ndim < 2 or self.lattice.shape[0] == 0:
                raise ValueError("lattice must be a non-empty 2D array")
        else:
            self.lattice = None

        # Handle coordinates
        if fractional_coordinates is not None:
            frac_coords: npt.NDArray[np.float64] = np.asarray(
                fractional_coordinates, dtype=np.float64
            )
            if frac_coords.ndim < 2 or frac_coords.shape[0] == 0:
                raise ValueError("fractional_coordinates must be a non-empty 2D array")
            self.fractional_coordinates = frac_coords
            if self.lattice is not None:
                self.cartesian_coordinates = np.dot(frac_coords, self.lattice)
            else:
                self.cartesian_coordinates = None
        elif cartesian_coordinates is not None:
            cart_coords: npt.NDArray[np.float64] = np.asarray(
                cartesian_coordinates, dtype=np.float64
            )
            if cart_coords.ndim < 2 or cart_coords.shape[0] == 0:
                raise ValueError("cartesian_coordinates must be a non-empty 2D array")
            self.cartesian_coordinates = cart_coords
            if self.lattice is not None:
                self.fractional_coordinates = np.dot(cart_coords, np.linalg.inv(self.lattice))
            else:
                self.fractional_coordinates = None
        else:
            self.cartesian_coordinates = None
            self.fractional_coordinates = None

        # Validate consistency between atoms and coordinates
        if self.atoms is not None and self.fractional_coordinates is not None:
            if self.atoms.shape[0] != self.fractional_coordinates.shape[0]:
                raise ValueError("atoms and fractional_coordinates must have the same length")

        # Initialize private attributes for lazy-loaded properties
        self._wyckoff_positions = None
        self._group = None

        # Handle rotations
        if rotations is not None:
            self._rotations = np.asarray(rotations, dtype=np.float64)
        else:
            self._rotations = np.empty(shape=(0, 3, 3), dtype=np.float64)

    @property
    def has_complete_data(self) -> bool:
        """Check if the structure has complete data for calculations.

        Returns
        -------
        bool
            True if atoms, lattice, and coordinates are all set, False otherwise.
        """
        return (
            self.atoms is not None
            and self.lattice is not None
            and self.fractional_coordinates is not None
            and len(self.atoms) > 0
        )

    @property
    def atoms_list(self) -> list[str] | None:
        """Return the atoms as a typed list of strings.

        Returns
        -------
        list[str] | None
            List of atomic symbols, or None if atoms is not set.
        """
        if self.atoms is None:
            return None
        return array_str_to_list(self.atoms)

    @override
    def __repr__(self) -> str:
        """Unambiguous representation with essential details for debugging."""
        return (
            f"Structure(natoms={self.natoms}, "
            f"species={list(self.species)}, "
            f"volume={self.volume * 1e30:.3f} A^3, "
            f"angles=({self.alpha:.2f}, {self.beta:.2f}, {self.gamma:.2f}), "
            f"spacegroup='{self.get_space_group_international() if self.has_complete_data else 'N/A'}')"
        )

    @override
    def __str__(self) -> str:
        """Human-readable summary of the structure."""
        header = f"Structure with {self.natoms} atoms and {self.nspecies} species"
        species_line = f"Species: {', '.join(self.species)}"
        volume_line = f"Volume: {self.volume * 1e30:.3f} A^3"
        angle_line = f"Angles (α, β, γ): {self.alpha:.2f}°, {self.beta:.2f}°, {self.gamma:.2f}°"

        # Only show first few fractional coords for readability
        if self.atoms is not None and self.fractional_coordinates is not None:
            lines: list[str] = []
            atoms_list = array_str_to_list(self.atoms)
            for i, atom_str in enumerate(atoms_list):
                coord = self.fractional_coordinates[i, :]
                lines.append(f"  {atom_str}: {coord}")
            frac_preview = "\n".join(lines)
        else:
            frac_preview = "  N/A"

        return "\n".join(
            [header, species_line, volume_line, angle_line, "Fractional coordinates:", frac_preview]
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structure):
            return False
        atoms_equal = False
        if self.atoms is not None and other.atoms is not None:
            atoms_equal = arrays_equal(self.atoms, other.atoms)

        fractional_coordinates_equal = True
        if self.fractional_coordinates is not None and other.fractional_coordinates is not None:
            fractional_coordinates_equal = bool(
                np.allclose(self.fractional_coordinates, other.fractional_coordinates)
            )

        lattice_equal = True
        if self.lattice is not None and other.lattice is not None:
            lattice_equal = bool(np.allclose(self.lattice, other.lattice))

        structure_equal = atoms_equal and fractional_coordinates_equal and lattice_equal
        return structure_equal

    @property
    def rotations(self) -> npt.NDArray[np.float64]:
        return self._rotations

    @property
    def wyckoff_positions(self) -> npt.NDArray[np.str_] | None:
        if self._wyckoff_positions is None:
            _ = self.get_wyckoff_positions()
        return self._wyckoff_positions

    @property
    def group(self) -> npt.NDArray[np.intp] | None:
        if self._group is None:
            _ = self.get_wyckoff_positions()
        return self._group

    @property
    def volume(self) -> float:
        """Volume of the unit cell.

        Returns
        -------
        float
            Volume of the unit cell (m^3).
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate volume")
        det_val = det_to_float(self.lattice)
        return abs(det_val) * 1e-30

    @property
    def masses(self) -> list[float]:
        """List of masses of each atom.

        Returns
        -------
        list[float]
            Masses of each atom (kg).
        """
        if self.atoms is None:
            raise ValueError("atoms must be set to calculate masses")
        result: list[float] = []
        atoms_list = array_str_to_list(self.atoms)
        for atom_str in atoms_list:
            mass = elements.atomic_mass(atom_str)
            if mass is None:
                raise ValueError(f"Unknown atomic mass for element: {atom_str}")
            result.append(float(mass) * 1.0e-3)
        return result

    @property
    def density(self) -> float:
        """Density of the cell.

        Returns
        -------
        float
            Density of the cell (kg/m^3).
        """
        total_mass = sum_to_float(self.masses)
        return total_mass / (self.volume * N_AVOGADRO)

    @property
    def a(self) -> float:
        """The magnitude of the first crystal lattice vector.

        Returns
        -------
        float
            The magnitude of the first crystal lattice vector.
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate lattice parameter a")
        return norm_to_float(self.lattice[0, :])

    @property
    def b(self) -> float:
        """The magnitude of the second crystal lattice vector.

        Returns
        -------
        float
            The magnitude of the second crystal lattice vector.
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate lattice parameter b")
        return norm_to_float(self.lattice[1, :])

    @property
    def c(self) -> float:
        """The magnitude of the third crystal lattice vector.

        Returns
        -------
        float
            The magnitude of the third crystal lattice vector.
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate lattice parameter c")
        return norm_to_float(self.lattice[2, :])

    @property
    def alpha(self) -> float:
        """The angle between the second and third crystal lattice vectors.

        Returns
        -------
        float
            The angle in degrees.
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate angle alpha")
        dot_val = dot_scalar(self.lattice[1, :], self.lattice[2, :])
        return arccos_deg(dot_val / (self.b * self.c))

    @property
    def beta(self) -> float:
        """The angle between the first and third crystal lattice vectors.

        Returns
        -------
        float
            The angle in degrees.
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate angle beta")
        dot_val = dot_scalar(self.lattice[0, :], self.lattice[2, :])
        return arccos_deg(dot_val / (self.a * self.c))

    @property
    def gamma(self) -> float:
        """The angle between the first and second crystal lattice vectors.

        Returns
        -------
        float
            The angle in degrees.
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate angle gamma")
        dot_val = dot_scalar(self.lattice[0, :], self.lattice[1, :])
        return arccos_deg(dot_val / (self.a * self.b))

    @property
    def species(self) -> npt.NDArray[np.str_]:
        """List of different species present in the cell.

        Returns
        -------
        npt.NDArray[np.str_]
            Array of unique species in the cell.
        """
        if self.atoms is None:
            return np.array([], dtype=np.str_)
        return np.unique(self.atoms)

    @property
    def nspecies(self) -> int:
        """Number of species present in the cell.

        Returns
        -------
        int
            Number of species present in the cell.
        """
        return len(self.species)

    @property
    def natoms(self) -> int:
        """Number of atoms.

        Returns
        -------
        int
            Number of atoms.
        """
        if self.atoms is None:
            return 0
        return len(self.atoms)

    @property
    def atomic_numbers(self) -> list[int]:
        """List of atomic numbers.

        Returns
        -------
        list[int]
            List of atomic numbers.
        """
        if self.atoms is None:
            raise ValueError("atoms must be set to get atomic numbers")
        result: list[int] = []
        atoms_list = array_str_to_list(self.atoms)
        for atom_str in atoms_list:
            atomic_num = elements.atomic_number(atom_str)
            result.append(int(atomic_num))
        return result

    @property
    def reciprocal_lattice(self) -> npt.NDArray[np.float64]:
        """The reciprocal lattice matrix corresponding to the crystal lattice.

        Returns
        -------
        npt.NDArray[np.float64]
            The reciprocal lattice matrix.
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to calculate reciprocal lattice")
        reciprocal_lattice: npt.NDArray[np.float64] = np.zeros_like(self.lattice)

        a = self.lattice[0, :]
        b = self.lattice[1, :]
        c = self.lattice[2, :]
        volume = self.volume * 1e30

        a_star: npt.NDArray[np.float64] = np.cross(b, c) / volume
        b_star: npt.NDArray[np.float64] = np.cross(c, a) / volume
        c_star: npt.NDArray[np.float64] = np.cross(a, b) / volume

        reciprocal_lattice[0, :] = a_star
        reciprocal_lattice[1, :] = b_star
        reciprocal_lattice[2, :] = c_star

        return reciprocal_lattice

    @property
    def _spglib_cell(
        self,
    ) -> tuple[
        Sequence[Sequence[float]], Sequence[Sequence[float]], Sequence[int]
    ]:
        """Return the structure in spglib format.

        Returns
        -------
        tuple
            (lattice, frac_coords, atomic_numbers)
        """
        if self.lattice is None:
            raise ValueError("lattice must be set for spglib cell")
        if self.fractional_coordinates is None:
            raise ValueError("fractional_coordinates must be set for spglib cell")
        return (
            self.lattice.tolist(),
            self.fractional_coordinates.tolist(),
            self.atomic_numbers,
        )

    def get_space_group_number(self, symprec: float = 1e-5) -> int:
        """Return the Space Group Number of the material.

        Parameters
        ----------
        symprec : float, optional
            Tolerance for symmetry, by default 1e-5

        Returns
        -------
        int
            The Space Group Number
        """
        dataset = spglib.get_symmetry_dataset(self._spglib_cell, symprec)
        if dataset is None:
            raise ValueError("Failed to determine space group")
        return int(dataset.number)

    def get_space_group_international(self, symprec: float = 1e-5) -> str:
        """Return the international Space Group symbol of the material.

        Parameters
        ----------
        symprec : float, optional
            Tolerance for symmetry, by default 1e-5

        Returns
        -------
        str
            The international Space Group symbol
        """
        dataset = spglib.get_symmetry_dataset(self._spglib_cell, symprec)
        if dataset is None:
            raise ValueError("Failed to determine space group")
        return str(dataset.international)

    def get_wyckoff_positions(self, symprec: float = 1e-5) -> npt.NDArray[np.str_] | None:
        """Return the Wyckoff positions.

        Parameters
        ----------
        symprec : float, optional
            Tolerance for symmetry, by default 1e-5

        Returns
        -------
        npt.NDArray[np.str_] | None
            The Wyckoff positions
        """
        if self.lattice is None or self.fractional_coordinates is None or self.atoms is None:
            raise ValueError(
                "Lattice, fractional coordinates, and atoms must be set to get wyckoff positions"
            )

        wyckoff_positions: npt.NDArray[np.str_] = np.empty(shape=(self.natoms,), dtype="<U4")

        spglib_dataset = spglib.get_symmetry_dataset(self._spglib_cell, symprec)
        if spglib_dataset is None:
            return None

        # spglib.SymmetryDataset has wyckoffs: list[str]
        wyckoffs_list: list[str] = spglib_dataset.wyckoffs

        group: npt.NDArray[np.intp] = np.zeros(shape=(self.natoms,), dtype=np.intp)
        counter = 0
        unique_wyckoffs_list: list[str] = list(dict.fromkeys(wyckoffs_list))  # Preserve order
        atoms_list = array_str_to_list(self.atoms)

        for iwyckoff in unique_wyckoffs_list:
            # Find indices where wyckoff matches
            idx_list: list[int] = [i for i, w in enumerate(wyckoffs_list) if w == iwyckoff]
            atoms_at_wyckoff: list[str] = [atoms_list[i] for i in idx_list]
            unique_species_list: list[str] = list(dict.fromkeys(atoms_at_wyckoff))

            for ispc in unique_species_list:
                # Find indices within idx_list where species matches
                idx2_list: list[int] = [
                    idx_list[j] for j, a in enumerate(atoms_at_wyckoff) if a == ispc
                ]
                multiplicity = len(idx2_list)
                for i_idx in idx2_list:
                    wyckoff_positions[i_idx] = str(multiplicity) + iwyckoff
                    group[i_idx] = counter
                counter += 1

        self._wyckoff_positions = wyckoff_positions
        self._group = group
        return wyckoff_positions

    def _get_lattice_corners(
        self, lattice: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return the corners of the crystal lattice.

        Parameters
        ----------
        lattice : npt.NDArray[np.float64]
            The crystal lattice

        Returns
        -------
        npt.NDArray[np.float64]
            The corners of the crystal lattice
        """
        origin: npt.NDArray[np.float64] = np.array([0.0, 0.0, 0.0])
        edges: list[npt.NDArray[np.float64]] = []
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    new_point: npt.NDArray[np.float64] = (
                        origin + lattice[0, :] * x + lattice[1, :] * y + lattice[2, :] * z
                    )
                    edges.append(new_point)
        return np.array(edges)

    @property
    def lattice_corners(self) -> npt.NDArray[np.float64]:
        """Return the corners of the crystal lattice.

        Returns
        -------
        npt.NDArray[np.float64]
            The corners of the crystal lattice
        """
        if self.lattice is None:
            raise ValueError("lattice must be set to get lattice corners")
        return self._get_lattice_corners(self.lattice)

    @property
    def cell_convex_hull(self) -> ConvexHull:
        """Return the cell convex hull.

        Returns
        -------
        ConvexHull
            The cell convex hull
        """
        return ConvexHull(self.lattice_corners)

    def plot_cell_convex_hull(self) -> None:
        """Plot the convex hull."""
        hull = self.cell_convex_hull
        # Convert simplices to pyvista face format (prepend count to each face)
        faces: npt.NDArray[np.intp] = np.hstack([[3] + list(face) for face in hull.simplices])
        surface = pv.PolyData(hull.points, faces)
        surface.plot()

    def get_spglib_symmetry_dataset(
        self, symprec: float = 1e-5
    ) -> spglib.SymmetryDataset | None:
        """Return the spglib symmetry dataset.

        Parameters
        ----------
        symprec : float, optional
            Tolerance for symmetry, by default 1e-5

        Returns
        -------
        spglib.SymmetryDataset | None
            spglib symmetry dataset, or None if symmetry could not be determined
        """
        return spglib.get_symmetry_dataset(self._spglib_cell, symprec)

    def transform(
        self,
        transformation_matrix: npt.NDArray[np.float64] | None = None,
    ) -> Structure:
        """Transform the crystal lattice by a transformation matrix.

        Parameters
        ----------
        transformation_matrix : npt.NDArray[np.float64] | None, optional
            The transformation matrix, by default identity matrix

        Returns
        -------
        Structure
            The transformed structure

        Raises
        ------
        ValueError
            If the transform is not proper
        """
        if transformation_matrix is None:
            transformation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

        if self.lattice is None:
            raise ValueError("lattice must be set to transform structure")
        if self.cartesian_coordinates is None:
            raise ValueError("cartesian_coordinates must be set to transform structure")
        if self.atoms is None:
            raise ValueError("atoms must be set to transform structure")

        det_val = det_to_float(transformation_matrix)
        scale: float = round(det_val, 2)
        if not scale.is_integer() or (1 / scale).is_integer():
            raise ValueError("This transform is not proper.")
        scale_int = int(scale)
        new_lattice = dot_matrix(self.lattice, transformation_matrix)
        temp_structure = Structure(
            atoms=["X"], fractional_coordinates=[[0, 0, 0]], lattice=new_lattice
        )
        new_atoms_list: list[list[str]] = []
        new_fractional_list: list[npt.NDArray[np.float64]] = []
        atoms_list = array_str_to_list(self.atoms)
        for iatom in range(len(atoms_list)):
            cart_coord: npt.NDArray[np.float64] = self.cartesian_coordinates[iatom, :]
            new_atoms_cartesian: list[npt.NDArray[np.float64]] = []
            for x in range(-1 * scale_int, scale_int):
                for y in range(-1 * scale_int, scale_int):
                    for z in range(-1 * scale_int, scale_int):
                        p: npt.NDArray[np.float64] = (
                            x * self.lattice[0, :]
                            + y * self.lattice[1, :]
                            + z * self.lattice[2, :]
                            + cart_coord
                        )
                        if temp_structure.is_point_inside(p):
                            new_atoms_cartesian.append(p)
            new_atoms_cart_arr: npt.NDArray[np.float64] = np.array(new_atoms_cartesian)
            inv_lattice = inv_matrix(new_lattice)
            new_atoms_fractional = dot_matrix(new_atoms_cart_arr, inv_lattice)
            new_atoms_fractional[new_atoms_fractional >= 1] -= 1
            new_atoms_fractional = np.unique(new_atoms_fractional, axis=0)
            new_fractional_list.append(new_atoms_fractional)
            new_atoms_list.append([atoms_list[iatom]] * len(new_atoms_fractional))
        new_atoms_flat = np.reshape(new_atoms_list, (-1,))
        new_fractional_flat: npt.NDArray[np.float64] = np.reshape(
            np.array(new_fractional_list, dtype=object), (-1, 3)
        ).astype(np.float64)
        return Structure(
            atoms=new_atoms_flat.tolist(),
            fractional_coordinates=new_fractional_flat,
            lattice=new_lattice,
        )

    def is_point_inside(
        self, point: npt.NDArray[np.float64], lattice: npt.NDArray[np.float64] | None = None
    ) -> bool:
        """Determine if a point is inside the unit cell.

        Parameters
        ----------
        point : npt.NDArray[np.float64]
            The point in question
        lattice : npt.NDArray[np.float64] | None, optional
            The crystal lattice matrix, by default None (uses self.lattice)

        Returns
        -------
        bool
            True if the point is inside the unit cell
        """
        if lattice is None:
            if self.lattice is None:
                raise ValueError("lattice must be set to check if point is inside")
            lattice = self.lattice
        corners: npt.NDArray[np.float64] = self._get_lattice_corners(lattice)
        # Build list of points including the test point
        edges_list: list[npt.NDArray[np.float64]] = list(corners)
        edges_list.append(point)
        edges_arr: npt.NDArray[np.float64] = np.array(edges_list)
        new_convex_hull = ConvexHull(edges_arr)
        return bool(new_convex_hull.area == self.cell_convex_hull.area)

    def supercell(self, matrix: npt.NDArray[np.float64]) -> Structure:
        """Transform the Structure to a supercell.

        Parameters
        ----------
        matrix : npt.NDArray[np.float64]
            The transformation matrix

        Returns
        -------
        Structure
            The transformed structure
        """
        return self.transform(matrix)

    def save(self, path: Path) -> None:
        """Save the structure to a file."""
        serializer = get_serializer(path)
        serializer.save(self, path)

    @classmethod
    def load(cls, path: Path) -> Structure:
        """Load a structure from a file."""
        serializer = get_serializer(path)
        result = serializer.load(path)
        if not isinstance(result, Structure):
            raise TypeError(f"Expected Structure, got {type(result)}")
        return result
