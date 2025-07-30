"""
Test module for pyprocar.core.structure.Structure class.

This module contains unit tests for the Structure class, testing various
properties and methods using simple crystal structures.
"""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.structure import Structure


class TestStructure:
    """Test class for Structure object."""

    @pytest.fixture
    def simple_cubic_structure(self):
        """
        Create a simple cubic structure with one atom.
        
        Returns
        -------
        Structure
            A simple cubic structure with lattice parameter 2.0 Å
        """
        # Simple cubic lattice with lattice parameter 2.0
        lattice = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        # One atom at origin
        atoms = ['H']
        fractional_coordinates = np.array([[0.0, 0.0, 0.0]])
        
        return Structure(
            atoms=atoms,
            fractional_coordinates=fractional_coordinates,
            lattice=lattice
        )

    @pytest.fixture
    def nacl_structure(self):
        """
        Create a simple NaCl-like structure.
        
        Returns
        -------
        Structure
            A simple NaCl structure with two atoms
        """
        # Simple cubic lattice for NaCl-like structure
        lattice = np.array([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0]
        ])
        
        # Na at origin, Cl at (0.5, 0.5, 0.5)
        atoms = ['Na', 'Cl']
        fractional_coordinates = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])
        
        return Structure(
            atoms=atoms,
            fractional_coordinates=fractional_coordinates,
            lattice=lattice
        )

    @pytest.fixture
    def hexagonal_structure(self):
        """
        Create a simple hexagonal structure.
        
        Returns
        -------
        Structure
            A simple hexagonal structure
        """
        # Hexagonal lattice
        a = 3.0
        c = 5.0
        lattice = np.array([
            [a, 0.0, 0.0],
            [-a/2, a*np.sqrt(3)/2, 0.0],
            [0.0, 0.0, c]
        ])
        
        atoms = ['C']
        fractional_coordinates = np.array([[1/3, 2/3, 0.0]])
        
        return Structure(
            atoms=atoms,
            fractional_coordinates=fractional_coordinates,
            lattice=lattice
        )

    def test_structure_initialization_with_fractional_coordinates(self, simple_cubic_structure):
        """Test Structure initialization with fractional coordinates."""
        struct = simple_cubic_structure
        
        assert struct.atoms.tolist() == ['H']
        assert np.allclose(struct.fractional_coordinates, [[0.0, 0.0, 0.0]])
        assert np.allclose(struct.cartesian_coordinates, [[0.0, 0.0, 0.0]])
        assert struct.has_complete_data is True

    def test_structure_initialization_with_cartesian_coordinates(self):
        """Test Structure initialization with cartesian coordinates."""
        lattice = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        
        atoms = ['H']
        cartesian_coordinates = np.array([[1.0, 1.0, 1.0]])
        
        struct = Structure(
            atoms=atoms,
            cartesian_coordinates=cartesian_coordinates,
            lattice=lattice
        )
        
        assert struct.atoms.tolist() == ['H']
        assert np.allclose(struct.cartesian_coordinates, [[1.0, 1.0, 1.0]])
        assert np.allclose(struct.fractional_coordinates, [[0.5, 0.5, 0.5]])

    def test_structure_initialization_incomplete_data(self):
        """Test Structure initialization with incomplete data."""
        struct = Structure(atoms=['H'])
        
        assert struct.has_complete_data is False
        assert struct.atoms.tolist() == ['H']
        assert struct.cartesian_coordinates is None
        assert struct.fractional_coordinates is None

    def test_structure_equality(self, simple_cubic_structure):
        """Test Structure equality comparison."""
        struct1 = simple_cubic_structure
        
        # Create identical structure
        lattice = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0]
        ])
        atoms = ['H']
        fractional_coordinates = np.array([[0.0, 0.0, 0.0]])
        
        struct2 = Structure(
            atoms=atoms,
            fractional_coordinates=fractional_coordinates,
            lattice=lattice
        )
        
        assert struct1 == struct2

    def test_structure_inequality(self, simple_cubic_structure, nacl_structure):
        """Test Structure inequality comparison."""
        assert simple_cubic_structure != nacl_structure

    def test_volume_property(self, simple_cubic_structure):
        """Test volume property calculation."""
        struct = simple_cubic_structure
        expected_volume = 8.0 * 1e-30  # 2^3 * 1e-30
        assert np.isclose(struct.volume, expected_volume)

    def test_lattice_parameters(self, simple_cubic_structure, hexagonal_structure):
        """Test lattice parameter properties (a, b, c, alpha, beta, gamma)."""
        # Test cubic structure
        cubic = simple_cubic_structure
        assert np.isclose(cubic.a, 2.0)
        assert np.isclose(cubic.b, 2.0)
        assert np.isclose(cubic.c, 2.0)
        assert np.isclose(cubic.alpha, 90.0)
        assert np.isclose(cubic.beta, 90.0)
        assert np.isclose(cubic.gamma, 90.0)
        
        # Test hexagonal structure
        hex_struct = hexagonal_structure
        assert np.isclose(hex_struct.a, 3.0)
        assert np.isclose(hex_struct.b, 3.0)
        assert np.isclose(hex_struct.c, 5.0)
        assert np.isclose(hex_struct.alpha, 90.0)
        assert np.isclose(hex_struct.beta, 90.0)
        assert np.isclose(hex_struct.gamma, 120.0)

    def test_masses_property(self, nacl_structure):
        """Test masses property calculation."""
        struct = nacl_structure
        masses = struct.masses
        
        assert len(masses) == 2  # Two atoms
        assert all(mass > 0 for mass in masses)  # All masses should be positive

    def test_density_property(self, simple_cubic_structure):
        """Test density property calculation."""
        struct = simple_cubic_structure
        density = struct.density
        
        assert density > 0  # Density should be positive

    def test_species_properties(self, nacl_structure):
        """Test species-related properties."""
        struct = nacl_structure
        
        assert set(struct.species) == {'Na', 'Cl'}
        assert struct.nspecies == 2
        assert struct.natoms == 2

    def test_atomic_numbers_property(self, nacl_structure):
        """Test atomic numbers property."""
        struct = nacl_structure
        atomic_numbers = struct.atomic_numbers
        
        assert len(atomic_numbers) == 2
        assert all(isinstance(num, int) for num in atomic_numbers)
        assert all(num > 0 for num in atomic_numbers)

    def test_reciprocal_lattice_property(self, simple_cubic_structure):
        """Test reciprocal lattice property."""
        struct = simple_cubic_structure
        recip_lattice = struct.reciprocal_lattice
        
        assert recip_lattice.shape == (3, 3)
  
    def test_lattice_corners_property(self, simple_cubic_structure):
        """Test lattice corners property."""
        struct = simple_cubic_structure
        corners = struct.lattice_corners
        
        assert corners.shape == (8, 3)  # 8 corners for a 3D lattice
        
        # Check that we have all expected corners
        expected_corners = [
            [0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2],
            [2, 2, 0], [2, 0, 2], [0, 2, 2], [2, 2, 2]
        ]
        
        for expected_corner in expected_corners:
            assert any(np.allclose(corner, expected_corner) for corner in corners)

    def test_cell_convex_hull_property(self, simple_cubic_structure):
        """Test cell convex hull property."""
        struct = simple_cubic_structure
        convex_hull = struct.cell_convex_hull
        
        assert hasattr(convex_hull, 'vertices')
        assert hasattr(convex_hull, 'simplices')
        assert len(convex_hull.vertices) == 8  # 8 vertices for a cube

    def test_spglib_cell_property(self, nacl_structure):
        """Test _spglib_cell property."""
        struct = nacl_structure
        spglib_cell = struct._spglib_cell
        
        assert len(spglib_cell) == 3  # (lattice, frac_coords, atomic_numbers)
        lattice, frac_coords, atomic_numbers = spglib_cell
        
        assert np.allclose(lattice, struct.lattice)
        assert np.allclose(frac_coords, struct.fractional_coordinates)
        assert atomic_numbers == struct.atomic_numbers

    def test_get_space_group_number(self, simple_cubic_structure):
        """Test get_space_group_number method."""
        struct = simple_cubic_structure
        space_group_num = struct.get_space_group_number()
        
        assert isinstance(space_group_num, int)
        assert 1 <= space_group_num <= 230  # Valid space group numbers

    def test_get_space_group_international(self, simple_cubic_structure):
        """Test get_space_group_international method."""
        struct = simple_cubic_structure
        space_group_int = struct.get_space_group_international()
        
        assert isinstance(space_group_int, str)
        assert len(space_group_int) > 0

    def test_get_wyckoff_positions(self, simple_cubic_structure):
        """Test get_wyckoff_positions method."""
        struct = simple_cubic_structure
        wyckoff_positions = struct.get_wyckoff_positions()
        
        assert wyckoff_positions is not None
        assert len(wyckoff_positions) == struct.natoms
        assert struct.wyckoff_positions is not None
        assert struct.group is not None

    def test_get_spglib_symmetry_dataset(self, simple_cubic_structure):
        """Test get_spglib_symmetry_dataset method."""
        struct = simple_cubic_structure
        dataset = struct.get_spglib_symmetry_dataset()
        
        assert isinstance(dataset, dict) or hasattr(dataset, 'number')

    def test_is_point_inside_method(self, simple_cubic_structure):
        """Test is_point_inside method."""
        struct = simple_cubic_structure
        
        # Point inside the cell
        point_inside = np.array([1.0, 1.0, 1.0])
        assert struct.is_point_inside(point_inside) is True
        
        # Point outside the cell
        point_outside = np.array([3.0, 3.0, 3.0])
        assert struct.is_point_inside(point_outside) is False

    def test_transform_method(self, simple_cubic_structure):
        """Test transform method."""
        struct = simple_cubic_structure
        
        # 2x2x2 supercell transformation
        transformation_matrix = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        
        transformed = struct.transform(transformation_matrix)
        
        assert isinstance(transformed, Structure)
        assert transformed.natoms >= struct.natoms  # Should have more atoms
        assert np.allclose(transformed.lattice, np.dot(struct.lattice, transformation_matrix))

    def test_transform_invalid_matrix(self, simple_cubic_structure):
        """Test transform method with invalid transformation matrix."""
        struct = simple_cubic_structure
        
        # Invalid transformation matrix (not proper)
        invalid_matrix = np.array([
            [1.5, 0, 0],
            [0, 1.5, 0],
            [0, 0, 1.5]
        ])
        
        with pytest.raises(ValueError):
            struct.transform(invalid_matrix)

    def test_supercell_method(self, simple_cubic_structure):
        """Test supercell method."""
        struct = simple_cubic_structure
        
        # 2x2x2 supercell
        matrix = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        
        supercell = struct.supercell(matrix)
        
        assert isinstance(supercell, Structure)
        assert supercell.natoms >= struct.natoms  # Should have more atoms

    def test_structure_with_empty_arrays(self):
        """Test Structure with empty arrays."""
        
        with pytest.raises(ValueError):
            struct = Structure(atoms=[], fractional_coordinates=[], lattice=np.eye(3))
        
        with pytest.raises(ValueError):
            struct = Structure()
        

    def test_structure_properties_with_single_atom(self, simple_cubic_structure):
        """Test various properties with single atom structure."""
        struct = simple_cubic_structure
        
        # Basic properties
        assert struct.natoms == 1
        assert struct.nspecies == 1
        assert list(struct.species) == ['H']
        
        # Should not raise errors
        assert struct.volume > 0
        assert len(struct.masses) == 1
        assert struct.density > 0
