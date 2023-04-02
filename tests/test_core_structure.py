import os
import pytest
import numpy as np
from typing import List
from scipy.interpolate import CubicSpline
import networkx as nx
from matplotlib import pylab as plt
import pyvista
from pyprocar.core import ElectronicBandStructure
from pyprocar.io import Parser
import pyprocar
# Helper function to create a dummy ElectronicBandStructure instance

DATA_DIR = f"{pyprocar.PROJECT_DIR}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}fermi"

def create_dummy_structure():
    parser = Parser(code = 'vasp', dir = DATA_DIR)
    return parser.structure

@pytest.fixture
def sample_structure():
    return create_dummy_structure()


def test_structure_initialization(sample_structure):
    assert len(sample_structure.atoms) == 1
    assert sample_structure.atoms == ['Fe']
    assert np.array_equal(sample_structure.fractional_coordinates, np.array([[0., 0., 0.]]))
    assert np.array_equal(sample_structure.lattice, np.array([[ 1.420026,  1.420026,  1.420026],
                                                    [ 1.420026, -1.420026, -1.420026],
                                                    [-1.420026,  1.420026, -1.420026]]))

def test_structure_properties(sample_structure):
    assert sample_structure.volume == pytest.approx(1.1453781128319115e-29, rel=1e-6)
    assert sample_structure.masses == [0.055845]
    assert sample_structure.density == pytest.approx(8096.261110086697, rel=1e-3)
    assert sample_structure.a == pytest.approx(2.4595571800688028, rel=1e-2)
    assert sample_structure.b == pytest.approx(2.4595571800688028, rel=1e-2)
    assert sample_structure.c == pytest.approx(2.4595571800688028, rel=1e-2)
    assert sample_structure.alpha == pytest.approx(109.47122063449069, rel=1e-6)
    assert sample_structure.beta == pytest.approx(109.47122063449069, rel=1e-6)
    assert sample_structure.gamma == pytest.approx(109.47122063449069, rel=1e-6)
    assert sample_structure.species == ['Fe']
    assert sample_structure.nspecies == 1
    assert sample_structure.natoms == 1
    assert sample_structure.atomic_numbers == [26]

def test_reciprocal_lattice(sample_structure):
    expected_reciprocal_lattice = np.array([[ 2.21234868,2.21234868,0.],
                                            [ 2.21234868, 0.,-2.21234868],
                                            [ 0.,2.21234868,-2.21234868]])
    np.testing.assert_almost_equal(sample_structure.reciprocal_lattice, expected_reciprocal_lattice, decimal=6)



def test_structure_methods(sample_structure):


    assert sample_structure.get_space_group_number() == 229
    assert sample_structure.get_space_group_international() == 'Im-3m'

    wyckoff_positions = sample_structure.get_wyckoff_positions()
    assert np.all(wyckoff_positions == np.array(['1a']))

    lattice_corners = sample_structure.lattice_corners
    assert np.allclose(lattice_corners, np.array([[ 0.,        0.,        0.      ],
                                                [-1.420026 , 1.420026, -1.420026],
                                                [ 1.420026, -1.420026, -1.420026],
                                                [ 0.,        0. ,      -2.840052],
                                                [ 1.420026,  1.420026 , 1.420026],
                                                [ 0.,        2.840052,  0.      ],
                                                [ 2.840052,  0.,        0.      ],
                                                [ 1.420026,  1.420026, -1.420026]]))

    cell_convex_hull = sample_structure.cell_convex_hull
    assert cell_convex_hull.area == 34.220695843854756
    assert cell_convex_hull.volume == 11.453781128319108

    spglib_symmetry_dataset = sample_structure.get_spglib_symmetry_dataset()
    assert spglib_symmetry_dataset['number'] == 229
    assert spglib_symmetry_dataset['international'] == 'Im-3m'

    # Test transform method
    transformation_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    transformed_structure = sample_structure.transform(transformation_matrix)

    assert np.allclose(transformed_structure.lattice,  np.array([[ 2.840052,  2.840052,  2.840052],
                                                        [ 2.840052, -2.840052, -2.840052],
                                                        [-2.840052,  2.840052, -2.840052]]))
    assert np.all(transformed_structure.atoms == np.array(['Fe']*8))
    assert transformed_structure.natoms == 8

    # Test is_point_inside method
    point_inside = np.array([0.5, 0.5, 0.5])
    point_outside = np.array([1.5, 1.5, 1.5])

    assert sample_structure.is_point_inside(point_inside) == True
    assert sample_structure.is_point_inside(point_outside) == False


    supercell_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    supercell_structure = sample_structure.supercell(supercell_matrix)

    assert np.allclose(supercell_structure.lattice, np.array([[ 2.840052,  2.840052,  2.840052],
                                                        [ 2.840052, -2.840052, -2.840052],
                                                        [-2.840052,  2.840052, -2.840052]]))
    assert np.all(supercell_structure.atoms == np.array(['Fe'] * 8))
    assert supercell_structure.natoms == 8