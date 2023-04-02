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

def create_dummy_electronic_band_structure():
    parser = Parser(code = 'vasp', dir = DATA_DIR)
    parser.ebs.ibz2fbz(parser.structure.rotations)
    return parser.ebs

@pytest.fixture
def electronic_band_structure():
    return create_dummy_electronic_band_structure()

def test_n_kx(electronic_band_structure):
    assert electronic_band_structure.n_kx == 15

def test_n_ky(electronic_band_structure):
    assert electronic_band_structure.n_ky == 15

def test_n_kz(electronic_band_structure):
    assert electronic_band_structure.n_kz == 15

def test_nkpoints(electronic_band_structure):
    assert electronic_band_structure.nkpoints == 3375

def test_nbands(electronic_band_structure):
    assert electronic_band_structure.nbands == 8

def test_natoms(electronic_band_structure):
    assert electronic_band_structure.natoms == 1

def test_nprincipals(electronic_band_structure):
    assert electronic_band_structure.nprincipals == 1

def test_norbitals(electronic_band_structure):
    assert electronic_band_structure.norbitals == 9

def test_nspins(electronic_band_structure):
    assert electronic_band_structure.nspins == 1

def test_is_non_collinear(electronic_band_structure):
    assert electronic_band_structure.is_non_collinear == False

def test_kpoints_cartesian(electronic_band_structure):
    expected_cartesian_kpoints = np.array([[0,0,0],[0.14748992,0.14748992,0]])
    np.testing.assert_array_almost_equal(electronic_band_structure.kpoints_cartesian[:2], expected_cartesian_kpoints)

def test_kpoints(electronic_band_structure):
    expected_cartesian_kpoints = np.array([[0,0,0],[0.06666667,-0,-0]])
    np.testing.assert_array_almost_equal(electronic_band_structure.kpoints[:2], expected_cartesian_kpoints)

def test_index_mesh(electronic_band_structure):
    assert electronic_band_structure.index_mesh.shape == (15,15,15)

def test_kpoints_mesh(electronic_band_structure):
    assert electronic_band_structure.kpoints_mesh.shape == (3,15,15,15)

def test_kpoints_cartesian_mesh(electronic_band_structure):
    assert electronic_band_structure.kpoints_cartesian_mesh.shape == (3,15,15,15)

def test_bands_mesh(electronic_band_structure):
    assert electronic_band_structure.bands_mesh.shape == (8,1,15,15,15)

def test_projected_mesh(electronic_band_structure):
    assert electronic_band_structure.projected_mesh.shape == (8,1,1,9,1,15,15,15)

def test_projected_phase_mesh(electronic_band_structure):
    assert electronic_band_structure.projected_phase == None

def test_weights_mesh(electronic_band_structure):
    assert electronic_band_structure.weights_mesh == None

def test_bands_gradient_mesh(electronic_band_structure):
    assert electronic_band_structure.bands_gradient_mesh.shape == (3,8,1,15,15,15)

def test_bands_hessian_mesh(electronic_band_structure):
    assert electronic_band_structure.bands_hessian_mesh.shape == (3,3,8,1,15,15,15)

