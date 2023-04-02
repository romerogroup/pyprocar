import os
import pytest
import numpy as np
from pyprocar.core import DensityOfStates
from pyprocar.core.dos import interpolate
from scipy.interpolate import CubicSpline
import pyprocar
DATA_DIR = f"{pyprocar.PROJECT_DIR}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}dos"


# Create a fixture to provide a DensityOfStates instance for testing
@pytest.fixture
def example_dos():
    parser = pyprocar.io.Parser(code = 'vasp', dir = DATA_DIR)
    return parser.dos


def test_dos_class_properties(example_dos):
    # Test n_dos, n_energies, and n_spins properties
    assert example_dos.n_dos == 301
    assert example_dos.n_energies == 301
    assert example_dos.n_spins == 1

    # Test is_non_collinear property
    assert example_dos.is_non_collinear == False


def test_dos_sum_method(example_dos):
    # Test dos_sum method with default parameters
    summed_dos = example_dos.dos_sum()
    expected_summed_dos = np.array([
        [0, 0.1, 0.4, 0.9, 2, 4, 2, 0.9, 0.4, 0.1, 0],
        [0, 0.05, 0.2, 0.45, 1, 2, 1, 0.45, 0.2, 0.05, 0]
    ])

    assert summed_dos.shape == (1,301)

def test_get_current_basis(example_dos):
    # Test get_current_basis method
    assert example_dos.get_current_basis() == 'spd basis'

