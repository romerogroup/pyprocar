import os
import pytest
import numpy as np
from typing import List
from scipy.interpolate import CubicSpline
import networkx as nx
from matplotlib import pylab as plt
import pyvista
from pyprocar.core import KPath
from pyprocar.io import Parser
import pyprocar
# Helper function to create a dummy ElectronicBandStructure instance

DATA_DIR = f"{pyprocar.PROJECT_DIR}{os.sep}data{os.sep}examples{os.sep}Fe{os.sep}vasp{os.sep}non-spin-polarized{os.sep}fermi"


@pytest.fixture
def sample_kpath():
    knames = [["$\\Gamma$", "X"], ["X", "M"], ["M", "$\\Gamma$"]]
    special_kpoints = [
        [np.array([0, 0, 0]), np.array([1, 0, 0])],
        [np.array([1, 0, 0]), np.array([1, 1, 0])],
        [np.array([1, 1, 0]), np.array([0, 0, 0])]
    ]
    ngrids = [10, 10, 10]
    return KPath(knames=knames, special_kpoints=special_kpoints, ngrids=ngrids)

def test_nsegments(sample_kpath):
    assert sample_kpath.nsegments == 3

def test_tick_positions(sample_kpath):
    assert sample_kpath.tick_positions == [0, 9, 19, 29]

def test_tick_names(sample_kpath):
    assert sample_kpath.tick_names == ["$\\Gamma$", "X","M", "$\\Gamma$"]

def test_kdistances(sample_kpath):
    print(sample_kpath.kdistances)
    assert np.allclose(sample_kpath.kdistances, [1.,1.,1.41421356])
