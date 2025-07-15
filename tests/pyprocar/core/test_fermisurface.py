import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from pyprocar.core.ebs import (
    ElectronicBandStructure,
    ElectronicBandStructureMesh,
    ElectronicBandStructurePath,
)
from pyprocar.core.fermisurface import FermiSurface
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)
user_logger = logging.getLogger("user")



@pytest.fixture
def mesh_3d_non_spin_polarized_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi3d" / "non-spin-polarized"

@pytest.fixture
def fermisurface_3d_non_spin_polarized(mesh_3d_non_spin_polarized_dir):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=mesh_3d_non_spin_polarized_dir)
    return fs


@pytest.fixture
def mesh_3d_spin_polarized_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi3d" / "spin-polarized"

@pytest.fixture
def fermisurface_3d_spin_polarized(mesh_3d_spin_polarized_dir):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=mesh_3d_spin_polarized_dir)
    return fs

@pytest.fixture
def mesh_3d_non_colinear_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi3d" / "non-colinear"

@pytest.fixture
def fermisurface_3d_non_colinear(mesh_3d_non_colinear_dir):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=mesh_3d_non_colinear_dir)
    return fs

@pytest.fixture
def mesh_2d_non_spin_polarized_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "non-spin-polarized"

@pytest.fixture
def fermisurface_2d_non_spin_polarized(mesh_2d_non_spin_polarized_dir):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=mesh_2d_non_spin_polarized_dir)
    return fs

@pytest.fixture
def mesh_2d_spin_polarized_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "spin-polarized"

@pytest.fixture
def fermisurface_2d_spin_polarized(mesh_2d_spin_polarized_dir):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=mesh_2d_spin_polarized_dir)
    return fs

@pytest.fixture
def mesh_2d_non_colinear_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "non-colinear"

@pytest.fixture
def fermisurface_2d_non_colinear(mesh_2d_non_colinear_dir):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=mesh_2d_non_colinear_dir)
    return fs

@pytest.fixture
def bisb_monolayer_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "bisb_monolayer"

@pytest.fixture
def fermisurface_bisb_monolayer(bisb_monolayer_dir):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=bisb_monolayer_dir)
    return fs


ALL_TEST_CASES = [
    "fermisurface_3d_non_spin_polarized",
    "fermisurface_3d_spin_polarized", 
    "fermisurface_3d_non_colinear",
    "fermisurface_2d_non_spin_polarized",
    "fermisurface_2d_spin_polarized",
    "fermisurface_2d_non_colinear",
    # "fermisurface_bisb_monolayer",  # TODO: Fix array to mesh conversion issue
]


def get_test_id(fixture_name: str) -> str:
    """Creates a nice, readable ID for each test run."""
    return fixture_name


@pytest.fixture(params=ALL_TEST_CASES, ids=get_test_id)
def fermisurface(request):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one fixture name at a time.
    """
    return request.getfixturevalue(request.param)

class TestFermiSurface:
    def test_select_bands(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        
        # Get the original number of surfaces and points
        original_surface_count = len(fs.band_spin_mask)
        original_point_count = fs.n_points
        
        # Select the first band-spin combination
        band_spin_mask_key = list(fs.band_spin_mask.keys())[0]
        band_spin_indices = [band_spin_mask_key]
        
        band_spin_mask_sum = fs.band_spin_mask[band_spin_mask_key].sum()

        # Test that we can select bands without error
        selected_fs = fs.select_bands(band_spin_indices, inplace=False)
        
        # The selected surface should have fewer or equal points than the original
        assert selected_fs.points.shape[0] + band_spin_mask_sum == original_point_count
        
  