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
from pyprocar.core.property_store import PointSet
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
    
    def test_fermi_surface_creations(self, mesh_3d_non_spin_polarized_dir):
        fs = FermiSurface.from_code(code="vasp", dirpath=mesh_3d_non_spin_polarized_dir, padding=10)
        assert fs.points.shape[0] > 0
        
        ebs = fs.ebs
        print(type(ebs))
        assert isinstance(fs.original_ebs, ElectronicBandStructureMesh)
        assert isinstance(fs.ebs, ElectronicBandStructureMesh)
        assert isinstance(fs.point_set, PointSet)
        assert isinstance(fs.isovalue, float)
        assert isinstance(fs.band_isosurfaces, dict)
        assert np.allclose(fs.transform_matrix_to_cart[:3, :3], fs.ebs.reciprocal_lattice.T)
        assert fs.isovalue == fs.ebs.fermi
        assert fs.fermi_shift == 0.0
        assert np.allclose(fs.ebs.reciprocal_lattice.shape, np.array([3, 3]))
        
    
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
        selected_fs = fs.select_bands(band_spin_indices)
        
        # The selected surface should have fewer or equal points than the original
        assert selected_fs.points.shape[0] + band_spin_mask_sum == original_point_count
        
    def test_set_band_colors(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        fs.set_band_colors(colors=["red", "blue", "green"])
        assert fs.point_data["bands"].shape[0] == fs.n_points
        assert fs.point_data["bands"].shape[1] == 4
        
        with pytest.raises(AssertionError):
            fs.set_band_colors(colors=["red", "blue", "green", "yellow", "purple"])
        
        with pytest.raises(AssertionError):
            fs.set_band_colors(surface_color_map={(0, 0): "red", (1, 0): "blue", (0, 1): "green", (1, 1): "yellow"})
            

        fs.set_band_colors(surface_color_map={(16, 0): "red", (17, 0): "blue", (18, 0): "green"})
        assert fs.point_data["bands"].shape[0] == fs.n_points
        assert fs.point_data["bands"].shape[1] == 4
        
        fs.set_band_colors()
        assert fs.point_data["bands"].shape[0] == fs.n_points
        assert fs.point_data["bands"].shape[1] == 4
        
        
    def test_set_spin_colors(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        fs.set_spin_colors(colors=("red", "blue"))
        assert fs.point_data["spin"].shape[0] == fs.n_points
        assert fs.point_data["spin"].shape[1] == 4
        
        
    def test_get_property(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        property = fs.get_property("bands")
        
        assert property.shape[0] == fs.n_points
        assert "bands" in fs.point_data
        
        property = fs.get_property("fermi_speed")
        assert property.shape[0] == fs.n_points
        assert "fermi_speed" in fs.point_data
        assert fs.point_data["fermi_speed"].shape[0] == fs.n_points
        assert len(fs.point_data["fermi_speed"].shape) == 1
        
        property = fs.get_property("fermi_velocity")
        assert property.shape[0] == fs.n_points
        assert "fermi_velocity" in fs.point_data
        assert fs.point_data["fermi_velocity"].shape[0] == fs.n_points
        assert fs.point_data["fermi_velocity"].shape[1] == 3
        
        property = fs.get_property("avg_inv_effective_mass")
        assert property.shape[0] == fs.n_points
        assert "avg_inv_effective_mass" in fs.point_data
        assert fs.point_data["avg_inv_effective_mass"].shape[0] == fs.n_points
        assert len(fs.point_data["avg_inv_effective_mass"].shape) == 1
        

        
        
        