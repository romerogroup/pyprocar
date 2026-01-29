import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.ebs import (
    ElectronicBandStructureMesh,
)
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.property_store import PointSet, Property
from tests.utils import DATA_DIR

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)
user_logger = logging.getLogger("user")


@pytest.fixture
def mesh_3d_non_spin_polarized_dir() -> Path:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi3d" / "non-spin-polarized"


@pytest.fixture
def fermisurface_3d_non_spin_polarized(mesh_3d_non_spin_polarized_dir: Path) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=str(mesh_3d_non_spin_polarized_dir))
    return fs


@pytest.fixture
def mesh_3d_spin_polarized_dir() -> Path:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi3d" / "spin-polarized"


@pytest.fixture
def fermisurface_3d_spin_polarized(mesh_3d_spin_polarized_dir: Path) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=str(mesh_3d_spin_polarized_dir))
    return fs


@pytest.fixture
def mesh_3d_non_colinear_dir() -> Path:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi3d" / "non-colinear"


@pytest.fixture
def fermisurface_3d_non_colinear(mesh_3d_non_colinear_dir: Path) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=str(mesh_3d_non_colinear_dir))
    return fs


@pytest.fixture
def mesh_2d_non_spin_polarized_dir() -> Path:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "non-spin-polarized"


@pytest.fixture
def fermisurface_2d_non_spin_polarized(mesh_2d_non_spin_polarized_dir: Path) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=str(mesh_2d_non_spin_polarized_dir))
    return fs


@pytest.fixture
def mesh_2d_spin_polarized_dir() -> Path:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "spin-polarized"


@pytest.fixture
def fermisurface_2d_spin_polarized(mesh_2d_spin_polarized_dir: Path) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=str(mesh_2d_spin_polarized_dir))
    return fs


@pytest.fixture
def mesh_2d_non_colinear_dir() -> Path:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "non-colinear"


@pytest.fixture
def fermisurface_2d_non_colinear(mesh_2d_non_colinear_dir: Path) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=str(mesh_2d_non_colinear_dir))
    return fs


@pytest.fixture
def bisb_monolayer_dir() -> Path:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "fermi2d" / "bisb_monolayer"


@pytest.fixture
def fermisurface_bisb_monolayer(bisb_monolayer_dir: Path) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.
    """
    fs = FermiSurface.from_code(code="vasp", dirpath=str(bisb_monolayer_dir))
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
def fermisurface(request: pytest.FixtureRequest) -> FermiSurface:
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one fixture name at a time.
    """
    fixture_value: FermiSurface = request.getfixturevalue(request.param)
    return fixture_value


class TestFermiSurface:
    def test_fermi_surface_creations(self, mesh_3d_non_spin_polarized_dir: Path) -> None:
        fs = FermiSurface.from_code(
            code="vasp", dirpath=str(mesh_3d_non_spin_polarized_dir), padding=10
        )
        assert fs.points.shape[0] > 0

        _ebs = fs.ebs
        assert isinstance(fs.original_ebs, ElectronicBandStructureMesh)
        assert isinstance(fs.ebs, ElectronicBandStructureMesh)
        assert isinstance(fs.point_set, PointSet)
        assert isinstance(fs.isovalue, float)
        assert isinstance(fs.band_isosurfaces, dict)
        reciprocal_lattice = fs.ebs.reciprocal_lattice
        assert reciprocal_lattice is not None
        assert np.allclose(fs.transform_matrix_to_cart[:3, :3], reciprocal_lattice.T)
        assert fs.isovalue == fs.ebs.fermi
        assert fs.fermi_shift == 0.0
        assert np.allclose(reciprocal_lattice.shape, np.array([3, 3]))

    def test_select_bands(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        fs = fermisurface_3d_non_spin_polarized

        # Get the original number of surfaces and points
        _original_surface_count = len(fs.band_spin_mask)
        original_point_count = fs.n_points

        # Select the first band-spin combination
        band_spin_mask_key = list(fs.band_spin_mask.keys())[0]
        band_spin_indices = [band_spin_mask_key]

        band_spin_mask_sum = fs.band_spin_mask[band_spin_mask_key].sum()

        # Test that we can select bands without error
        selected_result = fs.select_bands(band_spin_indices)
        assert isinstance(selected_result, FermiSurface)
        selected_fs = selected_result

        # The selected surface should have fewer or equal points than the original
        assert selected_fs.points.shape[0] + band_spin_mask_sum == original_point_count

    def test_set_band_colors(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        fs = fermisurface_3d_non_spin_polarized
        fs.set_band_colors(colors=["red", "blue", "green"])
        assert fs.point_data["bands"].shape[0] == fs.n_points
        assert fs.point_data["bands"].shape[1] == 4

        with pytest.raises(AssertionError):
            fs.set_band_colors(colors=["red", "blue", "green", "yellow", "purple"])

        with pytest.raises(AssertionError):
            fs.set_band_colors(
                surface_color_map={(0, 0): "red", (1, 0): "blue", (0, 1): "green", (1, 1): "yellow"}
            )

        fs.set_band_colors(surface_color_map={(16, 0): "red", (17, 0): "blue", (18, 0): "green"})
        assert fs.point_data["bands"].shape[0] == fs.n_points
        assert fs.point_data["bands"].shape[1] == 4

        fs.set_band_colors()
        assert fs.point_data["bands"].shape[0] == fs.n_points
        assert fs.point_data["bands"].shape[1] == 4

    def test_set_spin_colors(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        fs = fermisurface_3d_non_spin_polarized
        fs.set_spin_colors(colors=("red", "blue"))
        assert fs.point_data["spin"].shape[0] == fs.n_points
        assert fs.point_data["spin"].shape[1] == 4

    def test_get_property(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        fs = fermisurface_3d_non_spin_polarized
        _property = fs.get_property("bands")

        assert _property.shape[0] == fs.n_points
        assert "bands" in fs.point_data

        _property = fs.get_property("fermi_speed")
        assert _property.shape[0] == fs.n_points
        assert "fermi_speed" in fs.point_data
        assert fs.point_data["fermi_speed"].shape[0] == fs.n_points
        assert len(fs.point_data["fermi_speed"].shape) == 1

        _property = fs.get_property("fermi_velocity")
        assert _property.shape[0] == fs.n_points
        assert "fermi_velocity" in fs.point_data
        assert fs.point_data["fermi_velocity"].shape[0] == fs.n_points
        assert fs.point_data["fermi_velocity"].shape[1] == 3

        _property = fs.get_property("avg_inv_effective_mass")
        assert _property.shape[0] == fs.n_points
        assert "avg_inv_effective_mass" in fs.point_data
        assert fs.point_data["avg_inv_effective_mass"].shape[0] == fs.n_points
        assert len(fs.point_data["avg_inv_effective_mass"].shape) == 1

    def test_extend_surface(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        fs = fermisurface_3d_non_spin_polarized
        zone_directions: list[list[int] | tuple[int, int, int]] = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
        n_zones = len(zone_directions) + 1
        extended_fs = fs.extend_surface(zone_directions=zone_directions)
        assert extended_fs.points.shape[0] == fs.n_points * n_zones, (
            f"extended_fs.points.shape[0]: {extended_fs.points.shape[0]} does not match fs.n_points*n_zones: {fs.n_points * n_zones}"
        )
        assert extended_fs.n_points == fs.n_points * n_zones, (
            f"extended_fs.n_points: {extended_fs.n_points} does not match fs.n_points*n_zones: {fs.n_points * n_zones}"
        )

        # Testing properties
        for prop_name, _property in fs.point_set.property_store.items():
            extended_property_result = extended_fs.point_set.get_property(prop_name)
            assert isinstance(extended_property_result, Property)

            extended_values = extended_property_result.value
            old_values = _property.value
            assert np.allclose(extended_values[: fs.n_points], old_values), (
                f"extended_values: {extended_values} does not match old_values: {old_values}"
            )
            assert np.allclose(extended_values[-1 * fs.n_points :], old_values), (
                f"extended_values: {extended_values} does not match old_values: {old_values}"
            )

    def test_2dmesh_fermisurface_creation(self, fermisurface_2d_non_colinear: FermiSurface) -> None:
        fs = fermisurface_2d_non_colinear
        assert isinstance(fs.ebs, ElectronicBandStructureMesh)
        assert isinstance(fs.point_set, PointSet)
        assert isinstance(fs.isovalue, float)
        assert isinstance(fs.band_isosurfaces, dict)
        assert fs.isovalue == fs.ebs.fermi
        assert fs.fermi_shift == 0.0


class TestFermiSurfaceNormalization:
    """Tests for FermiSurface normalization system."""

    def test_normalize_raw(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        """Test that raw normalization returns unchanged values."""
        fs = fermisurface_3d_non_spin_polarized
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = fs.normalize("raw", values)

        np.testing.assert_array_equal(result, values)

    def test_normalize_max(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        """Test max normalization produces values in [-1, 1]."""
        fs = fermisurface_3d_non_spin_polarized
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = fs.normalize("max", values)

        assert np.max(np.abs(result)) == 1.0
        assert result[-1] == 1.0  # Max value normalized to 1

    def test_normalize_total(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        """Test total normalization produces values that sum to 1."""
        fs = fermisurface_3d_non_spin_polarized
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = fs.normalize("total", values)

        assert np.isclose(np.sum(result), 1.0)

    def test_fsnormmode_from_input(self) -> None:
        """Test FSNormMode.from_input conversion."""
        from pyprocar.core.fermisurface import FSNormMode

        assert FSNormMode.from_input("raw") == FSNormMode.RAW
        assert FSNormMode.from_input("max") == FSNormMode.MAX
        assert FSNormMode.from_input("MAX") == FSNormMode.MAX  # Case insensitive
        assert FSNormMode.from_input(None) == FSNormMode.RAW
        assert FSNormMode.from_input(FSNormMode.TOTAL) == FSNormMode.TOTAL

    def test_fsnormmode_list_modes(self) -> None:
        """Test FSNormMode.list_modes returns all modes."""
        from pyprocar.core.fermisurface import FSNormMode

        modes = FSNormMode.list_modes()
        assert "raw" in modes
        assert "max" in modes
        assert "total" in modes
        assert "integral" in modes
        assert len(modes) == 4

    def test_fsnormmode_invalid_input(self) -> None:
        """Test FSNormMode.from_input raises error on invalid input."""
        from pyprocar.core.fermisurface import FSNormMode

        with pytest.raises(ValueError):
            FSNormMode.from_input("invalid_mode")


class TestFermiSurfaceSerialization:
    """Tests for FermiSurface save/load functionality."""

    def test_save_load_roundtrip(
        self, fermisurface_3d_non_spin_polarized: FermiSurface, tmp_path: Path
    ) -> None:
        """Test that save/load preserves all data."""
        fs = fermisurface_3d_non_spin_polarized
        save_path = tmp_path / "test_fs.pkl"

        # Save
        fs.save(str(save_path))
        assert save_path.exists()

        # Load
        fs2 = FermiSurface.load(str(save_path))

        # Verify
        assert fs2.n_points == fs.n_points
        np.testing.assert_array_almost_equal(fs2.points, fs.points)
        assert fs2.isovalue == fs.isovalue

    def test_save_load_preserves_point_data(
        self, fermisurface_3d_non_spin_polarized: FermiSurface, tmp_path: Path
    ) -> None:
        """Test that point_data is preserved through save/load."""
        fs = fermisurface_3d_non_spin_polarized

        # Add some point data
        test_data = np.random.rand(fs.n_points)
        fs.point_data["test_scalar"] = test_data

        save_path = tmp_path / "test_fs_data.pkl"
        fs.save(str(save_path))
        fs2 = FermiSurface.load(str(save_path))

        assert "test_scalar" in fs2.point_data
        np.testing.assert_array_almost_equal(fs2.point_data["test_scalar"], test_data)

    def test_save_load_preserves_band_isosurfaces(
        self, fermisurface_3d_non_spin_polarized: FermiSurface, tmp_path: Path
    ) -> None:
        """Test that band_isosurfaces are preserved through save/load."""
        fs = fermisurface_3d_non_spin_polarized
        original_keys = set(fs.band_isosurfaces.keys())

        save_path = tmp_path / "test_fs_bands.pkl"
        fs.save(str(save_path))
        fs2 = FermiSurface.load(str(save_path))

        assert set(fs2.band_isosurfaces.keys()) == original_keys


class TestFermiSurfaceCache:
    """Tests for FermiSurface caching system."""

    def test_cache_invalidation(self, fermisurface_3d_non_spin_polarized: FermiSurface) -> None:
        """Test that cache invalidation works."""
        fs = fermisurface_3d_non_spin_polarized

        # Mark something as cached
        fs._mark_cached("test_prop")
        assert fs._is_cache_valid("test_prop")

        # Invalidate
        fs._invalidate_cache()
        assert not fs._is_cache_valid("test_prop")

    def test_cache_version_increments(
        self, fermisurface_3d_non_spin_polarized: FermiSurface
    ) -> None:
        """Test that cache version increments on invalidation."""
        fs = fermisurface_3d_non_spin_polarized

        initial_version = fs._ebs_cache_version
        fs._invalidate_cache()

        assert fs._ebs_cache_version == initial_version + 1

    def test_uncached_property_invalid(
        self, fermisurface_3d_non_spin_polarized: FermiSurface
    ) -> None:
        """Test that uncached properties are detected as invalid."""
        fs = fermisurface_3d_non_spin_polarized

        assert not fs._is_cache_valid("nonexistent_property")
