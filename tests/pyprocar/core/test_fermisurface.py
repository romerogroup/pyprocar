import logging

import numpy as np
import pytest

from pyprocar.core.ebs import (
    ElectronicBandStructureMesh,
)
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.property_store import PointSet
from pyprocar.utils import math
from pyprocar.utils.physics import HBAR_EV, METER_ANGSTROM
from tests.utils import DATA_DIR

pytestmark = pytest.mark.data

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

        key = list(fs.band_spin_mask.keys())[0]
        mask = fs.band_spin_mask[key]

        selected_fs = fs.select_bands([key])

        assert list(selected_fs.band_spin_mask) == [key]
        assert selected_fs.n_points == mask.sum()
        assert np.allclose(selected_fs.points, fs.points[mask])

    def test_set_band_colors(self, fermisurface_3d_non_spin_polarized):
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

    def test_set_spin_colors(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        fs.set_spin_colors(colors=("red", "blue"))
        assert fs.point_data["spin"].shape[0] == fs.n_points
        assert fs.point_data["spin"].shape[1] == 4

    def test_get_property(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        active_before = fs.active_scalars_name

        for name in ("bands", "fermi_speed", "fermi_velocity", "avg_inv_effective_mass"):
            prop = fs.get_property(name)
            assert prop.shape[0] == fs.n_points

        assert fs.active_scalars_name == active_before

        fs.set_values("fermi_speed", fs.get_property("fermi_speed").value)
        assert fs.point_data["fermi_speed"].shape == (fs.n_points,)
        fs.set_values("fermi_velocity", fs.get_property("fermi_velocity").value)
        assert fs.point_data["fermi_velocity"].shape == (fs.n_points, 3)

    def test_extend_surface(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        zone_directions = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
        n_zones = len(zone_directions) + 1
        extended_fs = fs.extend_surface(zone_directions=zone_directions)
        assert extended_fs.points.shape[0] == fs.n_points * n_zones, (
            f"extended_fs.points.shape[0]: {extended_fs.points.shape[0]} does not match fs.n_points*n_zones: {fs.n_points * n_zones}"
        )
        assert extended_fs.n_points == fs.n_points * n_zones, (
            f"extended_fs.n_points: {extended_fs.n_points} does not match fs.n_points*n_zones: {fs.n_points * n_zones}"
        )

        # Testing properties
        for prop_name, property in fs.point_set.property_store.items():
            extended_property = extended_fs.point_set.get_property(prop_name)

            extended_values = extended_property.value
            old_values = property.value
            assert np.allclose(extended_values[: fs.n_points], old_values), (
                f"extended_values: {extended_values} does not match old_values: {old_values}"
            )
            assert np.allclose(extended_values[-1 * fs.n_points :], old_values), (
                f"extended_values: {extended_values} does not match old_values: {old_values}"
            )

    def test_2dmesh_fermisurface_creation(self, fermisurface_2d_non_colinear):
        fs = fermisurface_2d_non_colinear
        assert isinstance(fs.ebs, ElectronicBandStructureMesh)
        assert isinstance(fs.point_set, PointSet)
        assert isinstance(fs.isovalue, float)
        assert isinstance(fs.band_isosurfaces, dict)
        assert fs.isovalue == fs.ebs.fermi
        assert fs.fermi_shift == 0.0


class TestFermiSurfaceNormalization:
    """Tests for FermiSurface normalization system."""

    def test_normalize_raw(self, fermisurface_3d_non_spin_polarized):
        """Test that raw normalization returns unchanged values."""
        fs = fermisurface_3d_non_spin_polarized
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = fs.normalize("raw", values)

        np.testing.assert_array_equal(result, values)

    def test_normalize_max(self, fermisurface_3d_non_spin_polarized):
        """Test max normalization produces values in [-1, 1]."""
        fs = fermisurface_3d_non_spin_polarized
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = fs.normalize("max", values)

        assert np.max(np.abs(result)) == 1.0
        assert result[-1] == 1.0  # Max value normalized to 1

    def test_projected_sum_total_is_per_point_fraction(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized
        v_d = dict(atoms=[1], orbitals=[4, 5, 6, 7, 8])

        total = fs.get_property("projected_sum", **v_d, norm_mode="total").value
        selected = fs.get_property("projected_sum", **v_d, norm_mode="raw").value
        everything = fs.get_property("projected_sum", norm_mode="raw").value

        assert sorted(fs.band_spin_mask) == [(16, 0), (17, 0), (18, 0)]
        on_surface = []
        for (iband, ispin), mask in fs.band_spin_mask.items():
            expected = selected[mask, iband, ispin] / everything[mask, iband, ispin]
            np.testing.assert_allclose(total[mask, iband, ispin], expected)
            on_surface.append(total[mask, iband, ispin])
        on_surface = np.concatenate(on_surface)
        assert on_surface.min() == pytest.approx(0.8438, abs=1e-4)
        assert on_surface.max() == pytest.approx(0.8862, abs=1e-4)

    def test_projected_sum_max_reaches_one_on_a_surface(self, fermisurface_3d_non_spin_polarized):
        fs = fermisurface_3d_non_spin_polarized

        values = fs.get_property(
            "projected_sum", atoms=[1], orbitals=[4, 5, 6, 7, 8], norm_mode="max"
        ).value

        on_surface = np.concatenate(
            [values[mask, iband, ispin] for (iband, ispin), mask in fs.band_spin_mask.items()]
        )
        assert on_surface.max() == 1.0
        assert on_surface.min() == pytest.approx(0.9452, abs=1e-4)
        assert np.count_nonzero(values) == on_surface.size


class TestFermiSurfaceSerialization:
    """Tests for FermiSurface save/load functionality."""

    def test_save_load_roundtrip(self, fermisurface_3d_non_spin_polarized, tmp_path):
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

    def test_save_load_preserves_point_data(self, fermisurface_3d_non_spin_polarized, tmp_path):
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

    def test_save_load_preserves_band_isosurfaces(self, fermisurface_3d_non_spin_polarized, tmp_path):
        """Test that band_isosurfaces are preserved through save/load."""
        fs = fermisurface_3d_non_spin_polarized
        original_keys = set(fs.band_isosurfaces.keys())

        save_path = tmp_path / "test_fs_bands.pkl"
        fs.save(str(save_path))
        fs2 = FermiSurface.load(str(save_path))

        assert set(fs2.band_isosurfaces.keys()) == original_keys


def _finite_difference_gradient(ebs, band, padding):
    """Central differences on the original periodic k-grid, dE/dk in eV*Angstrom with angular k."""
    bands = math.array_to_mesh(ebs.bands.value[:, band, 0], ebs.n_kx, ebs.n_ky, ebs.n_kz)
    d_frac = np.stack(
        [
            (np.roll(bands, -1, axis=i) - np.roll(bands, 1, axis=i)) * n / 2
            for i, n in enumerate(bands.shape)
        ],
        axis=-1,
    )
    d_cart = d_frac @ np.linalg.inv(2 * np.pi * ebs.reciprocal_lattice).T
    padded = np.pad(d_cart, [(padding, padding)] * 3 + [(0, 0)], mode="wrap")
    return math.mesh_to_array(padded)


def test_compute_gradients_on_fresh_surface_matches_finite_difference(
    fermisurface_3d_non_spin_polarized,
):
    fs = fermisurface_3d_non_spin_polarized
    padding = (fs.ebs.n_kx - fs.original_ebs.n_kx) // 2

    fs.compute_gradients(1)

    gradient = fs.point_set.get_property("bands").gradients[1] / METER_ANGSTROM
    expected = fs.interpolate_to_surface(_finite_difference_gradient(fs.original_ebs, 16, padding))
    speed = np.linalg.norm(gradient[:, 16, 0], axis=-1)
    assert gradient.shape == (2748, 20, 1, 3)
    assert np.allclose(gradient[:, 16, 0], expected, rtol=1e-3, atol=1e-2)
    assert np.allclose([speed.min(), speed.max()], [0.2078, 2.9329], atol=1e-3)


def test_fermi_speed_matches_finite_difference(fermisurface_3d_non_spin_polarized):
    fs = fermisurface_3d_non_spin_polarized
    padding = (fs.ebs.n_kx - fs.original_ebs.n_kx) // 2

    fermi_speed = fs.get_property("fermi_speed").value[:, 16, 0]

    grid_speed = np.linalg.norm(_finite_difference_gradient(fs.original_ebs, 16, padding), axis=-1)
    expected_speed = fs.interpolate_to_surface(grid_speed) * METER_ANGSTROM / HBAR_EV
    on_band_16 = fs.point_set.get_property("spin_band_index").value == 0
    assert on_band_16.sum() == 1800
    assert np.allclose(fermi_speed[on_band_16], expected_speed[on_band_16], rtol=1e-3)
    assert np.isclose(np.median(fermi_speed[on_band_16]), 4.419e5, rtol=1e-3)
    assert np.all(fermi_speed[~on_band_16] == 0)
