"""Tests for BandStructure2D class.

These tests verify the modernized BandStructure2D implementation following
the FermiSurface pattern with factory methods, cache invalidation, and
serialization support.
"""

import logging

import numpy as np
import pytest
import pyvista as pv

from pyprocar.core.bandstructure2D import (
    BandStructure2D,
    PlaneInfo,
    compute_plane_info,
    generate_band_2d_surfaces,
)
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.property_store import PointSet, Property
from tests.utils import DATA_DIR

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mesh_3d_non_spin_polarized_dir():
    """Directory for non-spin-polarized 3D mesh data."""
    return DATA_DIR / "examples" / "fermi3d" / "non-spin-polarized"


@pytest.fixture
def mesh_2d_non_spin_polarized_dir():
    """Directory for non-spin-polarized 2D mesh data."""
    return DATA_DIR / "examples" / "fermi2d" / "non-spin-polarized"


@pytest.fixture
def mesh_2d_spin_polarized_dir():
    """Directory for spin-polarized 2D mesh data."""
    return DATA_DIR / "examples" / "fermi2d" / "spin-polarized"


@pytest.fixture
def mesh_2d_non_colinear_dir():
    """Directory for non-colinear 2D mesh data."""
    return DATA_DIR / "examples" / "fermi2d" / "non-colinear"


@pytest.fixture
def ebs_mesh(mesh_3d_non_spin_polarized_dir):
    """ElectronicBandStructureMesh from 3D non-spin-polarized data."""
    ebs = ElectronicBandStructureMesh.from_code(
        code="vasp", dirpath=mesh_3d_non_spin_polarized_dir
    )
    ebs.reduce_bands_near_fermi()
    return ebs


@pytest.fixture
def bandstructure2d_3d(mesh_3d_non_spin_polarized_dir):
    """BandStructure2D from 3D non-spin-polarized data."""
    return BandStructure2D.from_code(
        code="vasp",
        dirpath=mesh_3d_non_spin_polarized_dir,
        grid_interpolation=(20, 20),
        padding=5,
    )


@pytest.fixture
def bandstructure2d_2d(mesh_2d_non_spin_polarized_dir):
    """BandStructure2D from 2D non-spin-polarized data."""
    return BandStructure2D.from_code(
        code="vasp",
        dirpath=mesh_2d_non_spin_polarized_dir,
        grid_interpolation=(20, 20),
        padding=5,
    )


# -----------------------------------------------------------------------------
# PlaneInfo dataclass tests
# -----------------------------------------------------------------------------


class TestPlaneInfo:
    """Tests for PlaneInfo dataclass."""

    def test_plane_info_fields(self):
        """Verify PlaneInfo has all required fields."""
        plane_info = PlaneInfo(
            normal=np.array([0, 0, 1]),
            origin=np.array([0, 0, 0]),
            u=np.array([1, 0, 0]),
            v=np.array([0, 1, 0]),
            u_limits=(-0.5, 0.5),
            v_limits=(-0.5, 0.5),
            grid_interpolation=(20, 20),
            u_grid=np.zeros((20, 20)),
            v_grid=np.zeros((20, 20)),
            uv_grid_points=np.zeros((400, 2)),
            as_cartesian=True,
        )

        assert isinstance(plane_info.normal, np.ndarray)
        assert isinstance(plane_info.origin, np.ndarray)
        assert isinstance(plane_info.u, np.ndarray)
        assert isinstance(plane_info.v, np.ndarray)
        assert isinstance(plane_info.u_limits, tuple)
        assert isinstance(plane_info.v_limits, tuple)
        assert isinstance(plane_info.grid_interpolation, tuple)
        assert isinstance(plane_info.u_grid, np.ndarray)
        assert isinstance(plane_info.v_grid, np.ndarray)
        assert isinstance(plane_info.uv_grid_points, np.ndarray)
        assert isinstance(plane_info.as_cartesian, bool)

    def test_compute_plane_info(self, ebs_mesh):
        """Test compute_plane_info function."""
        padded_ebs = ebs_mesh.pad(padding=5, inplace=False)
        padded_ebs = padded_ebs.expand_single_dimension(inplace=False)

        plane_info = compute_plane_info(
            ebs=padded_ebs,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            as_cartesian=True,
        )

        assert isinstance(plane_info, PlaneInfo)
        assert plane_info.normal.shape == (3,)
        assert plane_info.origin.shape == (3,)
        assert plane_info.u.shape == (3,)
        assert plane_info.v.shape == (3,)
        assert plane_info.grid_interpolation == (20, 20)

        # u and v should be orthonormal to normal
        assert np.abs(np.dot(plane_info.u, plane_info.normal)) < 1e-10
        assert np.abs(np.dot(plane_info.v, plane_info.normal)) < 1e-10
        assert np.isclose(np.linalg.norm(plane_info.u), 1.0)
        assert np.isclose(np.linalg.norm(plane_info.v), 1.0)


# -----------------------------------------------------------------------------
# Factory method tests
# -----------------------------------------------------------------------------


class TestBandStructure2DFactory:
    """Tests for BandStructure2D factory methods."""

    def test_from_ebs_creates_valid_surface(self, ebs_mesh):
        """Test from_ebs factory creates valid BandStructure2D."""
        bs2d = BandStructure2D.from_ebs(
            ebs=ebs_mesh,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            padding=5,
        )

        assert isinstance(bs2d, BandStructure2D)
        assert isinstance(bs2d, pv.PolyData)
        assert bs2d.points.shape[0] > 0
        assert bs2d.n_points > 0

    def test_from_code_creates_valid_surface(self, mesh_3d_non_spin_polarized_dir):
        """Test from_code factory creates valid BandStructure2D."""
        bs2d = BandStructure2D.from_code(
            code="vasp",
            dirpath=mesh_3d_non_spin_polarized_dir,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            padding=5,
        )

        assert isinstance(bs2d, BandStructure2D)
        assert bs2d.points.shape[0] > 0

    def test_from_code_delegates_to_from_ebs(self, mesh_3d_non_spin_polarized_dir):
        """Test from_code internally uses from_ebs (same results)."""
        bs2d_from_code = BandStructure2D.from_code(
            code="vasp",
            dirpath=mesh_3d_non_spin_polarized_dir,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            padding=5,
            reduce_bands_near_fermi=True,
        )

        # Load EBS manually and create via from_ebs
        ebs = ElectronicBandStructureMesh.from_code(
            code="vasp", dirpath=mesh_3d_non_spin_polarized_dir
        )
        ebs.reduce_bands_near_fermi()
        bs2d_from_ebs = BandStructure2D.from_ebs(
            ebs=ebs,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            padding=5,
        )

        # Both should have same number of bands and similar structure
        assert len(bs2d_from_code.band_surfaces) == len(bs2d_from_ebs.band_surfaces)


# -----------------------------------------------------------------------------
# Constructor validation tests
# -----------------------------------------------------------------------------


class TestBandStructure2DConstructor:
    """Tests for BandStructure2D constructor validation."""

    def test_constructor_validates_spin_band_index(self, ebs_mesh):
        """Test constructor raises error if spin_band_index missing."""
        # Create minimal valid inputs
        padded_ebs = ebs_mesh.pad(padding=5, inplace=False)
        padded_ebs = padded_ebs.expand_single_dimension(inplace=False)

        plane_info = compute_plane_info(
            ebs=padded_ebs,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            as_cartesian=True,
        )

        # Create PointSet missing spin_band_index
        point_set = PointSet(np.zeros((10, 3)))
        point_set.add_property(Property(name="spin_index", value=np.zeros(10)))
        # Note: spin_band_index is missing

        with pytest.raises(ValueError, match="spin_band_index"):
            BandStructure2D(
                points=np.zeros((10, 3)),
                faces=np.array([3, 0, 1, 2]),
                band_surfaces={},
                point_set=point_set,
                original_ebs=ebs_mesh,
                ebs=padded_ebs,
                plane_info=plane_info,
            )

    def test_constructor_validates_spin_index(self, ebs_mesh):
        """Test constructor raises error if spin_index missing."""
        padded_ebs = ebs_mesh.pad(padding=5, inplace=False)
        padded_ebs = padded_ebs.expand_single_dimension(inplace=False)

        plane_info = compute_plane_info(
            ebs=padded_ebs,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            as_cartesian=True,
        )

        # Create PointSet missing spin_index
        point_set = PointSet(np.zeros((10, 3)))
        point_set.add_property(Property(name="spin_band_index", value=np.zeros(10)))
        # Note: spin_index is missing

        with pytest.raises(ValueError, match="spin_index"):
            BandStructure2D(
                points=np.zeros((10, 3)),
                faces=np.array([3, 0, 1, 2]),
                band_surfaces={},
                point_set=point_set,
                original_ebs=ebs_mesh,
                ebs=padded_ebs,
                plane_info=plane_info,
            )


# -----------------------------------------------------------------------------
# Property accessor tests
# -----------------------------------------------------------------------------


class TestBandStructure2DProperties:
    """Tests for BandStructure2D property accessors."""

    def test_ebs_property(self, bandstructure2d_3d):
        """Test ebs property returns ElectronicBandStructureMesh."""
        assert isinstance(bandstructure2d_3d.ebs, ElectronicBandStructureMesh)

    def test_original_ebs_property(self, bandstructure2d_3d):
        """Test original_ebs property returns ElectronicBandStructureMesh."""
        assert isinstance(bandstructure2d_3d.original_ebs, ElectronicBandStructureMesh)

    def test_point_set_property(self, bandstructure2d_3d):
        """Test point_set property returns PointSet."""
        assert isinstance(bandstructure2d_3d.point_set, PointSet)

    def test_plane_info_property(self, bandstructure2d_3d):
        """Test plane_info property returns PlaneInfo."""
        assert isinstance(bandstructure2d_3d.plane_info, PlaneInfo)

    def test_normal_property(self, bandstructure2d_3d):
        """Test normal property returns array."""
        assert isinstance(bandstructure2d_3d.normal, np.ndarray)
        assert bandstructure2d_3d.normal.shape == (3,)

    def test_origin_property(self, bandstructure2d_3d):
        """Test origin property returns array."""
        assert isinstance(bandstructure2d_3d.origin, np.ndarray)
        assert bandstructure2d_3d.origin.shape == (3,)

    def test_u_v_properties(self, bandstructure2d_3d):
        """Test u and v properties return orthonormal basis vectors."""
        u = bandstructure2d_3d.u
        v = bandstructure2d_3d.v
        normal = bandstructure2d_3d.normal

        assert isinstance(u, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert u.shape == (3,)
        assert v.shape == (3,)

        # Verify orthonormality
        assert np.isclose(np.linalg.norm(u), 1.0)
        assert np.isclose(np.linalg.norm(v), 1.0)
        assert np.abs(np.dot(u, normal)) < 1e-5
        assert np.abs(np.dot(v, normal)) < 1e-5

    def test_n_grid_points_property(self, bandstructure2d_3d):
        """Test n_grid_points property returns integer."""
        assert isinstance(bandstructure2d_3d.n_grid_points, int)
        assert bandstructure2d_3d.n_grid_points > 0

    def test_band_surfaces_property(self, bandstructure2d_3d):
        """Test band_surfaces property returns dict."""
        assert isinstance(bandstructure2d_3d.band_surfaces, dict)
        for key, surface in bandstructure2d_3d.band_surfaces.items():
            assert isinstance(key, tuple)
            assert len(key) == 2  # (band_index, spin_index)
            assert isinstance(surface, pv.PolyData)

    def test_band_spin_surface_map_property(self, bandstructure2d_3d):
        """Test band_spin_surface_map property."""
        mapping = bandstructure2d_3d.band_spin_surface_map
        assert isinstance(mapping, dict)
        for key, idx in mapping.items():
            assert isinstance(key, tuple)
            assert isinstance(idx, int)

    def test_surface_band_spin_map_property(self, bandstructure2d_3d):
        """Test surface_band_spin_map property."""
        mapping = bandstructure2d_3d.surface_band_spin_map
        assert isinstance(mapping, dict)
        for idx, key in mapping.items():
            assert isinstance(idx, int)
            assert isinstance(key, tuple)

    def test_band_spin_mask_property(self, bandstructure2d_3d):
        """Test band_spin_mask property."""
        masks = bandstructure2d_3d.band_spin_mask
        assert isinstance(masks, dict)
        for key, mask in masks.items():
            assert isinstance(key, tuple)
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == bool

    def test_backwards_compatible_properties(self, bandstructure2d_3d):
        """Test backwards-compatible property accessors."""
        # u_grid, v_grid
        assert isinstance(bandstructure2d_3d.u_grid, np.ndarray)
        assert isinstance(bandstructure2d_3d.v_grid, np.ndarray)

        # uv_grid_points
        assert isinstance(bandstructure2d_3d.uv_grid_points, np.ndarray)

        # as_cartesian
        assert isinstance(bandstructure2d_3d.as_cartesian, bool)

        # grid_interpolation
        assert isinstance(bandstructure2d_3d.grid_interpolation, tuple)


# -----------------------------------------------------------------------------
# Cache invalidation tests
# -----------------------------------------------------------------------------


class TestBandStructure2DCache:
    """Tests for BandStructure2D caching system."""

    def test_cache_invalidation_marks_properties_stale(self, bandstructure2d_3d):
        """Test that cache invalidation works."""
        bs2d = bandstructure2d_3d

        # Mark something as cached
        bs2d._mark_cached("test_prop")
        assert bs2d._is_cache_valid("test_prop")

        # Invalidate cache
        bs2d._invalidate_cache()
        assert not bs2d._is_cache_valid("test_prop")

    def test_cache_version_increments(self, bandstructure2d_3d):
        """Test that cache version increments on invalidation."""
        bs2d = bandstructure2d_3d

        initial_version = bs2d._ebs_cache_version
        bs2d._invalidate_cache()

        assert bs2d._ebs_cache_version == initial_version + 1

    def test_uncached_property_invalid(self, bandstructure2d_3d):
        """Test that uncached properties are detected as invalid."""
        bs2d = bandstructure2d_3d

        assert not bs2d._is_cache_valid("nonexistent_property")


# -----------------------------------------------------------------------------
# Serialization tests
# -----------------------------------------------------------------------------


class TestBandStructure2DSerialization:
    """Tests for BandStructure2D save/load functionality."""

    def test_save_creates_file(self, bandstructure2d_3d, tmp_path):
        """Test that save creates a file."""
        save_path = tmp_path / "test_bs2d.pkl"

        bandstructure2d_3d.save(str(save_path))

        assert save_path.exists()

    def test_save_load_roundtrip(self, bandstructure2d_3d, tmp_path):
        """Test that save/load preserves surface geometry."""
        bs2d = bandstructure2d_3d
        save_path = tmp_path / "test_bs2d.pkl"

        # Save
        bs2d.save(str(save_path))

        # Load
        bs2d_loaded = BandStructure2D.load(str(save_path))

        # Verify geometry
        assert bs2d_loaded.n_points == bs2d.n_points
        np.testing.assert_array_almost_equal(bs2d_loaded.points, bs2d.points)

    def test_save_load_preserves_point_data(self, bandstructure2d_3d, tmp_path):
        """Test that point_data is preserved through save/load."""
        bs2d = bandstructure2d_3d
        save_path = tmp_path / "test_bs2d_data.pkl"

        # Add test data
        test_data = np.random.rand(bs2d.n_points)
        bs2d.point_data["test_scalar"] = test_data

        bs2d.save(str(save_path))
        bs2d_loaded = BandStructure2D.load(str(save_path))

        assert "test_scalar" in bs2d_loaded.point_data
        np.testing.assert_array_almost_equal(
            bs2d_loaded.point_data["test_scalar"], test_data
        )

    def test_save_load_preserves_band_surfaces(self, bandstructure2d_3d, tmp_path):
        """Test that band_surfaces are preserved through save/load."""
        bs2d = bandstructure2d_3d
        original_keys = set(bs2d.band_surfaces.keys())

        save_path = tmp_path / "test_bs2d_bands.pkl"
        bs2d.save(str(save_path))
        bs2d_loaded = BandStructure2D.load(str(save_path))

        assert set(bs2d_loaded.band_surfaces.keys()) == original_keys

    def test_save_load_preserves_plane_info(self, bandstructure2d_3d, tmp_path):
        """Test that plane_info is preserved through save/load."""
        bs2d = bandstructure2d_3d
        save_path = tmp_path / "test_bs2d_plane.pkl"

        bs2d.save(str(save_path))
        bs2d_loaded = BandStructure2D.load(str(save_path))

        np.testing.assert_array_almost_equal(
            bs2d_loaded.plane_info.normal, bs2d.plane_info.normal
        )
        np.testing.assert_array_almost_equal(
            bs2d_loaded.plane_info.origin, bs2d.plane_info.origin
        )
        assert bs2d_loaded.plane_info.grid_interpolation == bs2d.plane_info.grid_interpolation

    def test_save_load_preserves_point_set_properties(self, bandstructure2d_3d, tmp_path):
        """Test that PointSet properties are preserved through save/load."""
        bs2d = bandstructure2d_3d
        save_path = tmp_path / "test_bs2d_pointset.pkl"

        bs2d.save(str(save_path))
        bs2d_loaded = BandStructure2D.load(str(save_path))

        # Check required properties exist
        assert "spin_index" in bs2d_loaded.point_set.property_store
        assert "spin_band_index" in bs2d_loaded.point_set.property_store

        # Check values match
        orig_spin_index = bs2d.point_set.get_property("spin_index").value
        loaded_spin_index = bs2d_loaded.point_set.get_property("spin_index").value
        np.testing.assert_array_almost_equal(loaded_spin_index, orig_spin_index)


# -----------------------------------------------------------------------------
# Surface generation tests
# -----------------------------------------------------------------------------


class TestBandStructure2DSurfaceGeneration:
    """Tests for surface generation functionality."""

    def test_generate_band_2d_surfaces(self, ebs_mesh):
        """Test generate_band_2d_surfaces function."""
        padded_ebs = ebs_mesh.pad(padding=5, inplace=False)
        padded_ebs = padded_ebs.expand_single_dimension(inplace=False)

        plane_info = compute_plane_info(
            ebs=padded_ebs,
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            grid_interpolation=(20, 20),
            as_cartesian=True,
        )

        combined_surface, band_surfaces, point_set = generate_band_2d_surfaces(
            ebs=padded_ebs,
            plane_info=plane_info,
            original_ebs=ebs_mesh,
        )

        assert isinstance(combined_surface, pv.PolyData)
        assert isinstance(band_surfaces, dict)
        assert isinstance(point_set, PointSet)

        # Verify required properties
        assert "spin_index" in point_set.property_store
        assert "spin_band_index" in point_set.property_store

    def test_required_properties_in_point_set(self, bandstructure2d_3d):
        """Test that spin_band_index and spin_index are in point_set."""
        bs2d = bandstructure2d_3d

        assert "spin_band_index" in bs2d.point_set.property_store
        assert "spin_index" in bs2d.point_set.property_store


# -----------------------------------------------------------------------------
# Functional tests
# -----------------------------------------------------------------------------


class TestBandStructure2DFunctionality:
    """Tests for BandStructure2D methods."""

    def test_get_2d_brillouin_zone(self, bandstructure2d_3d):
        """Test get_2d_brillouin_zone method."""
        from pyprocar.core.brillouin_zone import BrillouinZone2D

        bz = bandstructure2d_3d.get_2d_brillouin_zone(e_min=-2, e_max=2)
        assert isinstance(bz, BrillouinZone2D)

    def test_set_scalars(self, bandstructure2d_3d):
        """Test set_scalars method."""
        bs2d = bandstructure2d_3d
        # Use n_points (surface point count) not n_grid_points
        values = np.random.rand(bs2d.n_points)

        bs2d.set_scalars("test_scalar", values)

        assert "test_scalar" in bs2d.point_data

    def test_transform_matrices_initialized(self, bandstructure2d_3d):
        """Test that transformation matrices are initialized."""
        bs2d = bandstructure2d_3d

        assert hasattr(bs2d, "transform_to_cart")
        assert hasattr(bs2d, "transform_to_frac")
        assert bs2d.transform_to_cart.shape == (4, 4)
        assert bs2d.transform_to_frac.shape == (4, 4)
