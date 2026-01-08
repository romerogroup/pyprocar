"""Tests for BXSF extractor."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.bxsf import Bxsf

# Minimal valid BXSF file: 2x2x2 k-grid (3x3x3 including boundary), 1 band
BXSF_STR_MINIMAL = """\
BEGIN_INFO
  Fermi Energy: 5.5000
END_INFO

BEGIN_BLOCK_BANDGRID_3D
  fermi_surface
  BEGIN_BANDGRID_3D_fermi
    1
    3 3 3
    0.0 0.0 0.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    BAND: 1
    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0

    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0

    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0
  END_BANDGRID_3D_fermi
END_BLOCK_BANDGRID_3D
"""


class TestBxsfExtractor:
    """Tests for Bxsf extractor class."""

    def test_fermi_energy(self) -> None:
        """Test Fermi energy extraction."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert bxsf.fermi_energy == pytest.approx(5.5)

    def test_reciprocal_lattice_shape(self) -> None:
        """Test reciprocal lattice shape."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert bxsf.reciprocal_lattice.shape == (3, 3)

    def test_reciprocal_lattice_values(self) -> None:
        """Test reciprocal lattice values are correctly parsed."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(bxsf.reciprocal_lattice, expected)

    def test_origin(self) -> None:
        """Test origin extraction."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(bxsf.origin, expected)

    def test_nkfs_dim(self) -> None:
        """Test full grid dimensions (including boundary)."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        np.testing.assert_array_equal(bxsf.nkfs_dim, np.array([3, 3, 3]))

    def test_nk_dim(self) -> None:
        """Test k-grid dimensions (excluding boundary)."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert bxsf.nk_dim == (2, 2, 2)

    def test_n_bands(self) -> None:
        """Test number of bands."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert bxsf.n_bands == 1

    def test_bands_shape(self) -> None:
        """Test bands array shape: (n_kpoints, n_bands, n_spins)."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        # 2x2x2 = 8 k-points, 1 band, 2 spins (pre-allocated)
        assert bxsf.bands.shape[0] == 8  # n_kpoints
        assert bxsf.bands.shape[1] == 1  # n_bands
        assert bxsf.bands.shape[2] == 2  # n_spins

    def test_kpoints_shape(self) -> None:
        """Test k-points array shape: (n_kpoints, 3)."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert bxsf.kpoints.shape == (8, 3)  # 2x2x2 = 8 k-points, 3D coordinates

    def test_kpoints_values_in_range(self) -> None:
        """Test k-points are in [0, 1) range (fractional coordinates)."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert np.all(bxsf.kpoints >= 0.0)
        assert np.all(bxsf.kpoints < 1.0)

    def test_mapping_interface_getitem(self) -> None:
        """Test Mapping __getitem__ protocol."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert bxsf["fermi_energy"] == pytest.approx(5.5)
        assert bxsf["nk_dim"] == (2, 2, 2)

    def test_mapping_interface_len(self) -> None:
        """Test Mapping __len__ protocol."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert len(bxsf) == 5

    def test_mapping_interface_iter(self) -> None:
        """Test Mapping __iter__ protocol."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        keys = list(bxsf)
        assert "fermi_energy" in keys
        assert "reciprocal_lattice" in keys
        assert "bands" in keys
        assert "kpoints" in keys
        assert "nk_dim" in keys

    def test_mapping_interface_contains(self) -> None:
        """Test Mapping __contains__ protocol for valid keys."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert "fermi_energy" in bxsf
        assert "bands" in bxsf
        assert "kpoints" in bxsf


class TestBxsfFromFile:
    """Tests for Bxsf file-based initialization."""

    def test_from_file(self, tmp_path: Path) -> None:
        """Test initialization from file path."""
        filepath = tmp_path / "test.bxsf"
        filepath.write_text(BXSF_STR_MINIMAL)

        bxsf = Bxsf(filepath)
        assert bxsf.fermi_energy == pytest.approx(5.5)

    def test_filepath_property(self, tmp_path: Path) -> None:
        """Test filepath property returns Path."""
        filepath = tmp_path / "test.bxsf"
        filepath.write_text(BXSF_STR_MINIMAL)

        bxsf = Bxsf(filepath)
        assert bxsf.filepath == filepath

    def test_from_str_filepath_is_none(self) -> None:
        """Test from_str sets filepath to None."""
        bxsf = Bxsf.from_str(BXSF_STR_MINIMAL)
        assert bxsf.filepath is None


class TestBxsfErrorHandling:
    """Tests for Bxsf error handling."""

    def test_missing_file_path_and_string_raises(self) -> None:
        """Test that missing both file path and string raises ValueError."""
        bxsf = Bxsf()
        with pytest.raises(ValueError, match="No file path or file string provided"):
            _ = bxsf.file_str

    def test_missing_fermi_energy_raises(self) -> None:
        """Test that missing Fermi energy raises ValueError."""
        file_content = """BEGIN_BLOCK_BANDGRID_3D
  fermi_surface
  BEGIN_BANDGRID_3D_fermi
    1
    3 3 3
    0.0 0.0 0.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    BAND: 1
    1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0
  END_BANDGRID_3D_fermi
END_BLOCK_BANDGRID_3D
"""
        bxsf = Bxsf.from_str(file_content)
        with pytest.raises(ValueError, match="No Fermi energy found"):
            _ = bxsf.fermi_energy
