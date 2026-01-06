"""Tests for FrmSrf extractor."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.frmsf import Frmsf


# Minimal valid FrmSrf file: 2x2x2 k-grid, 2 bands, method 1 (gamma-centered)
# Format: grid dims, method, n_bands, reciprocal lattice (3 lines), band data
FRMSF_STR_MINIMAL = """\
2 2 2
1
2
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0
"""


# FrmSrf file with Monkhorst-Pack method (method=0)
FRMSF_STR_MP = """\
2 2 2
0
1
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
"""


# FrmSrf file with shifted gamma-centered method (method=2)
FRMSF_STR_SHIFTED = """\
2 2 2
2
1
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
"""


class TestFrmsfExtractor:
    """Tests for Frmsf extractor class."""

    def test_nk_dim(self) -> None:
        """Test k-grid dimensions."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.nk_dim == (2, 2, 2)

    def test_kpoint_generation_method(self) -> None:
        """Test k-point generation method."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.kpoint_generation_method == 1

    def test_n_bands(self) -> None:
        """Test number of bands."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.n_bands == 2

    def test_reciprocal_lattice_shape(self) -> None:
        """Test reciprocal lattice shape."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.reciprocal_lattice.shape == (3, 3)

    def test_reciprocal_lattice_values(self) -> None:
        """Test reciprocal lattice values."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(frmsf.reciprocal_lattice, expected)

    def test_n_kpoints(self) -> None:
        """Test total k-points count."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.n_kpoints == 8  # 2*2*2

    def test_bands_shape(self) -> None:
        """Test bands array shape: (n_kpoints, n_bands)."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.bands.shape == (8, 2)  # 8 k-points, 2 bands

    def test_bands_values(self) -> None:
        """Test bands values are correctly parsed."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        # First band values (8 k-points)
        assert frmsf.bands[0, 0] == pytest.approx(1.0)
        assert frmsf.bands[7, 0] == pytest.approx(8.0)
        # Second band values
        assert frmsf.bands[0, 1] == pytest.approx(9.0)
        assert frmsf.bands[7, 1] == pytest.approx(16.0)

    def test_kpoints_shape(self) -> None:
        """Test k-points array shape."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.kpoints.shape == (8, 3)

    def test_projections_none_when_no_projection_data(self) -> None:
        """Test projections is None when no projection data in file."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.projections is None

    def test_mapping_interface_getitem(self) -> None:
        """Test Mapping __getitem__ protocol."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf["n_bands"] == 2
        assert frmsf["nk_dim"] == (2, 2, 2)

    def test_mapping_interface_len(self) -> None:
        """Test Mapping __len__ protocol."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert len(frmsf) == 4

    def test_mapping_interface_iter(self) -> None:
        """Test Mapping __iter__ protocol."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        keys = list(frmsf)
        assert "reciprocal_lattice" in keys
        assert "bands" in keys
        assert "kpoints" in keys
        assert "nk_dim" in keys


class TestFrmsfKpointGeneration:
    """Tests for different k-point generation methods."""

    def test_gamma_centered_kpoints(self) -> None:
        """Test gamma-centered (method=1) k-point generation."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        # For method 1 (gamma), k = (n-1)/N, so for 2x2x2:
        # First k-point should be (0,0,0)
        np.testing.assert_array_almost_equal(frmsf.kpoints[0], [0.0, 0.0, 0.0])

    def test_monkhorst_pack_kpoints(self) -> None:
        """Test Monkhorst-Pack (method=0) k-point generation."""
        frmsf = Frmsf.from_str(FRMSF_STR_MP)
        # For method 0 (MP), k = (2n-1-N)/N
        # For 2x2x2 grid, first k-point is (n1=1,n2=1,n3=1)
        # k = (2*1-1-2)/2 = -0.5
        np.testing.assert_array_almost_equal(frmsf.kpoints[0], [-0.5, -0.5, -0.5])

    def test_shifted_gamma_kpoints(self) -> None:
        """Test shifted gamma-centered (method=2) k-point generation."""
        frmsf = Frmsf.from_str(FRMSF_STR_SHIFTED)
        # For method 2 (shifted gamma), k = (2n-1)/(2N)
        # For 2x2x2 grid, first k-point is (n1=1,n2=1,n3=1)
        # k = (2*1-1)/(2*2) = 0.25
        np.testing.assert_array_almost_equal(frmsf.kpoints[0], [0.25, 0.25, 0.25])


class TestFrmsfFromFile:
    """Tests for Frmsf file-based initialization."""

    def test_from_file(self, tmp_path: Path) -> None:
        """Test initialization from file path."""
        filepath = tmp_path / "test.frmsf"
        filepath.write_text(FRMSF_STR_MINIMAL)

        frmsf = Frmsf(filepath)
        assert frmsf.nk_dim == (2, 2, 2)

    def test_filepath_property(self, tmp_path: Path) -> None:
        """Test filepath property returns Path."""
        filepath = tmp_path / "test.frmsf"
        filepath.write_text(FRMSF_STR_MINIMAL)

        frmsf = Frmsf(filepath)
        assert frmsf.filepath == filepath

    def test_from_str_filepath_is_none(self) -> None:
        """Test from_str sets filepath to None."""
        frmsf = Frmsf.from_str(FRMSF_STR_MINIMAL)
        assert frmsf.filepath is None


class TestFrmsfErrorHandling:
    """Tests for Frmsf error handling."""

    def test_missing_file_path_and_string_raises(self) -> None:
        """Test that missing both file path and string raises ValueError."""
        frmsf = Frmsf()
        with pytest.raises(ValueError, match="No file path or file string provided"):
            _ = frmsf.file_str
