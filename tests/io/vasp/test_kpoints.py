import json
import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import vasp
from pyprocar.utils.log_utils import set_verbose_level
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)

KPOINTS_DATA_DIR = DATA_DIR / "io" / "vasp" / "kpoints"


class TestKpoints(BaseTest):
    """Test class for VASP KPOINTS file parsing."""

    def test_kpoints_auto_mesh_gamma(self):
        """Test parsing of automatic mesh generation with Gamma centering."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-gamma")

        # Check basic properties
        assert kpoints.mode == "gamma"
        assert kpoints.automatic == True
        assert kpoints.ngrids == [0]
        assert kpoints.kgrid == [11, 11, 1]
        assert kpoints.kshift == [0, 0, 0]
        assert kpoints.cartesian == False
        assert kpoints.special_kpoints is None
        assert kpoints.knames is None
        assert kpoints.kpath is None
        assert "K-Spacing Value to Generate K-Mesh: 0.010" in kpoints.comment

    def test_kpoints_auto_mesh_monkhorst_pack(self):
        """Test parsing of automatic mesh generation with Monkhorst-Pack."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-monkhorst_pack")

        # Check basic properties
        assert kpoints.mode == "monkhorst-pack"
        assert kpoints.automatic == True
        assert kpoints.ngrids == [0]
        assert kpoints.kgrid == [15, 15, 15]
        assert kpoints.kshift == [0, 0, 0]
        assert kpoints.cartesian == False
        assert kpoints.special_kpoints is None
        assert kpoints.knames is None
        assert kpoints.kpath is None
        assert "k-points" in kpoints.comment

    def test_kpoints_explicit_mesh(self):
        """Test parsing of explicit k-point mesh."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_explicit-mesh")

        # Check basic properties
        assert (
            kpoints.mode is None
        )  # Explicit mesh doesn't set mode in current implementation
        assert kpoints.automatic == False
        assert kpoints.ngrids == [144]
        assert kpoints.kgrid is None
        assert kpoints.kshift is None
        assert kpoints.cartesian == False
        assert kpoints.special_kpoints is None
        assert kpoints.knames is None
        assert kpoints.kpath is None
        assert "Generated by PyProcar" in kpoints.comment

    def test_kpoints_bands_line_mode(self):
        """Test parsing of band structure k-path in line mode."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # Check basic properties
        assert kpoints.mode == "line"
        assert kpoints.automatic == False
        assert kpoints.ngrids == [50, 50, 50, 50, 50, 50]  # One per segment
        assert kpoints.kgrid is None
        assert kpoints.kshift is None
        assert kpoints.cartesian == False

        # Check special k-points and names
        assert kpoints.special_kpoints is not None
        assert kpoints.knames is not None
        assert kpoints.kpath is not None

        # Check the structure of special k-points (should be 6 segments, 2 k-points each)
        expected_shape = (6, 2, 3)
        assert kpoints.special_kpoints.shape == expected_shape

        # Check some specific k-points
        np.testing.assert_allclose(
            kpoints.special_kpoints[0, 0], [0.0, 0.0, 0.0]
        )  # GAMMA
        np.testing.assert_allclose(kpoints.special_kpoints[0, 1], [0.5, -0.5, 0.5])  # H

        # Check k-point names
        expected_knames_shape = (6, 2)
        assert kpoints.knames.shape == expected_knames_shape
        assert kpoints.knames[0, 0].strip() == "GAMMA"
        assert kpoints.knames[0, 1].strip() == "H"
        assert kpoints.knames[1, 0].strip() == "H"
        assert kpoints.knames[1, 1].strip() == "N"

    def test_kpoints_bands_kpath_object(self):
        """Test that KPath object is created correctly for band structure calculations."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # KPath should be created for line mode
        assert kpoints.kpath is not None

        # Test has_time_reversal parameter
        kpoints_no_tr = vasp.Kpoints(
            KPOINTS_DATA_DIR / "KPOINTS_bands", has_time_reversal=False
        )
        assert kpoints_no_tr.kpath is not None

    def test_kpoints_comment_parsing(self):
        """Test that comments are parsed correctly."""
        kpoints_gamma = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-gamma")
        kpoints_mp = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-monkhorst_pack")
        kpoints_explicit = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_explicit-mesh")
        kpoints_bands = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # Check that comments are strings and not empty
        assert isinstance(kpoints_gamma.comment, str)
        assert isinstance(kpoints_mp.comment, str)
        assert isinstance(kpoints_explicit.comment, str)
        assert isinstance(kpoints_bands.comment, str)

        assert len(kpoints_gamma.comment.strip()) > 0
        assert len(kpoints_mp.comment.strip()) > 0
        assert len(kpoints_explicit.comment.strip()) > 0
        assert len(kpoints_bands.comment.strip()) > 0

    def test_kpoints_file_not_found(self):
        """Test that appropriate error is raised when KPOINTS file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            vasp.Kpoints("nonexistent_kpoints_file")

    def test_kpoints_ngrids_single_value_expansion(self):
        """Test that single ngrids value is expanded for multiple segments in line mode."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # Should have expanded single value to match number of segments
        assert len(kpoints.ngrids) == 6  # 6 segments in the bands file
        assert all(val == 50 for val in kpoints.ngrids)

    def test_kpoints_automatic_detection(self):
        """Test automatic detection based on ngrids[0] == 0."""
        kpoints_auto = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-gamma")
        kpoints_explicit = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_explicit-mesh")

        assert kpoints_auto.automatic == True
        assert kpoints_explicit.automatic == False

    def test_kpoints_mode_detection(self):
        """Test detection of different k-point generation modes."""
        kpoints_gamma = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-gamma")
        kpoints_mp = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-monkhorst_pack")
        kpoints_bands = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        assert kpoints_gamma.mode == "gamma"
        assert kpoints_mp.mode == "monkhorst-pack"
        assert kpoints_bands.mode == "line"

    def test_kpoints_cartesian_coordinate_detection(self):
        """Test detection of cartesian vs reciprocal coordinates in line mode."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # The test file uses "reciprocal" coordinates
        assert kpoints.cartesian == False

    def test_kpoints_kshift_parsing(self):
        """Test parsing of k-point shifts."""
        kpoints_gamma = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-gamma")
        kpoints_mp = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-monkhorst_pack")

        # Both should have zero shift
        assert kpoints_gamma.kshift == [0, 0, 0]
        assert kpoints_mp.kshift == [0, 0, 0]

    def test_kpoints_special_points_array_structure(self):
        """Test the structure of special k-points array for band calculations."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # Should be 3D array: (n_segments, 2, 3)
        assert kpoints.special_kpoints.ndim == 3
        assert (
            kpoints.special_kpoints.shape[1] == 2
        )  # Start and end point for each segment
        assert kpoints.special_kpoints.shape[2] == 3  # x, y, z coordinates

        # Check data types
        assert kpoints.special_kpoints.dtype == np.float64

    def test_kpoints_knames_array_structure(self):
        """Test the structure of k-point names array for band calculations."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # Should be 2D array: (n_segments, 2)
        assert kpoints.knames.ndim == 2
        assert kpoints.knames.shape[1] == 2  # Start and end name for each segment

        # Check that names are strings and not empty
        for i in range(kpoints.knames.shape[0]):
            for j in range(kpoints.knames.shape[1]):
                assert isinstance(kpoints.knames[i, j], str)
                assert len(kpoints.knames[i, j].strip()) > 0

    def test_kpoints_comment_cleanup(self):
        """Test that k-point names are cleaned up properly (removing ! characters)."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / "KPOINTS_bands")

        # Check that ! characters are removed from k-point names
        for i in range(kpoints.knames.shape[0]):
            for j in range(kpoints.knames.shape[1]):
                assert "!" not in kpoints.knames[i, j]

    @pytest.mark.parametrize(
        "filename,expected_mode,expected_automatic",
        [
            ("KPOINTS_auto-mesh-gamma", "gamma", True),
            ("KPOINTS_auto-mesh-monkhorst_pack", "monkhorst-pack", True),
            ("KPOINTS_bands", "line", False),
            ("KPOINTS_explicit-mesh", None, False),
        ],
    )
    def test_kpoints_parametrized_properties(
        self, filename, expected_mode, expected_automatic
    ):
        """Parametrized test for different KPOINTS file properties."""
        kpoints = vasp.Kpoints(KPOINTS_DATA_DIR / filename)
        assert kpoints.mode == expected_mode
        assert kpoints.automatic == expected_automatic

    def test_kpoints_initialization_with_custom_time_reversal(self):
        """Test initialization with custom time reversal symmetry setting."""
        # Test with time reversal = True (default)
        kpoints_tr_true = vasp.Kpoints(
            KPOINTS_DATA_DIR / "KPOINTS_bands", has_time_reversal=True
        )

        # Test with time reversal = False
        kpoints_tr_false = vasp.Kpoints(
            KPOINTS_DATA_DIR / "KPOINTS_bands", has_time_reversal=False
        )

        # Both should create kpath objects for line mode
        assert kpoints_tr_true.kpath is not None
        assert kpoints_tr_false.kpath is not None

        # For non-line mode, kpath should be None regardless of time reversal setting
        kpoints_gamma = vasp.Kpoints(
            KPOINTS_DATA_DIR / "KPOINTS_auto-mesh-gamma", has_time_reversal=False
        )
        assert kpoints_gamma.kpath is None
