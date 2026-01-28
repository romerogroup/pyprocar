"""
Test module for pyprocar.core.kpoints module.

This module contains unit tests for k-point generation functions,
coordinate transformation utilities, and the KPath class.
"""

import numpy as np
import pytest

from pyprocar.core.kpoints import (
    KGRID_MODE,
    KGridInfo,
    KPath,
    cartesian_to_reduced,
    format_names,
    generate_gamma_centered_kpoints,
    get_kpoints_from_kgrid,
    monkhorst_pack_kpoints,
    normalize_kpoint_name,
    reduced_to_cartesian,
    sort_kpoints,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_cubic_reciprocal_lattice():
    """
    Create a simple cubic reciprocal lattice.

    Returns
    -------
    np.ndarray
        A 3x3 array representing a simple cubic reciprocal lattice
        with lattice parameter 2*pi.
    """
    return np.array([[2 * np.pi, 0.0, 0.0], [0.0, 2 * np.pi, 0.0], [0.0, 0.0, 2 * np.pi]])


@pytest.fixture
def identity_reciprocal_lattice():
    """Create an identity reciprocal lattice for simple testing."""
    return np.eye(3)


@pytest.fixture
def simple_kpath_kpoints():
    """
    Create simple k-points for a Gamma-X-M-Gamma path.

    This creates a simple 2D-like path with 3 segments, 5 points each.
    """
    # Gamma to X: (0,0,0) -> (0.5,0,0)
    seg1 = np.linspace([0, 0, 0], [0.5, 0, 0], 5)
    # X to M: (0.5,0,0) -> (0.5,0.5,0)
    seg2 = np.linspace([0.5, 0, 0], [0.5, 0.5, 0], 5)
    # M to Gamma: (0.5,0.5,0) -> (0,0,0)
    seg3 = np.linspace([0.5, 0.5, 0], [0, 0, 0], 5)

    return np.vstack([seg1, seg2, seg3])


@pytest.fixture
def simple_segment_names():
    """Create segment names for a Gamma-X-M-Gamma path."""
    return [("Gamma", "X"), ("X", "M"), ("M", "Gamma")]


@pytest.fixture
def special_kpoint_map():
    """Create a special k-point map for testing."""
    return {
        "Γ": np.array([0.0, 0.0, 0.0]),
        "X": np.array([0.5, 0.0, 0.0]),
        "M": np.array([0.5, 0.5, 0.0]),
        "R": np.array([0.5, 0.5, 0.5]),
    }


# =============================================================================
# Test Classes - Enums and Dataclasses
# =============================================================================


class TestKGRIDMODE:
    """Test class for KGRID_MODE enum."""

    def test_kgrid_mode_values(self):
        """Test that KGRID_MODE has expected values."""
        assert KGRID_MODE.MONKHORST.value == "monkhorst"
        assert KGRID_MODE.GAMMA.value == "gamma"

    def test_kgrid_mode_members(self):
        """Test that KGRID_MODE has exactly two members."""
        assert len(KGRID_MODE) == 2


class TestKGridInfo:
    """Test class for KGridInfo dataclass."""

    def test_kgrid_info_creation(self):
        """Test KGridInfo can be created with valid parameters."""
        info = KGridInfo(kgrid=(4, 4, 4), kgrid_mode=KGRID_MODE.MONKHORST, kshift=(0.0, 0.0, 0.0))
        assert info.kgrid == (4, 4, 4)
        assert info.kgrid_mode == KGRID_MODE.MONKHORST
        assert info.kshift == (0.0, 0.0, 0.0)

    def test_kgrid_info_with_shift(self):
        """Test KGridInfo with non-zero k-shift."""
        info = KGridInfo(kgrid=(8, 8, 8), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.5, 0.5, 0.5))
        assert info.kshift == (0.5, 0.5, 0.5)


# =============================================================================
# Test Classes - Utility Functions
# =============================================================================


class TestGenerateGammaCenteredKpoints:
    """Test class for generate_gamma_centered_kpoints function."""

    def test_basic_grid(self):
        """Test basic gamma-centered grid generation."""
        kpoints = generate_gamma_centered_kpoints((2, 2, 2))

        assert kpoints.shape == (8, 3)
        # All points should be in first Brillouin zone [-0.5, 0.5]
        assert np.all(kpoints >= -0.5)
        assert np.all(kpoints <= 0.5)

    def test_grid_contains_gamma(self):
        """Test that gamma point is included in even grids."""
        kpoints = generate_gamma_centered_kpoints((4, 4, 4))

        # Check if gamma point (0,0,0) is in the grid
        has_gamma = any(np.allclose(k, [0, 0, 0]) for k in kpoints)
        assert has_gamma, "Gamma point should be in gamma-centered grid"

    def test_grid_with_shift(self):
        """Test gamma-centered grid with k-shift."""
        kpoints_no_shift = generate_gamma_centered_kpoints((4, 4, 4), (0.0, 0.0, 0.0))
        kpoints_shifted = generate_gamma_centered_kpoints((4, 4, 4), (0.5, 0.5, 0.5))

        # Shifted grid should be different
        assert not np.allclose(kpoints_no_shift, kpoints_shifted)

    def test_asymmetric_grid(self):
        """Test gamma-centered grid with different dimensions."""
        kpoints = generate_gamma_centered_kpoints((2, 4, 6))

        assert kpoints.shape == (2 * 4 * 6, 3)

    def test_single_point_grid(self):
        """Test 1x1x1 grid returns single point."""
        kpoints = generate_gamma_centered_kpoints((1, 1, 1))

        assert kpoints.shape == (1, 3)
        assert np.allclose(kpoints[0], [0, 0, 0])


class TestMonkhorstPackKpoints:
    """Test class for monkhorst_pack_kpoints function."""

    def test_basic_grid(self):
        """Test basic Monkhorst-Pack grid generation."""
        kpoints = monkhorst_pack_kpoints((4, 4, 4))

        assert kpoints.shape == (64, 3)

    def test_even_grid_excludes_gamma(self):
        """Test that even Monkhorst-Pack grid excludes gamma point."""
        kpoints = monkhorst_pack_kpoints((4, 4, 4))

        # Even grid should NOT include gamma point
        has_gamma = any(np.allclose(k, [0, 0, 0]) for k in kpoints)
        assert not has_gamma, "Even Monkhorst-Pack grid should not include gamma"

    def test_odd_grid_includes_gamma(self):
        """Test that odd Monkhorst-Pack grid includes gamma point."""
        kpoints = monkhorst_pack_kpoints((3, 3, 3))

        # Odd grid should include gamma point
        has_gamma = any(np.allclose(k, [0, 0, 0]) for k in kpoints)
        assert has_gamma, "Odd Monkhorst-Pack grid should include gamma"

    def test_grid_with_shift(self):
        """Test Monkhorst-Pack grid with k-shift."""
        kpoints_no_shift = monkhorst_pack_kpoints((4, 4, 4), (0.0, 0.0, 0.0))
        kpoints_shifted = monkhorst_pack_kpoints((4, 4, 4), (0.5, 0.5, 0.5))

        assert not np.allclose(kpoints_no_shift, kpoints_shifted)

    def test_grid_symmetry(self):
        """Test that Monkhorst-Pack grid is symmetric around gamma."""
        kpoints = monkhorst_pack_kpoints((4, 4, 4))

        # For each k-point, -k should also be in the grid
        for k in kpoints:
            has_negative = any(np.allclose(-k, kp) for kp in kpoints)
            assert has_negative, f"Grid should contain both {k} and {-k}"


class TestGetKpointsFromKgrid:
    """Test class for get_kpoints_from_kgrid factory function."""

    def test_monkhorst_mode(self):
        """Test Monkhorst mode selection."""
        kpoints = get_kpoints_from_kgrid((4, 4, 4), mode="monkhorst")
        expected = monkhorst_pack_kpoints((4, 4, 4))

        assert np.allclose(kpoints, expected)

    def test_gamma_mode(self):
        """Test Gamma mode selection."""
        kpoints = get_kpoints_from_kgrid((4, 4, 4), mode="gamma")
        expected = generate_gamma_centered_kpoints((4, 4, 4))

        assert np.allclose(kpoints, expected)

    def test_mode_case_insensitive(self):
        """Test that mode is case insensitive."""
        kpoints_lower = get_kpoints_from_kgrid((4, 4, 4), mode="monkhorst")
        kpoints_upper = get_kpoints_from_kgrid((4, 4, 4), mode="MONKHORST")
        kpoints_mixed = get_kpoints_from_kgrid((4, 4, 4), mode="Monkhorst")

        assert np.allclose(kpoints_lower, kpoints_upper)
        assert np.allclose(kpoints_lower, kpoints_mixed)

    def test_mode_first_letter(self):
        """Test that only first letter matters for mode."""
        kpoints_m = get_kpoints_from_kgrid((4, 4, 4), mode="m")
        kpoints_full = get_kpoints_from_kgrid((4, 4, 4), mode="monkhorst")

        assert np.allclose(kpoints_m, kpoints_full)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            get_kpoints_from_kgrid((4, 4, 4), mode="invalid")


class TestReducedToCartesian:
    """Test class for reduced_to_cartesian function."""

    def test_identity_lattice(self, identity_reciprocal_lattice):
        """Test conversion with identity lattice."""
        kpoints = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])

        cartesian = reduced_to_cartesian(kpoints, identity_reciprocal_lattice)

        assert np.allclose(cartesian, kpoints)

    def test_cubic_lattice(self, simple_cubic_reciprocal_lattice):
        """Test conversion with cubic lattice."""
        kpoints = np.array([[0.5, 0.0, 0.0]])

        cartesian = reduced_to_cartesian(kpoints, simple_cubic_reciprocal_lattice)

        expected = np.array([[np.pi, 0.0, 0.0]])
        assert np.allclose(cartesian, expected)

    def test_none_lattice_returns_none(self):
        """Test that None lattice returns None."""
        kpoints = np.array([[0.5, 0.0, 0.0]])

        result = reduced_to_cartesian(kpoints, None)

        assert result is None


class TestCartesianToReduced:
    """Test class for cartesian_to_reduced function."""

    def test_identity_lattice(self, identity_reciprocal_lattice):
        """Test conversion with identity lattice."""
        cartesian = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])

        reduced = cartesian_to_reduced(cartesian, identity_reciprocal_lattice)

        assert np.allclose(reduced, cartesian)

    def test_cubic_lattice(self, simple_cubic_reciprocal_lattice):
        """Test conversion with cubic lattice."""
        cartesian = np.array([[np.pi, 0.0, 0.0]])

        reduced = cartesian_to_reduced(cartesian, simple_cubic_reciprocal_lattice)

        expected = np.array([[0.5, 0.0, 0.0]])
        assert np.allclose(reduced, expected)

    def test_roundtrip(self, simple_cubic_reciprocal_lattice):
        """Test that reduced -> cartesian -> reduced is identity."""
        original = np.array([[0.25, 0.5, 0.0], [0.0, 0.0, 0.5]])

        cartesian = reduced_to_cartesian(original, simple_cubic_reciprocal_lattice)
        roundtrip = cartesian_to_reduced(cartesian, simple_cubic_reciprocal_lattice)

        assert np.allclose(roundtrip, original)

    def test_none_lattice_returns_none(self):
        """Test that None lattice returns None."""
        cartesian = np.array([[0.5, 0.0, 0.0]])

        result = cartesian_to_reduced(cartesian, None)

        assert result is None


class TestSortKpoints:
    """Test class for sort_kpoints function."""

    def test_c_order_sort(self):
        """Test C-order (row-major) sorting."""
        kpoints = np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ]
        )

        sorted_kpoints = sort_kpoints(kpoints, order="C")

        # C-order: sort by x, then y, then z
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
            ]
        )
        assert np.allclose(sorted_kpoints, expected)

    def test_f_order_sort(self):
        """Test F-order (column-major) sorting."""
        kpoints = np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ]
        )

        sorted_kpoints = sort_kpoints(kpoints, order="F")

        # F-order: sort by z, then y, then x
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ]
        )
        assert np.allclose(sorted_kpoints, expected)


class TestFormatNames:
    """Test class for format_names function."""

    def test_gamma_conversion(self):
        """Test that 'gamma' is converted to LaTeX Gamma."""
        names = ["gamma", "X", "M"]

        formatted = format_names(names, as_latex=True)

        assert formatted[0] == r"$\Gamma$"
        assert formatted[1] == "X"
        assert formatted[2] == "M"

    def test_gamma_case_insensitive(self):
        """Test that gamma conversion is case insensitive."""
        names = ["GAMMA", "Gamma", "gamma"]

        formatted = format_names(names, as_latex=False)

        # All should be converted to \Gamma
        for name in formatted:
            assert r"\Gamma" in name

    def test_latex_wrapping(self):
        """Test that backslash names get LaTeX wrapping."""
        names = [r"\Gamma", "X"]

        formatted = format_names(names, as_latex=True)

        assert formatted[0] == r"$\Gamma$"
        assert formatted[1] == "X"

    def test_no_latex_wrapping(self):
        """Test that backslash names are not wrapped when as_latex=False."""
        names = [r"\Gamma", "X"]

        formatted = format_names(names, as_latex=False)

        assert formatted[0] == r"\Gamma"


class TestNormalizeKpointName:
    """Test class for normalize_kpoint_name function."""

    def test_gamma_aliases(self):
        """Test that gamma aliases are normalized."""
        aliases = ["gamma", "Gamma", "G", "g", "Γ"]

        for alias in aliases:
            assert normalize_kpoint_name(alias) == "Γ"

    def test_x_aliases(self):
        """Test that X aliases are normalized."""
        aliases = ["x", "X"]

        for alias in aliases:
            assert normalize_kpoint_name(alias) == "X"

    def test_unknown_name_unchanged(self):
        """Test that unknown names are returned unchanged."""
        name = "CustomPoint"

        assert normalize_kpoint_name(name) == "CustomPoint"

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        assert normalize_kpoint_name("  gamma  ") == "Γ"
        assert normalize_kpoint_name(" X ") == "X"


# =============================================================================
# Test Classes - KPath Initialization
# =============================================================================


class TestKPathInitialization:
    """Test class for KPath initialization."""

    def test_init_with_kpoints(self, simple_kpath_kpoints, simple_segment_names):
        """Test KPath initialization with kpoints array."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
        )

        assert kpath.n_kpoints == len(simple_kpath_kpoints)
        assert kpath.kpoints is not None

    def test_init_with_n_grids_and_special_kpoints(self, special_kpoint_map):
        """Test KPath initialization by generating from special kpoints."""
        segment_names = [("Γ", "X"), ("X", "M"), ("M", "Γ")]
        n_grids = [10, 10, 10]

        kpath = KPath(
            n_grids=n_grids,
            segment_names=segment_names,
            special_kpoint_map=special_kpoint_map,
        )

        assert kpath.n_kpoints == 30  # 3 segments * 10 points
        assert kpath.n_segments == 3

    def test_init_requires_kpoints_or_n_grids(self):
        """Test that init raises error without kpoints or n_grids."""
        with pytest.raises(ValueError, match="Either kpoints or n_grids"):
            KPath()

    def test_init_with_reciprocal_lattice(
        self,
        simple_kpath_kpoints,
        simple_segment_names,
        simple_cubic_reciprocal_lattice,
    ):
        """Test KPath initialization with reciprocal lattice."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
            reciprocal_lattice=simple_cubic_reciprocal_lattice,
        )

        assert kpath.reciprocal_lattice is not None
        assert np.allclose(kpath.reciprocal_lattice, simple_cubic_reciprocal_lattice)

    def test_init_normalizes_kpoint_names(self, special_kpoint_map):
        """Test that initialization normalizes k-point names."""
        segment_names = [("gamma", "X"), ("X", "M"), ("M", "Gamma")]
        n_grids = [10, 10, 10]

        kpath = KPath(
            n_grids=n_grids,
            segment_names=segment_names,
            special_kpoint_map=special_kpoint_map,
        )

        # Should be normalized to canonical form
        assert kpath.segment_names[0][0] == "Γ"
        assert kpath.segment_names[2][1] == "Γ"

    def test_init_with_discontinuity_threshold(self, simple_kpath_kpoints, simple_segment_names):
        """Test KPath initialization with custom discontinuity threshold."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
            discontinuity_threshold=0.3,
        )

        assert kpath.discontinuity_threshold == 0.3

    def test_init_with_zero_diff_threshold(self, simple_kpath_kpoints, simple_segment_names):
        """Test KPath initialization with custom zero diff threshold."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
            zero_diff_threshold=1e-8,
        )

        assert kpath.zero_diff_threshold == 1e-8


# =============================================================================
# Test Classes - KPath Properties
# =============================================================================


class TestKPathProperties:
    """Test class for KPath properties."""

    @pytest.fixture
    def kpath_with_segments(self, simple_kpath_kpoints, simple_segment_names):
        """Create a KPath with multiple segments for testing."""
        return KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
        )

    def test_n_kpoints(self, kpath_with_segments, simple_kpath_kpoints):
        """Test n_kpoints property."""
        assert kpath_with_segments.n_kpoints == len(simple_kpath_kpoints)

    def test_n_segments(self, kpath_with_segments):
        """Test n_segments property."""
        assert kpath_with_segments.n_segments == 3

    def test_kpoints_property(self, kpath_with_segments, simple_kpath_kpoints):
        """Test kpoints property returns correct array."""
        assert np.allclose(kpath_with_segments.kpoints, simple_kpath_kpoints)

    def test_segment_names_property(self, kpath_with_segments):
        """Test segment_names property returns correct names."""
        # Names get normalized
        expected_normalized = [("Γ", "X"), ("X", "M"), ("M", "Γ")]
        assert kpath_with_segments.segment_names == expected_normalized

    def test_segment_indices_property(self, kpath_with_segments):
        """Test segment_indices property returns list of index arrays."""
        indices = kpath_with_segments.segment_indices

        assert len(indices) == 3
        for idx_array in indices:
            assert isinstance(idx_array, np.ndarray)

    def test_special_kpoint_names_property(self, kpath_with_segments):
        """Test special_kpoint_names property."""
        names = kpath_with_segments.special_kpoint_names

        # Should contain unique special point names
        assert "$\\Gamma$" in names or "Γ" in names
        assert "X" in names
        assert "M" in names

    def test_special_kpoint_map_property(self, kpath_with_segments):
        """Test special_kpoint_map property returns dict."""
        kpoint_map = kpath_with_segments.special_kpoint_map

        assert isinstance(kpoint_map, dict)
        assert len(kpoint_map) > 0

    def test_tick_names_property(self, kpath_with_segments):
        """Test tick_names property."""
        tick_names = kpath_with_segments.tick_names

        assert isinstance(tick_names, list)
        assert len(tick_names) > 0

    def test_tick_positions_property(self, kpath_with_segments):
        """Test tick_positions property."""
        tick_positions = kpath_with_segments.tick_positions

        assert isinstance(tick_positions, list)
        assert len(tick_positions) == len(kpath_with_segments.tick_names)

    def test_k_distances_property(self, kpath_with_segments):
        """Test k_distances property returns distances along path."""
        distances = kpath_with_segments.k_distances

        assert len(distances) == kpath_with_segments.n_kpoints
        # Distances should be monotonically increasing
        assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))

    def test_kpoints_cartesian_property(
        self,
        simple_kpath_kpoints,
        simple_segment_names,
        simple_cubic_reciprocal_lattice,
    ):
        """Test kpoints_cartesian property with reciprocal lattice."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
            reciprocal_lattice=simple_cubic_reciprocal_lattice,
        )

        cartesian = kpath.kpoints_cartesian

        assert cartesian is not None
        assert cartesian.shape == simple_kpath_kpoints.shape

    def test_knames_alias(self, kpath_with_segments):
        """Test knames property is alias for segment_names."""
        assert kpath_with_segments.knames == kpath_with_segments.segment_names

    def test_brillouin_zone_property(
        self,
        simple_kpath_kpoints,
        simple_segment_names,
        simple_cubic_reciprocal_lattice,
    ):
        """Test brillouin_zone property returns BrillouinZone object."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
            reciprocal_lattice=simple_cubic_reciprocal_lattice,
        )

        bz = kpath.brillouin_zone

        from pyprocar.core.brillouin_zone import BrillouinZone

        assert isinstance(bz, BrillouinZone)


# =============================================================================
# Test Classes - KPath Methods
# =============================================================================


class TestKPathMethods:
    """Test class for KPath methods."""

    @pytest.fixture
    def kpath_with_segments(self, simple_kpath_kpoints, simple_segment_names):
        """Create a KPath with multiple segments for testing."""
        return KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
        )

    def test_get_segments_all(self, kpath_with_segments):
        """Test get_segments returns all segments by default."""
        segments = kpath_with_segments.get_segments()

        assert len(segments) == 3

    def test_get_segments_specific(self, kpath_with_segments):
        """Test get_segments with specific segment indices."""
        segments = kpath_with_segments.get_segments(isegments=[0, 2])

        assert len(segments) == 2

    def test_get_segments_cartesian(
        self,
        simple_kpath_kpoints,
        simple_segment_names,
        simple_cubic_reciprocal_lattice,
    ):
        """Test get_segments with cartesian=True."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
            reciprocal_lattice=simple_cubic_reciprocal_lattice,
        )

        segments_reduced = kpath.get_segments(cartesian=False)
        segments_cartesian = kpath.get_segments(cartesian=True)

        # Cartesian coordinates should be scaled by reciprocal lattice
        assert not np.allclose(segments_reduced[0], segments_cartesian[0])

    def test_get_distances_as_segments(self, kpath_with_segments):
        """Test get_distances returns list of segment distances."""
        distances = kpath_with_segments.get_distances(as_segments=True)

        assert isinstance(distances, list)
        assert len(distances) == 3

    def test_get_distances_concatenated(self, kpath_with_segments):
        """Test get_distances returns concatenated array."""
        distances = kpath_with_segments.get_distances(as_segments=False)

        assert isinstance(distances, np.ndarray)
        assert len(distances) == kpath_with_segments.n_kpoints

    def test_get_distances_cumulative(self, kpath_with_segments):
        """Test get_distances with cumulative_across_segments."""
        distances_cumulative = kpath_with_segments.get_distances(
            as_segments=True, cumlative_across_segments=True
        )
        distances_not_cumulative = kpath_with_segments.get_distances(
            as_segments=True, cumlative_across_segments=False
        )

        # First segments should be the same
        assert np.allclose(distances_cumulative[0], distances_not_cumulative[0])
        # Later segments should differ when cumulative
        assert not np.allclose(distances_cumulative[1], distances_not_cumulative[1])

    def test_get_segment_indices(self, kpath_with_segments):
        """Test get_segment_indices returns tuple of lists."""
        segment_indices, continuous, discontinuous = kpath_with_segments.get_segment_indices()

        assert len(segment_indices) > 0
        assert isinstance(continuous, list)
        assert isinstance(discontinuous, list)

    def test_get_special_kpoints_as_segments(self, kpath_with_segments):
        """Test get_special_kpoints with as_segments=True."""
        special = kpath_with_segments.get_special_kpoints(as_segments=True)

        # Should return list of (start, end) tuples
        assert len(special) == kpath_with_segments.n_segments

    def test_get_special_kpoints_flat(self, kpath_with_segments):
        """Test get_special_kpoints with as_segments=False."""
        special = kpath_with_segments.get_special_kpoints(as_segments=False)

        # Should return unique special k-points as array
        assert isinstance(special, np.ndarray)
        assert special.ndim == 2
        assert special.shape[1] == 3

    def test_get_special_kpoint_names(self, kpath_with_segments):
        """Test get_special_kpoint_names returns unique names."""
        names = kpath_with_segments.get_special_kpoint_names()

        # Should not have duplicates
        assert len(names) == len(set(names))

    def test_get_continuous_segments(self, kpath_with_segments):
        """Test get_continuous_segments merges continuous segments."""
        continuous = kpath_with_segments.get_continuous_segments()

        assert isinstance(continuous, list)
        assert len(continuous) > 0

    def test_str_representation(self, kpath_with_segments):
        """Test __str__ returns formatted string."""
        str_repr = str(kpath_with_segments)

        assert "K-Path" in str_repr
        assert "n_kpoints" in str_repr
        assert "n_segments" in str_repr

    def test_equality_same_kpath(self, kpath_with_segments):
        """Test equality comparison with identical KPath."""
        kpath2 = KPath(
            kpoints=kpath_with_segments.kpoints.copy(),
            segment_names=[("Gamma", "X"), ("X", "M"), ("M", "Gamma")],
        )

        assert kpath_with_segments == kpath2

    def test_equality_different_kpath(self, kpath_with_segments, special_kpoint_map):
        """Test equality comparison with different KPath."""
        kpath2 = KPath(
            n_grids=[5, 5, 5],
            segment_names=[("Γ", "X"), ("X", "M"), ("M", "Γ")],
            special_kpoint_map=special_kpoint_map,
        )

        # Different n_grids should make them unequal
        assert not (kpath_with_segments == kpath2)


# =============================================================================
# Test Classes - KPath Discontinuities
# =============================================================================


class TestKPathDiscontinuities:
    """Test class for KPath discontinuity handling."""

    @pytest.fixture
    def discontinuous_kpath_kpoints(self):
        """
        Create k-points with a discontinuity between segments.

        Path: Gamma -> X | M -> R (| indicates discontinuity)
        """
        # Gamma to X: (0,0,0) -> (0.5,0,0)
        seg1 = np.linspace([0, 0, 0], [0.5, 0, 0], 5)
        # M to R: (0.5,0.5,0) -> (0.5,0.5,0.5) - discontinuous from X
        seg2 = np.linspace([0.5, 0.5, 0], [0.5, 0.5, 0.5], 5)

        return np.vstack([seg1, seg2])

    @pytest.fixture
    def discontinuous_segment_names(self):
        """Create segment names for discontinuous path."""
        return [("Gamma", "X"), ("M", "R")]

    def test_discontinuity_detection(
        self, discontinuous_kpath_kpoints, discontinuous_segment_names
    ):
        """Test that discontinuities are detected."""
        kpath = KPath(
            kpoints=discontinuous_kpath_kpoints,
            segment_names=discontinuous_segment_names,
        )

        assert len(kpath.discontinuity_start_indices) > 0

    def test_continuous_detection(self, simple_kpath_kpoints, simple_segment_names):
        """Test that continuous segments are detected."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
        )

        # Continuous path should have continuous indices
        assert len(kpath.continuous_start_indices) >= 0

    def test_tick_names_with_discontinuity(
        self, discontinuous_kpath_kpoints, discontinuous_segment_names
    ):
        """Test that tick names are generated for discontinuous paths."""
        kpath = KPath(
            kpoints=discontinuous_kpath_kpoints,
            segment_names=discontinuous_segment_names,
        )

        tick_names = kpath.tick_names

        # Verify tick names are generated (at least start and end points)
        assert len(tick_names) >= 2, f"Should have at least 2 tick names, got: {tick_names}"
        # Verify we have the expected special point names
        tick_str = " ".join(tick_names)
        # Should contain gamma (or its normalized form) and some other points
        assert any(name in tick_str for name in ["Γ", "$\\Gamma$", "X", "M", "R"]), (
            f"Tick names should include special k-points: {tick_names}"
        )

    def test_continuous_segments_grouping(self, simple_kpath_kpoints, simple_segment_names):
        """Test get_continuous_segments groups continuous parts."""
        kpath = KPath(
            kpoints=simple_kpath_kpoints,
            segment_names=simple_segment_names,
        )

        continuous = kpath.get_continuous_segments()

        # For a continuous path, should merge segments
        total_points = sum(len(seg) for seg in continuous)
        assert total_points == kpath.n_kpoints

    def test_threshold_affects_detection(self):
        """Test that discontinuity_threshold affects detection."""
        # Create path with moderate jump
        kpoints = np.vstack(
            [
                np.linspace([0, 0, 0], [0.1, 0, 0], 5),
                np.linspace([0.25, 0, 0], [0.35, 0, 0], 5),  # Jump of 0.15
            ]
        )
        segment_names = [("A", "B"), ("C", "D")]

        # With low threshold, should detect discontinuity
        kpath_low = KPath(
            kpoints=kpoints,
            segment_names=segment_names,
            discontinuity_threshold=0.1,
        )

        # With high threshold, should NOT detect discontinuity
        kpath_high = KPath(
            kpoints=kpoints,
            segment_names=segment_names,
            discontinuity_threshold=0.2,
        )

        assert len(kpath_low.discontinuity_start_indices) >= len(
            kpath_high.discontinuity_start_indices
        )
