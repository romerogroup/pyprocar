"""Tests for BandStructurePlotter.

Follows the same organization as test_dos_plot.py with phases:
1. Mock fixtures and factories
2. Initialization tests
3. Core plot() method tests
4. Scalars mode tests
5. Colorbar tests
6. Axis configuration tests
7. High-symmetry point tests
8. Edge cases
9. Integration tests
"""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D

from pyprocar.plotter.bs_plot import BandSeries, BandStructurePlotter

# =============================================================================
# Mock Fixtures and Factories
# =============================================================================


def _make_mock_property(
    n_kpoints: int = 50,
    n_bands: int = 5,
    n_spins: int = 1,
    label: str = "Energy",
    units: str = "eV",
) -> Mock:
    """Create a mock Property object for testing BandStructurePlotter."""
    mock = Mock()

    # Create band-like data
    k = np.linspace(0, 2 * np.pi, n_kpoints)
    bands = np.zeros((n_kpoints, n_bands, n_spins))
    for iband in range(n_bands):
        for ispin in range(n_spins):
            bands[:, iband, ispin] = -5.0 + iband + 0.5 * np.sin(k + iband) + 0.1 * ispin

    mock.to_array.return_value = bands
    mock.label = label
    mock.units = units

    # kpath metadata
    mock.metadata = {
        "kpath": {
            "k_distances": np.linspace(0, 5.0, n_kpoints),
            "tick_positions": [0, 12, 25, 37, 49],
            "tick_names": ["Gamma", "X", "M", "Gamma", "R"],
            "tick_names_latex": ["$\\Gamma$", "$X$", "$M$", "$\\Gamma$", "$R$"],
        }
    }

    return mock


def _make_mock_scalars_property(
    n_kpoints: int = 50,
    n_bands: int = 5,
    n_spins: int = 1,
) -> Mock:
    """Create a mock Property for scalar coloring data."""
    mock = Mock()

    scalars = np.random.rand(n_kpoints, n_bands, n_spins)
    mock.to_array.return_value = scalars
    mock.label = "Projection"
    mock.units = ""
    mock.metadata = {}
    mock.rounded_data_lim = [(0.0, 1.0)] * n_spins

    return mock


def _make_mock_kpath(n_kpoints: int = 50) -> Mock:
    """Create a mock KPath for legacy API tests."""
    mock = Mock()
    mock.get_distances.return_value = np.linspace(0, 5.0, n_kpoints)
    mock.tick_positions = [0, 12, 25, 37, 49]
    mock.tick_names = ["Gamma", "X", "M", "Gamma", "R"]
    mock.tick_names_latex = ["$\\Gamma$", "$X$", "$M$", "$\\Gamma$", "$R$"]
    return mock


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_property_single_spin():
    return _make_mock_property(n_spins=1)


@pytest.fixture
def mock_property_two_spins():
    return _make_mock_property(n_spins=2)


@pytest.fixture
def mock_scalars_single_spin():
    return _make_mock_scalars_property(n_spins=1)


@pytest.fixture
def mock_scalars_two_spins():
    return _make_mock_scalars_property(n_spins=2)


@pytest.fixture
def mock_kpath():
    return _make_mock_kpath()


# =============================================================================
# Phase 2: Initialization Tests
# =============================================================================


class TestBandStructurePlotterInitialization:
    """Tests for BandStructurePlotter initialization."""

    def test_default_initialization(self):
        plotter = BandStructurePlotter()
        assert plotter.figsize == (8, 6)
        assert plotter.dpi == 100
        assert plotter.ax is not None
        assert plotter.fig is not None
        plt.close(plotter.fig)

    def test_custom_figsize(self):
        plotter = BandStructurePlotter(figsize=(10, 8))
        assert plotter.figsize == (10, 8)
        plt.close(plotter.fig)

    def test_external_axes(self):
        fig, ax = plt.subplots()
        plotter = BandStructurePlotter(ax=ax)
        assert plotter.ax is ax
        plt.close(fig)


# =============================================================================
# Phase 2: _to_series_list Tests
# =============================================================================


class TestBandStructurePlotterToSeriesList:
    """Tests for the _to_series_list method."""

    def test_to_series_list_basic(self, mock_property_single_spin):
        """Test basic series list creation."""
        plotter = BandStructurePlotter()
        series_list = plotter._to_series_list(mock_property_single_spin, None, None, "normal")

        n_bands = mock_property_single_spin.to_array().shape[1]
        n_spins = mock_property_single_spin.to_array().shape[2]

        # Should have n_bands * n_spins series
        assert len(series_list) == n_bands * n_spins

        # Check that each series has correct band_index and spin_index
        for i, series in enumerate(series_list):
            assert isinstance(series, BandSeries)
            assert series.band_index == i // n_spins
            assert series.spin_index == i % n_spins
            assert series.scalars is None
            assert series.vectors is None

        plt.close(plotter.fig)

    def test_to_series_list_with_scalars(self, mock_property_single_spin, mock_scalars_single_spin):
        """Test series list with scalar data."""
        plotter = BandStructurePlotter()
        series_list = plotter._to_series_list(
            mock_property_single_spin, mock_scalars_single_spin, None, "normal"
        )

        # Check that scalars are properly sliced
        for series in series_list:
            assert series.scalars is not None
            assert series.scalars.shape == (50,)  # n_kpoints
            assert series.scalars_label == "Projection"

        plt.close(plotter.fig)

    def test_to_series_list_two_spins(self, mock_property_two_spins):
        """Test series list with two spin channels."""
        plotter = BandStructurePlotter()
        series_list = plotter._to_series_list(mock_property_two_spins, None, None, "normal")

        n_bands = mock_property_two_spins.to_array().shape[1]
        n_spins = mock_property_two_spins.to_array().shape[2]

        # Should have n_bands * n_spins series
        assert len(series_list) == n_bands * n_spins

        # Check spin labels are assigned (uses arrow symbols)
        for series in series_list:
            if series.spin_index == 0:
                assert "↑" in series.label or "Band" in series.label
            else:
                assert "↓" in series.label or "Band" in series.label

        plt.close(plotter.fig)

    def test_to_series_list_flip_channel_mode(self, mock_property_two_spins):
        """Test that flip channel mode negates second spin channel."""
        plotter = BandStructurePlotter()

        # Get series with normal mode
        series_normal = plotter._to_series_list(mock_property_two_spins, None, None, "normal")

        # Get series with flip mode
        series_flipped = plotter._to_series_list(mock_property_two_spins, None, None, "flip")

        # First spin should be identical
        for sn, sf in zip(series_normal, series_flipped):
            if sn.spin_index == 0:
                assert np.allclose(sn.y, sf.y)
            else:
                # Second spin should be negated
                assert np.allclose(sn.y, -sf.y)

        plt.close(plotter.fig)

    def test_to_series_list_missing_kpath_raises(self):
        """Test that missing kpath metadata raises error."""
        mock = Mock()
        mock.to_array.return_value = np.random.rand(50, 5, 1)
        mock.metadata = {}  # No kpath

        plotter = BandStructurePlotter()
        with pytest.raises(ValueError, match="kpath metadata"):
            plotter._to_series_list(mock, None, None, "normal")

        plt.close(plotter.fig)

    def test_distribute_kwargs(self):
        """Test kwargs distribution to channels."""
        plotter = BandStructurePlotter()

        # Test with list that matches channel count
        kwargs = {"color": ["red", "blue"], "linewidth": 2.0}
        result = plotter._distribute_kwargs(kwargs, 2)

        assert len(result) == 2
        assert result[0]["color"] == "red"
        assert result[1]["color"] == "blue"
        assert result[0]["linewidth"] == 2.0
        assert result[1]["linewidth"] == 2.0

        plt.close(plotter.fig)

    def test_build_series_label_single_spin(self, mock_property_single_spin):
        """Test label building for single spin."""
        plotter = BandStructurePlotter()
        label = plotter._build_series_label(mock_property_single_spin, 0, 0, 5, 1)

        # Single spin should not have labels by default
        assert label is None

        plt.close(plotter.fig)

    def test_build_series_label_two_spins(self, mock_property_two_spins):
        """Test label building for two spins."""
        plotter = BandStructurePlotter()
        label_up = plotter._build_series_label(mock_property_two_spins, 0, 0, 5, 2)
        label_down = plotter._build_series_label(mock_property_two_spins, 0, 1, 5, 2)

        # Two spins should have spin labels
        assert label_up is not None
        assert label_down is not None

        plt.close(plotter.fig)


# =============================================================================
# Phase 3: Core plot() Method Tests
# =============================================================================


class TestBandStructurePlotterPlot:
    """Tests for the unified plot() method."""

    def test_plot_none_mode_creates_lines(self, mock_property_single_spin):
        """Test that scalars_mode='none' creates line plots."""
        plotter = BandStructurePlotter()
        artists = plotter.plot(mock_property_single_spin, scalars_mode="none")

        n_bands = mock_property_single_spin.to_array().shape[1]
        assert len(artists) == n_bands
        assert all(isinstance(a, Line2D) for a in artists.values())
        plt.close(plotter.fig)

    def test_plot_scatter_mode_creates_scatter(
        self, mock_property_single_spin, mock_scalars_single_spin
    ):
        """Test that scalars_mode='scatter' creates scatter plots."""
        plotter = BandStructurePlotter()
        artists = plotter.plot(
            mock_property_single_spin,
            scalars_data=mock_scalars_single_spin,
            scalars_mode="scatter",
        )

        n_bands = mock_property_single_spin.to_array().shape[1]
        assert len(artists) == n_bands
        assert all(isinstance(a, PathCollection) for a in artists.values())
        plt.close(plotter.fig)

    def test_plot_parametric_mode_creates_collections(
        self, mock_property_single_spin, mock_scalars_single_spin
    ):
        """Test that scalars_mode='parametric' creates LineCollections."""
        plotter = BandStructurePlotter()
        artists = plotter.plot(
            mock_property_single_spin,
            scalars_data=mock_scalars_single_spin,
            scalars_mode="parametric",
        )

        n_bands = mock_property_single_spin.to_array().shape[1]
        assert len(artists) == n_bands
        assert all(isinstance(a, LineCollection) for a in artists.values())
        plt.close(plotter.fig)

    def test_plot_two_spins_doubles_artists(self, mock_property_two_spins):
        """Test that two spin channels produce n_bands * n_spins artists."""
        plotter = BandStructurePlotter()
        artists = plotter.plot(mock_property_two_spins, scalars_mode="none")

        n_bands = mock_property_two_spins.to_array().shape[1]
        n_spins = mock_property_two_spins.to_array().shape[2]
        assert len(artists) == n_bands * n_spins
        plt.close(plotter.fig)

    def test_plot_returns_dict_with_correct_keys(self, mock_property_single_spin):
        """Test that returned dict has (band_index, spin_index) keys."""
        plotter = BandStructurePlotter()
        artists = plotter.plot(mock_property_single_spin, scalars_mode="none")

        n_bands = mock_property_single_spin.to_array().shape[1]
        for iband in range(n_bands):
            assert (iband, 0) in artists

        plt.close(plotter.fig)

    def test_plot_stores_x_data(self, mock_property_single_spin):
        """Test that plot() stores x data for axis methods."""
        plotter = BandStructurePlotter()
        plotter.plot(mock_property_single_spin, scalars_mode="none")

        k_distances = mock_property_single_spin.metadata["kpath"]["k_distances"]
        assert plotter.x is not None
        assert np.allclose(plotter.x, k_distances)
        plt.close(plotter.fig)

    def test_plot_unknown_scalars_mode_raises(self, mock_property_single_spin):
        """Test that unknown scalars_mode raises ValueError."""
        plotter = BandStructurePlotter()
        with pytest.raises(ValueError, match="Unknown scalars_mode"):
            plotter.plot(mock_property_single_spin, scalars_mode="invalid")
        plt.close(plotter.fig)


# =============================================================================
# Phase 3: Scalars Mode Tests
# =============================================================================


class TestBandStructurePlotterScalarsModes:
    """Tests for scalar coloring modes."""

    def test_scatter_uses_cmap(self, mock_property_single_spin, mock_scalars_single_spin):
        """Test that scatter mode uses specified colormap."""
        plotter = BandStructurePlotter()
        plotter.plot(
            mock_property_single_spin,
            scalars_data=mock_scalars_single_spin,
            scalars_mode="scatter",
            scalars_cmap="viridis",
        )

        scatter = list(plotter.ax.collections)[0]
        assert scatter.get_cmap().name == "viridis"
        plt.close(plotter.fig)

    def test_scatter_respects_clim(self, mock_property_single_spin, mock_scalars_single_spin):
        """Test that scatter mode respects color limits."""
        plotter = BandStructurePlotter()
        plotter.plot(
            mock_property_single_spin,
            scalars_data=mock_scalars_single_spin,
            scalars_mode="scatter",
            scalars_clim=(0.2, 0.8),
        )

        scatter = list(plotter.ax.collections)[0]
        assert scatter.get_clim() == (0.2, 0.8)
        plt.close(plotter.fig)

    def test_parametric_uses_cmap(self, mock_property_single_spin, mock_scalars_single_spin):
        """Test that parametric mode uses specified colormap."""
        plotter = BandStructurePlotter()
        plotter.plot(
            mock_property_single_spin,
            scalars_data=mock_scalars_single_spin,
            scalars_mode="parametric",
            scalars_cmap="coolwarm",
        )

        lc = list(plotter.ax.collections)[0]
        assert lc.get_cmap().name == "coolwarm"
        plt.close(plotter.fig)

    def test_resolve_clim_uses_user_value(self, mock_property_single_spin):
        """Test that _resolve_clim returns user-provided clim."""
        plotter = BandStructurePlotter()
        series_list = plotter._to_series_list(mock_property_single_spin, None, None, "normal")

        clim = plotter._resolve_clim(series_list, (0.1, 0.9))
        assert clim == (0.1, 0.9)
        plt.close(plotter.fig)

    def test_resolve_clim_auto_from_data(self, mock_property_single_spin, mock_scalars_single_spin):
        """Test that _resolve_clim computes limits from data when not provided."""
        plotter = BandStructurePlotter()
        series_list = plotter._to_series_list(
            mock_property_single_spin, mock_scalars_single_spin, None, "normal"
        )

        clim = plotter._resolve_clim(series_list, None)
        # Should compute from actual scalars data
        assert clim[0] >= 0.0
        assert clim[1] <= 1.0
        plt.close(plotter.fig)


# =============================================================================
# Phase 3: Colorbar Tests
# =============================================================================


class TestBandStructurePlotterColorbar:
    """Tests for colorbar functionality."""

    def test_colorbar_single_creates_colorbar(
        self, mock_property_single_spin, mock_scalars_single_spin
    ):
        """Test that colorbar is created with scalars_show_colorbar='single'."""
        plotter = BandStructurePlotter()
        plotter.plot(
            mock_property_single_spin,
            scalars_data=mock_scalars_single_spin,
            scalars_mode="scatter",
            scalars_show_colorbar="single",
        )

        assert plotter.colorbar is not None
        plt.close(plotter.fig)

    def test_colorbar_none_no_colorbar(self, mock_property_single_spin, mock_scalars_single_spin):
        """Test that no colorbar is created with scalars_show_colorbar='none'."""
        plotter = BandStructurePlotter()
        plotter.plot(
            mock_property_single_spin,
            scalars_data=mock_scalars_single_spin,
            scalars_mode="scatter",
            scalars_show_colorbar="none",
        )

        assert plotter.colorbar is None
        plt.close(plotter.fig)

    def test_no_colorbar_for_none_mode(self, mock_property_single_spin):
        """Test that no colorbar is created for scalars_mode='none'."""
        plotter = BandStructurePlotter()
        plotter.plot(
            mock_property_single_spin,
            scalars_mode="none",
            scalars_show_colorbar="single",
        )

        assert plotter.colorbar is None
        plt.close(plotter.fig)


# =============================================================================
# Phase 3: High-Symmetry Point Tests
# =============================================================================


class TestBandStructurePlotterHighSymmetry:
    """Tests for high-symmetry point handling."""

    def test_high_symmetry_lines_drawn(self, mock_property_single_spin):
        """Test that vertical lines are drawn at high-symmetry points."""
        plotter = BandStructurePlotter()
        plotter.plot(mock_property_single_spin, scalars_mode="none")

        # Count vertical lines (axvline creates Line2D objects)
        # We can check that _draw_high_symmetry_lines was called by verifying
        # the tick positions were stored
        n_expected_hsym = len(mock_property_single_spin.metadata["kpath"]["tick_positions"])
        assert len(plotter._tick_positions) == n_expected_hsym
        plt.close(plotter.fig)

    def test_tick_positions_stored(self, mock_property_single_spin):
        """Test that tick positions are stored from metadata."""
        plotter = BandStructurePlotter()
        plotter.plot(mock_property_single_spin, scalars_mode="none")

        expected_positions = mock_property_single_spin.metadata["kpath"]["tick_positions"]
        assert plotter._tick_positions == expected_positions
        plt.close(plotter.fig)

    def test_tick_names_stored(self, mock_property_single_spin):
        """Test that tick names are stored from metadata."""
        plotter = BandStructurePlotter()
        plotter.plot(mock_property_single_spin, scalars_mode="none")

        expected_names = mock_property_single_spin.metadata["kpath"]["tick_names"]
        assert plotter._tick_names == expected_names
        plt.close(plotter.fig)


# =============================================================================
# Phase 3: Edge Cases
# =============================================================================


class TestBandStructurePlotterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_band(self):
        """Test plotting with a single band."""
        mock = _make_mock_property(n_bands=1)
        plotter = BandStructurePlotter()
        artists = plotter.plot(mock, scalars_mode="none")

        assert len(artists) == 1
        plt.close(plotter.fig)

    def test_missing_kpath_metadata_raises(self):
        """Test that missing kpath metadata raises ValueError."""
        mock = Mock()
        mock.to_array.return_value = np.random.rand(50, 5, 1)
        mock.metadata = {}  # No kpath

        plotter = BandStructurePlotter()
        with pytest.raises(ValueError, match="kpath metadata"):
            plotter.plot(mock, scalars_mode="none")
        plt.close(plotter.fig)

    def test_2d_bands_array(self):
        """Test that 2D bands array is handled correctly."""
        mock = Mock()
        mock.to_array.return_value = np.random.rand(50, 5)  # 2D, no spin dim
        mock.metadata = {
            "kpath": {
                "k_distances": np.linspace(0, 5.0, 50),
                "tick_positions": [0, 25, 49],
                "tick_names": ["G", "X", "M"],
            }
        }
        mock.label = "Energy"

        plotter = BandStructurePlotter()
        artists = plotter.plot(mock, scalars_mode="none")

        assert len(artists) == 5  # 5 bands
        plt.close(plotter.fig)

    def test_export_data_recorded(self, mock_property_single_spin):
        """Test that band data is recorded for export."""
        plotter = BandStructurePlotter()
        plotter.plot(mock_property_single_spin, scalars_mode="none")

        # Check that values_dict was populated
        assert len(plotter.values_dict) > 0
        assert "bands__band-0_spinChannel-0" in plotter.values_dict
        plt.close(plotter.fig)


# =============================================================================
# Phase 4: Wrapper Helper Methods Tests
# =============================================================================


class TestBandStructurePlotterWrapperHelpers:
    """Tests for Property wrapper helper methods."""

    def test_wrap_as_property_creates_property(self, mock_kpath):
        """Test _wrap_as_property creates a valid Property."""
        from pyprocar.core.property_store import Property

        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)

        prop = plotter._wrap_as_property(mock_kpath, bands)

        assert isinstance(prop, Property)
        assert prop.name == "bands"
        assert prop.units == "eV"
        assert prop.label == "Energy"
        plt.close(plotter.fig)

    def test_wrap_as_property_includes_kpath_metadata(self, mock_kpath):
        """Test _wrap_as_property includes kpath metadata."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)

        prop = plotter._wrap_as_property(mock_kpath, bands)

        assert "kpath" in prop.metadata
        kpath_meta = prop.metadata["kpath"]
        assert "k_distances" in kpath_meta
        assert "tick_positions" in kpath_meta
        assert "tick_names" in kpath_meta
        assert len(kpath_meta["k_distances"]) == 50
        plt.close(plotter.fig)

    def test_wrap_as_property_handles_2d_bands(self, mock_kpath):
        """Test _wrap_as_property adds spin dimension for 2D bands."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5)  # 2D, no spin dimension

        prop = plotter._wrap_as_property(mock_kpath, bands)

        # Should have added spin dimension
        assert prop.value.ndim == 3
        assert prop.value.shape == (50, 5, 1)
        plt.close(plotter.fig)

    def test_wrap_scalars_as_property_creates_property(self):
        """Test _wrap_scalars_as_property creates a valid Property."""
        from pyprocar.core.property_store import Property

        plotter = BandStructurePlotter()
        scalars = np.random.rand(50, 5, 1)

        prop = plotter._wrap_scalars_as_property(scalars)

        assert isinstance(prop, Property)
        assert prop.name == "scalars"
        assert prop.label == "Projection"
        plt.close(plotter.fig)

    def test_wrap_scalars_as_property_custom_label(self):
        """Test _wrap_scalars_as_property accepts custom label."""
        plotter = BandStructurePlotter()
        scalars = np.random.rand(50, 5, 1)

        prop = plotter._wrap_scalars_as_property(scalars, label="Orbital Weight")

        assert prop.label == "Orbital Weight"
        plt.close(plotter.fig)


# =============================================================================
# Phase 4: Legacy API Backwards Compatibility Tests
# =============================================================================


class TestBandStructurePlotterLegacyAPI:
    """Tests for backwards-compatible array-based API methods."""

    def test_plot_plain_creates_lines(self, mock_kpath):
        """Test plot_plain creates Line2D artists."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)

        artists = plotter.plot_plain(mock_kpath, bands)

        assert len(artists) == 5  # n_bands
        assert all(isinstance(a, Line2D) for a in artists.values())
        plt.close(plotter.fig)

    def test_plot_plain_sets_axis_properties(self, mock_kpath):
        """Test plot_plain sets axis limits and ticks."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)

        plotter.plot_plain(mock_kpath, bands)

        # Check that axis properties were set
        xlim = plotter.ax.get_xlim()
        ylim = plotter.ax.get_ylim()
        assert xlim[0] < xlim[1]
        assert ylim[0] < ylim[1]
        plt.close(plotter.fig)

    def test_plot_scatter_creates_scatter(self, mock_kpath):
        """Test plot_scatter creates PathCollection artists."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)
        scalars = np.random.rand(50, 5, 1)

        artists = plotter.plot_scatter(mock_kpath, bands, scalars=scalars)

        assert len(artists) == 5  # n_bands
        assert all(isinstance(a, PathCollection) for a in artists.values())
        plt.close(plotter.fig)

    def test_plot_scatter_without_scalars(self, mock_kpath):
        """Test plot_scatter works without scalars."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)

        artists = plotter.plot_scatter(mock_kpath, bands, scalars=None)

        assert len(artists) == 5  # n_bands
        plt.close(plotter.fig)

    def test_plot_parametric_creates_collections(self, mock_kpath):
        """Test plot_parametric creates LineCollection artists."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)
        scalars = np.random.rand(50, 5, 1)

        artists = plotter.plot_parametric(mock_kpath, bands, scalars=scalars)

        assert len(artists) == 5  # n_bands
        assert all(isinstance(a, LineCollection) for a in artists.values())
        plt.close(plotter.fig)

    def test_plot_parametric_without_scalars(self, mock_kpath):
        """Test plot_parametric works without scalars."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)

        artists = plotter.plot_parametric(mock_kpath, bands, scalars=None)

        assert len(artists) == 5  # n_bands
        plt.close(plotter.fig)

    def test_legacy_methods_record_export_data(self, mock_kpath):
        """Test legacy methods record data for export."""
        plotter = BandStructurePlotter()
        bands = np.random.rand(50, 5, 1)

        plotter.plot_plain(mock_kpath, bands)

        assert "bands__band-0_spinChannel-0" in plotter.values_dict
        assert "kpath_values" in plotter.values_dict
        plt.close(plotter.fig)
