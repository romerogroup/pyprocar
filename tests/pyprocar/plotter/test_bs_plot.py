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

from pyprocar.plotter.bs_plot import BandStructurePlotter

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
# Phase 2: Per-series data through plot()
# =============================================================================


def _make_literal_property(bands: np.ndarray) -> Mock:
    """Property over three k-points with literal band energies."""
    mock = Mock()
    mock.to_array.return_value = bands
    mock.label = "Energy"
    mock.units = "eV"
    mock.metadata = {
        "kpath": {
            "k_distances": np.array([0.0, 0.5, 1.0]),
            "tick_positions": [0, 2],
            "tick_names": ["G", "X"],
        }
    }
    return mock


# bands[k, band, spin]
LITERAL_BANDS = np.array(
    [
        [[-1.0, -1.5], [2.0, 2.5]],
        [[-0.5, -1.0], [2.5, 3.0]],
        [[-1.0, -1.5], [3.0, 3.5]],
    ]
)


def _literal_scalars(rounded_data_lim) -> Mock:
    mock = Mock()
    mock.to_array.return_value = np.array(
        [
            [[0.1, 0.2], [0.6, 0.7]],
            [[0.3, 0.4], [0.8, 0.9]],
            [[0.5, 0.0], [1.25, 0.25]],
        ]
    )
    mock.label = "Projection"
    mock.units = ""
    mock.metadata = {}
    mock.rounded_data_lim = rounded_data_lim
    return mock


class TestBandStructurePlotterSeries:
    """Per-(band, spin) data reaches the artists."""

    def test_one_line_per_band_and_spin_with_its_energies(self):
        plotter = BandStructurePlotter()
        artists = plotter.plot(_make_literal_property(LITERAL_BANDS))

        assert sorted(artists) == [(0, 0), (0, 1), (1, 0), (1, 1)]
        np.testing.assert_array_equal(artists[(1, 1)].get_ydata(), [2.5, 3.0, 3.5])
        np.testing.assert_array_equal(artists[(0, 1)].get_xdata(), [0.0, 0.5, 1.0])
        plt.close(plotter.fig)

    def test_flip_negates_second_spin_only(self):
        plotter = BandStructurePlotter()
        artists = plotter.plot(_make_literal_property(LITERAL_BANDS), channel_mode="flip")

        np.testing.assert_array_equal(artists[(0, 0)].get_ydata(), [-1.0, -0.5, -1.0])
        np.testing.assert_array_equal(artists[(0, 1)].get_ydata(), [1.5, 1.0, 1.5])
        plt.close(plotter.fig)

    def test_invalid_channel_mode_raises(self):
        plotter = BandStructurePlotter()
        with pytest.raises(ValueError, match="Invalid channel mode"):
            plotter.plot(_make_literal_property(LITERAL_BANDS), channel_mode="sideways")
        plt.close(plotter.fig)

    def test_list_kwarg_matching_spin_count_splits_per_spin(self):
        plotter = BandStructurePlotter()
        artists = plotter.plot(
            _make_literal_property(LITERAL_BANDS), color=["red", "blue"], linewidth=2.0
        )

        assert [artists[k].get_color() for k in sorted(artists)] == [
            "red",
            "blue",
            "red",
            "blue",
        ]
        assert {a.get_linewidth() for a in artists.values()} == {2.0}
        plt.close(plotter.fig)

    def test_2d_bands_are_one_spin(self):
        plotter = BandStructurePlotter()
        artists = plotter.plot(_make_literal_property(LITERAL_BANDS[:, :, 0]))

        assert sorted(artists) == [(0, 0), (1, 0)]
        np.testing.assert_array_equal(artists[(1, 0)].get_ydata(), [2.0, 2.5, 3.0])
        plt.close(plotter.fig)

    def test_scatter_colors_each_series_by_its_own_scalars(self):
        plotter = BandStructurePlotter()
        artists = plotter.plot(
            _make_literal_property(LITERAL_BANDS),
            scalars_data=_literal_scalars(None),
            scalars_mode="scatter",
        )

        np.testing.assert_array_equal(artists[(1, 0)].get_array(), [0.6, 0.8, 1.25])
        np.testing.assert_array_equal(artists[(0, 1)].get_array(), [0.2, 0.4, 0.0])
        plt.close(plotter.fig)

    def test_clim_spans_per_band_limits_of_every_spin(self):
        # rows are spins; each holds the band minima then the band maxima
        lims = np.array([[0.1, 0.5, 0.4, 0.9], [0.0, 0.2, 0.3, 1.5]])
        plotter = BandStructurePlotter()
        artists = plotter.plot(
            _make_literal_property(LITERAL_BANDS),
            scalars_data=_literal_scalars(lims),
            scalars_mode="parametric",
        )

        assert {a.get_clim() for a in artists.values()} == {(0.0, 1.5)}
        assert plotter.colorbar.mappable.get_clim() == (0.0, 1.5)
        plt.close(plotter.fig)

    def test_clim_falls_back_to_scalar_range_without_limits(self):
        plotter = BandStructurePlotter()
        artists = plotter.plot(
            _make_literal_property(LITERAL_BANDS),
            scalars_data=_literal_scalars(None),
            scalars_mode="scatter",
        )

        assert {a.get_clim() for a in artists.values()} == {(0.0, 1.25)}
        plt.close(plotter.fig)

    def test_user_clim_overrides_limits(self):
        plotter = BandStructurePlotter()
        artists = plotter.plot(
            _make_literal_property(LITERAL_BANDS),
            scalars_data=_literal_scalars(np.array([[0.1, 0.9], [0.0, 1.5]])),
            scalars_mode="scatter",
            scalars_clim=(0.25, 0.75),
        )

        assert {a.get_clim() for a in artists.values()} == {(0.25, 0.75)}
        plt.close(plotter.fig)

    @pytest.mark.parametrize("shape", [(3, 1, 1), (3, 1), (3, 2, 1)])
    def test_scalars_with_fewer_bands_or_spins_raise(self, shape):
        scalars = _literal_scalars(None)
        scalars.to_array.return_value = np.zeros(shape)
        plotter = BandStructurePlotter()

        with pytest.raises(ValueError, match=r"scalars shape \(3, .*\) does not match"):
            plotter.plot(
                _make_literal_property(LITERAL_BANDS),
                scalars_data=scalars,
                scalars_mode="scatter",
            )
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

    def test_scatter_uses_cmap(
        self, mock_property_single_spin, mock_scalars_single_spin
    ):
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

    def test_scatter_respects_clim(
        self, mock_property_single_spin, mock_scalars_single_spin
    ):
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

    def test_parametric_uses_cmap(
        self, mock_property_single_spin, mock_scalars_single_spin
    ):
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

    def test_colorbar_none_no_colorbar(
        self, mock_property_single_spin, mock_scalars_single_spin
    ):
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
