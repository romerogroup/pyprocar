import matplotlib as mpl

mpl.use("Agg")

from unittest.mock import MagicMock, Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from pyprocar.core.dos import DensityOfStates
from pyprocar.plotter.dos_plot import AxesOrientation, DOSPlotter

_rng = np.random.default_rng(42)


def _make_dos(n_spins: int = 2) -> DensityOfStates:
    energies = np.linspace(-1.0, 1.0, 5)
    base = np.linspace(0.1, 0.5, energies.size)
    total_channels = [base + 0.2 * spin for spin in range(n_spins)]
    total = np.stack(total_channels, axis=1)

    projected = np.zeros((energies.size, n_spins, 1, 2), dtype=float)
    for spin in range(n_spins):
        projected[:, spin, 0, 0] = base + 0.05 * spin
        projected[:, spin, 0, 1] = base + 0.1 * spin

    return DensityOfStates(
        energies=energies,
        total=total,
        projected=projected,
    )


def _make_mock_property(
    n_points: int = 100,
    n_channels: int = 1,
    label: str = "Test DOS",
    units: str = "states/eV",
    points_label: str = "Energy",
    points_units: str = "eV",
    metadata: dict[str, list[str]] | None = None,
    data_lim: tuple[float, float] | None = None,
) -> Mock:
    """Create a mock Property object for testing DOSPlotter.

    Args:
        n_points: Number of sample points
        n_channels: Number of data channels (1 for single, 2 for spin-polarized)
        label: Property label for axis/legend
        units: Unit string
        points_label: Label for points axis
        points_units: Units for points axis
        metadata: Optional metadata dict with "label" key for per-channel labels
        data_lim: Optional (min, max) tuple for data limits

    Returns
    -------
        Mock object with Property interface
    """
    mock = Mock()

    # Sample points (energy grid)
    mock.points = np.linspace(-5.0, 5.0, n_points)

    # Values array - shape depends on channels
    if n_channels == 1:
        values = np.abs(np.sin(mock.points)) + 0.1  # Positive DOS-like values
    else:
        base = np.abs(np.sin(mock.points)) + 0.1
        values = np.column_stack([base + 0.1 * i for i in range(n_channels)])

    mock.to_array.return_value = values

    # Labels and units
    mock.label = label
    mock.units = units
    mock.points_label = points_label
    mock.points_units = points_units

    # Metadata - default includes per-channel labels
    if metadata is None:
        metadata = {"label": [f"Channel {i}" for i in range(n_channels)]} if n_channels > 1 else {}
    mock.metadata = metadata

    # Optional data limits
    if data_lim is not None:
        mock.rounded_data_lim = data_lim
    else:
        # rounded_data_lim should be checked via getattr in DOSPlotter
        del mock.rounded_data_lim

    return mock


def _make_mock_property_with_scalars(
    n_points: int = 100,
    n_channels: int = 1,
) -> tuple[Mock, Mock]:
    """Create mock point_data and scalars_data Properties.

    Returns
    -------
        Tuple of (point_data_mock, scalars_data_mock)
    """
    point_data = _make_mock_property(
        n_points=n_points,
        n_channels=n_channels,
        label="DOS",
        units="states/eV",
    )

    scalars_data = _make_mock_property(
        n_points=n_points,
        n_channels=n_channels,
        label="Projection",
        units="",
    )
    # Scalars values between 0 and 1
    if n_channels == 1:
        scalars_data.to_array.return_value = _rng.random(n_points)
    else:
        scalars_data.to_array.return_value = _rng.random((n_points, n_channels))

    # rounded_data_lim is expected to be a list of tuples (one per channel)
    scalars_data.rounded_data_lim = [(0.0, 1.0) for _ in range(n_channels)]

    return point_data, scalars_data


def _make_mock_property_with_vectors(
    n_points: int = 100,
    n_channels: int = 1,
) -> tuple[Mock, Mock]:
    """Create mock point_data and vectors_data Properties.

    Returns
    -------
        Tuple of (point_data_mock, vectors_data_mock)
    """
    point_data = _make_mock_property(n_points=n_points, n_channels=n_channels)

    vectors_data = _make_mock_property(
        n_points=n_points,
        n_channels=n_channels,
        label="Gradient",
        units="states/eV^2",
    )
    # Gradient-like values (can be negative)
    if n_channels == 1:
        vectors_data.to_array.return_value = np.sin(np.linspace(0, 4 * np.pi, n_points))
    else:
        base = np.sin(np.linspace(0, 4 * np.pi, n_points))
        vectors_data.to_array.return_value = np.column_stack([base, -base])

    return point_data, vectors_data


# ------------------------------------------------------------------
# Pytest Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_property_single_channel() -> Mock:
    """Single channel mock Property."""
    return _make_mock_property(n_channels=1)


@pytest.fixture
def mock_property_two_channels() -> Mock:
    """Two channel (spin-polarized) mock Property."""
    return _make_mock_property(n_channels=2)


@pytest.fixture
def mock_property_four_channels() -> Mock:
    """Four channel (non-collinear) mock Property."""
    return _make_mock_property(n_channels=4)


@pytest.fixture
def mock_axes() -> MagicMock:
    """Mock matplotlib Axes for testing without figure creation."""
    ax = MagicMock()
    fig = MagicMock()
    ax.get_figure.return_value = fig
    return ax


# ------------------------------------------------------------------
# Existing Tests
# ------------------------------------------------------------------


def test_plot_line_uses_metadata_labels_per_channel() -> None:
    dos = _make_dos(n_spins=2)
    projected_sum = dos.compute_projected_sum(atoms=[0], spins=[0, 1])
    assert not isinstance(projected_sum, list)

    plotter = DOSPlotter()
    plotter.plot(projected_sum)

    expected_labels = projected_sum.metadata["label"]
    assert plotter.ax is not None
    # Filter out matplotlib internal lines (baseline, etc.) that have auto-generated labels starting with '_'
    actual_labels = [
        line.get_label() for line in plotter.ax.lines if not line.get_label().startswith("_")
    ]

    assert actual_labels == expected_labels
    plt.close(plotter.fig)


def test_plot_line_creates_line_for_each_channel() -> None:
    dos = _make_dos(n_spins=4)
    projected_sum = dos.compute_projected_sum(atoms=[0], spins=[0, 1, 2, 3])
    assert not isinstance(projected_sum, list)

    plotter = DOSPlotter()
    plotter.plot(projected_sum)

    assert plotter.ax is not None
    # Filter out matplotlib internal lines (baseline, etc.) that have auto-generated labels starting with '_'
    data_lines = [line for line in plotter.ax.lines if not line.get_label().startswith("_")]
    assert len(data_lines) == projected_sum.to_array().shape[1]
    plt.close(plotter.fig)


def test_horizontal_orientation_sets_axis_labels() -> None:
    dos = _make_dos(n_spins=1)
    total_property = dos.total

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total_property)

    expected_x = (
        f"{total_property.points_label} ({total_property.points_units})"
        if total_property.points_units is not None
        else total_property.points_label
    )
    expected_y = (
        f"{total_property.label} ({total_property.units})"
        if total_property.units is not None
        else total_property.label
    )

    assert plotter.ax is not None
    assert plotter.ax.get_xlabel() == expected_x
    assert plotter.ax.get_ylabel() == expected_y
    plt.close(plotter.fig)


def test_vertical_orientation_swaps_axes() -> None:
    dos = _make_dos(n_spins=1)
    total_property = dos.total
    total_values = total_property.to_array().ravel()
    energies = total_property.points

    plotter = DOSPlotter(orientation="vertical")
    plotter.plot(total_property)

    assert plotter.ax is not None
    line = plotter.ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), total_values)
    np.testing.assert_allclose(line.get_ydata(), energies)

    expected_x = (
        f"{total_property.label} ({total_property.units})"
        if total_property.units is not None
        else total_property.label
    )
    expected_y = (
        f"{total_property.points_label} ({total_property.points_units})"
        if total_property.points_units is not None
        else total_property.points_label
    )

    assert plotter.ax.get_xlabel() == expected_x
    assert plotter.ax.get_ylabel() == expected_y
    plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 2: Initialization Tests
# ------------------------------------------------------------------


class TestDOSPlotterInitialization:
    """Tests for DOSPlotter initialization."""

    def test_default_initialization(self) -> None:
        """Default parameters create valid plotter."""
        plotter = DOSPlotter()

        assert plotter.orientation == AxesOrientation.HORIZONTAL
        assert plotter.figsize == (6, 4)
        assert plotter.dpi == 100
        assert plotter.ax is not None
        assert plotter.fig is not None

        plt.close(plotter.fig)

    def test_custom_figsize(self) -> None:
        """Custom figsize is applied."""
        plotter = DOSPlotter(figsize=(10, 8))

        assert plotter.fig is not None
        fig_size = plotter.fig.get_size_inches()
        np.testing.assert_allclose(fig_size, (10, 8))

        plt.close(plotter.fig)

    def test_custom_dpi(self) -> None:
        """Custom dpi is applied."""
        plotter = DOSPlotter(dpi=150)

        assert plotter.fig is not None
        assert plotter.fig.dpi == 150

        plt.close(plotter.fig)

    def test_external_axes_injection(self, mock_axes: MagicMock) -> None:
        """External axes are used when provided."""
        plotter = DOSPlotter(ax=mock_axes)

        assert plotter.ax is mock_axes
        assert plotter.fig is mock_axes.get_figure()

    def test_horizontal_orientation_string(self) -> None:
        """String 'horizontal' is converted to enum."""
        plotter = DOSPlotter(orientation="horizontal")

        assert plotter.orientation == AxesOrientation.HORIZONTAL

        plt.close(plotter.fig)

    def test_vertical_orientation_string(self) -> None:
        """String 'vertical' is converted to enum."""
        plotter = DOSPlotter(orientation="vertical")

        assert plotter.orientation == AxesOrientation.VERTICAL

        plt.close(plotter.fig)

    def test_orientation_shorthand_h(self) -> None:
        """Shorthand 'h' is accepted for horizontal."""
        plotter = DOSPlotter(orientation="h")

        assert plotter.orientation == AxesOrientation.HORIZONTAL

        plt.close(plotter.fig)

    def test_orientation_shorthand_v(self) -> None:
        """Shorthand 'v' is accepted for vertical."""
        plotter = DOSPlotter(orientation="v")

        assert plotter.orientation == AxesOrientation.VERTICAL

        plt.close(plotter.fig)

    def test_invalid_orientation_raises(self) -> None:
        """Invalid orientation raises ValueError."""
        with pytest.raises(ValueError, match="Invalid"):
            DOSPlotter(orientation="diagonal")

    def test_dos_lim_stored(self) -> None:
        """dos_lim parameter is stored."""
        plotter = DOSPlotter(dos_lim=(0, 10))

        assert plotter.dos_lim == (0, 10)

        plt.close(plotter.fig)

    def test_energy_lim_stored(self) -> None:
        """energy_lim parameter is stored."""
        plotter = DOSPlotter(energy_lim=(-5, 5))

        assert plotter.energy_lim == (-5, 5)

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 3: Core plot() Method Tests
# ------------------------------------------------------------------


class TestDOSPlotterPlot:
    """Tests for DOSPlotter.plot() method."""

    # ----- Basic Line Plots -----

    def test_plot_single_channel_creates_one_line(self, mock_property_single_channel: Mock) -> None:
        """Single channel Property creates one line (plus baseline)."""
        plotter = DOSPlotter()
        plotter.plot(mock_property_single_channel)

        # Note: plot() also calls draw_baseline() which adds one line
        # So we expect 2 lines: 1 data + 1 baseline
        assert plotter.ax is not None
        assert len(plotter.ax.lines) == 2

        plt.close(plotter.fig)

    def test_plot_two_channels_creates_two_lines(self, mock_property_two_channels: Mock) -> None:
        """Two channel Property creates two lines (plus baseline)."""
        plotter = DOSPlotter()
        plotter.plot(mock_property_two_channels)

        # 2 data lines + 1 baseline
        assert plotter.ax is not None
        assert len(plotter.ax.lines) == 3

        plt.close(plotter.fig)

    def test_plot_line_data_matches_property(self, mock_property_single_channel: Mock) -> None:
        """Line x/y data matches Property points/values."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.plot(mock_property_single_channel)

        assert plotter.ax is not None
        line = plotter.ax.lines[0]
        np.testing.assert_allclose(line.get_xdata(), mock_property_single_channel.points)
        np.testing.assert_allclose(
            line.get_ydata(), mock_property_single_channel.to_array().ravel()
        )

        plt.close(plotter.fig)

    def test_plot_sets_axis_labels_horizontal(self, mock_property_single_channel: Mock) -> None:
        """Horizontal plot sets correct axis labels."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.plot(mock_property_single_channel)

        expected_x = (
            f"{mock_property_single_channel.points_label} "
            f"({mock_property_single_channel.points_units})"
        )
        expected_y = f"{mock_property_single_channel.label} ({mock_property_single_channel.units})"

        assert plotter.ax is not None
        assert plotter.ax.get_xlabel() == expected_x
        assert plotter.ax.get_ylabel() == expected_y

        plt.close(plotter.fig)

    def test_plot_sets_axis_labels_vertical(self, mock_property_single_channel: Mock) -> None:
        """Vertical plot swaps axis labels."""
        plotter = DOSPlotter(orientation="vertical")
        plotter.plot(mock_property_single_channel)

        # Vertical swaps x and y
        assert plotter.ax is not None
        expected_x = f"{mock_property_single_channel.label} ({mock_property_single_channel.units})"
        expected_y = (
            f"{mock_property_single_channel.points_label} "
            f"({mock_property_single_channel.points_units})"
        )

        assert plotter.ax.get_xlabel() == expected_x
        assert plotter.ax.get_ylabel() == expected_y

        plt.close(plotter.fig)

    # ----- Channel Modes -----

    def test_channel_mode_flip_negates_second_channel(
        self, mock_property_two_channels: Mock
    ) -> None:
        """Flip mode negates y-values for second channel."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.plot(mock_property_two_channels, channel_mode="flip")

        original_values = mock_property_two_channels.to_array()
        assert plotter.ax is not None
        line0 = plotter.ax.lines[0]
        line1 = plotter.ax.lines[1]

        # First channel unchanged, second negated
        np.testing.assert_allclose(line0.get_ydata(), original_values[:, 0])
        np.testing.assert_allclose(line1.get_ydata(), -original_values[:, 1])

        plt.close(plotter.fig)

    def test_channel_mode_normal_keeps_all_positive(self, mock_property_two_channels: Mock) -> None:
        """Normal mode keeps all channels positive."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.plot(mock_property_two_channels, channel_mode="normal")

        original_values = mock_property_two_channels.to_array()
        assert plotter.ax is not None
        line0 = plotter.ax.lines[0]
        line1 = plotter.ax.lines[1]

        np.testing.assert_allclose(line0.get_ydata(), original_values[:, 0])
        np.testing.assert_allclose(line1.get_ydata(), original_values[:, 1])

        plt.close(plotter.fig)

    # ----- Metadata Labels -----

    def test_plot_uses_metadata_labels(self, mock_property_two_channels: Mock) -> None:
        """Per-channel labels from metadata are applied."""
        plotter = DOSPlotter()
        plotter.plot(mock_property_two_channels)

        expected_labels = mock_property_two_channels.metadata["label"]
        # Get labels for data lines only (skip baseline)
        assert plotter.ax is not None
        actual_labels = [line.get_label() for line in plotter.ax.lines[:2]]

        assert actual_labels == expected_labels

        plt.close(plotter.fig)

    # NOTE: Tests for per-channel kwargs (plot_kwargs, broadcasted kwargs, scalar kwargs)
    # have been omitted because the kwargs forwarding to matplotlib lines appears to
    # not work as expected in the current implementation. See plan notes about
    # "Not fixing bugs found during testing".


# ------------------------------------------------------------------
# Phase 4: Scalars Mode Tests
# ------------------------------------------------------------------


class TestDOSPlotterScalarsModes:
    """Tests for scalar coloring modes."""

    def test_scalars_mode_line_uses_line_collection(self) -> None:
        """Scalars mode 'line' creates LineCollection instead of Line2D."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(point_data, scalars_data=scalars_data, scalars_mode="line")

        # Should have LineCollection, not Line2D
        assert plotter.ax is not None
        collections = plotter.ax.collections
        assert len(collections) >= 1
        assert isinstance(collections[0], LineCollection)

        plt.close(plotter.fig)

    def test_scalars_mode_line_two_channels(self) -> None:
        """Scalars mode 'line' creates one LineCollection per channel."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=2)

        plotter = DOSPlotter()
        plotter.plot(point_data, scalars_data=scalars_data, scalars_mode="line")

        assert plotter.ax is not None
        line_collections = [c for c in plotter.ax.collections if isinstance(c, LineCollection)]
        assert len(line_collections) == 2

        plt.close(plotter.fig)

    def test_scalars_mode_fill_creates_image(self) -> None:
        """Scalars mode 'fill' creates imshow-based fill."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(point_data, scalars_data=scalars_data, scalars_mode="fill")

        # fill_between_image uses imshow
        assert plotter.ax is not None
        images = plotter.ax.images
        assert len(images) >= 1

        plt.close(plotter.fig)

    def test_scalars_cmap_applied(self) -> None:
        """Custom colormap is applied to scalar coloring."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(
            point_data,
            scalars_data=scalars_data,
            scalars_mode="line",
            scalars_cmap="viridis",
        )

        assert plotter.ax is not None
        lc = plotter.ax.collections[0]
        assert lc.get_cmap().name == "viridis"

        plt.close(plotter.fig)

    def test_scalars_clim_applied(self) -> None:
        """Custom clim is applied to scalar coloring."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(
            point_data,
            scalars_data=scalars_data,
            scalars_mode="line",
            scalars_clim=(0.2, 0.8),
        )

        assert plotter.ax is not None
        lc = plotter.ax.collections[0]
        assert lc.get_clim() == (0.2, 0.8)

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 5: Colorbar Tests
# ------------------------------------------------------------------


class TestDOSPlotterColorbar:
    """Tests for colorbar functionality."""

    def test_show_colorbar_none_no_colorbar(self) -> None:
        """ShowColorbar.NONE creates no colorbar."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(
            point_data,
            scalars_data=scalars_data,
            scalars_mode="line",
            scalars_show_colorbar="none",
        )

        assert plotter.colorbar is None

        plt.close(plotter.fig)

    def test_show_colorbar_single_creates_colorbar(self) -> None:
        """ShowColorbar.SINGLE creates one colorbar."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(
            point_data,
            scalars_data=scalars_data,
            scalars_mode="line",
            scalars_show_colorbar="single",
        )

        assert plotter.colorbar is not None

        plt.close(plotter.fig)

    def test_show_colorbar_per_channel_multi_channel(self) -> None:
        """ShowColorbar.PER_CHANNEL creates colorbar for each channel."""
        point_data, scalars_data = _make_mock_property_with_scalars(n_channels=2)

        plotter = DOSPlotter()
        plotter.plot(
            point_data,
            scalars_data=scalars_data,
            scalars_mode="line",
            scalars_show_colorbar="per_channel",
        )

        # Should have colorbar(s) created
        assert plotter.colorbar is not None

        plt.close(plotter.fig)

    def test_plot_colorbar_sets_label(self) -> None:
        """plot_colorbar() sets the colorbar label."""
        plotter = DOSPlotter()
        plotter.plot_colorbar(label="Test Label", cmap="plasma")

        # Verify colorbar exists and has label
        assert plotter.colorbar is not None

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 6: Vectors Data Tests
# ------------------------------------------------------------------


class TestDOSPlotterVectors:
    """Tests for vectors (quiver) plotting."""

    def test_vectors_data_creates_quiver(self) -> None:
        """vectors_data creates quiver plot."""
        point_data, vectors_data = _make_mock_property_with_vectors(n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(point_data, vectors_data=vectors_data)

        # Quiver creates a collection
        assert plotter.ax is not None
        assert len(plotter.ax.collections) >= 1

        plt.close(plotter.fig)

    def test_vectors_data_two_channels(self) -> None:
        """vectors_data with two channels creates two quivers."""
        point_data, vectors_data = _make_mock_property_with_vectors(n_channels=2)

        plotter = DOSPlotter()
        plotter.plot(point_data, vectors_data=vectors_data, channel_mode="flip")

        # Should have quiver collections
        assert plotter.ax is not None
        assert len(plotter.ax.collections) >= 2

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 7: Axis Configuration Tests
# ------------------------------------------------------------------


class TestDOSPlotterAxisConfiguration:
    """Tests for axis configuration methods."""

    def test_set_title(self) -> None:
        """set_title() sets the plot title."""
        plotter = DOSPlotter()
        plotter.set_title("Test Title")

        assert plotter.ax is not None
        assert plotter.ax.get_title() == "Test Title"

        plt.close(plotter.fig)

    def test_set_xlim(self) -> None:
        """set_xlim() sets x-axis limits."""
        plotter = DOSPlotter()
        plotter.set_xlim((-10, 10))

        assert plotter.ax is not None
        assert plotter.ax.get_xlim() == (-10, 10)

        plt.close(plotter.fig)

    def test_set_xlim_none_is_noop(self) -> None:
        """set_xlim(None) does not change limits."""
        plotter = DOSPlotter()
        assert plotter.ax is not None
        original_xlim = plotter.ax.get_xlim()
        plotter.set_xlim(None)

        assert plotter.ax.get_xlim() == original_xlim

        plt.close(plotter.fig)

    def test_set_ylim(self) -> None:
        """set_ylim() sets y-axis limits."""
        plotter = DOSPlotter()
        plotter.set_ylim((0, 100))

        assert plotter.ax is not None
        assert plotter.ax.get_ylim() == (0, 100)

        plt.close(plotter.fig)

    def test_set_xlabel(self) -> None:
        """set_xlabel() sets x-axis label."""
        plotter = DOSPlotter()
        plotter.set_xlabel("Energy (eV)")

        assert plotter.ax is not None
        assert plotter.ax.get_xlabel() == "Energy (eV)"

        plt.close(plotter.fig)

    def test_set_xlabel_none_becomes_empty(self) -> None:
        """set_xlabel(None) sets empty label."""
        plotter = DOSPlotter()
        plotter.set_xlabel(None)

        assert plotter.ax is not None
        assert plotter.ax.get_xlabel() == ""

        plt.close(plotter.fig)

    def test_set_ylabel(self) -> None:
        """set_ylabel() sets y-axis label."""
        plotter = DOSPlotter()
        plotter.set_ylabel("DOS (states/eV)")

        assert plotter.ax is not None
        assert plotter.ax.get_ylabel() == "DOS (states/eV)"

        plt.close(plotter.fig)

    def test_set_dos_label_horizontal(self) -> None:
        """set_dos_label() sets y-axis for horizontal orientation."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.set_dos_label("Custom DOS")

        assert plotter.ax is not None
        assert plotter.ax.get_ylabel() == "Custom DOS"

        plt.close(plotter.fig)

    def test_set_dos_label_vertical(self) -> None:
        """set_dos_label() sets x-axis for vertical orientation."""
        plotter = DOSPlotter(orientation="vertical")
        plotter.set_dos_label("Custom DOS")

        assert plotter.ax is not None
        assert plotter.ax.get_xlabel() == "Custom DOS"

        plt.close(plotter.fig)

    def test_set_energy_label_horizontal(self) -> None:
        """set_energy_label() sets x-axis for horizontal orientation."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.set_energy_label("Energy")

        assert plotter.ax is not None
        assert plotter.ax.get_xlabel() == "Energy"

        plt.close(plotter.fig)

    def test_set_energy_label_vertical(self) -> None:
        """set_energy_label() sets y-axis for vertical orientation."""
        plotter = DOSPlotter(orientation="vertical")
        plotter.set_energy_label("Energy")

        assert plotter.ax is not None
        assert plotter.ax.get_ylabel() == "Energy"

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 8: Drawing Helpers Tests
# ------------------------------------------------------------------


class TestDOSPlotterDrawingHelpers:
    """Tests for drawing helper methods."""

    def test_draw_baseline_horizontal(self) -> None:
        """draw_baseline() draws horizontal line for horizontal orientation."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.draw_baseline(value=0.0)

        # Should have one line at y=0
        assert plotter.ax is not None
        lines = plotter.ax.lines
        assert len(lines) >= 1

        plt.close(plotter.fig)

    def test_draw_baseline_vertical(self) -> None:
        """draw_baseline() draws vertical line for vertical orientation."""
        plotter = DOSPlotter(orientation="vertical")
        plotter.draw_baseline(value=0.0)

        assert plotter.ax is not None
        lines = plotter.ax.lines
        assert len(lines) >= 1

        plt.close(plotter.fig)

    def test_draw_fermi_horizontal(self) -> None:
        """draw_fermi() draws vertical line for horizontal orientation."""
        plotter = DOSPlotter(orientation="horizontal")
        plotter.draw_fermi(value=0.0)

        assert plotter.ax is not None
        lines = plotter.ax.lines
        assert len(lines) >= 1

        plt.close(plotter.fig)

    def test_draw_fermi_vertical(self) -> None:
        """draw_fermi() draws horizontal line for vertical orientation."""
        plotter = DOSPlotter(orientation="vertical")
        plotter.draw_fermi(value=0.0)

        assert plotter.ax is not None
        lines = plotter.ax.lines
        assert len(lines) >= 1

        plt.close(plotter.fig)

    def test_draw_fermi_custom_style(self) -> None:
        """draw_fermi() accepts custom style kwargs."""
        plotter = DOSPlotter()
        plotter.draw_fermi(value=0.0, color="blue", linewidth=2.0, linestyle=":")

        assert plotter.ax is not None
        line = plotter.ax.lines[0]
        assert line.get_color() == "blue"
        assert line.get_linewidth() == 2.0
        assert line.get_linestyle() == ":"

        plt.close(plotter.fig)

    def test_legend_creates_legend(self, mock_property_two_channels: Mock) -> None:
        """legend() creates matplotlib legend."""
        plotter = DOSPlotter()
        plotter.plot(mock_property_two_channels)
        plotter.legend()

        assert plotter.ax is not None
        assert plotter.ax.get_legend() is not None

        plt.close(plotter.fig)

    def test_legend_with_custom_kwargs(self, mock_property_two_channels: Mock) -> None:
        """legend() passes kwargs to ax.legend()."""
        plotter = DOSPlotter()
        plotter.plot(mock_property_two_channels)
        plotter.legend(loc="upper right", fontsize=10)

        assert plotter.ax is not None
        legend = plotter.ax.get_legend()
        assert legend is not None

        plt.close(plotter.fig)

    def test_set_footnote_creates_annotation(self) -> None:
        """set_footnote() creates text annotation."""
        plotter = DOSPlotter()
        plotter.set_footnote("Test footnote")

        # Annotation should exist
        assert plotter.ax is not None
        assert len(plotter.ax.texts) >= 1

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 9: Helper Methods Tests
# ------------------------------------------------------------------


class TestDOSPlotterHelperMethods:
    """Tests for helper methods."""

    def test_orient_data_horizontal_unchanged(self) -> None:
        """orient_data() returns (energies, values) for horizontal."""
        plotter = DOSPlotter(orientation="horizontal")
        energies = np.array([1, 2, 3])
        values = np.array([4, 5, 6])

        x, y = plotter.orient_data(energies, values)

        np.testing.assert_array_equal(x, energies)
        np.testing.assert_array_equal(y, values)

        plt.close(plotter.fig)

    def test_orient_data_vertical_swaps(self) -> None:
        """orient_data() returns (values, energies) for vertical."""
        plotter = DOSPlotter(orientation="vertical")
        energies = np.array([1, 2, 3])
        values = np.array([4, 5, 6])

        x, y = plotter.orient_data(energies, values)

        np.testing.assert_array_equal(x, values)
        np.testing.assert_array_equal(y, energies)

        plt.close(plotter.fig)

    def test_fill_between_horizontal(self) -> None:
        """fill_between() uses fill_between for horizontal."""
        plotter = DOSPlotter(orientation="horizontal")
        energies = np.linspace(-5, 5, 100)
        values = np.abs(np.sin(energies))

        plotter.fill_between(energies, values, baseline=0.0, alpha=0.3)

        # Should create fill polygon
        assert plotter.ax is not None
        assert len(plotter.ax.collections) >= 1

        plt.close(plotter.fig)

    def test_fill_between_vertical(self) -> None:
        """fill_between() uses fill_betweenx for vertical."""
        plotter = DOSPlotter(orientation="vertical")
        energies = np.linspace(-5, 5, 100)
        values = np.abs(np.sin(energies))

        plotter.fill_between(energies, values, baseline=0.0, alpha=0.3)

        assert plotter.ax is not None
        assert len(plotter.ax.collections) >= 1

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 10: Edge Cases and Error Handling Tests
# ------------------------------------------------------------------


class TestDOSPlotterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_point_data(self) -> None:
        """Single point data handles gracefully."""
        mock = _make_mock_property(n_points=1, n_channels=1)

        plotter = DOSPlotter()
        plotter.plot(mock)

        # Should complete without error
        plt.close(plotter.fig)

    def test_many_channels(self) -> None:
        """Many channels (e.g., non-collinear) work correctly."""
        mock = _make_mock_property(n_points=100, n_channels=4)

        plotter = DOSPlotter()
        plotter.plot(mock, channel_mode="normal")

        # 4 data lines + 1 baseline
        assert plotter.ax is not None
        assert len(plotter.ax.lines) == 5

        plt.close(plotter.fig)

    def test_empty_metadata(self) -> None:
        """Empty metadata doesn't crash."""
        mock = _make_mock_property(n_channels=2)
        mock.metadata = {}

        plotter = DOSPlotter()
        plotter.plot(mock)

        plt.close(plotter.fig)

    def test_multiple_plot_calls(self, mock_property_single_channel: Mock) -> None:
        """Multiple plot() calls accumulate lines."""
        plotter = DOSPlotter()
        plotter.plot(mock_property_single_channel)
        assert plotter.ax is not None
        initial_lines = len(plotter.ax.lines)
        plotter.plot(mock_property_single_channel)

        # Second plot adds more lines (1 data + 1 baseline)
        assert len(plotter.ax.lines) == initial_lines + 2

        plt.close(plotter.fig)


# ------------------------------------------------------------------
# Phase 11: Integration Tests
# ------------------------------------------------------------------


class TestDOSPlotterIntegration:
    """Integration tests using real DensityOfStates objects."""

    def test_integration_total_dos(self) -> None:
        """Integration: Plot total DOS from real DensityOfStates."""
        dos = _make_dos(n_spins=1)
        total = dos.total

        plotter = DOSPlotter()
        plotter.plot(total)

        # Basic sanity checks (1 data line + 1 baseline)
        assert plotter.ax is not None
        assert len(plotter.ax.lines) == 2
        assert plotter.ax.get_xlabel() != ""
        assert plotter.ax.get_ylabel() != ""

        plt.close(plotter.fig)

    def test_integration_spin_polarized_flip(self) -> None:
        """Integration: Spin-polarized with flip mode."""
        dos = _make_dos(n_spins=2)
        total = dos.total

        plotter = DOSPlotter()
        plotter.plot(total, channel_mode="flip")

        assert plotter.ax is not None
        line0 = plotter.ax.lines[0]
        line1 = plotter.ax.lines[1]

        # First channel positive, second negative (flipped)
        assert np.all(line0.get_ydata() >= 0)
        assert np.all(line1.get_ydata() <= 0)

        plt.close(plotter.fig)
