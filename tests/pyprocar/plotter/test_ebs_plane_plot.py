"""Tests for EBSPlanePlotter class."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from pyprocar.core.property_store import Property
from pyprocar.plotter.ebs_plane_plot import (
    EBSPlanePlotter,
    PlaneScalarsMode,
    PlaneSeries,
)


class TestPlaneScalarsMode:
    """Tests for PlaneScalarsMode enum."""

    def test_from_string_pcolormesh(self):
        mode = PlaneScalarsMode.from_string("pcolormesh")
        assert mode is PlaneScalarsMode.PCOLORMESH

    def test_from_string_contour(self):
        mode = PlaneScalarsMode.from_string("contour")
        assert mode is PlaneScalarsMode.CONTOUR

    def test_from_string_contourf(self):
        mode = PlaneScalarsMode.from_string("contourf")
        assert mode is PlaneScalarsMode.CONTOURF

    def test_from_string_case_insensitive(self):
        mode = PlaneScalarsMode.from_string("PCOLORMESH")
        assert mode is PlaneScalarsMode.PCOLORMESH

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid scalars mode"):
            PlaneScalarsMode.from_string("invalid")

    def test_from_string_passthrough(self):
        mode = PlaneScalarsMode.from_string(PlaneScalarsMode.CONTOUR)
        assert mode is PlaneScalarsMode.CONTOUR


class TestPlaneSeries:
    """Tests for PlaneSeries dataclass."""

    def test_creation_scalars_only(self):
        u_grid = np.zeros((5, 5))
        v_grid = np.zeros((5, 5))

        series = PlaneSeries(
            u_grid=u_grid,
            v_grid=v_grid,
            scalars=np.ones((5, 5)),
            scalars_label="test",
            scalars_unit="eV",
            scalars_lim=(0.0, 1.0),
            vectors_u=None,
            vectors_v=None,
            vectors_magnitude=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
        )

        assert series.scalars_label == "test"
        assert series.scalars_unit == "eV"
        assert series.scalars_lim == (0.0, 1.0)
        assert series.vectors_u is None
        assert series.vectors_v is None

    def test_creation_vectors_only(self):
        u_grid = np.zeros((5, 5))
        v_grid = np.zeros((5, 5))

        series = PlaneSeries(
            u_grid=u_grid,
            v_grid=v_grid,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors_u=np.ones((5, 5)),
            vectors_v=np.ones((5, 5)),
            vectors_magnitude=np.ones((5, 5)),
            vectors_label="velocity",
            vectors_unit="m/s",
            vectors_lim=(0.0, 2.0),
        )

        assert series.vectors_label == "velocity"
        assert series.vectors_unit == "m/s"
        assert series.vectors_lim == (0.0, 2.0)
        assert series.scalars is None

    def test_creation_with_additional_kwargs(self):
        u_grid = np.zeros((5, 5))
        v_grid = np.zeros((5, 5))

        series = PlaneSeries(
            u_grid=u_grid,
            v_grid=v_grid,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors_u=None,
            vectors_v=None,
            vectors_magnitude=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            additional_kwargs={"custom_key": "custom_value"},
        )

        assert series.additional_kwargs == {"custom_key": "custom_value"}

    def test_default_additional_kwargs(self):
        u_grid = np.zeros((5, 5))
        v_grid = np.zeros((5, 5))

        series = PlaneSeries(
            u_grid=u_grid,
            v_grid=v_grid,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors_u=None,
            vectors_v=None,
            vectors_magnitude=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
        )

        assert series.additional_kwargs == {}


class MockEBSMesh:
    """Mock ElectronicBandStructureMesh for testing."""

    def __init__(self, n_points: int = 100):
        # Create mock slice points on a plane
        u = np.linspace(-0.5, 0.5, 10)
        v = np.linspace(-0.5, 0.5, 10)
        uu, vv = np.meshgrid(u, v)
        self.slice_points = np.column_stack([uu.ravel(), vv.ravel(), np.zeros(100)])
        self.n_points = 100

    def slice(self, normal=None, origin=None, scalars=None, vectors=None):
        """Return mock slice with points."""

        class MockSlice:
            def __init__(self, points, scalars_val=None, vectors_val=None):
                self.points = points
                self._scalars = scalars_val
                self._vectors = vectors_val

            @property
            def active_scalars(self):
                return self._scalars

            @property
            def active_vectors(self):
                return self._vectors

        s_val = np.random.rand(100) if scalars is not None else None
        v_val = np.random.rand(100, 3) if vectors is not None else None
        return MockSlice(self.slice_points, s_val, v_val)


class TestEBSPlanePlotterInit:
    """Tests for EBSPlanePlotter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)

        assert plotter.ebs_mesh is mock_mesh
        assert plotter.normal == (0, 0, 1)
        assert plotter.origin == (0, 0, 0)
        assert plotter.grid_interpolation == (20, 20)
        assert plotter.fig is not None
        assert plotter.ax is not None
        assert isinstance(plotter.values_dict, dict)
        plt.close(plotter.fig)

    def test_init_with_custom_normal(self):
        """Test initialization with custom normal."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh, normal=(1, 0, 0))

        assert plotter.normal == (1, 0, 0)
        plt.close(plotter.fig)

    def test_init_with_existing_ax(self):
        """Test initialization with provided axes."""
        mock_mesh = MockEBSMesh()
        fig, ax = plt.subplots()
        plotter = EBSPlanePlotter(mock_mesh, ax=ax)

        assert plotter.ax is ax
        assert plotter.fig is fig
        plt.close(fig)


class TestEBSPlanePlotterToSeries:
    """Tests for _to_series method."""

    @pytest.fixture
    def plotter(self):
        """Create a plotter with mock mesh."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)
        yield plotter
        plt.close(plotter.fig)

    def test_to_series_with_property_scalars(self, plotter):
        """Test _to_series with Property input for scalars."""
        scalars_prop = Property(
            name="bands",
            value=np.random.rand(100),
            units="eV",
            label="Energy",
        )

        series = plotter._to_series(scalars_data=scalars_prop)

        assert series.scalars_label == "Energy"
        assert series.scalars_unit == "eV"
        assert series.scalars is not None
        assert series.scalars.shape == plotter.u_grid.shape

    def test_to_series_with_tuple_scalars(self, plotter):
        """Test _to_series with legacy tuple input."""
        scalars_tuple = ("bands", np.random.rand(100))

        series = plotter._to_series(scalars_data=scalars_tuple)

        assert series.scalars_label == "bands"
        assert series.scalars_unit is None
        assert series.scalars is not None

    def test_to_series_with_vectors(self, plotter):
        """Test _to_series with vector data."""
        vectors_prop = Property(
            name="velocity",
            value=np.random.rand(100, 3),
            units="m/s",
            label="Velocity",
        )

        series = plotter._to_series(vectors_data=vectors_prop)

        assert series.vectors_label == "Velocity"
        assert series.vectors_unit == "m/s"
        assert series.vectors_u is not None
        assert series.vectors_v is not None
        assert series.vectors_magnitude is not None

    def test_to_series_with_both(self, plotter):
        """Test _to_series with both scalars and vectors."""
        scalars_tuple = ("energy", np.random.rand(100))
        vectors_tuple = ("velocity", np.random.rand(100, 3))

        series = plotter._to_series(scalars_data=scalars_tuple, vectors_data=vectors_tuple)

        assert series.scalars is not None
        assert series.vectors_u is not None

    def test_to_series_with_none(self, plotter):
        """Test _to_series with no data."""
        series = plotter._to_series()

        assert series.scalars is None
        assert series.vectors_u is None


class TestEBSPlanePlotterRenderMethods:
    """Tests for internal render methods."""

    @pytest.fixture
    def plotter(self):
        """Create a plotter with mock mesh."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)
        yield plotter
        plt.close(plotter.fig)

    def test_add_pcolormesh(self, plotter):
        """Test _add_pcolormesh creates a QuadMesh."""
        series = PlaneSeries(
            u_grid=plotter.u_grid,
            v_grid=plotter.v_grid,
            scalars=np.random.rand(*plotter.u_grid.shape),
            scalars_label="test",
            scalars_unit="eV",
            scalars_lim=(0.0, 1.0),
            vectors_u=None,
            vectors_v=None,
            vectors_magnitude=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
        )

        result = plotter._add_pcolormesh(series, "plasma", (0.0, 1.0), 0.7)

        # Check that it returns a QuadMesh
        assert result is not None

    def test_add_contour(self, plotter):
        """Test _add_contour creates a contour plot."""
        series = PlaneSeries(
            u_grid=plotter.u_grid,
            v_grid=plotter.v_grid,
            scalars=np.random.rand(*plotter.u_grid.shape),
            scalars_label="test",
            scalars_unit="eV",
            scalars_lim=(0.0, 1.0),
            vectors_u=None,
            vectors_v=None,
            vectors_magnitude=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
        )

        result = plotter._add_contour(series, "plasma", (0.0, 1.0), 10)

        assert result is not None

    def test_add_contourf(self, plotter):
        """Test _add_contourf creates a filled contour plot."""
        series = PlaneSeries(
            u_grid=plotter.u_grid,
            v_grid=plotter.v_grid,
            scalars=np.random.rand(*plotter.u_grid.shape),
            scalars_label="test",
            scalars_unit="eV",
            scalars_lim=(0.0, 1.0),
            vectors_u=None,
            vectors_v=None,
            vectors_magnitude=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
        )

        result = plotter._add_contourf(series, "plasma", (0.0, 1.0), 0.7, 10)

        assert result is not None

    def test_add_quiver(self, plotter):
        """Test _add_quiver creates a quiver plot."""
        series = PlaneSeries(
            u_grid=plotter.u_grid,
            v_grid=plotter.v_grid,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors_u=np.random.rand(*plotter.u_grid.shape),
            vectors_v=np.random.rand(*plotter.u_grid.shape),
            vectors_magnitude=np.random.rand(*plotter.u_grid.shape) + 0.1,
            vectors_label="velocity",
            vectors_unit="m/s",
            vectors_lim=(0.0, 1.0),
        )

        result = plotter._add_quiver(series, "plasma", (0.0, 1.0), 1, None, 1.0)

        assert result is not None


class TestEBSPlanePlotterPlot:
    """Integration tests for plot() method."""

    @pytest.fixture
    def plotter(self):
        """Create a plotter with mock mesh."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)
        yield plotter
        plt.close(plotter.fig)

    def test_plot_scalars_pcolormesh(self, plotter):
        """Test pcolormesh rendering mode."""
        scalars_tuple = ("bands", np.random.rand(100))

        artists = plotter.plot(scalars_data=scalars_tuple, scalars_mode="pcolormesh")

        assert "scalars" in artists
        assert plotter.scalar_name == "bands"
        assert plotter.scalar_plot is not None

    def test_plot_scalars_contour(self, plotter):
        """Test contour rendering mode."""
        scalars_tuple = ("bands", np.random.rand(100))

        artists = plotter.plot(scalars_data=scalars_tuple, scalars_mode="contour")

        assert "scalars" in artists

    def test_plot_scalars_contourf(self, plotter):
        """Test contourf rendering mode."""
        scalars_tuple = ("bands", np.random.rand(100))

        artists = plotter.plot(
            scalars_data=scalars_tuple, scalars_mode="contourf", scalars_show_colorbar="single"
        )

        assert "scalars" in artists

    def test_plot_vectors(self, plotter):
        """Test vector plotting."""
        vectors_tuple = ("velocity", np.random.rand(100, 3))

        artists = plotter.plot(vectors_data=vectors_tuple)

        assert "vectors" in artists
        assert plotter.vector_name == "velocity"
        assert plotter.vector_plot is not None

    def test_plot_with_property(self, plotter):
        """Test plot with Property input."""
        scalars_prop = Property(
            name="bands",
            value=np.random.rand(100),
            units="eV",
            label="Energy",
        )

        artists = plotter.plot(scalars_data=scalars_prop)

        assert "scalars" in artists
        assert plotter.scalar_name == "Energy"

    def test_plot_returns_dict(self, plotter):
        """Test that plot returns a dict of artists."""
        scalars_tuple = ("bands", np.random.rand(100))

        result = plotter.plot(scalars_data=scalars_tuple)

        assert isinstance(result, dict)


class TestEBSPlanePlotterExport:
    """Tests for export functionality."""

    @pytest.fixture
    def plotter_with_data(self, tmp_path):
        """Create plotter with plotted data."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)
        scalars_tuple = ("bands", np.random.rand(100))
        plotter.plot(scalars_data=scalars_tuple)
        yield plotter, tmp_path
        plt.close(plotter.fig)

    def test_export_data_csv(self, plotter_with_data):
        """Test CSV export."""
        plotter, tmp_path = plotter_with_data
        filepath = tmp_path / "output.csv"

        plotter.export_data(str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "u_grid" in content
        assert "v_grid" in content

    def test_export_data_json(self, plotter_with_data):
        """Test JSON export."""
        import json

        plotter, tmp_path = plotter_with_data
        filepath = tmp_path / "output.json"

        plotter.export_data(str(filepath))

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "u_grid" in data
        assert "v_grid" in data

    def test_export_data_txt(self, plotter_with_data):
        """Test TXT export."""
        plotter, tmp_path = plotter_with_data
        filepath = tmp_path / "output.txt"

        plotter.export_data(str(filepath))

        assert filepath.exists()

    def test_export_data_dat(self, plotter_with_data):
        """Test DAT export."""
        plotter, tmp_path = plotter_with_data
        filepath = tmp_path / "output.dat"

        plotter.export_data(str(filepath))

        assert filepath.exists()

    def test_export_data_invalid_type(self, plotter_with_data):
        """Test export with invalid file type."""
        plotter, tmp_path = plotter_with_data
        filepath = tmp_path / "output.xyz"

        with pytest.raises(ValueError, match="File type must be one of"):
            plotter.export_data(str(filepath))

    def test_export_data_no_values(self):
        """Test export raises error when no values recorded."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)

        with pytest.raises(ValueError, match="No values recorded"):
            plotter.export_data("output.csv")
        plt.close(plotter.fig)


class TestEBSPlanePlotterAxisMethods:
    """Tests for axis configuration methods."""

    @pytest.fixture
    def plotter(self):
        """Create a plotter with mock mesh."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)
        yield plotter
        plt.close(plotter.fig)

    def test_set_xlim_with_values(self, plotter):
        """Test set_xlim with explicit values."""
        plotter.set_xlim((-1.0, 1.0))
        xlim = plotter.ax.get_xlim()
        assert xlim == (-1.0, 1.0)

    def test_set_xlim_auto(self, plotter):
        """Test set_xlim with auto values."""
        plotter.set_xlim(None)
        xlim = plotter.ax.get_xlim()
        assert xlim is not None

    def test_set_ylim_with_values(self, plotter):
        """Test set_ylim with explicit values."""
        plotter.set_ylim((-1.0, 1.0))
        ylim = plotter.ax.get_ylim()
        assert ylim == (-1.0, 1.0)

    def test_set_xlabel(self, plotter):
        """Test set_xlabel."""
        plotter.set_xlabel("Custom X Label")
        assert plotter.ax.get_xlabel() == "Custom X Label"

    def test_set_ylabel(self, plotter):
        """Test set_ylabel."""
        plotter.set_ylabel("Custom Y Label")
        assert plotter.ax.get_ylabel() == "Custom Y Label"

    def test_set_aspect(self, plotter):
        """Test set_aspect."""
        plotter.set_aspect("equal")
        # Just ensure it doesn't raise

    def test_draw_origin(self, plotter):
        """Test draw_origin adds a marker."""
        plotter.draw_origin()
        # Check that a line was added
        assert len(plotter.ax.lines) > 0

    def test_grid(self, plotter):
        """Test grid configuration."""
        plotter.grid(True)
        # Just ensure it doesn't raise


class TestEBSPlanePlotterLegacyAPI:
    """Tests for legacy API methods."""

    @pytest.fixture
    def plotter(self):
        """Create a plotter with mock mesh."""
        mock_mesh = MockEBSMesh()
        plotter = EBSPlanePlotter(mock_mesh)
        yield plotter
        plt.close(plotter.fig)

    def test_plot_scalars_with_tuple(self, plotter):
        """Test legacy plot_scalars with tuple."""
        scalars_tuple = ("bands", np.random.rand(100))

        plotter.plot_scalars(scalars=scalars_tuple)

        assert plotter.scalar_name == "bands"
        assert plotter.scalar_plot is not None

    def test_plot_scalars_with_property(self, plotter):
        """Test legacy plot_scalars with Property."""
        scalars_prop = Property(
            name="bands",
            value=np.random.rand(100),
            units="eV",
            label="Energy",
        )

        plotter.plot_scalars(scalars=scalars_prop)

        assert plotter.scalar_name == "Energy"

    def test_plot_scalars_with_grid_points(self, plotter):
        """Test legacy plot_scalars with pre-computed grid points."""
        grid_points = np.random.rand(plotter.n_points)

        plotter.plot_scalars(grid_points=grid_points, name="test_field")

        assert plotter.scalar_name == "test_field"

    def test_plot_vectors_quiver_with_tuple(self, plotter):
        """Test legacy plot_vectors_quiver with tuple."""
        vectors_tuple = ("velocity", np.random.rand(100, 3))

        plotter.plot_vectors_quiver(vectors=vectors_tuple)

        assert plotter.vector_name == "velocity"
        assert plotter.vector_plot is not None

    def test_plot_vectors_quiver_with_property(self, plotter):
        """Test legacy plot_vectors_quiver with Property."""
        vectors_prop = Property(
            name="velocity",
            value=np.random.rand(100, 3),
            units="m/s",
            label="Velocity Field",
        )

        plotter.plot_vectors_quiver(vectors=vectors_prop)

        assert plotter.vector_name == "Velocity Field"
