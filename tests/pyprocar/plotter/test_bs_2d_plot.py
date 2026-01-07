"""Tests for BS2DPlotter Property-based API."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv


def _load_bs_2d_plot_module():
    """Load bs_2d_plot module directly to avoid pyprocar/__init__.py import issues."""
    bs_2d_plot_path = Path(__file__).parents[3] / "pyprocar" / "plotter" / "bs_2d_plot.py"
    spec = importlib.util.spec_from_file_location("bs_2d_plot", bs_2d_plot_path)
    bs_2d_plot = importlib.util.module_from_spec(spec)
    sys.modules["bs_2d_plot"] = bs_2d_plot
    spec.loader.exec_module(bs_2d_plot)
    return bs_2d_plot


# Load module once
_bs_2d_plot = _load_bs_2d_plot_module()
BS2DPlotter = _bs_2d_plot.BS2DPlotter
BS2DSeries = _bs_2d_plot.BS2DSeries


class TestBS2DSeries:
    """Tests for BS2DSeries dataclass."""

    def test_bs2d_series_creation_minimal(self):
        """Test BS2DSeries can be instantiated with minimal required fields."""
        mesh = pv.Plane()
        series = BS2DSeries(
            mesh=mesh,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=0,
            spin_index=0,
        )
        assert series.band_index == 0
        assert series.spin_index == 0
        assert series.mesh is mesh
        assert series.scalars is None
        assert series.vectors is None

    def test_bs2d_series_creation_with_scalars(self):
        """Test BS2DSeries with scalar data."""
        mesh = pv.Plane()
        scalars = np.random.rand(mesh.n_points)
        series = BS2DSeries(
            mesh=mesh,
            scalars=scalars,
            scalars_label="Projection",
            scalars_unit="eV",
            scalars_lim=(0.0, 1.0),
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label="Band 0 up",
            band_index=0,
            spin_index=0,
        )
        assert series.scalars is scalars
        assert series.scalars_label == "Projection"
        assert series.scalars_unit == "eV"
        assert series.scalars_lim == (0.0, 1.0)
        assert series.label == "Band 0 up"

    def test_bs2d_series_creation_with_vectors(self):
        """Test BS2DSeries with vector data."""
        mesh = pv.Plane()
        vectors = np.random.rand(mesh.n_points, 3)
        series = BS2DSeries(
            mesh=mesh,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=vectors,
            vectors_label="Spin",
            vectors_unit="hbar",
            vectors_lim=(0.0, 1.0),
            label="Band 1 down",
            band_index=1,
            spin_index=1,
        )
        assert series.vectors is vectors
        assert series.vectors_label == "Spin"
        assert series.vectors_unit == "hbar"
        assert series.vectors_lim == (0.0, 1.0)

    def test_bs2d_series_additional_kwargs_default(self):
        """Test BS2DSeries additional_kwargs defaults to empty dict."""
        mesh = pv.Plane()
        series = BS2DSeries(
            mesh=mesh,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=0,
            spin_index=0,
        )
        assert series.additional_kwargs == {}


class TestBS2DPlotterInit:
    """Tests for BS2DPlotter initialization."""

    def test_plotter_init_has_meshes_list(self):
        """Test BS2DPlotter initializes with _meshes list."""
        # Create a minimal mock bandstructure2d
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            assert hasattr(plotter, "_meshes")
            assert isinstance(plotter._meshes, list)
            assert len(plotter._meshes) == 0
        finally:
            plotter.close()

    def test_plotter_init_has_values_dict(self):
        """Test BS2DPlotter initializes with values_dict."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            assert hasattr(plotter, "values_dict")
            assert isinstance(plotter.values_dict, dict)
            assert len(plotter.values_dict) == 0
        finally:
            plotter.close()

    def test_plotter_has_plot_method(self):
        """Test BS2DPlotter has plot method."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            assert hasattr(plotter, "plot")
            assert callable(plotter.plot)
        finally:
            plotter.close()

    def test_plotter_has_to_series_list_method(self):
        """Test BS2DPlotter has _to_series_list method."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            assert hasattr(plotter, "_to_series_list")
            assert callable(plotter._to_series_list)
        finally:
            plotter.close()


class TestBS2DPlotterBuildSeriesLabel:
    """Tests for _build_series_label method."""

    def test_build_series_label_single_surface(self):
        """Test label generation for single surface returns None."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            label = plotter._build_series_label(iband=0, ispin=0, n_surfaces=1)
            assert label is None
        finally:
            plotter.close()

    def test_build_series_label_multiple_surfaces_spin_up(self):
        """Test label generation for spin up in multi-surface plot."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            label = plotter._build_series_label(iband=0, ispin=0, n_surfaces=2)
            assert label == "Band 0 up"
        finally:
            plotter.close()

    def test_build_series_label_multiple_surfaces_spin_down(self):
        """Test label generation for spin down in multi-surface plot."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            label = plotter._build_series_label(iband=1, ispin=1, n_surfaces=2)
            assert label == "Band 1 down"
        finally:
            plotter.close()


class TestBS2DPlotterResolveClim:
    """Tests for _resolve_clim method."""

    def test_resolve_clim_empty_series(self):
        """Test clim resolution with no series."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        try:
            clim = plotter._resolve_clim([])
            assert clim == (0.0, 1.0)
        finally:
            plotter.close()

    def test_resolve_clim_no_scalars(self):
        """Test clim resolution when series have no scalars."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh = pv.Plane()
        series = BS2DSeries(
            mesh=mesh,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=0,
            spin_index=0,
        )
        try:
            clim = plotter._resolve_clim([series])
            assert clim == (0.0, 1.0)
        finally:
            plotter.close()

    def test_resolve_clim_with_scalars(self):
        """Test clim resolution with scalar data."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh = pv.Plane()
        scalars = np.array([0.5, 1.0, 1.5, 2.0])
        series = BS2DSeries(
            mesh=mesh,
            scalars=scalars,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=0,
            spin_index=0,
        )
        try:
            clim = plotter._resolve_clim([series])
            assert clim == (0.5, 2.0)
        finally:
            plotter.close()

    def test_resolve_clim_multiple_series(self):
        """Test clim resolution with multiple series."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh = pv.Plane()
        series1 = BS2DSeries(
            mesh=mesh,
            scalars=np.array([1.0, 2.0]),
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=0,
            spin_index=0,
        )
        series2 = BS2DSeries(
            mesh=mesh,
            scalars=np.array([0.5, 3.0]),
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=1,
            spin_index=0,
        )
        try:
            clim = plotter._resolve_clim([series1, series2])
            assert clim == (0.5, 3.0)
        finally:
            plotter.close()


class TestBS2DPlotterRecordSeriesData:
    """Tests for _record_series_data method."""

    def test_record_series_data_basic(self):
        """Test recording series data for export."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh = pv.Plane()
        series = BS2DSeries(
            mesh=mesh,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=0,
            spin_index=0,
        )
        try:
            plotter._record_series_data(series)
            assert "band_0_spin_0_points" in plotter.values_dict
            assert np.array_equal(plotter.values_dict["band_0_spin_0_points"], mesh.points)
        finally:
            plotter.close()

    def test_record_series_data_with_scalars(self):
        """Test recording series data with scalars."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh = pv.Plane()
        scalars = np.array([1.0, 2.0, 3.0])
        series = BS2DSeries(
            mesh=mesh,
            scalars=scalars,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=0,
            spin_index=0,
        )
        try:
            plotter._record_series_data(series)
            assert "band_0_spin_0_scalars" in plotter.values_dict
            assert np.array_equal(plotter.values_dict["band_0_spin_0_scalars"], scalars)
        finally:
            plotter.close()

    def test_record_series_data_with_vectors(self):
        """Test recording series data with vectors."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh = pv.Plane()
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        series = BS2DSeries(
            mesh=mesh,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=vectors,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label=None,
            band_index=1,
            spin_index=1,
        )
        try:
            plotter._record_series_data(series)
            assert "band_1_spin_1_vectors" in plotter.values_dict
            assert np.array_equal(plotter.values_dict["band_1_spin_1_vectors"], vectors)
        finally:
            plotter.close()


class TestBS2DPlotterExport:
    """Tests for export functionality."""

    def test_export_npz(self, tmp_path):
        """Test export to NPZ format."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        plotter.values_dict = {"test": np.array([1, 2, 3])}

        output_path = tmp_path / "test.npz"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            loaded = np.load(output_path)
            assert "test" in loaded
            assert np.array_equal(loaded["test"], np.array([1, 2, 3]))
        finally:
            plotter.close()

    def test_export_vtk(self, tmp_path):
        """Test export to VTK format."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh = pv.Plane()
        plotter._meshes = [mesh]

        output_path = tmp_path / "test.vtk"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            loaded_mesh = pv.read(str(output_path))
            assert loaded_mesh.n_points == mesh.n_points
        finally:
            plotter.close()

    def test_export_unsupported_format_raises(self, tmp_path):
        """Test export raises ValueError for unsupported formats."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        output_path = tmp_path / "test.xyz"
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                plotter.export_data(str(output_path))
        finally:
            plotter.close()

    def test_export_multiple_meshes_merged(self, tmp_path):
        """Test export merges multiple meshes."""
        mock_bs2d = type("MockBS2D", (), {})()
        plotter = BS2DPlotter(mock_bs2d, off_screen=True)
        mesh1 = pv.Plane(center=(0, 0, 0))
        mesh2 = pv.Plane(center=(2, 0, 0))
        plotter._meshes = [mesh1, mesh2]

        output_path = tmp_path / "test.vtk"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            loaded_mesh = pv.read(str(output_path))
            # Merged mesh should have points from both planes
            assert loaded_mesh.n_points == mesh1.n_points + mesh2.n_points
        finally:
            plotter.close()
