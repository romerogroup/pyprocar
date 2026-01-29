"""Tests for FermiPlotter Property-based API."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

_rng = np.random.default_rng(42)


def _load_fs_plot_module() -> types.ModuleType:
    """Load fs_plot module directly to avoid pyprocar/__init__.py import issues."""
    fs_plot_path = Path(__file__).parents[3] / "pyprocar" / "plotter" / "fs_plot.py"
    spec = importlib.util.spec_from_file_location("fs_plot", fs_plot_path)
    assert spec is not None
    fs_plot = importlib.util.module_from_spec(spec)
    sys.modules["fs_plot"] = fs_plot
    assert spec.loader is not None
    spec.loader.exec_module(fs_plot)
    return fs_plot


# Load module once
_fs_plot = _load_fs_plot_module()
FermiPlotter = _fs_plot.FermiPlotter
FermiSeries = _fs_plot.FermiSeries


class TestFermiSeries:
    """Tests for FermiSeries dataclass."""

    def test_fermi_series_creation_minimal(self) -> None:
        """Test FermiSeries can be instantiated with minimal required fields."""
        mesh = pv.Sphere()
        series = FermiSeries(
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

    def test_fermi_series_creation_with_scalars(self) -> None:
        """Test FermiSeries with scalar data."""
        mesh = pv.Sphere()
        scalars = _rng.random(mesh.n_points)
        series = FermiSeries(
            mesh=mesh,
            scalars=scalars,
            scalars_label="Projection",
            scalars_unit="eV",
            scalars_lim=(0.0, 1.0),
            vectors=None,
            vectors_label=None,
            vectors_unit=None,
            vectors_lim=None,
            label="Band 0 ↑",
            band_index=0,
            spin_index=0,
        )
        assert series.scalars is scalars
        assert series.scalars_label == "Projection"
        assert series.scalars_unit == "eV"
        assert series.scalars_lim == (0.0, 1.0)
        assert series.label == "Band 0 ↑"

    def test_fermi_series_creation_with_vectors(self) -> None:
        """Test FermiSeries with vector data."""
        mesh = pv.Sphere()
        vectors = _rng.random((mesh.n_points, 3))
        series = FermiSeries(
            mesh=mesh,
            scalars=None,
            scalars_label=None,
            scalars_unit=None,
            scalars_lim=None,
            vectors=vectors,
            vectors_label="Spin",
            vectors_unit="hbar",
            vectors_lim=(0.0, 1.0),
            label="Band 1 ↓",
            band_index=1,
            spin_index=1,
        )
        assert series.vectors is vectors
        assert series.vectors_label == "Spin"
        assert series.vectors_unit == "hbar"
        assert series.vectors_lim == (0.0, 1.0)

    def test_fermi_series_additional_kwargs_default(self) -> None:
        """Test FermiSeries additional_kwargs defaults to empty dict."""
        mesh = pv.Sphere()
        series = FermiSeries(
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

    def test_fermi_series_additional_kwargs_custom(self) -> None:
        """Test FermiSeries with custom additional_kwargs."""
        mesh = pv.Sphere()
        kwargs = {"opacity": 0.5, "style": "wireframe"}
        series = FermiSeries(
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
            additional_kwargs=kwargs,
        )
        assert series.additional_kwargs == kwargs


class TestFermiPlotterInit:
    """Tests for FermiPlotter initialization."""

    def test_plotter_init_default(self) -> None:
        """Test FermiPlotter initializes with expected attributes."""
        plotter = FermiPlotter(off_screen=True)
        try:
            assert hasattr(plotter, "_meshes")
            assert hasattr(plotter, "values_dict")
            meshes: list[object] = plotter._meshes
            values: dict[str, object] = plotter.values_dict
            assert isinstance(meshes, list)
            assert isinstance(values, dict)
            assert len(meshes) == 0
            assert len(values) == 0
        finally:
            plotter.close()

    def test_plotter_has_plot_method(self) -> None:
        """Test FermiPlotter has plot method."""
        plotter = FermiPlotter(off_screen=True)
        try:
            assert hasattr(plotter, "plot")
            assert callable(plotter.plot)
        finally:
            plotter.close()

    def test_plotter_has_to_series_list_method(self) -> None:
        """Test FermiPlotter has _to_series_list method."""
        plotter = FermiPlotter(off_screen=True)
        try:
            assert hasattr(plotter, "_to_series_list")
            assert callable(plotter._to_series_list)
        finally:
            plotter.close()


class TestFermiPlotterBuildSeriesLabel:
    """Tests for _build_series_label method."""

    def test_build_series_label_single_surface(self) -> None:
        """Test label generation for single surface returns None."""
        plotter = FermiPlotter(off_screen=True)
        try:
            label = plotter._build_series_label(iband=0, ispin=0, n_surfaces=1)
            assert label is None
        finally:
            plotter.close()

    def test_build_series_label_multiple_surfaces_spin_up(self) -> None:
        """Test label generation for spin up in multi-surface plot."""
        plotter = FermiPlotter(off_screen=True)
        try:
            label = plotter._build_series_label(iband=0, ispin=0, n_surfaces=2)
            assert label == "Band 0 ↑"
        finally:
            plotter.close()

    def test_build_series_label_multiple_surfaces_spin_down(self) -> None:
        """Test label generation for spin down in multi-surface plot."""
        plotter = FermiPlotter(off_screen=True)
        try:
            label = plotter._build_series_label(iband=1, ispin=1, n_surfaces=2)
            assert label == "Band 1 ↓"
        finally:
            plotter.close()


class TestFermiPlotterResolveClim:
    """Tests for _resolve_clim method."""

    def test_resolve_clim_empty_series(self) -> None:
        """Test clim resolution with no series."""
        plotter = FermiPlotter(off_screen=True)
        try:
            clim = plotter._resolve_clim([])
            assert clim == (0.0, 1.0)
        finally:
            plotter.close()

    def test_resolve_clim_no_scalars(self) -> None:
        """Test clim resolution when series have no scalars."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        series = FermiSeries(
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

    def test_resolve_clim_with_scalars(self) -> None:
        """Test clim resolution with scalar data."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        scalars = np.array([0.5, 1.0, 1.5, 2.0])
        series = FermiSeries(
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

    def test_resolve_clim_multiple_series(self) -> None:
        """Test clim resolution with multiple series."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        series1 = FermiSeries(
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
        series2 = FermiSeries(
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


class TestFermiPlotterRecordSeriesData:
    """Tests for _record_series_data method."""

    def test_record_series_data_basic(self) -> None:
        """Test recording series data for export."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        series = FermiSeries(
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

    def test_record_series_data_with_scalars(self) -> None:
        """Test recording series data with scalars."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        scalars = np.array([1.0, 2.0, 3.0])
        series = FermiSeries(
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

    def test_record_series_data_with_vectors(self) -> None:
        """Test recording series data with vectors."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        series = FermiSeries(
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


class TestFermiPlotterExport:
    """Tests for export functionality."""

    def test_export_npz(self, tmp_path: Path) -> None:
        """Test export to NPZ format."""
        plotter = FermiPlotter(off_screen=True)
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

    def test_export_vtk(self, tmp_path: Path) -> None:
        """Test export to VTK format."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        plotter._meshes = [mesh]

        output_path = tmp_path / "test.vtk"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            # Verify the file can be loaded back
            loaded_mesh = pv.read(str(output_path))
            assert isinstance(loaded_mesh, pv.PolyData)
            assert loaded_mesh.n_points == mesh.n_points
        finally:
            plotter.close()

    def test_export_vtp(self, tmp_path: Path) -> None:
        """Test export to VTP format."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        plotter._meshes = [mesh]

        output_path = tmp_path / "test.vtp"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            loaded_mesh = pv.read(str(output_path))
            assert isinstance(loaded_mesh, pv.PolyData)
            assert loaded_mesh.n_points == mesh.n_points
        finally:
            plotter.close()

    def test_export_ply(self, tmp_path: Path) -> None:
        """Test export to PLY format."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        plotter._meshes = [mesh]

        output_path = tmp_path / "test.ply"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            loaded_mesh = pv.read(str(output_path))
            assert isinstance(loaded_mesh, pv.PolyData)
            assert loaded_mesh.n_points == mesh.n_points
        finally:
            plotter.close()

    def test_export_stl(self, tmp_path: Path) -> None:
        """Test export to STL format."""
        plotter = FermiPlotter(off_screen=True)
        mesh = pv.Sphere()
        plotter._meshes = [mesh]

        output_path = tmp_path / "test.stl"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            loaded_mesh = pv.read(str(output_path))
            assert isinstance(loaded_mesh, pv.PolyData)
            assert loaded_mesh.n_points == mesh.n_points
        finally:
            plotter.close()

    def test_export_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Test export raises ValueError for unsupported formats."""
        plotter = FermiPlotter(off_screen=True)
        output_path = tmp_path / "test.xyz"
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                plotter.export_data(str(output_path))
        finally:
            plotter.close()

    def test_export_multiple_meshes_merged(self, tmp_path: Path) -> None:
        """Test export merges multiple meshes."""
        plotter = FermiPlotter(off_screen=True)
        mesh1 = pv.Sphere(center=(0, 0, 0))
        mesh2 = pv.Sphere(center=(2, 0, 0))
        plotter._meshes = [mesh1, mesh2]

        output_path = tmp_path / "test.vtk"
        try:
            plotter.export_data(str(output_path))
            assert output_path.exists()
            loaded_mesh = pv.read(str(output_path))
            assert isinstance(loaded_mesh, pv.PolyData)
            # Merged mesh should have points from both spheres
            assert loaded_mesh.n_points == mesh1.n_points + mesh2.n_points
        finally:
            plotter.close()

    def test_export_empty_meshes_no_error(self, tmp_path: Path) -> None:
        """Test export with empty meshes doesn't raise."""
        plotter = FermiPlotter(off_screen=True)
        plotter._meshes = []

        output_path = tmp_path / "test.vtk"
        try:
            # Should not raise, just do nothing
            plotter.export_data(str(output_path))
            assert not output_path.exists()  # No file created for empty meshes
        finally:
            plotter.close()


class TestFermiPlotterScalarBarMethods:
    """Tests for scalar bar configuration methods."""

    def test_set_scalar_bar_title_no_bar(self) -> None:
        """Test set_scalar_bar_title when no scalar bar exists."""
        plotter = FermiPlotter(off_screen=True)
        try:
            # Should not raise even without scalar bar
            plotter.set_scalar_bar_title("Test Title")
        finally:
            plotter.close()

    def test_set_scalar_bar_label_font_size_no_bar(self) -> None:
        """Test set_scalar_bar_label_font_size when no scalar bar exists."""
        plotter = FermiPlotter(off_screen=True)
        try:
            # Should not raise even without scalar bar
            plotter.set_scalar_bar_label_font_size(12)
        finally:
            plotter.close()

    def test_set_scalar_bar_position_no_bar(self) -> None:
        """Test set_scalar_bar_position when no scalar bar exists."""
        plotter = FermiPlotter(off_screen=True)
        try:
            # Should not raise even without scalar bar
            plotter.set_scalar_bar_position((0.1, 0.1))
        finally:
            plotter.close()

    def test_scalar_bar_methods_exist(self) -> None:
        """Test all scalar bar methods exist."""
        plotter = FermiPlotter(off_screen=True)
        try:
            assert hasattr(plotter, "set_scalar_bar_title")
            assert hasattr(plotter, "set_scalar_bar_label_font_size")
            assert hasattr(plotter, "set_scalar_bar_position")
            assert callable(plotter.set_scalar_bar_title)
            assert callable(plotter.set_scalar_bar_label_font_size)
            assert callable(plotter.set_scalar_bar_position)
        finally:
            plotter.close()
