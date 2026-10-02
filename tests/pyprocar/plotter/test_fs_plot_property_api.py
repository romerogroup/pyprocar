"""Tests for FermiPlotter.plot() through the rendered meshes."""

from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from pyprocar.plotter.fs_plot import FermiPlotter

TRIANGLE_FACES = [3, 0, 1, 2]


def _triangle(z: float) -> pv.PolyData:
    return pv.PolyData(np.array([[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]]), TRIANGLE_FACES)


def _property(values, label="Projection"):
    return SimpleNamespace(
        to_array=lambda: np.asarray(values), label=label, units="", data_lim=None
    )


def _fermi_surface(props=None):
    """Two isosurfaces: (band 0, spin 0) owns points 0-2, (band 3, spin 1) owns points 3-5."""
    mask_a = np.array([True, True, True, False, False, False])
    return SimpleNamespace(
        band_isosurfaces={(0, 0): _triangle(0.0), (3, 1): _triangle(1.0)},
        band_spin_mask={(0, 0): mask_a, (3, 1): ~mask_a},
        get_property=lambda name: (props or {})[name],
        brillouin_zone=pv.Cube(),
    )


SCALARS = [0.5, 1.0, 1.5, 2.0, np.nan, -1.0]
VECTORS = np.tile([0.0, 0.0, 2.0], (6, 1))


@pytest.fixture
def plotter():
    p = FermiPlotter(off_screen=True)
    yield p
    p.close()


class TestFermiPlotterPlot:
    def test_one_mesh_per_isosurface_keyed_by_band_and_spin(self, plotter):
        fs = _fermi_surface()
        meshes = plotter.plot(fs)

        assert list(meshes) == [(0, 0), (3, 1)]
        assert meshes[(3, 1)].points[:, 2].tolist() == [1.0, 1.0, 1.0]
        assert meshes[(0, 0)] is not fs.band_isosurfaces[(0, 0)]

    def test_brillouin_zone_is_drawn_unless_disabled(self, plotter):
        plotter.plot(_fermi_surface())
        other = FermiPlotter(off_screen=True)
        other.plot(_fermi_surface(), show_brillouin_zone=False)

        assert len(plotter.actors) - len(other.actors) == 1
        other.close()

    def test_each_mesh_gets_its_masked_scalars(self, plotter):
        meshes = plotter.plot(_fermi_surface(), scalars_data=_property(SCALARS))

        assert meshes[(0, 0)].point_data["scalars"].tolist() == [0.5, 1.0, 1.5]
        assert meshes[(3, 1)].active_scalars_name == "scalars"
        np.testing.assert_array_equal(meshes[(3, 1)].point_data["scalars"], [2.0, np.nan, -1.0])

    def test_band_resolved_scalars_take_each_surface_band_and_spin(self, plotter):
        values = np.zeros((6, 4, 2))
        values[:3, 0, 0] = [0.1, 0.2, 0.3]
        values[3:, 3, 1] = [0.7, 0.8, 0.9]
        values[:, 1, 0] = 5.0

        meshes = plotter.plot(_fermi_surface(), scalars_data=_property(values))

        assert meshes[(0, 0)].point_data["scalars"].tolist() == [0.1, 0.2, 0.3]
        assert meshes[(3, 1)].point_data["scalars"].tolist() == [0.7, 0.8, 0.9]

    def test_spins_limits_the_drawn_surfaces(self, plotter):
        meshes = plotter.plot(_fermi_surface(), spins=[1])

        assert list(meshes) == [(3, 1)]

    def test_scalars_resolve_by_name(self, plotter):
        fs = _fermi_surface({"projected_sum": _property(SCALARS)})
        meshes = plotter.plot(fs, scalars_data="projected_sum")

        assert meshes[(0, 0)].point_data["scalars"].tolist() == [0.5, 1.0, 1.5]

    def test_clim_spans_finite_scalars_of_every_surface(self, plotter):
        plotter.plot(_fermi_surface(), scalars_data=_property(SCALARS))

        assert plotter.actors["surface_0_0"].mapper.scalar_range == (-1.0, 2.0)
        assert plotter.actors["surface_3_1"].mapper.scalar_range == (-1.0, 2.0)

    def test_user_clim_wins(self, plotter):
        plotter.plot(_fermi_surface(), scalars_data=_property(SCALARS), scalars_clim=(0.0, 0.5))

        assert plotter.actors["surface_3_1"].mapper.scalar_range == (0.0, 0.5)

    def test_single_scalar_bar_titled_by_property_label(self, plotter):
        plotter.plot(_fermi_surface(), scalars_data=_property(SCALARS, label="d orbitals"))

        assert list(plotter.scalar_bars.keys()) == ["d orbitals"]

    def test_scalars_mode_none_leaves_meshes_uncolored(self, plotter):
        meshes = plotter.plot(
            _fermi_surface(), scalars_data=_property(SCALARS), scalars_mode="none"
        )

        assert "scalars" not in meshes[(0, 0)].point_data

    def test_vectors_add_glyphs_scaled_to_a_hundredth_of_the_longest(self, plotter):
        plotter.plot(_fermi_surface(), vectors_data=_property(VECTORS))

        arrows = plotter.actors["vectors"].mapper.dataset
        assert arrows.bounds[5] == pytest.approx(1.01)

    def test_records_points_scalars_and_vectors_per_surface(self, plotter):
        plotter.plot(
            _fermi_surface(),
            scalars_data=_property(SCALARS),
            vectors_data=_property(VECTORS),
        )

        assert sorted(plotter.values_dict) == [
            "band_0_spin_0_points",
            "band_0_spin_0_scalars",
            "band_0_spin_0_vectors",
            "band_3_spin_1_points",
            "band_3_spin_1_scalars",
            "band_3_spin_1_vectors",
        ]
        assert plotter.values_dict["band_0_spin_0_scalars"].tolist() == [0.5, 1.0, 1.5]


class TestFermiPlotterAddTexture:
    def test_accepts_fermi_surface_keyword(self, plotter):
        surface = _triangle(0.0)
        surface.point_data["v"] = VECTORS[:3]
        surface.set_active_vectors("v")

        arrows = plotter.add_texture(fermi_surface=surface)

        assert arrows.bounds[5] == pytest.approx(0.01)


class TestFermiPlotterExport:
    @pytest.mark.parametrize("ext", [".vtk", ".vtp", ".ply", ".stl"])
    def test_mesh_formats_merge_plotted_surfaces(self, plotter, tmp_path, ext):
        plotter.plot(_fermi_surface())
        out = tmp_path / f"fs{ext}"
        plotter.export_data(str(out))

        assert pv.read(str(out)).n_points == 6

    def test_npz_holds_recorded_arrays(self, plotter, tmp_path):
        plotter.plot(_fermi_surface(), scalars_data=_property(SCALARS))
        out = tmp_path / "fs.npz"
        plotter.export_data(str(out))

        assert np.load(out)["band_0_spin_0_scalars"].tolist() == [0.5, 1.0, 1.5]

    def test_nothing_plotted_writes_nothing(self, plotter, tmp_path):
        out = tmp_path / "fs.vtk"
        plotter.export_data(str(out))

        assert not out.exists()

    def test_unsupported_format_raises(self, plotter, tmp_path):
        with pytest.raises(ValueError, match="Unsupported file format"):
            plotter.export_data(str(tmp_path / "fs.xyz"))


class TestFermiPlotterScalarBarMethods:
    """Tests for scalar bar configuration methods."""

    def test_set_scalar_bar_title_no_bar(self):
        """Test set_scalar_bar_title when no scalar bar exists."""
        plotter = FermiPlotter(off_screen=True)
        try:
            # Should not raise even without scalar bar
            plotter.set_scalar_bar_title("Test Title")
        finally:
            plotter.close()

    def test_set_scalar_bar_label_font_size_no_bar(self):
        """Test set_scalar_bar_label_font_size when no scalar bar exists."""
        plotter = FermiPlotter(off_screen=True)
        try:
            # Should not raise even without scalar bar
            plotter.set_scalar_bar_label_font_size(12)
        finally:
            plotter.close()

    def test_set_scalar_bar_position_no_bar(self):
        """Test set_scalar_bar_position when no scalar bar exists."""
        plotter = FermiPlotter(off_screen=True)
        try:
            # Should not raise even without scalar bar
            plotter.set_scalar_bar_position((0.1, 0.1))
        finally:
            plotter.close()

    def test_scalar_bar_methods_exist(self):
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
