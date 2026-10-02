"""Tests for BS2DPlotter.plot() through the rendered meshes."""

from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv

from pyprocar.plotter.bs_2d_plot import BS2DPlotter

TRIANGLE_FACES = [3, 0, 1, 2]


def _triangle(z: float) -> pv.PolyData:
    return pv.PolyData(np.array([[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]]), TRIANGLE_FACES)


def _property(values, label="Band speed"):
    return SimpleNamespace(
        to_array=lambda: np.asarray(values), label=label, units="", data_lim=None
    )


def _bandstructure2d(props=None):
    """Two band surfaces: (band 1, spin 0) owns points 0-2, (band 2, spin 0) owns points 3-5."""
    mask_a = np.array([True, True, True, False, False, False])
    return SimpleNamespace(
        band_surfaces={(1, 0): _triangle(-1.0), (2, 0): _triangle(2.0)},
        band_spin_mask={(1, 0): mask_a, (2, 0): ~mask_a},
        get_property=lambda name: (props or {})[name],
    )


SCALARS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
VECTORS = np.tile([0.0, 0.0, 2.0], (6, 1))


@pytest.fixture
def plotter():
    p = BS2DPlotter(_bandstructure2d({"band_speed": _property(SCALARS)}), off_screen=True)
    yield p
    p.close()


class TestBS2DPlotterPlot:
    def test_defaults_to_the_constructor_band_structure(self, plotter):
        meshes = plotter.plot()

        assert list(meshes) == [(1, 0), (2, 0)]
        assert meshes[(2, 0)].points[:, 2].tolist() == [2.0, 2.0, 2.0]

    def test_each_mesh_gets_its_masked_scalars_by_name(self, plotter):
        meshes = plotter.plot(scalars_data="band_speed")

        assert meshes[(1, 0)].point_data["scalars"].tolist() == [3.0, 4.0, 5.0]
        assert meshes[(2, 0)].point_data["scalars"].tolist() == [6.0, 7.0, 8.0]

    def test_clim_spans_scalars_of_every_surface(self, plotter):
        plotter.plot(scalars_data="band_speed")

        assert plotter.actors["surface_1_0"].mapper.scalar_range == (3.0, 8.0)

    def test_single_scalar_bar_titled_by_property_label(self, plotter):
        plotter.plot(scalars_data="band_speed")

        assert list(plotter.scalar_bars.keys()) == ["Band speed"]

    def test_vectors_add_glyphs_scaled_to_the_longest(self, plotter):
        plotter.plot(vectors_data=_property(VECTORS))

        arrows = plotter.actors["vectors"].mapper.dataset
        assert arrows.bounds[5] == pytest.approx(3.0)

    def test_records_points_and_scalars_per_surface(self, plotter):
        plotter.plot(scalars_data="band_speed")

        assert sorted(plotter.values_dict) == [
            "band_1_spin_0_points",
            "band_1_spin_0_scalars",
            "band_2_spin_0_points",
            "band_2_spin_0_scalars",
        ]


class TestBS2DPlotterExport:
    def test_vtk_merges_plotted_surfaces(self, plotter, tmp_path):
        plotter.plot()
        out = tmp_path / "bs2d.vtk"
        plotter.export_data(str(out))

        assert pv.read(str(out)).n_points == 6

    def test_npz_holds_recorded_arrays(self, plotter, tmp_path):
        plotter.plot(scalars_data="band_speed")
        out = tmp_path / "bs2d.npz"
        plotter.export_data(str(out))

        assert np.load(out)["band_2_spin_0_scalars"].tolist() == [6.0, 7.0, 8.0]

    def test_unsupported_format_raises(self, plotter, tmp_path):
        with pytest.raises(ValueError, match="Unsupported file format"):
            plotter.export_data(str(tmp_path / "bs2d.xyz"))
