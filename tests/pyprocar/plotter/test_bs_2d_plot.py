"""Tests for BS2DPlotter.plot() through the rendered meshes."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import pyvista as pv

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from tests.pyprocar.core.test_bandstructure2d_grid import (
    HEXAGONAL,
    tight_binding_graphene,
    two_band_mesh,
)

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

        assert plotter.actors["vectors_1_0"].mapper.dataset.bounds[5] == pytest.approx(0.0)
        assert plotter.actors["vectors_2_0"].mapper.dataset.bounds[5] == pytest.approx(3.0)

    def test_arrows_skip_points_where_the_band_has_no_velocity(self, plotter):
        vectors = VECTORS.copy()
        vectors[1] = np.nan

        plotter.plot(vectors_data=_property(vectors))

        arrows = plotter.actors["vectors_2_0"].mapper.dataset
        assert np.isfinite(arrows.points).all()
        assert arrows.bounds[5] == pytest.approx(3.0)
        assert np.isfinite(plotter.actors["vectors_1_0"].mapper.dataset.points).all()

    def test_arrows_survive_clipping_to_the_zone(self):
        bs2d = _bandstructure2d()
        bs2d.points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]])
        square = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        zone = SimpleNamespace(face_normals=square, centers=2 * square)

        def get_2d_brillouin_zone(e_min: float, e_max: float) -> SimpleNamespace:
            assert e_min < e_max
            return zone

        bs2d.get_2d_brillouin_zone = get_2d_brillouin_zone
        plotter = BS2DPlotter(bs2d, off_screen=True)

        plotter.plot(
            vectors_data=_property(VECTORS), show_brillouin_zone=False, clip_brillouin_zone=True
        )

        arrows = cast(pv.Actor, plotter.actors["vectors_2_0"]).mapper.dataset
        assert arrows.bounds[5] == pytest.approx(3.0)
        plotter.close()

    def test_band_surfaces_keep_their_scalars_when_arrows_are_added(self, plotter):
        meshes = plotter.plot(scalars_data="band_speed", vectors_data=_property(VECTORS))

        assert [m.active_scalars_name for m in meshes.values()] == ["scalars", "scalars"]
        assert plotter.actors["surface_1_0"].mapper.scalar_range == (3.0, 8.0)

    def test_records_points_and_scalars_per_surface(self, plotter):
        plotter.plot(scalars_data="band_speed")

        assert sorted(plotter.values_dict) == [
            "band_1_spin_0_points",
            "band_1_spin_0_scalars",
            "band_2_spin_0_points",
            "band_2_spin_0_scalars",
        ]


class TestBS2DPlotterBrillouinZone:
    def test_default_plot_draws_the_zone_spanning_the_finite_band_energies(self):
        requested = {}

        def zone(e_min, e_max):
            requested.update(e_min=e_min, e_max=e_max)
            return pv.Box(bounds=(-1.0, 1.0, -1.0, 1.0, e_min, e_max))

        bs2d = _bandstructure2d()
        bs2d.points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, np.nan], [0.0, 0.0, 2.0]])
        bs2d.get_2d_brillouin_zone = zone
        plotter = BS2DPlotter(bs2d, off_screen=True)

        meshes = plotter.plot()

        assert requested == {"e_min": -1.0, "e_max": 2.0}
        assert len(plotter.actors) == len(meshes) + 1
        plotter.close()

    @pytest.mark.parametrize(
        ("offset", "energy_range"),
        [(-5.0, (-8.0, -2.0)), (0.0, (-3.0, 3.0)), (5.0, (2.0, 8.0))],
        ids=["below-zero", "around-zero", "above-zero"],
    )
    def test_clipping_to_the_zone_keeps_graphene_bands_at_any_energy(self, offset, energy_range):
        """The zone prism spans the band energies, so clipping must keep the same in-zone
        part of each band wherever the energies sit relative to 0."""
        ebs = two_band_mesh(HEXAGONAL, tight_binding_graphene, offset=offset)
        bs2d = BandStructure2D.from_ebs(ebs, grid_interpolation=(40, 40), padding=3)
        plotter = BS2DPlotter(bs2d, off_screen=True)

        drawn = plotter.plot(show_brillouin_zone=False, clip_brillouin_zone=True)

        plotter.close()
        assert (bs2d.points[:, 2].min(), bs2d.points[:, 2].max()) == pytest.approx(
            energy_range, abs=0.01
        )
        assert {key: mesh.n_points for key, mesh in drawn.items()} == {(0, 0): 401, (1, 0): 401}


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
