"""Tests for BS2DPlotter.plot() through the rendered meshes."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import pyvista as pv

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from tests.pyprocar.core.test_bandstructure2d_grid import (
    HEXAGONAL,
    N_K,
    PADDING,
    tight_binding_graphene,
    two_band_mesh,
)
from tests.pyprocar.core.test_bandstructure2d_zone import HEXAGON_ZONE_AREA, projected_area
from tests.utils.user_warning import user_warning

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


OUTSIDE = "lies only outside the first Brillouin zone; it is not drawn"


def _clipped_bandstructure2d(zone, props=None):
    """``_bandstructure2d`` spanning E = -1 to 2, whose zone for any energy range is ``zone``."""
    bs2d = _bandstructure2d(props)
    bs2d.points = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]])

    def get_2d_brillouin_zone(e_min: float, e_max: float):
        assert e_min < e_max
        return zone

    bs2d.get_2d_brillouin_zone = get_2d_brillouin_zone
    return bs2d


def _outside_warnings(record: pytest.WarningsRecorder) -> list[str]:
    return [str(w.message) for w in record if OUTSIDE in str(w.message)]


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

    @pytest.mark.guards_existing_behaviour(
        reason="a pad that covers the zone keeps the whole zone at any energy, as today"
    )
    @pytest.mark.parametrize(
        ("offset", "energy_range"),
        [(-5.0, (-8.0, -2.0)), (0.0, (-3.0, 3.0)), (5.0, (2.0, 8.0))],
        ids=["below-zero", "around-zero", "above-zero"],
    )
    def test_clipping_to_the_zone_keeps_graphene_bands_at_any_energy(self, offset, energy_range):
        """The zone prism spans the band energies, so clipping must keep the same in-zone
        part of each band wherever the energies sit relative to 0."""
        ebs = two_band_mesh(HEXAGONAL, tight_binding_graphene, offset=offset)
        grid = N_K + 2 * PADDING
        bs2d = BandStructure2D.from_ebs(ebs, grid_interpolation=(grid, grid), padding=PADDING)
        plotter = BS2DPlotter(bs2d, off_screen=True)

        drawn = plotter.plot(show_brillouin_zone=False, clip_brillouin_zone=True)

        plotter.close()
        assert (bs2d.points[:, 2].min(), bs2d.points[:, 2].max()) == pytest.approx(
            energy_range, abs=0.01
        )
        assert sorted(drawn) == [(0, 0), (1, 0)]
        for mesh in drawn.values():
            assert projected_area(mesh) == pytest.approx(HEXAGON_ZONE_AREA, rel=1e-9)

    @pytest.mark.parametrize(
        "offset", [-5.0, 0.0, 5.0], ids=["below-zero", "around-zero", "above-zero"]
    )
    def test_clipping_a_pad_short_of_the_zone_keeps_the_whole_graphene_zone(self, offset):
        """A pad of 3 on the [0, 1) 24 x 24 grid starts at fractional -3/24, short of the zone
        corner at -2/3; the drawn box reaches both, so each clipped band is the whole hexagon."""
        ebs = two_band_mesh(HEXAGONAL, tight_binding_graphene, offset=offset)
        bs2d = BandStructure2D.from_ebs(ebs, grid_interpolation=(40, 40), padding=3)
        plotter = BS2DPlotter(bs2d, off_screen=True)

        drawn = plotter.plot(show_brillouin_zone=False, clip_brillouin_zone=True)

        plotter.close()
        assert sorted(drawn) == [(0, 0), (1, 0)]
        for mesh in drawn.values():
            assert projected_area(mesh) == pytest.approx(HEXAGON_ZONE_AREA, rel=1e-9)

    def test_a_band_outside_the_zone_is_skipped_with_a_warning(self):
        """H3: the zone spans -1.5 to 1.5 on each axis, so band 2 at E = 2 clips to nothing;
        today it is drawn as an empty mesh without a warning."""
        plotter = BS2DPlotter(
            _clipped_bandstructure2d(BrillouinZone(3 * np.eye(3))), off_screen=True
        )

        with user_warning(__file__, match=OUTSIDE) as record:
            meshes = plotter.plot(show_brillouin_zone=False, clip_brillouin_zone=True)

        actors = sorted(plotter.actors)
        plotter.close()
        assert _outside_warnings(record) == [f"band 2 spin 0 {OUTSIDE}"]
        assert list(meshes) == [(1, 0)]
        assert meshes[(1, 0)].n_points == 3
        assert actors == ["surface_1_0"]

    def test_the_first_drawn_band_carries_the_scalar_bar_when_band_one_is_skipped(self):
        """The zone keeps E >= 0, so band 1 at E = -1 is skipped and band 2 shows the bar."""
        upper_half = SimpleNamespace(
            face_normals=np.array([[0.0, 0.0, -1.0]]), centers=np.zeros((1, 3))
        )
        bs2d = _clipped_bandstructure2d(upper_half, {"band_speed": _property(SCALARS)})
        plotter = BS2DPlotter(bs2d, off_screen=True)

        with user_warning(__file__, match=OUTSIDE) as record:
            meshes = plotter.plot(
                scalars_data="band_speed", show_brillouin_zone=False, clip_brillouin_zone=True
            )

        bars = list(plotter.scalar_bars.keys())
        plotter.close()
        assert _outside_warnings(record) == [f"band 1 spin 0 {OUTSIDE}"]
        assert list(meshes) == [(2, 0)]
        assert bars == ["Band speed"]

    def test_add_surface_skips_a_surface_clipped_away_with_a_warning(self, plotter):
        """H3: today the empty clip is drawn silently."""
        plotter.add_brillouin_zone(BrillouinZone(np.eye(3)))
        outside = pv.Sphere(radius=0.2, center=(2.0, 2.0, 2.0))

        with user_warning(__file__, match=OUTSIDE) as record:
            plotter.add_surface(outside, clip_surface=True, show_scalar_bar=False)

        assert _outside_warnings(record) == [f"surface 'surface' {OUTSIDE}"]
        assert "surface" not in plotter.actors


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
