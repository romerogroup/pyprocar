from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import pyvista as pv
from matplotlib.quiver import Quiver

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from pyprocar.plotter.fs_plot import FS_AREA_SCALE_FACTOR, FermiPlotter, dHvA_frequency
from tests.utils import DATA_DIR

RADIUS = 0.3


def _sphere(x: float) -> pv.PolyData:
    return pv.Sphere(radius=RADIUS, center=(x, 0, 0), theta_resolution=180, phi_resolution=180)


def _area_text(plotter) -> str:
    return cast(pv.CornerAnnotation, plotter.actors["area_text"]).GetText(2)


def _number(text: str) -> float:
    return float(text.split(":")[1].split()[0])


def test_cross_section_area_sums_the_closed_loops():
    two_spheres = _sphere(-0.5).merge(_sphere(0.5))
    plotter = FermiPlotter(off_screen=True)

    plotter.add_box_slicer(two_spheres, normal=(0, 0, 1), show_cross_section_area=True)

    text = _area_text(plotter)
    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(2 * np.pi * RADIUS**2 * FS_AREA_SCALE_FACTOR, rel=1e-3)
    plotter.close()


def test_van_alphen_frequency_uses_the_largest_closed_orbit():
    small = pv.Sphere(radius=0.1, center=(-0.5, 0, 0), theta_resolution=180, phi_resolution=180)
    plotter = FermiPlotter(off_screen=True)

    plotter.add_box_slicer(
        small.merge(_sphere(0.5)), normal=(0, 0, 1), show_van_alphen_frequency=True
    )

    assert _number(_area_text(plotter)) == pytest.approx(onsager_gauss(RADIUS), rel=1e-3)
    plotter.close()


def onsager_gauss(radius: float) -> float:
    """F = hbar A / (2 pi e) in SI with CODATA 2018 constants, converted from tesla to gauss.

    ``radius`` is in 1/Angstrom without 2 pi, so the physical orbit area is
    pi (2 pi radius)^2 in 1/Angstrom^2, times 1e20 for 1/m^2.
    """
    hbar, e = 1.054571817e-34, 1.602176634e-19
    area = np.pi * (2 * np.pi * radius) ** 2 * 1e20
    return hbar * area / (2 * np.pi * e) * 1e4


def test_dhva_frequency_of_one_inverse_square_angstrom():
    assert dHvA_frequency(1.0) == pytest.approx(1.04758e8, rel=1e-5)


def _bowl() -> pv.PolyData:
    return cast(pv.PolyData, _sphere(-0.5).clip(normal=(1, 0, 0), origin=(-0.6, 0, 0)))


def _slice_text(surface: pv.PolyData, origin=(0, 0, 0), **flags: bool) -> str:
    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(surface, normal=(0, 0, 1), origin=origin, **flags)
    text = _area_text(plotter)
    plotter.close()
    return text


def test_open_curves_are_reported_and_not_counted():
    text = _slice_text(_bowl().merge(_sphere(0.5)), show_cross_section_area=True)

    assert text.endswith(" Ang^-2 (1 open curve not counted)")
    assert _number(text) == pytest.approx(np.pi * RADIUS**2 * FS_AREA_SCALE_FACTOR, rel=1e-3)


def test_van_alphen_frequency_without_a_closed_orbit_says_so():
    text = _slice_text(_bowl(), origin=(-0.7, 0, 0), show_van_alphen_frequency=True)

    assert text == "Van Alphen Frequency : no closed orbit (1 open curve not counted)"


def test_van_alphen_frequency_reports_the_open_curves_it_skipped():
    text = _slice_text(_bowl().merge(_sphere(0.5)), show_van_alphen_frequency=True)

    assert text.endswith(" Gauss (1 open curve not counted)")
    assert _number(text) == pytest.approx(onsager_gauss(RADIUS), rel=1e-3)


def _band_surface_with_nan_energies() -> pv.PolyData:
    """E = kx^2 + ky^2 with NaN energies beyond |k| = 0.4, as BandStructure2D leaves them."""
    k = np.linspace(-0.5, 0.5, 41)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    energy = kx**2 + ky**2
    energy[np.hypot(kx, ky) > 0.4] = np.nan
    grid = pv.StructuredGrid(kx[..., None], ky[..., None], energy[..., None])
    surface = cast(pv.PolyData, grid.extract_surface().triangulate())
    surface.point_data["energy"] = surface.points[:, 2]
    surface.set_active_scalars("energy")
    return surface


@pytest.mark.parametrize(
    ("normal", "origin", "area", "suffix"),
    [
        ((0, 0, 1), (0, 0, 0.04), np.pi * 0.04, " Ang^-2"),
        ((1, 0, 0), (0, 0, 0), 0.0, " Ang^-2 (1 open curve not counted)"),
    ],
)
def test_band_surface_area_skips_nan_points(normal, origin, area, suffix):
    plotter = BS2DPlotter(SimpleNamespace(), off_screen=True)

    plotter.add_box_slicer(
        _band_surface_with_nan_energies(), normal=normal, origin=origin, cross_section_area=True
    )

    text = _area_text(plotter)
    assert text.endswith(suffix)
    assert _number(text) == pytest.approx(area, rel=0.01, abs=1e-12)
    plotter.close()


def test_saved_slice_draws_the_active_vectors_as_arrows(tmp_path):
    sphere = _sphere(0.0)
    sphere.point_data["spin"] = np.tile([0.0, 1.0, 0.0], (sphere.n_points, 1))
    sphere.point_data["spin-norm"] = np.ones(sphere.n_points)
    sphere.set_active_vectors("spin")
    sphere.set_active_scalars("spin-norm")
    plotter = FermiPlotter(off_screen=True)

    slice_plotter = plotter.save_slice_2d(
        sphere, (0, 0, 1), (0, 0, 0), tmp_path / "slice.png", cmap="viridis"
    )

    (arrows,) = [c for c in slice_plotter.ax.collections if isinstance(c, Quiver)]
    assert arrows.get_cmap().name == "viridis"
    np.testing.assert_allclose(arrows.get_array(), 1.0)
    plotter.close()


def _periodic_surface(energy) -> FermiSurface:
    """The E_F = 0.1 surface of ``energy(k)`` on a 16^3 cubic mesh with B = I."""
    n = 16
    frac = np.arange(n) / n
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    energies = np.stack([energy(kpoints), np.full(len(kpoints), 5.0)], axis=1)
    return FermiSurface.from_ebs(
        ElectronicBandStructureMesh(
            kgrid_info=KGridInfo(kgrid=(n, n, n), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0, 0, 0)),
            kpoints=kpoints,
            bands=energies[..., np.newaxis],
            projected=np.ones((len(kpoints), 2, 1, 1, 1)),
            fermi=0.1,
            reciprocal_lattice=np.eye(3),
            orbital_names=["s"],
        )
    )


def test_orbit_inside_the_zone_keeps_its_area():
    def sphere_around_gamma(k):
        to_gamma = (k + 0.5) % 1.0 - 0.5
        return np.sum(to_gamma**2, axis=1)

    text = _slice_text(_periodic_surface(sphere_around_gamma), show_cross_section_area=True)

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(np.pi * 0.1 * (2 * np.pi) ** 2, rel=0.02)


def test_orbit_around_the_zone_corner_closes_across_the_zone_boundary():
    def cylinder_around_m(k):
        to_m = k[:, :2] % 1.0 - 0.5
        return np.sum(to_m**2, axis=1)

    text = _slice_text(_periodic_surface(cylinder_around_m), show_cross_section_area=True)

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(np.pi * 0.1 * (2 * np.pi) ** 2, rel=0.02)


def test_sheets_that_run_through_the_zone_stay_open():
    def planes_at_ky(k):
        to_gamma = (k[:, 1] + 0.5) % 1.0 - 0.5
        return to_gamma**2

    text = _slice_text(_periodic_surface(planes_at_ky), show_cross_section_area=True)

    assert text == "Cross sectional area : 0.0000 Ang^-2 (2 open curves not counted)"


def test_offset_oblique_cut_finds_the_orbit_in_the_neighbouring_zones():
    """The plane kx + ky = 3/4 cuts the sphere |k - M| = sqrt(0.1) mostly outside the first zone.

    Its distance to M is 1 / (4 sqrt 2), so the orbit has area pi (0.1 - 1/32).
    """

    def sphere_around_m(k):
        to_m = k % 1.0 - np.array([0.5, 0.5, 0.0])
        to_m[:, 2] = (k[:, 2] + 0.5) % 1.0 - 0.5
        return np.sum(to_m**2, axis=1)

    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(
        _periodic_surface(sphere_around_m),
        normal=(1, 1, 0),
        origin=(0.5, 0.25, 0),
        show_cross_section_area=True,
    )
    text = _area_text(plotter)
    plotter.close()

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(np.pi * (0.1 - 1 / 32) * (2 * np.pi) ** 2, rel=0.03)


@pytest.mark.data
def test_srvo3_band_16_offset_oblique_cut_finds_both_orbits():
    """SrVO3 band 16 on the plane kx + ky = 10/21 (fractional), normal (1, 1, 0).

    The plane holds the grid points (i, 10 - i, k). A matplotlib contour of the parsed
    energies on that periodic in-plane grid has two closed orbits per in-plane cell, each
    0.1632 of the cell |b1 - b2| |b3| = sqrt(2) / a^2.
    """
    a = 3.84652
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/non-spin-polarized"
    )
    band_16 = fs.select_bands([(16, 0)])
    assert isinstance(band_16, FermiSurface)

    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(
        band_16, normal=(1, 1, 0), origin=(10 / (21 * a), 0, 0), show_cross_section_area=True
    )
    text = _area_text(plotter)
    plotter.close()

    assert text.endswith(" Ang^-2")
    expected = 2 * 0.1632 * np.sqrt(2) / a**2 * (2 * np.pi) ** 2
    assert _number(text) == pytest.approx(expected, rel=0.03)


@pytest.mark.data
def test_srvo3_band_16_orbit_around_m_closes_across_the_zone_boundary():
    """SrVO3 band 16 at kz = 0 meets the zone boundary four times around M.

    The expected area is a matplotlib contour of the parsed band 16 energies on the
    exact i/21 mesh, tiled so the orbit around M = (1/2, 1/2, 0) lies inside one
    domain: 0.020664 1/A^2 without 2 pi.
    """
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/non-spin-polarized"
    )

    band_16 = fs.select_bands([(16, 0)])
    assert isinstance(band_16, FermiSurface)

    text = _slice_text(band_16, show_cross_section_area=True)

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(0.020664 * (2 * np.pi) ** 2, rel=0.01)


@pytest.mark.parametrize(
    ("flag", "text"),
    [
        ("show_cross_section_area", "Cross sectional area : 0.0000 Ang^-2"),
        ("show_van_alphen_frequency", "Van Alphen Frequency : no closed orbit"),
    ],
)
def test_moving_the_plane_to_an_empty_cut_clears_the_previous_result(flag, text):
    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(_sphere(0.0), normal=(1, 0, 0), origin=(0, 0, 0), **{flag: True})
    widget = plotter.plane_widgets[0]

    widget.SetOrigin(0.32, 0.0, 0.0)
    widget.InvokeEvent("EndInteractionEvent")

    assert widget.GetOrigin()[0] > RADIUS
    assert _area_text(plotter) == text
    assert "slice" not in plotter.actors
    plotter.close()
