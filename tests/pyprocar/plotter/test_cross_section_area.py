from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import pyvista as pv
from matplotlib.quiver import Quiver

from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from pyprocar.plotter.fs_plot import FS_AREA_SCALE_FACTOR, FermiPlotter, dHvA_frequency

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


def test_open_curves_are_reported_and_not_counted():
    bowl = cast(pv.PolyData, _sphere(-0.5).clip(normal=(1, 0, 0), origin=(-0.6, 0, 0)))
    plotter = FermiPlotter(off_screen=True)

    plotter.add_box_slicer(bowl.merge(_sphere(0.5)), normal=(0, 0, 1), show_cross_section_area=True)

    text = _area_text(plotter)
    assert text.endswith(" Ang^-2 (1 open curve not counted)")
    assert _number(text) == pytest.approx(np.pi * RADIUS**2 * FS_AREA_SCALE_FACTOR, rel=1e-3)
    plotter.close()


def _bowl() -> pv.PolyData:
    return cast(pv.PolyData, _sphere(-0.5).clip(normal=(1, 0, 0), origin=(-0.6, 0, 0)))


def _van_alphen_text(surface: pv.PolyData, origin=(0, 0, 0)) -> str:
    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(surface, normal=(0, 0, 1), origin=origin, show_van_alphen_frequency=True)
    text = _area_text(plotter)
    plotter.close()
    return text


def test_van_alphen_frequency_without_a_closed_orbit_says_so():
    text = _van_alphen_text(_bowl(), origin=(-0.7, 0, 0))

    assert text == "Van Alphen Frequency : no closed orbit (1 open curve not counted)"


def test_van_alphen_frequency_reports_the_open_curves_it_skipped():
    text = _van_alphen_text(_bowl().merge(_sphere(0.5)))

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
