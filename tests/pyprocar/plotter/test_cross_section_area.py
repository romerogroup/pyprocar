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

    expected = dHvA_frequency(np.pi * RADIUS**2 * FS_AREA_SCALE_FACTOR)
    assert _number(_area_text(plotter)) == pytest.approx(expected, rel=1e-3)
    plotter.close()


def test_open_curves_are_reported_and_not_counted():
    bowl = cast(pv.PolyData, _sphere(-0.5).clip(normal=(1, 0, 0), origin=(-0.6, 0, 0)))
    plotter = FermiPlotter(off_screen=True)

    plotter.add_box_slicer(bowl.merge(_sphere(0.5)), normal=(0, 0, 1), show_cross_section_area=True)

    text = _area_text(plotter)
    assert text.endswith(" Ang^-2 (1 open curve not counted)")
    assert _number(text) == pytest.approx(np.pi * RADIUS**2 * FS_AREA_SCALE_FACTOR, rel=1e-3)
    plotter.close()


def _band_surface_with_nan_energies() -> pv.PolyData:
    """E = kx^2 + ky^2 with NaN energies beyond |k| = 0.4, as BandStructure2D leaves them."""
    k = np.linspace(-0.5, 0.5, 41)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    energy = kx**2 + ky**2
    energy[np.hypot(kx, ky) > 0.4] = np.nan
    grid = pv.StructuredGrid(kx[..., None], ky[..., None], energy[..., None])
    surface = grid.extract_surface().triangulate()
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
    np.testing.assert_allclose(arrows.V, 1.0)
    plotter.close()
