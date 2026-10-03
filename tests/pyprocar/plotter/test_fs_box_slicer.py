from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv

from pyprocar.plotter.fs_plot import FS_AREA_SCALE_FACTOR, FermiPlotter


def test_box_slicer_draws_the_plane_slice_and_its_area():
    sphere = pv.Sphere(radius=0.5, theta_resolution=60, phi_resolution=60)
    plotter = FermiPlotter(off_screen=True)

    plotter.add_box_slicer(sphere, normal=(0, 0, 1), origin=(0, 0, 0), show_cross_section_area=True)

    slice_mesh = cast(pv.Actor, plotter.actors["slice"]).mapper.dataset
    np.testing.assert_allclose(slice_mesh.points[:, 2], 0.0, atol=1e-12)
    area = np.pi * 0.5**2 * FS_AREA_SCALE_FACTOR
    text = cast(pv.CornerAnnotation, plotter.actors["area_text"]).GetText(2)
    assert text.startswith("Cross sectional area : ")
    assert float(text.split(":")[1].split()[0]) == pytest.approx(area, rel=0.01)
    plotter.close()


def _inked_pixels(path) -> int:
    image = plt.imread(path)[..., :3]
    return int(np.count_nonzero(image.min(axis=-1) < 0.5))


def test_box_slicer_saves_the_3d_view_and_the_2d_slice(tmp_path):
    sphere = pv.Sphere(radius=0.5, theta_resolution=60, phi_resolution=60)
    plotter = FermiPlotter(off_screen=True)
    view, cut = tmp_path / "view.png", tmp_path / "slice.png"

    plotter.add_box_slicer(
        sphere, normal=(0, 0, 1), origin=(0, 0, 0), save_2d=view, save_2d_slice=cut
    )

    assert _inked_pixels(view) > 1000
    assert _inked_pixels(cut) > 100
    plotter.close()


def test_box_slicer_save_2d_needs_an_off_screen_plotter(tmp_path):
    sphere = pv.Sphere(radius=0.5)
    plotter = FermiPlotter(off_screen=False)

    with pytest.raises(ValueError, match="off_screen=True"):
        plotter.add_box_slicer(sphere, save_2d=tmp_path / "view.png")

    assert "slice" not in plotter.actors
    plotter.close()


def test_saved_slice_uses_the_surface_cmap_and_clim(tmp_path):
    sphere = pv.Sphere(radius=0.5, theta_resolution=60, phi_resolution=60)
    sphere.point_data["weight"] = np.ones(sphere.n_points)
    sphere.set_active_scalars("weight")
    plotter = FermiPlotter(off_screen=True)
    cut = tmp_path / "slice.png"

    plotter.add_box_slicer(
        sphere,
        normal=(0, 0, 1),
        save_2d_slice=cut,
        add_surface_args={"cmap": "Greens", "clim": (0.0, 1.0)},
    )

    image = plt.imread(cut)[..., :3]
    top_green = np.array([0.0, 0.267, 0.106])
    assert np.count_nonzero(np.abs(image - top_green).max(axis=-1) < 0.05) > 100
    plotter.close()


@pytest.mark.parametrize("method", ["add_box_slicer", "add_slicer"])
def test_slicers_accept_an_origin_outside_the_surface(method):
    sphere = pv.Sphere(radius=0.5)
    plotter = FermiPlotter(off_screen=True)

    getattr(plotter, method)(sphere, normal=(0, 0, 1), origin=(0, 0, 2.0))

    assert plotter.plane_widgets[0].GetOrigin() == (0.0, 0.0, 2.0)
    assert "slice" not in plotter.actors
    plotter.close()
