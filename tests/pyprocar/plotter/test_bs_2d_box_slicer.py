from types import SimpleNamespace
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pyvista as pv

from pyprocar.plotter.bs_2d_plot import BS2DPlotter


def test_box_slicer_draws_the_plane_slice_and_its_area():
    sphere = pv.Sphere(radius=0.5, theta_resolution=60, phi_resolution=60)
    sphere.point_data["energy"] = sphere.points[:, 2]
    sphere.set_active_scalars("energy")
    plotter = BS2DPlotter(SimpleNamespace(), off_screen=True)

    plotter.add_box_slicer(sphere, normal=(0, 0, 1), origin=(0, 0, 0), cross_section_area=True)

    slice_mesh = cast(pv.Actor, plotter.actors["slice"]).mapper.dataset
    np.testing.assert_allclose(slice_mesh.points[:, 2], 0.0, atol=1e-12)
    text = cast(pv.CornerAnnotation, plotter.actors["area_text"]).GetText(2)
    assert text.startswith("Cross sectional area : ")
    assert float(text.split(":")[1].split()[0]) == pytest.approx(np.pi * 0.5**2, rel=0.01)
    plotter.close()


def test_box_slicer_saves_the_3d_view_and_the_2d_slice(tmp_path):
    sphere = pv.Sphere(radius=0.5, theta_resolution=60, phi_resolution=60)
    sphere.point_data["energy"] = sphere.points[:, 2]
    sphere.set_active_scalars("energy")
    plotter = BS2DPlotter(SimpleNamespace(), off_screen=True)
    view, cut = tmp_path / "view.png", tmp_path / "slice.png"

    plotter.add_box_slicer(sphere, normal=(0, 0, 1), save_2d=view, save_2d_slice=cut)

    for path, minimum in [(view, 1000), (cut, 100)]:
        image = plt.imread(path)[..., :3]
        assert np.count_nonzero(image.min(axis=-1) < 0.5) > minimum
    plotter.close()


def test_box_slicer_accepts_an_origin_outside_the_energy_range():
    sphere = pv.Sphere(radius=0.5)
    sphere.point_data["energy"] = sphere.points[:, 2]
    plotter = BS2DPlotter(SimpleNamespace(), off_screen=True)

    plotter.add_box_slicer(sphere, normal=(0, 0, 1), origin=(0, 0, 2.0), cross_section_area=True)

    assert plotter.plane_widgets[0].GetOrigin() == (0.0, 0.0, 2.0)
    assert "slice" not in plotter.actors
    plotter.close()


def test_moving_the_plane_to_an_empty_cut_clears_the_previous_area():
    sphere = pv.Sphere(radius=0.5)
    sphere.point_data["energy"] = sphere.points[:, 2]
    plotter = BS2DPlotter(SimpleNamespace(), off_screen=True)
    plotter.add_box_slicer(sphere, normal=(0, 0, 1), origin=(0, 0, 0), cross_section_area=True)
    widget = plotter.plane_widgets[0]

    widget.SetOrigin(0.0, 0.0, 0.52)
    widget.InvokeEvent("EndInteractionEvent")

    assert widget.GetOrigin()[2] > 0.5
    text = cast(pv.CornerAnnotation, plotter.actors["area_text"]).GetText(2)
    assert text == "Cross sectional area : 0.0000 Ang^-2"
    assert "slice" not in plotter.actors
    plotter.close()
