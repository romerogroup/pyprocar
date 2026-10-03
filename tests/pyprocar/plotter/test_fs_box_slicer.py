from typing import cast

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
