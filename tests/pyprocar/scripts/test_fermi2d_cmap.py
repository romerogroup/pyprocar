import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.quiver import Quiver

import pyprocar
from pyprocar.core.fermisurface import FermiSurface
from tests.pyprocar.core.test_fermisurface_noncollinear import sphere_mesh


@pytest.fixture
def noncollinear_calc(monkeypatch):
    """fermi2D reads a non-collinear sphere whose band 0 has spin (0.1, 0.2, 0.3)."""
    projected = np.zeros((2, 4, 2, 1))
    projected[0, :, 0, 0] = [0.4, 0.1, 0.2, 0.3]
    ebs = sphere_mesh(4, projected)
    monkeypatch.setattr(
        FermiSurface, "from_code", classmethod(lambda cls, **kwargs: cls.from_ebs(ebs))
    )
    yield
    plt.close("all")


def _artist(ax, kind):
    (artist,) = [c for c in ax.collections if isinstance(c, kind)]
    return artist


def test_spin_texture_arrows_use_cmap(noncollinear_calc):
    _, ax = pyprocar.fermi2D(
        code="vasp", dirname="calc", mode="spin_texture", cmap="viridis", show=False
    )

    assert _artist(ax, Quiver).get_cmap().name == "viridis"


def test_plot_line_kwargs_cmap_colors_the_contours(noncollinear_calc):
    _, ax = pyprocar.fermi2D(
        code="vasp",
        dirname="calc",
        mode="parametric",
        atoms=[0],
        spins=[3],
        plot_line_kwargs={"cmap": "magma", "linewidths": 3.0},
        show=False,
    )

    lines = _artist(ax, LineCollection)
    assert lines.get_cmap().name == "magma"
    assert np.asarray(lines.get_linewidth()).tolist() == [3.0]


def test_plot_arrows_kwargs_cmap_colors_the_arrows(noncollinear_calc):
    _, ax = pyprocar.fermi2D(
        code="vasp",
        dirname="calc",
        mode="spin_texture",
        plot_arrows_kwargs={"cmap": "cividis", "scale": 2.0},
        show=False,
    )

    arrows = _artist(ax, Quiver)
    assert arrows.get_cmap().name == "cividis"
    assert arrows.scale == 2.0
