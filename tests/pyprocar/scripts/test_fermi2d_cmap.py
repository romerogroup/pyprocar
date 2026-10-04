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

TOTAL, SX, SY, SZ = 0.4, 0.1, 0.2, 0.3

pytestmark = pytest.mark.usefixtures("noncollinear_calc")


@pytest.fixture
def noncollinear_calc(monkeypatch):
    projected = np.zeros((2, 4, 2, 1))
    projected[0, :, 0, 0] = [TOTAL, SX, SY, SZ]
    ebs = sphere_mesh(4, projected)

    def from_code(cls: type[FermiSurface], **_kwargs: object) -> FermiSurface:
        return cls.from_ebs(ebs)

    monkeypatch.setattr(FermiSurface, "from_code", classmethod(from_code))
    yield
    plt.close("all")


def _artist(ax, kind):
    (artist,) = [c for c in ax.collections if isinstance(c, kind)]
    return artist


def test_spin_texture_arrows_use_cmap():
    _, ax = pyprocar.fermi2D(
        code="vasp", dirname="calc", mode="spin_texture", cmap="viridis", show=False
    )

    assert _artist(ax, Quiver).get_cmap().name == "viridis"


def test_plot_line_kwargs_cmap_colors_the_contours():
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
    np.testing.assert_allclose(lines.get_array(), SZ)


def test_plot_arrows_kwargs_cmap_colors_the_arrows():
    _, ax = pyprocar.fermi2D(
        code="vasp",
        dirname="calc",
        mode="spin_texture",
        plot_arrows_kwargs={"cmap": "cividis", "alpha": 0.5},
        show=False,
    )

    arrows = _artist(ax, Quiver)
    assert arrows.get_cmap().name == "cividis"
    assert arrows.get_alpha() == 0.5


@pytest.mark.parametrize(
    ("spins", "value", "label"), [([1], SX, "Sx projection"), ([3], SZ, "Sz projection")]
)
def test_spin_texture_colours_arrows_and_contours_by_one_spin_component(spins, value, label):
    fig, ax = pyprocar.fermi2D(
        code="vasp", dirname="calc", mode="spin_texture", atoms=[0], spins=spins, show=False
    )

    np.testing.assert_allclose(_artist(ax, Quiver).get_array(), value)
    np.testing.assert_allclose(_artist(ax, LineCollection).get_array(), value)
    (colorbar,) = [a for a in fig.axes if a is not ax]
    assert colorbar.get_ylabel() == label


def test_spin_texture_without_spins_colours_arrows_by_the_spin_magnitude():
    _, ax = pyprocar.fermi2D(
        code="vasp", dirname="calc", mode="spin_texture", atoms=[0], show=False
    )

    np.testing.assert_allclose(_artist(ax, Quiver).get_array(), np.sqrt(SX**2 + SY**2 + SZ**2))
