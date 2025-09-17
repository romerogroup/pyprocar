import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from matplotlib.collections import LineCollection

from pyprocar.plotter.dos_plots import (
    DOSPlotter,
    ParametricLinePlot,
    ParametricPlot,
)
from tests.pyprocar.core.test_dos import dos_spin_polarized, rng


def test_parametric_plot_simple_interface(dos_spin_polarized):
    dataset = dos_spin_polarized.build_parametric_dataset(spins=[0])
    plotter = ParametricPlot(show_colorbar=False)
    plotter.plot(
        dataset.energies,
        dataset.totals,
        dataset.normalized(),
        labels=dataset.spin_labels,
    )
    assert "dos_total_0" in plotter.values_dict


def test_parametric_plot_scaling_behaviour():
    energies = [0.0, 1.0, 2.0]
    dos_values = np.array([[1.0, 2.0, 3.0]])
    scalars = np.array([[0.2, 0.4, 0.6]])

    plot_plain = ParametricPlot(show_colorbar=False)
    plot_plain.plot(energies, dos_values, scalars, labels=["up"], scale=False)
    assert np.allclose(plot_plain.values_dict["dos_weight_0"], scalars[0])

    plot_scaled = ParametricPlot(show_colorbar=False)
    plot_scaled.plot(energies, dos_values, scalars, labels=["up"], scale=True)
    assert np.allclose(plot_scaled.values_dict["dos_weight_0"], dos_values[0] * scalars[0])


def test_dos_plotter_parametric_wraps_dataset(dos_spin_polarized):
    dataset = dos_spin_polarized.build_parametric_dataset(spins=[0])

    plotter = DOSPlotter(show_fermi=False)
    plotter.parametric(
        dataset.energies,
        dataset.totals,
        dataset.normalized(),
        labels=dataset.spin_labels,
    )

    assert plotter.plotters, "Expected DOSPlotter to register a ParametricPlot instance"
    inner = plotter.plotters[-1]
    assert isinstance(inner, ParametricPlot)
    assert "dos_total_0" in inner.values_dict


def test_parametric_line_plot_creates_collections():
    energies = [0.0, 1.0, 2.0, 3.0]
    dos_values = np.array([[0.0, 1.0, 2.0, 3.0]])
    scalars = np.array([[0.1, 0.2, 0.3, 0.4]])

    plotter = ParametricLinePlot(show_colorbar=False)
    plotter.plot(energies, dos_values, scalars, labels=["line"])

    collections = [c for c in plotter.ax.collections if isinstance(c, LineCollection)]
    assert collections, "Expected a LineCollection to be added to the axis"


def test_dos_plotter_parametric_line(dos_spin_polarized):
    dataset = dos_spin_polarized.build_parametric_dataset(spins=[0])

    plotter = DOSPlotter(show_fermi=False)
    plotter.parametric_line(
        dataset.energies,
        dataset.totals,
        dataset.normalized(),
        labels=dataset.spin_labels,
    )

    assert isinstance(plotter.plotters[-1], ParametricLinePlot)
