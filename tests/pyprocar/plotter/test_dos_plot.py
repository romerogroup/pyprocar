import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyprocar.core.dos import DensityOfStates
from pyprocar.plotter.dos_plot import DOSPlotter


def _make_dos(n_spins: int = 2) -> DensityOfStates:
    energies = np.linspace(-1.0, 1.0, 5)
    base = np.linspace(0.1, 0.5, energies.size)
    total_channels = [base + 0.2 * spin for spin in range(n_spins)]
    total = np.stack(total_channels, axis=1)

    projected = np.zeros((energies.size, n_spins, 1, 2), dtype=float)
    for spin in range(n_spins):
        projected[:, spin, 0, 0] = base + 0.05 * spin
        projected[:, spin, 0, 1] = base + 0.1 * spin

    return DensityOfStates(
        energies=energies,
        total=total,
        projected=projected,
    )


def test_plot_line_uses_metadata_labels_per_channel():
    dos = _make_dos(n_spins=2)
    projected_sum = dos.compute_projected_sum(atoms=[0], spins=[0, 1])

    plotter = DOSPlotter()
    plotter.plot(projected_sum)

    expected_labels = projected_sum.metadata["label"]
    actual_labels = [line.get_label() for line in plotter.ax.lines]

    assert actual_labels == expected_labels
    plt.close(plotter.fig)


def test_plot_line_creates_line_for_each_channel():
    dos = _make_dos(n_spins=4)
    projected_sum = dos.compute_projected_sum(atoms=[0], spins=[0, 1, 2, 3])

    plotter = DOSPlotter()
    plotter.plot(projected_sum)

    assert len(plotter.ax.lines) == projected_sum.to_array().shape[1]
    plt.close(plotter.fig)


def test_horizontal_orientation_sets_axis_labels():
    dos = _make_dos(n_spins=1)
    total_property = dos.total

    plotter = DOSPlotter(orientation="horizontal")
    plotter.plot(total_property)

    expected_x = (
        f"{total_property.points_label} ({total_property.points_units})"
        if total_property.points_units is not None
        else total_property.points_label
    )
    expected_y = (
        f"{total_property.label} ({total_property.units})"
        if total_property.units is not None
        else total_property.label
    )

    assert plotter.ax.get_xlabel() == expected_x
    assert plotter.ax.get_ylabel() == expected_y
    plt.close(plotter.fig)


def test_vertical_orientation_swaps_axes():
    dos = _make_dos(n_spins=1)
    total_property = dos.total
    total_values = total_property.to_array().ravel()
    energies = total_property.points

    plotter = DOSPlotter(orientation="vertical")
    plotter.plot(total_property)

    line = plotter.ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), total_values)
    np.testing.assert_allclose(line.get_ydata(), energies)

    expected_x = (
        f"{total_property.label} ({total_property.units})"
        if total_property.units is not None
        else total_property.label
    )
    expected_y = (
        f"{total_property.points_label} ({total_property.points_units})"
        if total_property.points_units is not None
        else total_property.points_label
    )

    assert plotter.ax.get_xlabel() == expected_x
    assert plotter.ax.get_ylabel() == expected_y
    plt.close(plotter.fig)
