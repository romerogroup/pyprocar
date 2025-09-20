"""High level helper orchestrating DOS plotting backends."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Tuple

import numpy as np

from pyprocar.core.property_store import Property
from pyprocar.plotter.dos_plots.base import BasePlotter
from pyprocar.plotter.dos_plots.parametric import ParametricLinePlot, ParametricPlot

logger = logging.getLogger(__name__)


class DOSPlotter(BasePlotter):
    """Facade that mirrors the public API of the bands plotter stack."""

    def __init__(self, figsize=(6, 4), dpi: int = 100, ax=None, **kwargs) -> None:
        super().__init__(figsize=figsize, dpi=dpi, ax=ax, **kwargs)
        self._plotters: list[BasePlotter] = []

    def _plot(self, *args, **kwargs):  # pragma: no cover - not used directly
        raise NotImplementedError("Use dedicated helper methods like parametric().")

    @property
    def plotters(self) -> Sequence[BasePlotter]:
        return tuple(self._plotters)

    def _add_plotter(self, plotter: BasePlotter, *args, **kwargs):
        self._plotters.append(plotter)
        return plotter.plot(*args, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parametric(
        self,
        energies: Iterable[float],
        dos_values: Iterable[Iterable[float]] | np.ndarray,
        scalars: Iterable[Iterable[float]] | np.ndarray | None = None,
        labels: Iterable[str] | None = None,
        *,
        scale: bool = False,
        **kwargs,
    ):
        plotter = ParametricPlot(ax=self.ax, **self.instance_plot_params)
        return self._add_plotter(
            plotter,
            energies,
            dos_values,
            scalars,
            labels=labels,
            scale=scale,
            **kwargs,
        )

    def parametric_line(
        self,
        energies: Iterable[float],
        dos_values: Iterable[Iterable[float]] | np.ndarray,
        scalars: Iterable[Iterable[float]] | np.ndarray | Property | None = None,
        labels: Iterable[str] | None = None,
        *,
        scale: bool = False,
        **kwargs,
    ):

        plotter = ParametricLinePlot(ax=self.ax, **self.instance_plot_params)
        return self._add_plotter(
            plotter,
            energies,
            dos_values,
            scalars,
            labels=labels,
            scale=scale,
            **kwargs,
        )
