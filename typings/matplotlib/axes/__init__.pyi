"""Type stubs for matplotlib.axes."""

from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

class Axes:
    """Matplotlib Axes class."""

    def plot(
        self,
        *args: npt.NDArray[np.float64] | Sequence[float],
        color: str | tuple[float, ...] | None = ...,
        linewidth: float | None = ...,
        linestyle: str | None = ...,
        marker: str | None = ...,
        alpha: float | None = ...,
        label: str | None = ...,
        **kwargs: object,
    ) -> list[Line2D]: ...
    def set_xlim(
        self,
        left: float | None = ...,
        right: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool | None = ...,
        xmin: float | None = ...,
        xmax: float | None = ...,
    ) -> tuple[float, float]: ...
    def set_ylim(
        self,
        bottom: float | None = ...,
        top: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool | None = ...,
        ymin: float | None = ...,
        ymax: float | None = ...,
    ) -> tuple[float, float]: ...
    def set_xticks(
        self,
        ticks: Sequence[float] | npt.NDArray[np.float64],
        labels: Sequence[str] | None = ...,
        *,
        minor: bool = ...,
        **kwargs: object,
    ) -> list[object]: ...
    def set_xticklabels(
        self,
        labels: Sequence[str],
        *,
        fontdict: dict[str, object] | None = ...,
        minor: bool = ...,
        **kwargs: object,
    ) -> list[object]: ...
    def set_yticks(
        self,
        ticks: Sequence[float] | npt.NDArray[np.float64],
        labels: Sequence[str] | None = ...,
        *,
        minor: bool = ...,
        **kwargs: object,
    ) -> list[object]: ...
    def set_yticklabels(
        self,
        labels: Sequence[str],
        *,
        fontdict: dict[str, object] | None = ...,
        minor: bool = ...,
        **kwargs: object,
    ) -> list[object]: ...
    def axhline(
        self,
        y: float = ...,
        xmin: float = ...,
        xmax: float = ...,
        **kwargs: object,
    ) -> Line2D: ...
    def axvline(
        self,
        x: float = ...,
        ymin: float = ...,
        ymax: float = ...,
        **kwargs: object,
    ) -> Line2D: ...
    def add_collection(
        self,
        collection: LineCollection,
        autolim: bool = ...,
    ) -> LineCollection: ...
    def legend(
        self,
        *args: object,
        **kwargs: object,
    ) -> object: ...
    def set_xlabel(self, xlabel: str, **kwargs: object) -> object: ...
    def set_ylabel(self, ylabel: str, **kwargs: object) -> object: ...
    def set_title(self, label: str, **kwargs: object) -> object: ...
    def grid(self, visible: bool | None = ..., **kwargs: object) -> None: ...

__all__ = ["Axes"]
