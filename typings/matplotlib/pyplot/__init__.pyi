"""Type stubs for matplotlib.pyplot."""

from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | str = ...,
    sharey: bool | str = ...,
    squeeze: bool = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, object] | None = ...,
    gridspec_kw: dict[str, object] | None = ...,
    **fig_kw: object,
) -> tuple[Figure, Axes] | tuple[Figure, npt.NDArray[np.object_]]: ...

def figure(
    num: int | str | Figure | None = ...,
    figsize: tuple[float, float] | None = ...,
    dpi: float | None = ...,
    facecolor: str | tuple[float, ...] | None = ...,
    edgecolor: str | tuple[float, ...] | None = ...,
    frameon: bool = ...,
    FigureClass: type[Figure] = ...,
    clear: bool = ...,
    **kwargs: object,
) -> Figure: ...

def plot(
    *args: npt.NDArray[np.float64] | Sequence[float],
    **kwargs: object,
) -> list[Line2D]: ...

def show(*, block: bool | None = ...) -> None: ...

def savefig(
    fname: str,
    *,
    dpi: float | str | None = ...,
    facecolor: str | tuple[float, ...] | None = ...,
    edgecolor: str | tuple[float, ...] | None = ...,
    orientation: str = ...,
    papertype: str | None = ...,
    format: str | None = ...,
    transparent: bool = ...,
    bbox_inches: str | object | None = ...,
    pad_inches: float | None = ...,
    metadata: dict[str, object] | None = ...,
    **kwargs: object,
) -> None: ...

def close(fig: Figure | str | int | None = ...) -> None: ...

def xlabel(xlabel: str, **kwargs: object) -> object: ...
def ylabel(ylabel: str, **kwargs: object) -> object: ...
def title(label: str, **kwargs: object) -> object: ...
def legend(*args: object, **kwargs: object) -> object: ...
def grid(visible: bool | None = ..., **kwargs: object) -> None: ...
def xlim(
    left: float | None = ..., right: float | None = ...
) -> tuple[float, float]: ...
def ylim(
    bottom: float | None = ..., top: float | None = ...
) -> tuple[float, float]: ...

def gca(**kwargs: object) -> Axes: ...
def gcf() -> Figure: ...

__all__ = [
    "subplots",
    "figure",
    "plot",
    "show",
    "savefig",
    "close",
    "xlabel",
    "ylabel",
    "title",
    "legend",
    "grid",
    "xlim",
    "ylim",
    "gca",
    "gcf",
]
