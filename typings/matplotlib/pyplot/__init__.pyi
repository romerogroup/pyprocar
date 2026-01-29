"""Type stubs for matplotlib.pyplot."""

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes as Axes
from matplotlib.figure import Figure as Figure
from matplotlib.lines import Line2D

@overload
def subplots(
    nrows: Literal[1] = ...,
    ncols: Literal[1] = ...,
    *,
    sharex: bool | str = ...,
    sharey: bool | str = ...,
    squeeze: bool = ...,
    width_ratios: Sequence[float] | None = ...,
    height_ratios: Sequence[float] | None = ...,
    subplot_kw: dict[str, object] | None = ...,
    gridspec_kw: dict[str, object] | None = ...,
    **fig_kw: object,
) -> tuple[Figure, Axes]: ...
@overload
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
) -> tuple[Figure, npt.NDArray[np.object_]]: ...

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
    *args: npt.NDArray[np.float64] | npt.NDArray[np.integer[object]] | Sequence[float] | str,
    **kwargs: object,
) -> list[Line2D]: ...

def get_cmap(name: str | None = ..., lut: int | None = ...) -> Colormap: ...
def show(*, block: bool | None = ...) -> None: ...
def ioff() -> None: ...
def ion() -> None: ...

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
def clf() -> None: ...
def tight_layout(
    *,
    pad: float = ...,
    h_pad: float | None = ...,
    w_pad: float | None = ...,
    rect: Sequence[float] | None = ...,
) -> None: ...

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

class _Cycler:
    """Cycler object for property cycling."""
    ...

def cycler(
    *args: object,
    **kwargs: object,
) -> _Cycler: ...

class _RcParams(dict[str, object]):
    """Runtime configuration parameters."""
    def __setitem__(self, key: str, val: object) -> None: ...
    def __getitem__(self, key: str) -> object: ...

rcParams: _RcParams

class _ColormapRegistry:
    """Registry of colormaps accessible via plt.colormaps."""

    def __getitem__(self, name: str) -> Colormap: ...
    def __contains__(self, name: str) -> bool: ...
    def __iter__(self) -> object: ...

colormaps: _ColormapRegistry

from matplotlib.colors import Colormap as Colormap
from matplotlib.colors import Normalize as Normalize
from matplotlib.ticker import NullLocator as NullLocator

import matplotlib.cm as cm

__all__ = [
    "Axes",
    "Figure",
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
    "colormaps",
]
