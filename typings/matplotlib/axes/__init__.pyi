"""Type stubs for matplotlib.axes."""

from collections.abc import Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

from matplotlib.collections import LineCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.quiver import Quiver
from matplotlib.ticker import Locator

class _Transform:
    """Matplotlib Transform (simplified)."""
    def inverted(self) -> _Transform: ...
    def transform(self, values: object) -> object: ...
    def transform_bbox(self, bbox: object) -> _Bbox: ...

class _Bbox:
    """Matplotlib Bbox (simplified)."""
    width: float
    height: float
    x0: float
    y0: float
    x1: float
    y1: float

class _Text:
    """Matplotlib Text artist (simplified)."""
    def get_window_extent(self, renderer: object | None = ...) -> _Bbox: ...
    def remove(self) -> None: ...
    def set_label(self, s: str) -> None: ...
    def get_text(self) -> str: ...

class _Axis:
    """Matplotlib Axis (simplified)."""
    def set_major_locator(self, locator: Locator) -> None: ...
    def set_minor_locator(self, locator: Locator) -> None: ...
    def set_major_formatter(self, formatter: object) -> None: ...
    def set_minor_formatter(self, formatter: object) -> None: ...

class Axes:
    """Matplotlib Axes class."""

    def get_figure(self) -> Figure | None: ...
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
        left: float | Sequence[float] | None = ...,
        right: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool | None = ...,
        xmin: float | None = ...,
        xmax: float | None = ...,
    ) -> tuple[float, float]: ...
    def set_ylim(
        self,
        bottom: float | Sequence[float] | None = ...,
        top: float | None = ...,
        *,
        emit: bool = ...,
        auto: bool | None = ...,
        ymin: float | None = ...,
        ymax: float | None = ...,
    ) -> tuple[float, float]: ...
    def get_xlim(self) -> tuple[float, float]: ...
    def get_ylim(self) -> tuple[float, float]: ...
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
    def get_xticks(self, *, minor: bool = ...) -> npt.NDArray[np.float64]: ...
    def get_yticks(self, *, minor: bool = ...) -> npt.NDArray[np.float64]: ...
    def get_xticklabels(self, *, minor: bool = ...) -> list[_Text]: ...
    def get_yticklabels(self, *, minor: bool = ...) -> list[_Text]: ...
    def tick_params(
        self,
        axis: str = ...,
        *,
        which: str = ...,
        reset: bool = ...,
        direction: str = ...,
        length: float = ...,
        width: float = ...,
        color: str = ...,
        pad: float = ...,
        labelsize: float | str = ...,
        labelcolor: str = ...,
        labelfontfamily: str = ...,
        colors: str = ...,
        zorder: float = ...,
        bottom: bool = ...,
        top: bool = ...,
        left: bool = ...,
        right: bool = ...,
        labelbottom: bool = ...,
        labeltop: bool = ...,
        labelleft: bool = ...,
        labelright: bool = ...,
        labelrotation: float = ...,
        grid_color: str = ...,
        grid_alpha: float = ...,
        grid_linewidth: float = ...,
        grid_linestyle: str = ...,
        **kwargs: object,
    ) -> None: ...
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
    def add_patch(self, p: Patch) -> Patch: ...
    def legend(
        self,
        *args: object,
        **kwargs: object,
    ) -> object: ...
    def set_xlabel(self, xlabel: str, **kwargs: object) -> object: ...
    def set_ylabel(self, ylabel: str, **kwargs: object) -> object: ...
    def set_title(self, label: str, **kwargs: object) -> object: ...
    def get_xlabel(self) -> str: ...
    def get_ylabel(self) -> str: ...
    def get_title(self) -> str: ...
    def get_legend(self) -> object | None: ...
    @property
    def texts(self) -> list[_Text]: ...
    @property
    def images(self) -> list[object]: ...
    def grid(self, visible: bool | None = ..., **kwargs: object) -> None: ...
    def pcolormesh(
        self,
        *args: npt.NDArray[np.float64] | npt.NDArray[np.floating[object]] | None,
        cmap: str | object | None = ...,
        norm: object | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        shading: str | None = ...,
        **kwargs: object,
    ) -> object: ...
    def contour(
        self,
        *args: npt.NDArray[np.float64] | npt.NDArray[np.floating[object]] | None,
        levels: int | Sequence[float] | None = ...,
        cmap: str | object | None = ...,
        norm: object | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        **kwargs: object,
    ) -> object: ...
    def contourf(
        self,
        *args: npt.NDArray[np.float64] | npt.NDArray[np.floating[object]] | None,
        levels: int | Sequence[float] | None = ...,
        cmap: str | object | None = ...,
        norm: object | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        alpha: float | None = ...,
        **kwargs: object,
    ) -> object: ...
    def imshow(
        self,
        X: npt.ArrayLike,
        *,
        cmap: str | object | None = ...,
        norm: object | None = ...,
        aspect: str | float | None = ...,
        interpolation: str | None = ...,
        alpha: float | npt.ArrayLike | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        origin: str | None = ...,
        extent: Sequence[float] | None = ...,
        filternorm: bool = ...,
        filterrad: float = ...,
        resample: bool | None = ...,
        url: str | None = ...,
        zorder: float | None = ...,
        clim: tuple[float | None, float | None] | None = ...,
        **kwargs: object,
    ) -> AxesImage: ...
    def fill_between(
        self,
        x: npt.ArrayLike,
        y1: npt.ArrayLike,
        y2: npt.ArrayLike | float = ...,
        where: npt.ArrayLike | None = ...,
        interpolate: bool = ...,
        step: str | None = ...,
        **kwargs: object,
    ) -> object: ...
    def fill_betweenx(
        self,
        y: npt.ArrayLike,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike | float = ...,
        where: npt.ArrayLike | None = ...,
        interpolate: bool = ...,
        step: str | None = ...,
        **kwargs: object,
    ) -> object: ...
    def annotate(
        self,
        text: str,
        xy: tuple[float, float],
        xytext: tuple[float, float] | None = ...,
        xycoords: str | tuple[str, str] | object = ...,
        textcoords: str | tuple[str, str] | object | None = ...,
        arrowprops: dict[str, object] | None = ...,
        annotation_clip: bool | None = ...,
        **kwargs: object,
    ) -> object: ...
    def set_aspect(
        self,
        aspect: str | float,
        adjustable: str | None = ...,
        anchor: str | tuple[float, float] | None = ...,
        share: bool = ...,
    ) -> None: ...
    def get_position(self, original: bool = ...) -> object: ...
    def set_position(
        self,
        pos: Sequence[float] | object,
        which: str = ...,
    ) -> None: ...
    def sharey(self, other: "Axes") -> None: ...
    def get_yaxis(self) -> object: ...
    def scatter(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        s: float | npt.ArrayLike | None = ...,
        c: npt.ArrayLike | Sequence[str] | str | None = ...,
        marker: str | None = ...,
        cmap: str | object | None = ...,
        norm: object | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        alpha: float | None = ...,
        linewidths: float | Sequence[float] | None = ...,
        edgecolors: str | Sequence[str] | None = ...,
        label: str | None = ...,
        **kwargs: object,
    ) -> PathCollection: ...
    def quiver(
        self,
        *args: npt.ArrayLike,
        **kwargs: object,
    ) -> Quiver: ...
    def autoscale_view(
        self,
        tight: bool | None = ...,
        scalex: bool = ...,
        scaley: bool = ...,
    ) -> None: ...
    @property
    def lines(self) -> list[Line2D]: ...
    @property
    def collections(self) -> list[LineCollection | PathCollection]: ...
    @property
    def axes(self) -> "Axes": ...
    @property
    def figure(self) -> object: ...
    @figure.setter
    def figure(self, fig: object) -> None: ...
    @property
    def xaxis(self) -> _Axis: ...
    @property
    def yaxis(self) -> _Axis: ...
    @property
    def transData(self) -> _Transform: ...
    def text(
        self,
        x: float,
        y: float,
        s: str,
        fontdict: dict[str, object] | None = ...,
        **kwargs: object,
    ) -> _Text: ...

__all__ = ["Axes"]
