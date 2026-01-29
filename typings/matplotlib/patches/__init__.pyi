"""Type stubs for matplotlib.patches."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class Patch:
    """Base class for patches."""

    def __init__(
        self,
        *,
        edgecolor: str | tuple[float, ...] | None = ...,
        facecolor: str | tuple[float, ...] | None = ...,
        color: str | tuple[float, ...] | None = ...,
        linewidth: float | None = ...,
        linestyle: str | None = ...,
        antialiased: bool | None = ...,
        hatch: str | None = ...,
        fill: bool = ...,
        capstyle: str | None = ...,
        joinstyle: str | None = ...,
        label: str | None = ...,
        alpha: float | None = ...,
        **kwargs: object,
    ) -> None: ...
    def set_clip_path(self, path: object | None, transform: object | None = ...) -> None: ...
    def get_path(self) -> object: ...
    def set_label(self, s: str) -> None: ...

class Polygon(Patch):
    """A polygon patch."""

    def __init__(
        self,
        xy: npt.NDArray[np.float64] | Sequence[Sequence[float]],
        closed: bool = ...,
        *,
        fill: bool = ...,
        facecolor: str | tuple[float, ...] | None = ...,
        edgecolor: str | tuple[float, ...] | None = ...,
        linewidth: float | None = ...,
        linestyle: str | None = ...,
        alpha: float | None = ...,
        label: str | None = ...,
        zorder: float | None = ...,
        **kwargs: object,
    ) -> None: ...

class Rectangle(Patch):
    """A rectangle patch."""

    def __init__(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        *,
        angle: float = ...,
        rotation_point: str | tuple[float, float] = ...,
        fill: bool = ...,
        facecolor: str | tuple[float, ...] | None = ...,
        edgecolor: str | tuple[float, ...] | None = ...,
        linewidth: float | None = ...,
        linestyle: str | None = ...,
        **kwargs: object,
    ) -> None: ...

class FancyBboxPatch(Patch):
    """A fancy box patch."""

    def __init__(
        self,
        xy: tuple[float, float],
        width: float,
        height: float,
        boxstyle: str = ...,
        *,
        mutation_scale: float = ...,
        mutation_aspect: float = ...,
        **kwargs: object,
    ) -> None: ...

__all__ = ["Patch", "Polygon", "Rectangle", "FancyBboxPatch"]
