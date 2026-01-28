"""Type stubs for matplotlib.lines."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class Line2D:
    """A line in 2D space."""

    def __init__(
        self,
        xdata: Sequence[float] | npt.NDArray[np.float64],
        ydata: Sequence[float] | npt.NDArray[np.float64],
        *,
        linewidth: float | None = ...,
        linestyle: str | None = ...,
        color: str | tuple[float, ...] | None = ...,
        marker: str | None = ...,
        markersize: float | None = ...,
        markeredgewidth: float | None = ...,
        markeredgecolor: str | tuple[float, ...] | None = ...,
        markerfacecolor: str | tuple[float, ...] | None = ...,
        antialiased: bool | None = ...,
        dash_capstyle: str | None = ...,
        solid_capstyle: str | None = ...,
        dash_joinstyle: str | None = ...,
        solid_joinstyle: str | None = ...,
        pickradius: float = ...,
        drawstyle: str | None = ...,
        fillstyle: str | None = ...,
        **kwargs: object,
    ) -> None: ...
    def get_xdata(self, orig: bool = ...) -> npt.NDArray[np.float64]: ...
    def get_ydata(self, orig: bool = ...) -> npt.NDArray[np.float64]: ...
    def set_xdata(self, x: Sequence[float] | npt.NDArray[np.float64]) -> None: ...
    def set_ydata(self, y: Sequence[float] | npt.NDArray[np.float64]) -> None: ...
    def set_data(
        self,
        x: Sequence[float] | npt.NDArray[np.float64],
        y: Sequence[float] | npt.NDArray[np.float64],
    ) -> None: ...
    def get_color(self) -> str: ...
    def set_color(self, color: str | tuple[float, ...]) -> None: ...
    def get_linewidth(self) -> float: ...
    def set_linewidth(self, w: float) -> None: ...
    def get_linestyle(self) -> str: ...
    def set_linestyle(self, ls: str) -> None: ...
    def get_label(self) -> str: ...
    def set_label(self, s: str) -> None: ...

__all__ = ["Line2D"]
