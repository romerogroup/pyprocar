"""Type stubs for matplotlib.collections."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from matplotlib.colors import Colormap

class LineCollection:
    """Collection of lines."""

    def __init__(
        self,
        segments: Sequence[Sequence[Sequence[float]]] | npt.NDArray[np.float64],
        *,
        array: Sequence[float] | npt.NDArray[np.float64] | None = ...,
        linewidths: Sequence[float] | float | None = ...,
        linewidth: float | None = ...,
        colors: (
            Sequence[str]
            | Sequence[tuple[float, float, float]]
            | Sequence[tuple[float, float, float, float]]
            | str
            | None
        ) = ...,
        antialiaseds: Sequence[bool] | bool | None = ...,
        linestyles: str | Sequence[str] = ...,
        linestyle: str | None = ...,
        offsets: npt.NDArray[np.float64] | None = ...,
        offset_transform: object | None = ...,
        norm: object | None = ...,
        cmap: object | None = ...,
        clim: tuple[float | None, float | None] | None = ...,
        pickradius: float = ...,
        zorder: float = ...,
        facecolors: str | Sequence[str] | None = ...,
        label: str | None = ...,
        alpha: float | None = ...,
        **kwargs: object,
    ) -> None: ...
    def set_array(self, A: npt.NDArray[np.float64] | None) -> None: ...
    def get_array(self) -> npt.NDArray[np.float64] | None: ...
    def set_linewidth(self, lw: Sequence[float] | npt.NDArray[np.float64] | float) -> None: ...
    def set_linewidths(self, lw: Sequence[float] | float) -> None: ...
    def set_cmap(self, cmap: object) -> None: ...
    def set_norm(self, norm: object) -> None: ...
    def set_colors(
        self,
        c: (
            Sequence[str]
            | Sequence[tuple[float, float, float]]
            | Sequence[tuple[float, float, float, float]]
            | str
        ),
    ) -> None: ...
    def get_cmap(self) -> Colormap: ...
    def get_clim(self) -> tuple[float, float]: ...

class PathCollection:
    """Collection of paths (returned by scatter)."""

    def set_array(self, A: npt.NDArray[np.float64] | None) -> None: ...
    def get_array(self) -> npt.NDArray[np.float64] | None: ...
    def get_cmap(self) -> Colormap: ...
    def get_clim(self) -> tuple[float, float]: ...

__all__ = ["LineCollection", "PathCollection"]
