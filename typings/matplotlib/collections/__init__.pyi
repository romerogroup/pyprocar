"""Type stubs for matplotlib.collections."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class LineCollection:
    """Collection of lines."""

    def __init__(
        self,
        segments: Sequence[Sequence[Sequence[float]]] | npt.NDArray[np.float64],
        *,
        linewidths: Sequence[float] | float | None = ...,
        colors: (
            Sequence[str]
            | Sequence[tuple[float, float, float]]
            | Sequence[tuple[float, float, float, float]]
            | str
            | None
        ) = ...,
        antialiaseds: Sequence[bool] | bool | None = ...,
        linestyles: str | Sequence[str] = ...,
        offsets: npt.NDArray[np.float64] | None = ...,
        offset_transform: object | None = ...,
        norm: object | None = ...,
        cmap: object | None = ...,
        pickradius: float = ...,
        zorder: float = ...,
        facecolors: str | Sequence[str] | None = ...,
        **kwargs: object,
    ) -> None: ...
    def set_array(self, A: npt.NDArray[np.float64] | None) -> None: ...
    def get_array(self) -> npt.NDArray[np.float64] | None: ...
    def set_linewidths(self, lw: Sequence[float] | float) -> None: ...
    def set_colors(
        self,
        c: (
            Sequence[str]
            | Sequence[tuple[float, float, float]]
            | Sequence[tuple[float, float, float, float]]
            | str
        ),
    ) -> None: ...

__all__ = ["LineCollection"]
