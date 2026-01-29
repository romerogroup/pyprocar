"""Type stubs for matplotlib.colors."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class Colormap:
    """Colormap base class."""

    name: str
    N: int

    def __call__(
        self,
        X: float | npt.NDArray[np.float64],
        alpha: float | None = ...,
        bytes: bool = ...,
    ) -> tuple[float, float, float, float] | npt.NDArray[np.float64]: ...

class Normalize:
    """Normalize data to [0, 1]."""

    def __init__(
        self,
        vmin: float | None = ...,
        vmax: float | None = ...,
        clip: bool = ...,
    ) -> None: ...
    def __call__(
        self, value: float | npt.NDArray[np.float64], clip: bool | None = ...
    ) -> float | npt.NDArray[np.float64]: ...

class _ColorConverter:
    """Color converter singleton."""

    def to_rgba(
        self,
        c: str | tuple[float, ...] | Sequence[float],
        alpha: float | None = ...,
    ) -> tuple[float, float, float, float]: ...
    def to_rgb(
        self,
        c: str | tuple[float, ...] | Sequence[float],
    ) -> tuple[float, float, float]: ...

colorConverter: _ColorConverter

def to_rgba(
    c: str | tuple[float, ...] | Sequence[float],
    alpha: float | None = ...,
) -> tuple[float, float, float, float]: ...

def to_rgb(
    c: str | tuple[float, ...] | Sequence[float],
) -> tuple[float, float, float]: ...

class LinearSegmentedColormap(Colormap):
    """Colormap created from a list of linear segments."""

    @classmethod
    def from_list(
        cls,
        name: str,
        colors: Sequence[str] | Sequence[tuple[float, ...]] | Sequence[tuple[float, str]],
        N: int = ...,
        gamma: float = ...,
    ) -> LinearSegmentedColormap: ...

__all__ = ["Colormap", "LinearSegmentedColormap", "Normalize", "colorConverter", "to_rgba", "to_rgb"]
