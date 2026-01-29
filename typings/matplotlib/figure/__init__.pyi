"""Type stubs for matplotlib.figure."""

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar

class Figure:
    """Matplotlib Figure class."""

    axes: list[Axes]

    def __init__(
        self,
        figsize: tuple[float, float] | None = ...,
        dpi: float | None = ...,
        facecolor: str | tuple[float, ...] | None = ...,
        edgecolor: str | tuple[float, ...] | None = ...,
        linewidth: float = ...,
        frameon: bool | None = ...,
        subplotpars: object | None = ...,
        tight_layout: bool | dict[str, object] | None = ...,
        constrained_layout: bool | dict[str, object] | None = ...,
        **kwargs: object,
    ) -> None: ...
    def add_subplot(
        self,
        *args: int,
        **kwargs: object,
    ) -> Axes: ...
    def add_axes(
        self,
        rect: Sequence[float] | Axes,
        **kwargs: object,
    ) -> Axes: ...
    def savefig(
        self,
        fname: str,
        *,
        dpi: float | str | None = ...,
        facecolor: str | tuple[float, ...] | None = ...,
        edgecolor: str | tuple[float, ...] | None = ...,
        orientation: str = ...,
        format: str | None = ...,
        transparent: bool = ...,
        bbox_inches: str | object | None = ...,
        pad_inches: float | None = ...,
        **kwargs: object,
    ) -> None: ...
    def tight_layout(
        self,
        *,
        pad: float = ...,
        h_pad: float | None = ...,
        w_pad: float | None = ...,
        rect: Sequence[float] | None = ...,
    ) -> None: ...
    def set_size_inches(
        self,
        w: float | tuple[float, float],
        h: float | None = ...,
        forward: bool = ...,
    ) -> None: ...
    def suptitle(
        self,
        t: str,
        **kwargs: object,
    ) -> object: ...
    def colorbar(
        self,
        mappable: object,
        cax: Axes | None = ...,
        ax: Axes | None = ...,
        **kwargs: object,
    ) -> Colorbar: ...

__all__ = ["Figure"]
