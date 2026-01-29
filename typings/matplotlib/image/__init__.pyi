"""Type stubs for matplotlib.image."""

from matplotlib.patches import Patch

class AxesImage:
    """Image displayed on axes."""

    def set_clip_path(self, path: Patch | None, transform: object | None = ...) -> None: ...
    def set_clim(self, vmin: float | None = ..., vmax: float | None = ...) -> None: ...
    def get_clim(self) -> tuple[float, float]: ...

__all__ = ["AxesImage"]
