"""Type stubs for matplotlib.colorbar."""

from matplotlib.axes import Axes

class Colorbar:
    """Matplotlib Colorbar class."""

    ax: Axes

    def set_label(self, label: str, **kwargs: object) -> None: ...

__all__ = ["Colorbar"]
