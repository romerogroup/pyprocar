"""Type stubs for matplotlib.cm (colormap module)."""

import numpy as np
import numpy.typing as npt

from matplotlib.colors import Colormap, Normalize

class ScalarMappable:
    """Mixin class to map scalar data to RGBA."""

    def __init__(
        self,
        norm: Normalize | None = ...,
        cmap: Colormap | str | None = ...,
    ) -> None: ...
    def to_rgba(
        self,
        x: float | npt.NDArray[np.float64],
        alpha: float | None = ...,
        bytes: bool = ...,
        norm: bool = ...,
    ) -> tuple[float, float, float, float] | npt.NDArray[np.float64]: ...
    def set_array(self, A: npt.ArrayLike | None) -> None: ...
    def get_array(self) -> npt.NDArray[np.float64] | None: ...
    def set_clim(self, vmin: float | None = ..., vmax: float | None = ...) -> None: ...
    def get_clim(self) -> tuple[float, float]: ...

# Standard colormaps
bwr: Colormap
coolwarm: Colormap
jet: Colormap
viridis: Colormap
plasma: Colormap
inferno: Colormap
magma: Colormap
cividis: Colormap
hot: Colormap
cool: Colormap
spring: Colormap
summer: Colormap
autumn: Colormap
winter: Colormap
gray: Colormap
bone: Colormap
copper: Colormap
pink: Colormap
hsv: Colormap
rainbow: Colormap
seismic: Colormap
spectral: Colormap

def get_cmap(name: str | None = ..., lut: int | None = ...) -> Colormap: ...

__all__ = ["ScalarMappable", "get_cmap", "bwr", "coolwarm", "jet", "viridis"]
