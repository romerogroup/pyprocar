"""Type stubs for skimage.measure module."""

from _typeshed import Incomplete

import numpy as np
import numpy.typing as npt

def marching_cubes(
    volume: npt.NDArray[np.float64],
    level: float | None = ...,
    spacing: tuple[float, float, float] = ...,
    gradient_direction: str = ...,
    step_size: int = ...,
    allow_degenerate: bool = ...,
    method: str = ...,
    mask: npt.NDArray[np.bool_] | None = ...,
) -> tuple[
    npt.NDArray[np.float64],  # verts
    npt.NDArray[np.intp],  # faces
    npt.NDArray[np.float64],  # normals
    npt.NDArray[np.float64],  # values
]: ...

# Deprecated alias for marching_cubes
def marching_cubes_lewiner(
    volume: npt.NDArray[np.float64],
    level: float | None = ...,
    spacing: tuple[float, float, float] = ...,
    gradient_direction: str = ...,
    step_size: int = ...,
    allow_degenerate: bool = ...,
    **kwargs: Incomplete,
) -> tuple[
    npt.NDArray[np.float64],  # verts
    npt.NDArray[np.intp],  # faces
    npt.NDArray[np.float64],  # normals
    npt.NDArray[np.float64],  # values
]: ...
def find_contours(
    image: npt.NDArray[np.float64],
    level: float | None = ...,
    fully_connected: str = ...,
    positive_orientation: str = ...,
    mask: npt.NDArray[np.bool_] | None = ...,
) -> list[npt.NDArray[np.float64]]: ...
def regionprops(
    label_image: npt.NDArray[np.intp],
    intensity_image: npt.NDArray[np.float64] | None = ...,
    cache: bool = ...,
    **kwargs: Incomplete,
) -> list[Incomplete]: ...
def __getattr__(name: str) -> Incomplete: ...
