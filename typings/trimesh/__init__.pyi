"""Type stubs for trimesh library."""

from _typeshed import Incomplete

import numpy as np
import numpy.typing as npt

from trimesh import path as path

class Trimesh:
    """Trimesh mesh class."""

    vertices: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int64]

    def __init__(
        self,
        vertices: npt.NDArray[np.float64] | None = ...,
        faces: npt.NDArray[np.int64] | None = ...,
        **kwargs: Incomplete,
    ) -> None: ...

def __getattr__(name: str) -> Incomplete: ...
