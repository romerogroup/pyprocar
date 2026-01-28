"""Type stubs for trimesh.path module."""

from _typeshed import Incomplete

import numpy as np
import numpy.typing as npt

from trimesh.path import entities as entities

class Path3D:
    """3D path representation."""

    def __init__(
        self,
        entities: list[entities.Line] | None = ...,
        vertices: npt.NDArray[np.float64] | None = ...,
        **kwargs: Incomplete,
    ) -> None: ...

def __getattr__(name: str) -> Incomplete: ...
