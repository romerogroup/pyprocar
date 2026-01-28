"""Type stubs for pyvista.core.grid module."""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from pyvista.core.pointset import PolyData

_Self = TypeVar("_Self")

class ImageData:
    """PyVista ImageData class (uniform rectilinear grid)."""

    points: npt.NDArray[np.float64]
    n_points: int
    n_cells: int
    bounds: tuple[float, float, float, float, float, float]
    center: tuple[float, float, float]
    dimensions: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]

    @property
    def point_data(self) -> _DataSetAttributes: ...
    @property
    def cell_data(self) -> _DataSetAttributes: ...
    @property
    def field_data(self) -> _DataSetAttributes: ...

    def __init__(
        self,
        dimensions: tuple[int, int, int] | Sequence[int] | None = ...,
        spacing: tuple[float, float, float] | Sequence[float] = ...,
        origin: tuple[float, float, float] | Sequence[float] = ...,
        deep: bool = ...,
    ) -> None: ...
    def copy(self: _Self, deep: bool = ...) -> _Self: ...
    def contour(
        self,
        isosurfaces: int | Sequence[float] = ...,
        scalars: str | npt.NDArray[np.float64] | None = ...,
        compute_normals: bool = ...,
        compute_gradients: bool = ...,
        compute_scalars: bool = ...,
        rng: tuple[float, float] | None = ...,
        preference: str = ...,
        method: str = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
    def plot(self, **kwargs: Incomplete) -> Incomplete: ...
    def cast_to_unstructured_grid(self) -> UnstructuredGrid: ...

class _DataSetAttributes:
    """Attribute data for point/cell data."""

    def __getitem__(self, name: str) -> npt.NDArray[np.float64]: ...
    def __setitem__(self, name: str, value: npt.NDArray[np.float64]) -> None: ...
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[npt.NDArray[np.float64]]: ...
    def items(self) -> list[tuple[str, npt.NDArray[np.float64]]]: ...
    def update(self, other: dict[str, npt.NDArray[np.float64]]) -> None: ...
    @property
    def active_scalars(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def active_scalars_name(self) -> str | None: ...

class UnstructuredGrid:
    """PyVista UnstructuredGrid class."""

    points: npt.NDArray[np.float64]
    n_points: int
    n_cells: int
    bounds: tuple[float, float, float, float, float, float]
    center: tuple[float, float, float]

    @property
    def point_data(self) -> _DataSetAttributes: ...
    @property
    def cell_data(self) -> _DataSetAttributes: ...
    @property
    def field_data(self) -> _DataSetAttributes: ...

    def __init__(
        self,
        *args: Incomplete,
        deep: bool = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def copy(self: _Self, deep: bool = ...) -> _Self: ...
    def transform(
        self: _Self,
        trans: npt.NDArray[np.float64],
        transform_all_input_vectors: bool = ...,
        inplace: bool = ...,
        progress_bar: bool = ...,
    ) -> _Self: ...
    def contour(
        self,
        isosurfaces: int | Sequence[float] = ...,
        scalars: str | npt.NDArray[np.float64] | None = ...,
        compute_normals: bool = ...,
        compute_gradients: bool = ...,
        compute_scalars: bool = ...,
        rng: tuple[float, float] | None = ...,
        preference: str = ...,
        method: str = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
    def plot(self, **kwargs: Incomplete) -> Incomplete: ...
    def extract_surface(self) -> PolyData: ...
