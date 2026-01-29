"""Type stubs for pyvista.core.pointset module."""

from _typeshed import Incomplete
from collections.abc import Iterator, Sequence
from typing import TypeVar, overload

import numpy as np
import numpy.typing as npt

_Self = TypeVar("_Self")

class _PointSet:
    """Base class for point-based datasets."""

    points: npt.NDArray[np.float64]
    n_points: int
    n_cells: int
    bounds: tuple[float, float, float, float, float, float]
    center: tuple[float, float, float]
    length: float

    @property
    def centers(self) -> npt.NDArray[np.float64]: ...

    @property
    def point_data(self) -> DataSetAttributes: ...
    @property
    def cell_data(self) -> DataSetAttributes: ...
    @property
    def field_data(self) -> DataSetAttributes: ...
    @property
    def cell(self) -> _CellArray: ...
    def set_active_scalars(
        self, name: str | None, preference: str = ...
    ) -> tuple[npt.NDArray[np.float64], str] | tuple[None, str]: ...
    def set_active_vectors(
        self, name: str | None, preference: str = ...
    ) -> tuple[npt.NDArray[np.float64], str] | tuple[None, str]: ...
    def shallow_copy(self, to_copy: _PointSet | Incomplete) -> None: ...
    def compute_cell_sizes(
        self,
        length: bool = ...,
        area: bool = ...,
        volume: bool = ...,
        progress_bar: bool = ...,
        vertex_count: bool = ...,
    ) -> _PointSet: ...
    @overload
    def remove_points(
        self: _Self,
        remove_mask: npt.NDArray[np.bool_],
        inplace: bool = ...,
        mode: str = ...,
    ) -> _Self: ...
    @overload
    def remove_points(
        self: _Self,
        remove_mask: npt.NDArray[np.bool_],
        inplace: bool,
        mode: str,
        keep_scalars: bool = ...,
        *,
        adjacent_cells: bool = ...,
    ) -> tuple[_Self, npt.NDArray[np.intp]]: ...
    def remove_points(
        self: _Self,
        remove_mask: npt.NDArray[np.bool_],
        inplace: bool = ...,
        mode: str = ...,
        keep_scalars: bool = ...,
        adjacent_cells: bool = ...,
    ) -> _Self | tuple[_Self, npt.NDArray[np.intp]]: ...

class _CellArray:
    """Cell connectivity array."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[npt.NDArray[np.intp]]: ...
    def __getitem__(self, index: int) -> npt.NDArray[np.intp]: ...

class DataSetAttributes:
    """Attribute data for point/cell data."""

    def __getitem__(self, name: str) -> npt.NDArray[np.float64]: ...
    def __setitem__(self, name: str, value: npt.NDArray[np.float64] | npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.floating[Incomplete]]) -> None: ...
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...
    def keys(self) -> list[str]: ...
    def values(self) -> list[npt.NDArray[np.float64]]: ...
    def items(self) -> list[tuple[str, npt.NDArray[np.float64]]]: ...
    def update(self, other: dict[str, npt.NDArray[np.float64]] | DataSetAttributes) -> None: ...
    def pop(self, name: str, default: npt.NDArray[np.float64] | None = ...) -> npt.NDArray[np.float64] | None: ...
    @property
    def active_scalars(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def active_scalars_name(self) -> str | None: ...
    @property
    def scalars(self) -> npt.NDArray[np.float64] | None: ...
    @scalars.setter
    def scalars(self, value: npt.NDArray[np.float64]) -> None: ...

class PointSet(_PointSet):
    """Point set without cell connectivity."""

    def __init__(
        self,
        points: npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = ...,
        force_float: bool = ...,
        deep: bool = ...,
    ) -> None: ...

class PolyData(_PointSet):
    """PyVista PolyData class for surface meshes."""

    faces: npt.NDArray[np.intp]
    lines: npt.NDArray[np.intp]
    n_faces: int
    n_lines: int

    def __getitem__(self, name: str) -> npt.NDArray[np.float64]: ...
    def __setitem__(self, name: str, value: npt.NDArray[np.float64] | npt.NDArray[np.floating[Incomplete]]) -> None: ...

    @property
    def active_scalars(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def active_scalars_name(self) -> str | None: ...
    @property
    def active_vectors(self) -> npt.NDArray[np.float64] | None: ...
    @property
    def active_vectors_name(self) -> str | None: ...
    @property
    def face_normals(self) -> npt.NDArray[np.float64]: ...

    def __init__(
        self,
        var_inp: (
            npt.NDArray[np.float64]
            | Sequence[Sequence[float]]
            | str
            | Incomplete
            | None
        ) = ...,
        faces: npt.NDArray[np.intp] | Sequence[int] | None = ...,
        n_faces: int | None = ...,
        lines: npt.NDArray[np.intp] | Sequence[int] | None = ...,
        n_lines: int | None = ...,
        strips: npt.NDArray[np.intp] | Sequence[int] | None = ...,
        n_strips: int | None = ...,
        deep: bool = ...,
        force_ext: str | None = ...,
        force_float: bool = ...,
    ) -> None: ...
    def copy(self: _Self, deep: bool = ...) -> _Self: ...
    def merge(
        self: _Self,
        dataset: PolyData | Incomplete,
        merge_points: bool = ...,
        tolerance: float = ...,
        inplace: bool = ...,
        main_has_priority: bool = ...,
        progress_bar: bool = ...,
    ) -> _Self: ...
    def cell_centers(self) -> PolyData: ...
    def compute_normals(
        self,
        cell_normals: bool = ...,
        point_normals: bool = ...,
        split_vertices: bool = ...,
        flip_normals: bool = ...,
        consistent_normals: bool = ...,
        auto_orient_normals: bool = ...,
        non_manifold_traversal: bool = ...,
        feature_angle: float = ...,
        inplace: bool = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
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
    def slice(
        self,
        normal: str | tuple[float, float, float] | npt.NDArray[np.float64] | npt.NDArray[np.floating[Incomplete]] = ...,
        origin: tuple[float, float, float] | npt.NDArray[np.float64] | npt.NDArray[np.floating[Incomplete]] | None = ...,
        generate_triangles: bool = ...,
        contour: bool = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
    def glyph(
        self,
        orient: str | bool = ...,
        scale: str | bool = ...,
        factor: float = ...,
        geom: PolyData | None = ...,
        indices: npt.NDArray[np.intp] | None = ...,
        tolerance: float | None = ...,
        absolute: bool = ...,
        clamping: bool = ...,
        rng: tuple[float, float] | None = ...,
        color_mode: str = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
    def delaunay_2d(
        self,
        tol: float = ...,
        alpha: float = ...,
        offset: float = ...,
        bound: bool = ...,
        inplace: bool = ...,
        edge_source: PolyData | None = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
    def set_active_scalars(
        self, name: str | None, preference: str = ...
    ) -> tuple[npt.NDArray[np.float64], str] | tuple[None, str]: ...
    def set_active_vectors(
        self, name: str | None, preference: str = ...
    ) -> tuple[npt.NDArray[np.float64], str] | tuple[None, str]: ...
    def plot(self, **kwargs: Incomplete) -> Incomplete: ...
    def save(
        self,
        filename: str,
        binary: bool = ...,
        texture: str | None = ...,
        recompute_normals: bool = ...,
    ) -> None: ...
    @property
    def area(self) -> float: ...
    def transform(
        self: _Self,
        trans: npt.NDArray[np.float64],
        transform_all_input_vectors: bool = ...,
        inplace: bool = ...,
        progress_bar: bool = ...,
    ) -> _Self: ...
    def interpolate(
        self,
        target: PolyData | UnstructuredGrid | Incomplete,
        sharpness: float = ...,
        radius: float | None = ...,
        strategy: str = ...,
        null_value: float = ...,
        n_points: int | None = ...,
        pass_cell_data: bool = ...,
        pass_point_data: bool = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
    def clip(
        self: _Self,
        normal: str | tuple[float, float, float] | npt.NDArray[np.float64] = ...,
        origin: tuple[float, float, float] | npt.NDArray[np.float64] | None = ...,
        invert: bool = ...,
        value: float = ...,
        inplace: bool = ...,
        return_clipped: bool = ...,
        progress_bar: bool = ...,
        crinkle: bool = ...,
    ) -> _Self: ...
    def clip_box(
        self: _Self,
        bounds: tuple[float, ...] | None = ...,
        invert: bool = ...,
        factor: float = ...,
        progress_bar: bool = ...,
        merge_points: bool = ...,
        crinkle: bool = ...,
    ) -> _Self: ...
    def translate(
        self: _Self,
        xyz: npt.NDArray[np.float64] | Sequence[float],
        inplace: bool = ...,
        transform_all_input_vectors: bool = ...,
    ) -> _Self: ...
    def __add__(self, other: PolyData) -> PolyData: ...
    def __iadd__(self: _Self, other: PolyData) -> _Self: ...

class StructuredGrid(_PointSet):
    """PyVista StructuredGrid class."""

    dimensions: tuple[int, int, int]

    def __init__(
        self,
        *args: npt.NDArray[np.float64] | Incomplete,
        deep: bool = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def copy(self: _Self, deep: bool = ...) -> _Self: ...
    @property
    def x(self) -> npt.NDArray[np.float64]: ...
    @property
    def y(self) -> npt.NDArray[np.float64]: ...
    @property
    def z(self) -> npt.NDArray[np.float64]: ...
    def plot(self, **kwargs: Incomplete) -> Incomplete: ...
    def slice(
        self,
        normal: str | tuple[float, float, float] | npt.NDArray[np.float64] | npt.NDArray[np.floating[Incomplete]] = ...,
        origin: tuple[float, float, float] | npt.NDArray[np.float64] | npt.NDArray[np.floating[Incomplete]] | None = ...,
        generate_triangles: bool = ...,
        contour: bool = ...,
        progress_bar: bool = ...,
    ) -> PolyData: ...
    def slice_orthogonal(
        self,
        x: float | None = ...,
        y: float | None = ...,
        z: float | None = ...,
        generate_triangles: bool = ...,
        contour: bool = ...,
        progress_bar: bool = ...,
    ) -> Incomplete: ...
    def cast_to_unstructured_grid(self) -> UnstructuredGrid: ...

# Forward reference for UnstructuredGrid
from pyvista.core.grid import UnstructuredGrid as UnstructuredGrid
