"""Type stubs for pyvista to fix incomplete type annotations.

This provides more complete type information for pyvista methods that have
**kwargs: Unknown in their signatures, which triggers reportUnknownMemberType.
"""

from _typeshed import Incomplete
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pyvista.core.grid import ImageData as ImageData
from pyvista.core.grid import UnstructuredGrid as UnstructuredGrid
from pyvista.core.pointset import PointSet as PointSet
from pyvista.core.pointset import PolyData as PolyData
from pyvista.core.pointset import StructuredGrid as StructuredGrid
from pyvista.plotting.plotter import Plotter as Plotter

class MultiBlock:
    """PyVista MultiBlock container."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int | str) -> PolyData | UnstructuredGrid | Incomplete: ...
    def __setitem__(
        self, index: int | str, value: PolyData | UnstructuredGrid | Incomplete
    ) -> None: ...
    def append(self, mesh: PolyData | UnstructuredGrid | Incomplete) -> None: ...
    def keys(self) -> list[str | None]: ...
    def values(self) -> list[PolyData | UnstructuredGrid | Incomplete]: ...
    def items(self) -> list[tuple[str | None, PolyData | UnstructuredGrid | Incomplete]]: ...

class _GlobalTheme:
    """Global PyVista theme settings."""

    allow_empty_mesh: bool
    background: str | tuple[float, float, float]
    cmap: str
    color: str | tuple[float, float, float]
    edge_color: str | tuple[float, float, float]
    font: Incomplete
    lighting: bool
    show_edges: bool
    show_scalar_bar: bool

global_theme: _GlobalTheme

__all__ = [
    "ImageData",
    "MultiBlock",
    "Plotter",
    "PointSet",
    "PolyData",
    "StructuredGrid",
    "UnstructuredGrid",
    "global_theme",
]
