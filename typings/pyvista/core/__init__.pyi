"""Type stubs for pyvista.core module."""

from _typeshed import Incomplete

from pyvista.core.grid import ImageData as ImageData
from pyvista.core.grid import UnstructuredGrid as UnstructuredGrid
from pyvista.core.pointset import PointSet as PointSet
from pyvista.core.pointset import PolyData as PolyData
from pyvista.core.pointset import StructuredGrid as StructuredGrid

def __getattr__(name: str) -> Incomplete: ...
