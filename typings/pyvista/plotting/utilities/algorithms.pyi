"""Type stubs for pyvista.plotting.utilities.algorithms module."""

from _typeshed import Incomplete

from pyvista.core.pointset import PolyData

class AddIDsAlgorithm:
    """VTK algorithm that adds point/cell IDs."""
    def GetOutputPort(self, port: int = ...) -> Incomplete: ...
    def Update(self) -> None: ...

def add_ids_algorithm(
    inp: PolyData | Incomplete,
    point_ids: bool = ...,
    cell_ids: bool = ...,
) -> AddIDsAlgorithm: ...

def algorithm_to_mesh_handler(
    mesh_or_algo: AddIDsAlgorithm | Incomplete,
    port: int = ...,
) -> tuple[PolyData, AddIDsAlgorithm | None]: ...

def set_algorithm_input(
    alg: Incomplete,
    inp: AddIDsAlgorithm | Incomplete,
    port: int = ...,
) -> None: ...

def __getattr__(name: str) -> Incomplete: ...
