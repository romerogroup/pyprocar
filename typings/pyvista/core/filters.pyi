"""Type stubs for pyvista.core.filters module."""

from _typeshed import Incomplete

from pyvista import MultiBlock
from pyvista.core.pointset import PolyData

__all__ = ["_get_output"]

def _get_output(
    algorithm: Incomplete,
    *,
    iport: int = ...,
    iconnection: int = ...,
    oport: int = ...,
    active_scalars: str | None = ...,
    active_scalars_field: str = ...,
) -> PolyData | MultiBlock: ...
def __getattr__(name: str) -> Incomplete: ...
