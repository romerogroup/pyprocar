"""Type stubs for pyvista.plotting module."""

from _typeshed import Incomplete

from pyvista.plotting.plotter import ColorLike as ColorLike
from pyvista.plotting.plotter import Plotter as Plotter

def __getattr__(name: str) -> Incomplete: ...

__all__ = ["ColorLike", "Plotter"]
