"""Type stubs for mayavi.mlab."""

from _typeshed import Incomplete

def figure(
    figure: Incomplete = ...,
    bgcolor: tuple[float, float, float] = ...,
    fgcolor: Incomplete = ...,
    engine: Incomplete = ...,
    size: tuple[int, int] = ...,
) -> Incomplete: ...

class pipeline:
    @staticmethod
    def surface(*args: Incomplete, **kwargs: Incomplete) -> Incomplete: ...

def quiver3d(*args: Incomplete, **kwargs: Incomplete) -> Incomplete: ...
def colorbar(orientation: str = ..., **kwargs: Incomplete) -> Incomplete: ...
def show(**kwargs: Incomplete) -> None: ...

def __getattr__(name: str) -> Incomplete: ...
