"""Type stubs for matplotlib.ticker."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class Locator:
    """Base class for tick locators."""

    def __call__(self) -> list[float]: ...
    def tick_values(self, vmin: float, vmax: float) -> npt.NDArray[np.float64]: ...
    def set_params(self, **kwargs: object) -> None: ...
    def get_ticks(self) -> list[float]: ...
    def get_ticklabels(self) -> list[str]: ...

class MaxNLocator(Locator):
    """Locator with at most N ticks."""

    def __init__(
        self,
        nbins: int | str = ...,
        *,
        steps: Sequence[float] | None = ...,
        integer: bool = ...,
        symmetric: bool = ...,
        prune: str | None = ...,
        min_n_ticks: int = ...,
    ) -> None: ...

class AutoLocator(MaxNLocator):
    """Automatic tick locator."""
    ...

class MultipleLocator(Locator):
    """Locator at multiples of a base."""

    def __init__(self, base: float = ..., offset: float = ...) -> None: ...

class NullLocator(Locator):
    """Locator that places no ticks."""

    def __init__(self) -> None: ...

class FixedLocator(Locator):
    """Locator with fixed tick positions."""

    def __init__(
        self,
        locs: Sequence[float],
        nbins: int | None = ...,
    ) -> None: ...

class Formatter:
    """Base class for tick formatters."""

    def __call__(self, x: float, pos: int | None = ...) -> str: ...

class ScalarFormatter(Formatter):
    """Default scalar tick formatter."""

    def __init__(
        self,
        useOffset: bool | float = ...,
        useMathText: bool | None = ...,
        useLocale: bool | None = ...,
    ) -> None: ...

class FuncFormatter(Formatter):
    """Formatter using a user-defined function."""

    def __init__(self, func: object) -> None: ...

__all__ = [
    "Locator",
    "MaxNLocator",
    "AutoLocator",
    "MultipleLocator",
    "NullLocator",
    "FixedLocator",
    "Formatter",
    "ScalarFormatter",
    "FuncFormatter",
]
