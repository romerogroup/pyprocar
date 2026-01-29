import numpy as np
import numpy.typing as npt
from _typeshed import Incomplete
from typing import Self

class KernelDensity:
    def __init__(
        self,
        *,
        bandwidth: float | str = ...,
        algorithm: str = ...,
        kernel: str = ...,
        metric: str = ...,
        atol: float = ...,
        rtol: float = ...,
        breadth_first: bool = ...,
        leaf_size: int = ...,
        metric_params: dict[str, Incomplete] | None = ...,
    ) -> None: ...
    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = ...,
        sample_weight: npt.ArrayLike | None = ...,
    ) -> Self: ...
    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
    def score(self, X: npt.ArrayLike, y: npt.ArrayLike | None = ...) -> float: ...
    def sample(
        self, n_samples: int = ..., random_state: int | None = ...
    ) -> npt.NDArray[np.float64]: ...

def __getattr__(name: str) -> Incomplete: ...
