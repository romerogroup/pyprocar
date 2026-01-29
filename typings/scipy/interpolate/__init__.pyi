from _typeshed import Incomplete

import numpy as np
import numpy.typing as npt

def griddata(
    points: npt.ArrayLike,
    values: npt.ArrayLike,
    xi: npt.ArrayLike | tuple[npt.ArrayLike, ...],
    method: str = ...,
    fill_value: float = ...,
    rescale: bool = ...,
) -> npt.NDArray[np.floating[Incomplete]]: ...

def __getattr__(name: str) -> Incomplete: ...
