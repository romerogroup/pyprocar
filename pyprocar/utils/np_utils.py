from typing import Any, cast

import numpy as np

COMPLEX_DTYPE: type[np.complexfloating[Any, Any]] = cast(
    type[np.complexfloating[Any, Any]],
    getattr(np, "complex_", np.complex128),
)
INT_DTYPE: type[np.signedinteger[Any]] = cast(
    type[np.signedinteger[Any]], getattr(np, "int_", np.int64)
)
FLOAT_DTYPE: type[np.floating[Any]] = cast(
    type[np.floating[Any]], getattr(np, "float_", np.float64)
)