import numpy as np

COMPLEX_DTYPE = np.complex_ if hasattr(np, "complex_") else np.complex128
INT_DTYPE = np.int if hasattr(np, "int") else np.int64
FLOAT_DTYPE = np.float_ if hasattr(np, "float_") else np.float64