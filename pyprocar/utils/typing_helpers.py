"""Type-safe helper functions for numpy operations.

This module provides wrapper functions around numpy operations that return
explicit Python types, avoiding the `Any` type parameters that numpy's
type stubs use (e.g., `floating[Any]`). This enables strict type checking
with basedpyright's `typeCheckingMode: "all"`.

JUSTIFICATION FOR CAST USAGE:
These functions use cast() for widening numpy scalar types to Python types.
This is necessary because numpy's type stubs use generic type parameters
(e.g., `floating[Any]`) that basedpyright interprets as `Any` in strict mode.
The runtime types are well-defined (float64, int64, etc.), but the static
type system cannot express this due to limitations in numpy's stubs.
This is not type narrowing (which would be unsafe) but type widening from
a numpy-specific scalar type to a Python primitive type.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt


def to_float(val: object) -> float:
    """Convert numpy scalar or Python numeric to Python float.

    Cast justification: numpy scalars are always convertible to float,
    but numpy's stubs return floating[Any] which triggers reportAny.

    Parameters
    ----------
    val : object
        Value to convert (numpy scalar, Python int/float).

    Returns
    -------
    float
        Python float value.
    """
    return float(cast(float, val))


def to_int(val: object) -> int:
    """Convert numpy scalar or Python numeric to Python int.

    Cast justification: numpy scalars are always convertible to int,
    but numpy's stubs return integer[Any] which triggers reportAny.

    Parameters
    ----------
    val : object
        Value to convert (numpy scalar, Python int/float).

    Returns
    -------
    int
        Python int value.
    """
    return int(cast(int, val))


def array_str_to_list(arr: npt.NDArray[np.str_]) -> list[str]:
    """Convert numpy string array to list of Python strings.

    Cast justification: tolist() returns list but type checker sees Any
    due to numpy's generic array types.

    Parameters
    ----------
    arr : npt.NDArray[np.str_]
        Numpy array of strings.

    Returns
    -------
    list[str]
        List of Python strings.
    """
    return cast(list[str], arr.tolist())


def array_float64_to_list(arr: npt.NDArray[np.float64]) -> list[float]:
    """Convert numpy float64 array to list of Python floats.

    Cast justification: tolist() returns list but type checker sees Any
    due to numpy's generic array types.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        Numpy array of float64.

    Returns
    -------
    list[float]
        List of Python floats.
    """
    return cast(list[float], arr.tolist())


def array_int_to_list(arr: npt.NDArray[np.intp]) -> list[int]:
    """Convert numpy int array to list of Python ints.

    Cast justification: tolist() returns list but type checker sees Any
    due to numpy's generic array types.

    Parameters
    ----------
    arr : npt.NDArray[np.intp]
        Numpy array of integers.

    Returns
    -------
    list[int]
        List of Python integers.
    """
    return cast(list[int], arr.tolist())


def det_to_float(arr: npt.NDArray[np.float64]) -> float:
    """Compute determinant and return as Python float.

    Cast justification: np.linalg.det returns floating[Any] which
    triggers reportAny, but the runtime value is always a scalar.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        Square matrix.

    Returns
    -------
    float
        Determinant as Python float.
    """
    return float(cast(float, np.linalg.det(arr)))


def norm_to_float(arr: npt.NDArray[np.float64]) -> float:
    """Compute norm and return as Python float.

    Cast justification: np.linalg.norm returns floating[Any] which
    triggers reportAny, but the runtime value is always a scalar.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        Array to compute norm of.

    Returns
    -------
    float
        Norm as Python float.
    """
    # np.linalg.norm returns floating[Any], convert via np.float64 scalar
    result = np.float64(np.linalg.norm(arr))
    return result.item()


def dot_scalar(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Compute dot product of 1D arrays and return as Python float.

    Cast justification: np.dot with 1D arrays returns floating[Any] which
    triggers reportAny, but the runtime value is always a scalar.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        First 1D array.
    b : npt.NDArray[np.float64]
        Second 1D array.

    Returns
    -------
    float
        Dot product as Python float.
    """
    return float(cast(float, np.dot(a, b)))


def dot_matrix(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute dot product of matrices and return as float64 array.

    Cast justification: np.dot returns ndarray[Any, dtype[Any]] which
    triggers reportAny, but when both inputs are float64, output is float64.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        First array/matrix.
    b : npt.NDArray[np.float64]
        Second array/matrix.

    Returns
    -------
    npt.NDArray[np.float64]
        Result array.
    """
    return cast(npt.NDArray[np.float64], np.dot(a, b))


def matmul(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Matrix multiplication returning float64 array.

    Cast justification: np.matmul returns ndarray with Any dtype,
    but when both inputs are float64, output is float64.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        First matrix.
    b : npt.NDArray[np.float64]
        Second matrix.

    Returns
    -------
    npt.NDArray[np.float64]
        Result matrix.
    """
    return cast(npt.NDArray[np.float64], np.matmul(a, b))


def arrays_equal(a: npt.NDArray[np.str_], b: npt.NDArray[np.str_]) -> bool:
    """Check if two string arrays are equal, returning Python bool.

    Parameters
    ----------
    a : npt.NDArray[np.str_]
        First array.
    b : npt.NDArray[np.str_]
        Second array.

    Returns
    -------
    bool
        True if arrays are equal.
    """
    return bool(np.array_equal(a, b))


def sum_to_float(arr: npt.NDArray[np.float64] | list[float]) -> float:
    """Sum array and return as Python float.

    Cast justification: np.sum returns floating[Any] for float arrays,
    which triggers reportAny.

    Parameters
    ----------
    arr : array-like
        Array to sum.

    Returns
    -------
    float
        Sum as Python float.
    """
    return float(cast(float, np.sum(arr)))


def arccos_deg(val: float) -> float:
    """Compute arccos and convert to degrees.

    Cast justification: np.arccos and np.rad2deg return floating[Any].

    Parameters
    ----------
    val : float
        Cosine value.

    Returns
    -------
    float
        Angle in degrees as Python float.
    """
    radians = cast(float, np.arccos(val))
    return float(cast(float, np.rad2deg(radians)))


def inv_matrix(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute matrix inverse.

    Cast justification: np.linalg.inv returns ndarray with Any dtype.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        Square matrix.

    Returns
    -------
    npt.NDArray[np.float64]
        Inverse matrix.
    """
    return cast(npt.NDArray[np.float64], np.linalg.inv(arr))


def cross_product(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute cross product of two 3D vectors.

    Cast justification: np.cross returns ndarray with Any dtype.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        First 3D vector.
    b : npt.NDArray[np.float64]
        Second 3D vector.

    Returns
    -------
    npt.NDArray[np.float64]
        Cross product vector.
    """
    return np.cross(a, b)


def shape_to_int(shape_element: object) -> int:
    """Convert numpy shape element to Python int.

    Cast justification: numpy shape tuple elements are always int-like but
    typed as Any in some contexts due to numpy's generic types.

    Parameters
    ----------
    shape_element : object
        Element from numpy array shape tuple.

    Returns
    -------
    int
        Python int value.
    """
    return int(cast(int, shape_element))


def get_shape_dim(arr: npt.NDArray[np.generic], dim: int) -> int:
    """Get array dimension size as Python int.

    Cast justification: numpy shape tuple returns Any elements in strict mode.

    Parameters
    ----------
    arr : npt.NDArray[np.generic]
        Array to get shape from (any dtype).
    dim : int
        Dimension index.

    Returns
    -------
    int
        Size of the dimension.
    """
    return int(cast(int, arr.shape[dim]))


def vdot_to_complex(
    a: npt.NDArray[np.complex128], b: npt.NDArray[np.complex128]
) -> complex:
    """Compute vdot and return as Python complex.

    Cast justification: np.vdot returns complexfloating[Any] which
    doesn't overlap with complex for direct casting.

    Parameters
    ----------
    a : npt.NDArray[np.complex128]
        First array.
    b : npt.NDArray[np.complex128]
        Second array.

    Returns
    -------
    complex
        vdot result as Python complex.
    """
    result = np.vdot(a, b)
    # Convert through intermediate numpy complex128 then to Python complex
    return complex(np.complex128(result))


def norm_axis(arr: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
    """Compute norm along an axis and return as float64 array.

    Cast justification: np.linalg.norm with axis returns floating[Any] array
    which triggers reportAny.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        Array to compute norms for.
    axis : int
        Axis along which to compute norms.

    Returns
    -------
    npt.NDArray[np.float64]
        Array of norms.
    """
    return cast(npt.NDArray[np.float64], np.linalg.norm(arr, axis=axis))


def where_indices(
    condition: npt.NDArray[np.bool_],
) -> npt.NDArray[np.intp]:
    """Get indices where condition is True.

    Cast justification: np.where returns tuple of arrays with Any element types.

    Parameters
    ----------
    condition : npt.NDArray[np.bool_]
        Boolean condition array.

    Returns
    -------
    npt.NDArray[np.intp]
        Array of indices where condition is True.
    """
    result = np.where(condition)
    return result[0]


def get_row(arr: npt.NDArray[np.float64], idx: int) -> npt.NDArray[np.float64]:
    """Get a row from a 2D array with proper typing.

    Cast justification: numpy array indexing returns elements with Any type
    in strict type checking mode.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        2D array.
    idx : int
        Row index.

    Returns
    -------
    npt.NDArray[np.float64]
        The row as a 1D array.
    """
    return cast(npt.NDArray[np.float64], arr[idx])


def array_element_to_int(arr: npt.NDArray[np.intp], idx: int) -> int:
    """Get an element from an int array as Python int.

    Cast justification: numpy array indexing returns np.intp which
    basedpyright sees as Any.

    Parameters
    ----------
    arr : npt.NDArray[np.intp]
        Integer array.
    idx : int
        Index.

    Returns
    -------
    int
        Element as Python int.
    """
    return int(cast(int, arr[idx]))


def array_element_to_float(arr: npt.NDArray[np.float64], idx: int) -> float:
    """Get an element from a float64 array as Python float.

    Cast justification: numpy array indexing returns np.float64 which
    basedpyright sees as Any.

    Parameters
    ----------
    arr : npt.NDArray[np.float64]
        Float array.
    idx : int
        Index.

    Returns
    -------
    float
        Element as Python float.
    """
    return float(cast(float, arr[idx]))


def array_sub(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Subtract two arrays with proper typing.

    Cast justification: numpy array subtraction can return arrays with
    Any element types in strict mode.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        First array.
    b : npt.NDArray[np.float64]
        Second array.

    Returns
    -------
    npt.NDArray[np.float64]
        Result of a - b.
    """
    return a - b


__all__ = [
    "arccos_deg",
    "array_element_to_float",
    "array_element_to_int",
    "array_float64_to_list",
    "array_int_to_list",
    "array_str_to_list",
    "array_sub",
    "arrays_equal",
    "cross_product",
    "det_to_float",
    "dot_matrix",
    "dot_scalar",
    "get_row",
    "get_shape_dim",
    "inv_matrix",
    "matmul",
    "norm_axis",
    "norm_to_float",
    "shape_to_int",
    "sum_to_float",
    "to_float",
    "to_int",
    "vdot_to_complex",
    "where_indices",
]
