from __future__ import annotations

import numpy as np
import numpy.typing as npt


def sort_coordinates(
    array1: npt.NDArray[np.float64],
    arrays: list[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    # Get the indices that would sort the first array lexicographically
    sorted_indices = np.lexsort((array1[:, 0], array1[:, 1], array1[:, 2]))

    # Apply the same sorting to both arrays
    sorted_array1 = array1[sorted_indices]
    for array in arrays:
        array = array[sorted_indices]

    return sorted_array1, arrays
