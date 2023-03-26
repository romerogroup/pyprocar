import numpy as np

def sort_coordinates(array1, arrays):
    # Get the indices that would sort the first array lexicographically
    sorted_indices = np.lexsort((array1[:, 0], array1[:, 1], array1[:, 2]))

    # Apply the same sorting to both arrays
    sorted_array1 = array1[sorted_indices]
    for array in arrays:
        array = array[sorted_indices]

    return sorted_array1, arrays
