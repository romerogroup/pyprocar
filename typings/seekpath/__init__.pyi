"""Type stubs for seekpath library."""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt

class PathResult(TypedDict):
    """Result from get_path function."""

    point_coords: dict[str, list[float]]
    path: list[tuple[str, str]]
    has_inversion_symmetry: bool
    augmented_path: bool
    bravais_lattice: str
    bravais_lattice_extended: str
    volume_original_wrt_conv: float
    volume_original_wrt_prim: float
    primitive_lattice: npt.NDArray[np.float64]
    primitive_positions: npt.NDArray[np.float64]
    primitive_types: list[int]
    conv_lattice: npt.NDArray[np.float64]
    conv_positions: npt.NDArray[np.float64]
    conv_types: list[int]

# Structure type: (lattice, positions, numbers)
_Structure = tuple[
    npt.NDArray[np.float64] | Sequence[Sequence[float]],
    npt.NDArray[np.float64] | Sequence[Sequence[float]],
    Sequence[int] | npt.NDArray[np.int64],
]

def get_path(
    structure: _Structure,
    with_time_reversal: bool = ...,
    recipe: str = ...,
    threshold: float = ...,
    symprec: float = ...,
    angle_tolerance: float = ...,
) -> PathResult: ...
def get_explicit_k_path(
    structure: _Structure,
    with_time_reversal: bool = ...,
    reference_distance: float = ...,
    recipe: str = ...,
    threshold: float = ...,
    symprec: float = ...,
    angle_tolerance: float = ...,
) -> dict[str, Incomplete]: ...
def __getattr__(name: str) -> Incomplete: ...
