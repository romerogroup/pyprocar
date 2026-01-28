"""Type stubs for spglib library."""

from _typeshed import Incomplete
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class SpacegroupType:
    """Spacegroup type information."""

    number: int
    international_short: str
    international_full: str
    international: str
    schoenflies: str
    hall_number: int
    hall_symbol: str
    choice: str
    pointgroup_international: str
    pointgroup_schoenflies: str
    arithmetic_crystal_class_number: int
    arithmetic_crystal_class_symbol: str

class SymmetryDataset:
    """Result from get_symmetry_dataset."""

    number: int
    international: str
    hall: str
    hall_number: int
    choice: str
    transformation_matrix: npt.NDArray[np.float64]
    origin_shift: npt.NDArray[np.float64]
    rotations: npt.NDArray[np.int32]
    translations: npt.NDArray[np.float64]
    wyckoffs: list[str]
    site_symmetry_symbols: list[str]
    equivalent_atoms: npt.NDArray[np.int32]
    crystallographic_orbits: npt.NDArray[np.int32]
    primitive_lattice: npt.NDArray[np.float64]
    mapping_to_primitive: npt.NDArray[np.int32]
    std_lattice: npt.NDArray[np.float64]
    std_types: npt.NDArray[np.int32]
    std_positions: npt.NDArray[np.float64]
    std_rotation_matrix: npt.NDArray[np.float64]
    std_mapping_to_primitive: npt.NDArray[np.int32]
    pointgroup: str

# Cell type: (lattice, positions, numbers)
_Cell = tuple[
    Sequence[Sequence[float]] | npt.NDArray[np.float64],
    Sequence[Sequence[float]] | npt.NDArray[np.float64],
    Sequence[int] | npt.NDArray[np.int64],
]

def get_spacegroup(
    cell: _Cell,
    symprec: float = ...,
) -> str | None: ...
def get_symmetry(
    cell: _Cell,
    symprec: float = ...,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]] | None: ...
def get_symmetry_dataset(
    cell: _Cell,
    symprec: float = ...,
    angle_tolerance: float = ...,
    hall_number: int = ...,
) -> SymmetryDataset | None: ...
def get_spacegroup_type(hall_number: int) -> SpacegroupType: ...
def standardize_cell(
    cell: _Cell,
    to_primitive: bool = ...,
    no_idealize: bool = ...,
    symprec: float = ...,
    angle_tolerance: float = ...,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
] | None: ...
def find_primitive(
    cell: _Cell,
    symprec: float = ...,
    angle_tolerance: float = ...,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
] | None: ...
def refine_cell(
    cell: _Cell,
    symprec: float = ...,
    angle_tolerance: float = ...,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64]
] | None: ...
def __getattr__(name: str) -> Incomplete: ...
