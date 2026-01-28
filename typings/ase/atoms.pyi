"""Type stubs for ase.atoms module."""

from _typeshed import Incomplete
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

class Cell:
    """ASE Cell class representing unit cell."""

    def __array__(self) -> npt.NDArray[np.float64]: ...
    def __getitem__(self, index: int) -> npt.NDArray[np.float64]: ...

class Atoms:
    """ASE Atoms class representing a collection of atoms."""

    positions: npt.NDArray[np.float64]
    cell: Cell
    numbers: npt.NDArray[np.int64]
    pbc: npt.NDArray[np.bool_]

    def __init__(
        self,
        symbols: str | Sequence[str] | None = ...,
        positions: npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = ...,
        numbers: npt.NDArray[np.int64] | Sequence[int] | None = ...,
        tags: npt.NDArray[np.int64] | Sequence[int] | None = ...,
        momenta: npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = ...,
        masses: npt.NDArray[np.float64] | Sequence[float] | None = ...,
        magmoms: npt.NDArray[np.float64] | Sequence[float] | None = ...,
        charges: npt.NDArray[np.float64] | Sequence[float] | None = ...,
        scaled_positions: (
            npt.NDArray[np.float64] | Sequence[Sequence[float]] | None
        ) = ...,
        cell: (
            npt.NDArray[np.float64]
            | Sequence[Sequence[float]]
            | Sequence[float]
            | None
        ) = ...,
        pbc: bool | Sequence[bool] | npt.NDArray[np.bool_] | None = ...,
        celldisp: npt.NDArray[np.float64] | Sequence[float] | None = ...,
        constraint: Incomplete = ...,
        calculator: Incomplete = ...,
        info: dict[str, Incomplete] | None = ...,
        velocities: npt.NDArray[np.float64] | Sequence[Sequence[float]] | None = ...,
    ) -> None: ...
    def get_positions(self) -> npt.NDArray[np.float64]: ...
    def get_scaled_positions(
        self, wrap: bool = ...
    ) -> npt.NDArray[np.float64]: ...
    def get_cell(self, complete: bool = ...) -> Cell: ...
    def get_atomic_numbers(self) -> npt.NDArray[np.int64]: ...
    def get_chemical_symbols(self) -> list[str]: ...
    def get_masses(self) -> npt.NDArray[np.float64]: ...
    def get_pbc(self) -> npt.NDArray[np.bool_]: ...
    def set_positions(
        self, newpositions: npt.NDArray[np.float64] | Sequence[Sequence[float]]
    ) -> None: ...
    def set_scaled_positions(
        self, scaled: npt.NDArray[np.float64] | Sequence[Sequence[float]]
    ) -> None: ...
    def set_cell(
        self,
        cell: npt.NDArray[np.float64] | Sequence[Sequence[float]] | Sequence[float],
        scale_atoms: bool = ...,
        apply_constraint: bool = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def copy(self) -> Atoms: ...

def __getattr__(name: str) -> Incomplete: ...
