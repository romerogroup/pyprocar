"""Type stubs for ase.io module."""

from _typeshed import Incomplete
from pathlib import Path

from ase.atoms import Atoms

def read(
    filename: str | Path,
    index: int | slice | str | None = ...,
    format: str | None = ...,
    parallel: bool = ...,
    do_not_split_by_at_sign: bool = ...,
    **kwargs: Incomplete,
) -> Atoms | list[Atoms]: ...
def write(
    filename: str | Path,
    images: Atoms | list[Atoms],
    format: str | None = ...,
    parallel: bool = ...,
    append: bool = ...,
    **kwargs: Incomplete,
) -> None: ...
def __getattr__(name: str) -> Incomplete: ...
