"""Type stubs for dill library."""

from _typeshed import Incomplete
from typing import IO

def dump(
    obj: object,
    file: IO[bytes],
    protocol: int | None = ...,
    byref: bool | None = ...,
    fmode: int | None = ...,
    recurse: bool | None = ...,
    **kwargs: Incomplete,
) -> None: ...
def dumps(
    obj: object,
    protocol: int | None = ...,
    byref: bool | None = ...,
    fmode: int | None = ...,
    recurse: bool | None = ...,
    **kwargs: Incomplete,
) -> bytes: ...
def load(
    file: IO[bytes],
    ignore: bool | None = ...,
    **kwargs: Incomplete,
) -> object: ...
def loads(
    data: bytes,
    ignore: bool | None = ...,
    **kwargs: Incomplete,
) -> object: ...
def __getattr__(name: str) -> Incomplete: ...
