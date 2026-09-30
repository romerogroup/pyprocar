import os
import sys

import pytest

from tests.utils import DATA_DIR

_DATA_PREFIXES = (str(DATA_DIR) + os.sep, str(DATA_DIR.resolve()) + os.sep)
_unmarked_test: str | None = None


def _forbid_unmarked_data_access(event: str, args: tuple[object, ...]) -> None:
    if _unmarked_test is None or event not in {"open", "os.listdir", "os.scandir"}:
        return
    path = args[0] if args else None
    if isinstance(path, str | bytes | os.PathLike):
        full = os.path.abspath(os.fsdecode(path)) + os.sep
        if full.startswith(_DATA_PREFIXES):
            raise RuntimeError(f"{_unmarked_test} reads {full}; mark it with pytest.mark.data")


sys.addaudithook(_forbid_unmarked_data_access)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "data: needs DFT fixtures from data/")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item):
    global _unmarked_test
    _unmarked_test = None if item.get_closest_marker("data") else item.nodeid
    try:
        return (yield)
    finally:
        _unmarked_test = None
