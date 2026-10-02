import os
import sys

import pytest

from tests.utils import DATA_DIR

_DATA_PREFIXES = (str(DATA_DIR) + os.sep, str(DATA_DIR.resolve()) + os.sep)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_MUTATING_EVENTS = {
    "os.remove",
    "os.rename",
    "os.replace",
    "os.mkdir",
    "os.rmdir",
    "os.truncate",
    "shutil.rmtree",
}
_current_test: tuple[str, bool] | None = None


def _data_path(path: object) -> str | None:
    if not isinstance(path, str | bytes | os.PathLike):
        return None
    full = os.path.abspath(os.fsdecode(path)) + os.sep
    return full if full.startswith(_DATA_PREFIXES) else None


def _is_write(event: str, args: tuple[object, ...]) -> bool:
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        return (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        )
    return event in _MUTATING_EVENTS


def _guard_data_dir(event: str, args: tuple[object, ...]) -> None:
    if _current_test is None or not args:
        return
    nodeid, marked = _current_test
    if _is_write(event, args):
        for path in args[:2]:
            if full := _data_path(path):
                raise RuntimeError(f"{nodeid} writes {full}; data/ is read-only, write to tmp_path")
    elif (
        not marked
        and event in {"open", "os.listdir", "os.scandir"}
        and (full := _data_path(args[0]))
    ):
        raise RuntimeError(f"{nodeid} reads {full}; mark it with pytest.mark.data")


sys.addaudithook(_guard_data_dir)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "data: needs DFT fixtures from data/")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item):
    global _current_test
    _current_test = (item.nodeid, item.get_closest_marker("data") is not None)
    try:
        return (yield)
    finally:
        _current_test = None
