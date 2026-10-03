"""Keep data/ read-only and reserve it for tests marked ``data``.

An audit hook watches every test. Writing, renaming, removing, creating or
truncating a path under data/ raises, so the write never happens. An unmarked
test that reads data/ raises too. Library code may swallow that RuntimeError
in an ``except Exception``, so each violation is also recorded and fails the
test phase it happened in.

Paths are resolved with realpath, so a symlinked alias of data/ counts. Not
caught: writes from subprocesses, opens relative to a ``dir_fd``, and
metadata changes such as chmod or utime.
"""

import os
import sys

import pytest

from tests.utils import DATA_DIR

pytest_plugins = ["pytester"]

_DATA_PREFIX = os.path.realpath(DATA_DIR) + os.sep
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
_violations: list[str] = []


def _data_path(path: object) -> str | None:
    if not isinstance(path, str | bytes | os.PathLike):
        return None
    full = os.path.realpath(os.fsdecode(path)) + os.sep
    return full if full.startswith(_DATA_PREFIX) else None


def _is_write(event: str, args: tuple[object, ...]) -> bool:
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        return (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        )
    return event in _MUTATING_EVENTS


def _violate(message: str) -> None:
    _violations.append(message)
    raise RuntimeError(message)


def _guard_data_dir(event: str, args: tuple[object, ...]) -> None:
    if _current_test is None or not args:
        return
    nodeid, marked = _current_test
    if _is_write(event, args):
        for path in args[:2]:
            if full := _data_path(path):
                _violate(f"{nodeid} writes {full}; data/ is read-only, write to tmp_path")
    elif (
        not marked
        and event in {"open", "os.listdir", "os.scandir"}
        and (full := _data_path(args[0]))
    ):
        _violate(f"{nodeid} reads {full}; mark it with pytest.mark.data")


sys.addaudithook(_guard_data_dir)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "data: needs DFT fixtures from data/")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item):
    global _current_test
    _current_test = (item.nodeid, item.get_closest_marker("data") is not None)
    _violations.clear()
    try:
        return (yield)
    finally:
        _current_test = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport():
    report: pytest.TestReport = yield
    if _violations and report.passed:
        report.outcome = "failed"
        report.longrepr = "\n".join(_violations)
    _violations.clear()
    return report
