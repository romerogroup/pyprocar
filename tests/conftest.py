"""Keep data/ read-only and reserve it for tests marked ``data``.

An audit hook watches every test. Writing, renaming, removing, creating,
truncating, linking, chmod-ing or touching a path under data/ raises, so the
write never happens. An unmarked test that reads data/ raises too. Library code
may swallow that RuntimeError in an ``except Exception``, so each violation is
also recorded and fails the test phase it happened in, whatever outcome the
phase reported.

Paths are resolved with realpath, so a symlinked alias of data/ counts. File
descriptors and ``dir_fd`` arguments are resolved through /proc/self/fd, which
exists on Linux only. Audit events carry neither the ``dir_fd`` of ``os.open``
nor ``follow_symlinks``, so ``os.open`` (with ``dir_fd``), ``os.chmod``,
``os.chown`` and ``os.utime`` (with ``follow_symlinks=False``), ``os.lchown``
and ``os.lchmod`` are wrapped. A wrapper checks the call itself and skips the
one audit event that call raises. Each wrapper joins the ``os.supports_*`` sets
its original is in, so capability checks such as shutil's still pass.

Not caught: writes from subprocesses. An audit hook sees only its own
interpreter, and the external programs tests run need not be Python.
"""

import functools
import os
import sys
import threading
from collections.abc import Callable

import pytest

from tests.utils import DATA_DIR

pytest_plugins = ["pytester"]

_DATA_ROOT = os.path.realpath(DATA_DIR)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
# event: (path position, dir_fd position, follows a final symlink) per mutated path
_MUTATING_EVENTS: dict[str, tuple[tuple[int, int | None, bool], ...]] = {
    "os.remove": ((0, 1, False),),
    "os.rename": ((0, 2, False), (1, 3, False)),
    "os.replace": ((0, 2, False), (1, 3, False)),
    "os.mkdir": ((0, 2, False),),
    "os.rmdir": ((0, 1, False),),
    "os.truncate": ((0, None, True),),
    "os.symlink": ((1, 2, False),),
    "os.link": ((0, 2, True), (1, 3, False)),
    "os.chmod": ((0, 2, True),),
    "os.chown": ((0, 3, True),),
    "os.utime": ((0, 3, True),),
    "shutil.rmtree": ((0, 1, False),),
}
_WATCHED_EVENTS = {"open", "os.listdir", "os.scandir", *_MUTATING_EVENTS}
_current_test: tuple[str, bool] | None = None
# (event, path) of the one audit event a wrapper below already checked
_checked_by_wrapper = threading.local()
_violations: list[str] = []


def _fd_path(fd: int) -> str:
    return f"/proc/self/fd/{fd}"


def _data_path(path: object, dir_fd: object = None, follow: bool = True) -> str | None:
    if isinstance(path, int):
        full = os.path.realpath(_fd_path(path))
    elif isinstance(path, str | bytes | os.PathLike):
        path = os.fsdecode(path)
        if isinstance(dir_fd, int) and dir_fd >= 0 and not os.path.isabs(path):
            path = os.path.join(_fd_path(dir_fd), path)
        if follow:
            full = os.path.realpath(path)
        else:
            head, tail = os.path.split(os.path.abspath(path))
            full = os.path.join(os.path.realpath(head), tail)
    else:
        return None
    return full if full == _DATA_ROOT or full.startswith(_DATA_ROOT + os.sep) else None


def _written_paths(event: str, args: tuple[object, ...]) -> list[tuple[object, object, bool]]:
    if event == "open":
        path, mode, flags = args
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        )
        return [(path, None, True)] if writes else []
    return [
        (args[path], None if dir_fd is None else args[dir_fd], follow)
        for path, dir_fd, follow in _MUTATING_EVENTS.get(event, ())
    ]


def _violate(message: str) -> None:
    _violations.append(message)
    raise RuntimeError(message)


def _check_write(nodeid: str, path: object, dir_fd: object, follow: bool = True) -> None:
    if full := _data_path(path, dir_fd, follow):
        _violate(f"{nodeid} writes {full}; data/ is read-only, write to tmp_path")


def _guard_data_dir(event: str, args: tuple[object, ...]) -> None:
    if _current_test is None or not args or event not in _WATCHED_EVENTS:
        return
    if getattr(_checked_by_wrapper, "call", None) == (event, args[0]):
        _checked_by_wrapper.call = None
        return
    nodeid, marked = _current_test
    if written := _written_paths(event, args):
        for path, dir_fd, follow in written:
            _check_write(nodeid, path, dir_fd, follow)
    elif not marked and (full := _data_path(args[0])):
        _violate(f"{nodeid} reads {full}; mark it with pytest.mark.data")


sys.addaudithook(_guard_data_dir)


def _call_checked[T](event: str, path: object, call: Callable[[], T]) -> T:
    # the audit event carries os.fspath of a PathLike, not the object itself
    _checked_by_wrapper.call = (event, os.fspath(path) if isinstance(path, os.PathLike) else path)
    try:
        return call()
    finally:
        _checked_by_wrapper.call = None


def _guard_os_open(os_open: Callable[..., int]) -> Callable[..., int]:
    @functools.wraps(os_open)
    def guarded(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None or dir_fd < 0:
            return os_open(path, flags, mode, dir_fd=dir_fd)
        _guard_data_dir("open", (os.path.join(_fd_path(dir_fd), os.fsdecode(path)), None, flags))
        return _call_checked("open", path, lambda: os_open(path, flags, mode, dir_fd=dir_fd))

    return guarded


def _guard_no_follow(fn: Callable[..., None], event: str, always: bool = False):
    @functools.wraps(fn)
    def guarded(path, *args, **kwargs):
        if not always and kwargs.get("follow_symlinks", True):
            return fn(path, *args, **kwargs)
        if _current_test is not None:
            _check_write(_current_test[0], path, kwargs.get("dir_fd"), follow=False)
        return _call_checked(event, path, lambda: fn(path, *args, **kwargs))

    return guarded


_GUARDED = {
    "open": _guard_os_open(os.open),
    "chmod": _guard_no_follow(os.chmod, "os.chmod"),
    "chown": _guard_no_follow(os.chown, "os.chown"),
    "utime": _guard_no_follow(os.utime, "os.utime"),
    "lchown": _guard_no_follow(os.lchown, "os.chown", always=True),
}
if hasattr(os, "lchmod"):
    _GUARDED["lchmod"] = _guard_no_follow(os.lchmod, "os.chmod", always=True)
for _name, _guarded in _GUARDED.items():
    _original = getattr(os, _name)
    for _support in (os.supports_dir_fd, os.supports_fd, os.supports_follow_symlinks):
        if _original in _support:
            _support.add(_guarded)
    setattr(os, _name, _guarded)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "data: needs DFT fixtures from data/")
    config.addinivalue_line(
        "markers",
        "guards_existing_behaviour(reason): passes on the base code on purpose (red_green.py)",
    )


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
    if _violations:
        text = "\n".join(_violations)
        if isinstance(report.longrepr, tuple):
            text += f"\n\n{report.longrepr[2]}"
        if hasattr(report, "wasxfail"):
            text += f"\n\nThe test was marked xfail: {report.wasxfail}"
            del report.wasxfail
        if addsection := getattr(report.longrepr, "addsection", None):
            addsection("data/ guard", text)
        else:
            report.longrepr = text
        report.outcome = "failed"
    _violations.clear()
    return report
