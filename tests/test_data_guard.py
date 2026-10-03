import pytest

from tests.utils import ROOT_DIR

INNER_CONFTEST = """
from pathlib import Path

import tests.utils

tests.utils.DATA_DIR = Path(__file__).parent / "data"

from tests.conftest import *
"""

INNER_TESTS = """
import os

import pytest

from tests.utils import DATA_DIR

TARGET = DATA_DIR / "no-such-dir" / "ebs.pkl"


def swallow(write):
    try:
        write()
    except Exception:
        pass


def test_direct_write():
    TARGET.write_bytes(b"")


def test_swallowed_write():
    swallow(lambda: TARGET.write_bytes(b""))


@pytest.mark.data
def test_swallowed_write_in_data_test():
    swallow(lambda: TARGET.write_bytes(b""))


def test_write_through_symlink(tmp_path):
    alias = tmp_path / "alias"
    os.symlink(DATA_DIR, alias)
    swallow(lambda: (alias / "no-such-dir" / "ebs.pkl").write_bytes(b""))


def test_tmp_path_write(tmp_path):
    (tmp_path / "ebs.pkl").write_bytes(b"")


def test_remove_symlink_to_data(tmp_path):
    alias = tmp_path / "alias"
    os.symlink(DATA_DIR, alias)
    os.remove(alias)


def test_os_open_keeps_dir_fd_support():
    assert os.open in os.supports_dir_fd


def test_write_then_skip():
    swallow(lambda: TARGET.write_bytes(b""))
    pytest.skip("skipped after the write")


@pytest.mark.xfail(reason="fails after the write")
def test_write_then_xfail():
    swallow(lambda: TARGET.write_bytes(b""))
    assert False


def test_write_then_fail():
    swallow(lambda: TARGET.write_bytes(b""))
    assert False, "unrelated failure"


def test_symlink_into_data(tmp_path):
    swallow(lambda: os.symlink(tmp_path, TARGET))


def test_hardlink_into_data(tmp_path):
    (tmp_path / "src").write_bytes(b"")
    swallow(lambda: os.link(tmp_path / "src", TARGET))


def test_hardlink_out_of_data(tmp_path):
    swallow(lambda: os.link(TARGET, tmp_path / "dst"))


def test_chmod():
    swallow(lambda: os.chmod(TARGET, 0o600))


def test_utime():
    swallow(lambda: os.utime(TARGET))


@pytest.mark.data
def test_chmod_by_fd():
    fd = os.open(DATA_DIR, os.O_RDONLY)
    swallow(lambda: os.chmod(fd, os.stat(fd).st_mode))
    os.close(fd)


@pytest.mark.data
def test_chmod_relative_to_dir_fd():
    fd = os.open(DATA_DIR, os.O_RDONLY)
    swallow(lambda: os.chmod("no-such-dir/ebs.pkl", 0o600, dir_fd=fd))
    os.close(fd)


@pytest.mark.data
def test_open_relative_to_dir_fd():
    fd = os.open(DATA_DIR, os.O_RDONLY)
    swallow(lambda: os.open("no-such-dir/ebs.pkl", os.O_WRONLY | os.O_CREAT, dir_fd=fd))
    os.close(fd)
"""

WRITTEN_PATH_BY_FAILING_TEST = {
    "test_direct_write": "data/no-such-dir/ebs.pkl",
    "test_swallowed_write": "data/no-such-dir/ebs.pkl",
    "test_swallowed_write_in_data_test": "data/no-such-dir/ebs.pkl",
    "test_write_through_symlink": "data/no-such-dir/ebs.pkl",
    "test_write_then_skip": "data/no-such-dir/ebs.pkl",
    "test_write_then_xfail": "data/no-such-dir/ebs.pkl",
    "test_write_then_fail": "data/no-such-dir/ebs.pkl",
    "test_symlink_into_data": "data/no-such-dir/ebs.pkl",
    "test_hardlink_into_data": "data/no-such-dir/ebs.pkl",
    "test_hardlink_out_of_data": "data/no-such-dir/ebs.pkl",
    "test_chmod": "data/no-such-dir/ebs.pkl",
    "test_utime": "data/no-such-dir/ebs.pkl",
    "test_chmod_by_fd": "data",
    "test_chmod_relative_to_dir_fd": "data/no-such-dir/ebs.pkl",
    "test_open_relative_to_dir_fd": "data/no-such-dir/ebs.pkl",
}


def test_writes_under_data_fail_even_when_swallowed(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PYTHONPATH", str(ROOT_DIR))
    pytester.mkdir("data")
    pytester.makeconftest(INNER_CONFTEST)
    pytester.makepyfile(test_inner=INNER_TESTS)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rA")

    result.assert_outcomes(passed=3, failed=len(WRITTEN_PATH_BY_FAILING_TEST))
    result.stdout.fnmatch_lines_random(
        [
            "FAILED test_inner.py::test_direct_write - RuntimeError: *",
            "PASSED test_inner.py::test_tmp_path_write",
            "PASSED test_inner.py::test_remove_symlink_to_data",
            "PASSED test_inner.py::test_os_open_keeps_dir_fd_support",
            "*Skipped: skipped after the write*",
            "*The test was marked xfail: fails after the write*",
            "*AssertionError: unrelated failure*",
            *(f"FAILED test_inner.py::{name}*" for name in WRITTEN_PATH_BY_FAILING_TEST),
            *(
                f"*::{name} writes */{path}; data/ is read-only, write to tmp_path"
                for name, path in WRITTEN_PATH_BY_FAILING_TEST.items()
            ),
        ]
    )
