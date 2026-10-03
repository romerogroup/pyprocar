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
import shutil
import sys

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


def test_wrappers_keep_os_capabilities():
    assert os.open in os.supports_dir_fd
    assert os.utime in os.supports_follow_symlinks
    assert os.chown in os.supports_follow_symlinks


def test_metadata_on_link_to_data_without_following(tmp_path):
    link = tmp_path / "link"
    os.symlink(TARGET, link)
    os.utime(link, follow_symlinks=False)
    os.chown(link, os.getuid(), os.getgid(), follow_symlinks=False)
    os.lchown(link, os.getuid(), os.getgid())
    try:
        os.chmod(link, 0o600, follow_symlinks=False)
    except NotImplementedError:
        pass


def test_copytree_keeps_links_to_data(tmp_path):
    (tmp_path / "src").mkdir()
    os.symlink(TARGET, tmp_path / "src" / "link")
    shutil.copytree(tmp_path / "src", tmp_path / "dst", symlinks=True)


def test_utime_through_link(tmp_path):
    link = tmp_path / "link"
    os.symlink(TARGET, link)
    swallow(lambda: os.utime(link))


def test_utime_without_following_in_data():
    swallow(lambda: os.utime(TARGET, follow_symlinks=False))


def test_lchown_in_data():
    swallow(lambda: os.lchown(TARGET, os.getuid(), os.getgid()))


def test_write_from_audit_hook_inside_wrapped_open(tmp_path):
    armed = [True]

    def hook(event, args):
        if armed[0] and event == "open" and args[0] == "inner-open":
            armed[0] = False
            swallow(lambda: TARGET.write_bytes(b""))

    sys.addaudithook(hook)
    fd = os.open(tmp_path, os.O_RDONLY)
    os.close(os.open("inner-open", os.O_WRONLY | os.O_CREAT, dir_fd=fd))
    os.close(fd)


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
    "test_utime_through_link": "data/no-such-dir/ebs.pkl",
    "test_utime_without_following_in_data": "data/no-such-dir/ebs.pkl",
    "test_lchown_in_data": "data/no-such-dir/ebs.pkl",
    "test_write_from_audit_hook_inside_wrapped_open": "data/no-such-dir/ebs.pkl",
}


def test_writes_under_data_fail_even_when_swallowed(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PYTHONPATH", str(ROOT_DIR))
    pytester.mkdir("data")
    pytester.makeconftest(INNER_CONFTEST)
    pytester.makepyfile(test_inner=INNER_TESTS)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rA")

    result.assert_outcomes(passed=5, failed=len(WRITTEN_PATH_BY_FAILING_TEST))
    result.stdout.fnmatch_lines_random(
        [
            "FAILED test_inner.py::test_direct_write - RuntimeError: *",
            "PASSED test_inner.py::test_tmp_path_write",
            "PASSED test_inner.py::test_remove_symlink_to_data",
            "PASSED test_inner.py::test_wrappers_keep_os_capabilities",
            "PASSED test_inner.py::test_metadata_on_link_to_data_without_following",
            "PASSED test_inner.py::test_copytree_keeps_links_to_data",
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
