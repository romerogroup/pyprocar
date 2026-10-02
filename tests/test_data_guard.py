import pytest

from tests.utils import ROOT_DIR

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
"""


def test_writes_under_data_fail_even_when_swallowed(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PYTHONPATH", str(ROOT_DIR))
    pytester.makeconftest("from tests.conftest import *")
    pytester.makepyfile(test_inner=INNER_TESTS)

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rA")

    result.assert_outcomes(passed=1, failed=4)
    result.stdout.fnmatch_lines_random(
        [
            "FAILED test_inner.py::test_direct_write - RuntimeError: *",
            "FAILED test_inner.py::test_swallowed_write",
            "FAILED test_inner.py::test_swallowed_write_in_data_test",
            "FAILED test_inner.py::test_write_through_symlink",
            "PASSED test_inner.py::test_tmp_path_write",
            "*::test_swallowed_write writes */no-such-dir/ebs.pkl/; data/ is read-only*",
            "*::test_write_through_symlink writes */no-such-dir/ebs.pkl/; data/ is read-only*",
        ]
    )
