import shutil
import stat

import pytest

from tests.utils import writable_copy

pytestmark = pytest.mark.guards_existing_behaviour(
    reason="tests the tests/utils helper itself; red_green copies the branch's tests/ into the "
    "base tree, so the helper exists there too"
)

WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _entries(root):
    return [p for p in [root, *root.rglob("*")] if not p.is_symlink()]


@pytest.fixture
def read_only_fixture(tmp_path):
    src = tmp_path / "fixture"
    outside = tmp_path / "outside"
    (src / "sub").mkdir(parents=True)
    outside.mkdir()
    (src / "PROCAR").write_text("procar")
    (src / "sub" / "OUTCAR").write_text("outcar")
    (src / "ebs.pkl").write_text("cache")
    (outside / "POSCAR").write_text("poscar")
    (src / "linked").symlink_to(outside)
    for path in [*_entries(src), *_entries(outside)]:
        path.chmod(path.stat().st_mode & ~WRITE_BITS)
    yield src
    for path in [*_entries(src), *_entries(outside)]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def test_copy_of_a_read_only_fixture_takes_writes_and_removal(read_only_fixture, tmp_path):
    dst = writable_copy(read_only_fixture, tmp_path / "calc")

    (dst / "sub" / "OUTCAR").write_text("rewritten")
    (dst / "sub" / "new.pkl").write_text("new")
    assert (dst / "PROCAR").read_text() == "procar"
    assert (dst / "sub" / "OUTCAR").read_text() == "rewritten"
    shutil.rmtree(dst)
    assert not dst.exists()


def test_copy_follows_a_symlink_and_leaves_its_target_read_only(read_only_fixture, tmp_path):
    dst = writable_copy(read_only_fixture, tmp_path / "calc")

    (dst / "linked" / "POSCAR").write_text("rewritten")
    assert not (dst / "linked").is_symlink()
    assert (tmp_path / "outside" / "POSCAR").read_text() == "poscar"
    assert [p.name for p in _entries(tmp_path / "outside") if p.stat().st_mode & WRITE_BITS] == []


def test_copy_skips_ignored_names(read_only_fixture, tmp_path):
    dst = writable_copy(
        read_only_fixture, tmp_path / "calc", ignore=shutil.ignore_patterns("*.pkl", "linked")
    )

    assert sorted(p.name for p in dst.rglob("*")) == ["OUTCAR", "PROCAR", "sub"]
