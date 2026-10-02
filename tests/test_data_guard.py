import pytest

from tests.utils import DATA_DIR


@pytest.mark.parametrize("_marker", ["unmarked", pytest.param("data", marks=pytest.mark.data)])
def test_tests_cannot_write_under_data(_marker: str):
    with pytest.raises(RuntimeError, match="data/ is read-only, write to tmp_path"):
        (DATA_DIR / "no-such-dir" / "ebs.pkl").write_bytes(b"")
