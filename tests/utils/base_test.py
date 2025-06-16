from pathlib import Path
from typing import ClassVar

import pytest


class BaseTest:
    """`pytest` based test framework extended to facilitate testing with
    the following methods:
    - tmp_path (attribute): Temporary directory.
    - get_structure: Load a Structure from `util.structures` with its name.
    - assert_str_content_equal: Check if two strings are equal (ignore whitespaces).
    - serialize_with_pickle: Test whether object(s) can be (de)serialized with pickle.
    - assert_msonable: Test if obj is MSONable and return its serialized string.
    """

    @pytest.fixture(autouse=True)
    def _tmp_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make all tests run a in a temporary directory accessible via self.tmp_path.

        References:
            https://docs.pytest.org/en/stable/how-to/tmp_path.html
        """
        monkeypatch.chdir(tmp_path)  # change to temporary directory
        self.tmp_path = tmp_path
