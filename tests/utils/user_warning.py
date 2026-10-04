import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


@contextmanager
def user_warning(caller: str, match: str) -> Iterator[pytest.WarningsRecorder]:
    with pytest.warns(UserWarning, match=match) as record:
        yield record
    misplaced = [
        f"{w.category.__name__} at {w.filename}:{w.lineno}"
        for w in record
        if re.search(match, str(w.message))
        and (w.category is not UserWarning or w.filename != caller)
    ]
    assert misplaced == [], f"expected a UserWarning naming a line in {caller}, got {misplaced}"
