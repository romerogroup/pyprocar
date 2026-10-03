import re
from pathlib import Path

import pyprocar

PACKAGE_DIR = Path(pyprocar.__file__).parent


def test_no_calls_to_cm_get_cmap():
    """matplotlib.cm.get_cmap is deprecated since 3.7 and removed in 3.11."""
    pattern = re.compile(r"\bcm\.get_cmap\(")
    offenders = [
        f"{path.relative_to(PACKAGE_DIR)}:{lineno}"
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line) and not line.lstrip().startswith("#")
    ]

    assert offenders == []
