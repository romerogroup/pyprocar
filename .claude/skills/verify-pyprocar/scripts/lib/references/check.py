"""Run every reference's analytic validation without a fixture; exit 1 if one fails.

usage: verify.sh exec python .claude/skills/verify-pyprocar/scripts/lib/references/check.py
"""

import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from references import orbits, spin_ibz, unfold  # noqa: E402

VALIDATIONS: dict[str, Callable[[], dict]] = {
    "unfold": unfold.validate,
    "spin_ibz": spin_ibz.validate,
    "orbits": orbits.validate,
}

failed = []
for name, validate in VALIDATIONS.items():
    try:
        print(name, json.dumps(validate(), default=str))
    except Exception:
        failed.append(name)
        print(name, "FAILED", traceback.format_exc(), sep="\n")
print("FAILED:", failed or "none")
sys.exit(1 if failed else 0)
