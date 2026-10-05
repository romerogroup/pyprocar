"""Step recorder for multi-entry-point drivers. verify.sh puts this dir on PYTHONPATH.

Each @step runs isolated: a crash is recorded in summary.json (error + innermost frame)
and the next step still runs. finish() prints the summary and exits 1 if any step failed.
"""

import json
import os
import shutil
import stat
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CALC, EV = Path(os.environ["CALC"]), Path(os.environ["EVIDENCE"])
REPO = Path(os.environ["REPO"])
SUMMARY: dict = {}


def _write():
    (EV / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


def step(name):
    def deco(fn):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            try:
                SUMMARY[name] = {"ok": True, **(fn() or {})}
            except Exception as e:
                frames = traceback.extract_tb(e.__traceback__)
                own = [f for f in frames if "/pyprocar/pyprocar/" in f.filename] or frames
                SUMMARY[name] = {
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}"[:300],
                    "where": f"{own[-1].filename.split('/pyprocar/')[-1]}:{own[-1].lineno}",
                }
        user_warnings = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        if user_warnings:
            SUMMARY[name]["warnings"] = user_warnings
        plt.close("all")
        _write()
        return fn

    return deco


def writable_copy(src, dst):
    """Copy a read-only fixture tree into the run; return dst with user write on every entry."""
    shutil.copytree(src, dst)
    for path in [dst, *dst.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    return dst


def png(name, fig=None):
    """Save a matplotlib figure into EVIDENCE; return its byte size."""
    p = EV / f"{name}.png"
    (fig or plt.gcf()).savefig(p, dpi=110)
    return p.stat().st_size


def distinct_colors(path):
    """Distinct RGBA values in a screenshot; ~1 means blank."""
    img = plt.imread(path)
    return int(len(np.unique(img.reshape(-1, img.shape[-1]), axis=0)))


def finish():
    _write()
    print(json.dumps(SUMMARY, indent=1, default=str))
    failed = [k for k, v in SUMMARY.items() if isinstance(v, dict) and v.get("ok") is False]
    print("FAILED steps:", failed or "none")
    sys.exit(1 if failed else 0)
