"""Compare verify.sh runs against the "Expected step status" tables in features/*.md.

usage: verify.sh compare <run-dir>...

Each table row is `| <driver>.py | <step> | <status> | <site> |`. Status `ok` means the step
recorded ok: true. Status `known-defect` means it recorded ok: false with the error type and
`where` named in the site cell, for example "`AttributeError` at `pyprocar/plotter/bs_plot.py:986`".
The step `side_effects.txt` lists the backticked files the run must leave in the calc copy, and
an empty site means none. A run deviates when a step's result differs from its row, when the
summary has a step the table lacks or lacks one the table lists, when side_effects.txt differs,
or when the exit code is not 1 with a known defect and 0 without. Exits 1 if any run deviates.
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

FEATURES = Path(__file__).resolve().parent.parent / "features"
EXAMPLES = Path(__file__).resolve().parent / "examples"
HEADING = "## Expected step status"
SIDE_EFFECTS = "side_effects.txt"


@dataclass(frozen=True)
class Expected:
    status: str
    error_type: str | None
    where: str | None
    files: tuple[str, ...]
    source: str


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _ticked(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def load_tables() -> dict[str, dict[str, Expected]]:
    tables: dict[str, dict[str, Expected]] = {}
    for md in sorted(FEATURES.glob("*.md")):
        lines = md.read_text().splitlines()
        if HEADING not in lines:
            continue
        for line in lines[lines.index(HEADING) + 1 :]:
            if line.startswith("## "):
                break
            if not line.startswith("|") or set(line) <= set("|- "):
                continue
            cells = _cells(line)
            if len(cells) != 4 or cells[0] == "Driver":
                continue
            driver, step = (_ticked(c)[0] if _ticked(c) else c for c in cells[:2])
            status, site = cells[2], cells[3]
            ticks = _ticked(site)
            if status == "known-defect" and len(ticks) < 2:
                sys.exit(
                    f"{md.name}: known-defect row {driver} {step} needs `Error` at `file:line`"
                )
            if status not in ("ok", "known-defect"):
                sys.exit(
                    f"{md.name}: row {driver} {step} has status {status!r}, not ok or known-defect"
                )
            tables.setdefault(driver, {})[step] = Expected(
                status=status,
                error_type=ticks[0] if status == "known-defect" else None,
                where=ticks[1] if status == "known-defect" else None,
                files=tuple(ticks) if step == SIDE_EFFECTS else (),
                source=md.name,
            )
    return tables


def driver_of(ev: Path) -> str | None:
    """The scripts/examples driver whose bytes the run's evidence/driver.py copy has.

    A run of an older version of a driver matches none, so its rows would not apply.
    """
    copy = ev / "driver.py"
    if not copy.is_file():
        return None
    body = copy.read_bytes()
    return next((p.name for p in sorted(EXAMPLES.glob("*.py")) if p.read_bytes() == body), None)


def compare(run: Path, tables: dict[str, dict[str, Expected]]) -> list[str]:
    ev = run / "evidence"
    driver = driver_of(ev)
    if driver is None:
        return [f"{ev / 'driver.py'} is missing or matches no driver in {EXAMPLES}"]
    table = tables.get(driver)
    if table is None:
        return [f"no Expected step status rows for {driver} in features/*.md"]
    summary_file = ev / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.is_file() else {}
    steps = {k: v for k, v in summary.items() if isinstance(v, dict) and "ok" in v}
    out = []
    for step, exp in table.items():
        if step == SIDE_EFFECTS:
            continue
        got = steps.get(step)
        if got is None:
            out.append(f"{step}: missing from summary.json (expected {exp.status})")
            continue
        error = str(got.get("error", "")).splitlines()[0] if not got["ok"] else ""
        crash = f"{error} at {got.get('where')}"
        if exp.status == "ok" and got["ok"] is not True:
            out.append(f"{step}: expected ok, got {crash}")
        elif exp.status == "known-defect" and got["ok"] is True:
            out.append(f"{step}: expected {exp.where}, got ok (fixed? update {exp.source})")
        elif exp.status == "known-defect" and (
            not error.startswith(f"{exp.error_type}:") or got.get("where") != exp.where
        ):
            out.append(f"{step}: expected {exp.error_type} at {exp.where}, got {crash}")
    for step in steps.keys() - table.keys():
        out.append(f"{step}: not in the table for {driver}")
    side = table.get(SIDE_EFFECTS)
    if side is None:
        out.append(f"{SIDE_EFFECTS}: no row for {driver}")
    else:
        listed = (ev / SIDE_EFFECTS).read_text().split() if (ev / SIDE_EFFECTS).is_file() else []
        got_files = sorted(p.removeprefix("./") for p in listed)
        if got_files != sorted(side.files):
            out.append(f"{SIDE_EFFECTS}: expected {sorted(side.files)}, got {got_files}")
    code_file = ev / "exit_code"
    code = code_file.read_text().strip() if code_file.is_file() else "missing"
    want = "1" if any(e.status == "known-defect" for e in table.values()) else "0"
    if code != want:
        out.append(f"exit code: expected {want}, got {code}")
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    tables = load_tables()
    deviated = 0
    for arg in argv:
        run = Path(arg).resolve()
        problems = compare(run, tables)
        print(f"{'DEVIATES' if problems else 'matches '}  {run.name}")
        for p in problems:
            print(f"  {p}")
        deviated += bool(problems)
    print(f"{deviated} of {len(argv)} run(s) deviate")
    return 1 if deviated else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
