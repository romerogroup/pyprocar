import json
import os
import sys
from pathlib import Path

import pytest
from red_green import LIBRARY, MARK, REPORT_ENV, TEST_ENV, function_id

_cases: list[str] = []
_report: dict[str, object] = {"error": "", "guard": None, "data": False}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    wanted = os.environ[TEST_ENV]
    config.hook.pytest_deselected(items=[i for i in items if function_id(i.nodeid) != wanted])
    items[:] = [i for i in items if function_id(i.nodeid) == wanted]


def pytest_itemcollected(item: pytest.Item) -> None:
    if function_id(item.nodeid) != os.environ[TEST_ENV]:
        return
    if mark := item.get_closest_marker(MARK):
        reason = mark.kwargs.get("reason", mark.args[0] if mark.args else None)
        _report["guard"] = reason if isinstance(reason, str) else ""
    if item.get_closest_marker("data"):
        _report["data"] = True


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if hasattr(report, "wasxfail"):
        result = "xfailed" if report.skipped else "passed"
    else:
        result = report.outcome
    if report.when == "call" or result != "passed":
        _cases.append(result)
    pytest_collectreport(report)


def pytest_collectreport(report: pytest.TestReport | pytest.CollectReport) -> None:
    if report.failed and not _report["error"]:
        _report["error"] = exception_line(report)[:200]


def exception_line(report: pytest.TestReport | pytest.CollectReport) -> str:
    crash = getattr(report.longrepr, "reprcrash", None)
    if message := getattr(crash, "message", "").strip():
        return message.splitlines()[0]
    lines = [stripped for line in str(report.longrepr).splitlines() if (stripped := line.strip())]
    return lines[-1] if lines else ""


def pytest_sessionfinish() -> None:
    lib = sys.modules.get(LIBRARY)
    report = {**_report, "cases": _cases, "library_file": lib.__file__ if lib else None}
    Path(os.environ[REPORT_ENV]).write_text(json.dumps(report), encoding="utf-8")
