import argparse
import ast
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from itertools import count
from pathlib import Path
from typing import Literal, TypeGuard, assert_never

MARK = "guards_existing_behaviour"
LIBRARY = "pyprocar"
TESTS = "tests"
TEST_ENV = "RED_GREEN_TEST"
REPORT_ENV = "RED_GREEN_REPORT"
TIMEOUT_SECONDS = 300
PYTEST_ARGS = (
    "-c", ".config/.pytest.toml", "--rootdir", ".", "-m", "not data", "-n", "0", "-q",
    "-p", "red_green_plugin", "-p", "no:cacheprovider", "--continue-on-collection-errors",
)  # fmt: skip


class Outcome(StrEnum):
    GREEN = "passes on base"
    RED = "fails on base"
    NOT_COLLECTED = "not collected on base"
    TIMED_OUT = "hangs on base"
    SKIPPED = "skipped on base"
    DATA = "marked data, not run"


@dataclass(frozen=True, slots=True)
class ChangedTest:
    nodeid: str
    line: int
    change: Literal["added", "changed"]


@dataclass(frozen=True, slots=True)
class BaseResult:
    outcome: Outcome
    detail: str
    guard: str | None


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


TestDef = ast.FunctionDef | ast.AsyncFunctionDef


def is_test_def(node: ast.stmt) -> TypeGuard[TestDef]:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test")


def test_functions(source: str) -> Iterator[tuple[str, TestDef]]:
    for node in ast.parse(source).body:
        if is_test_def(node):
            yield node.name, node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            yield from ((f"{node.name}::{f.name}", f) for f in node.body if is_test_def(f))


def body_fingerprint(func: TestDef) -> str:
    body = func.body[1:] if ast.get_docstring(func) is not None else func.body
    return "\n".join(ast.dump(part) for part in [*func.decorator_list, func.args, *body])


def is_test_file(path: str) -> bool:
    return Path(path).match("test_*.py")


def changed_tests(base: str, head: str) -> list[ChangedTest]:
    base_names: set[str] = set()
    base_bodies: set[str] = set()
    for path in filter(is_test_file, git("ls-tree", "-r", "--name-only", base, TESTS).split()):
        for name, func in test_functions(git("show", f"{base}:{path}")):
            base_names.add(f"{path}::{name}")
            base_bodies.add(body_fingerprint(func))
    found: list[ChangedTest] = []
    for path in filter(
        is_test_file,
        git("diff", "--name-only", "--diff-filter=AMR", base, head, "--", TESTS).split(),
    ):
        for name, func in test_functions(git("show", f"{head}:{path}")):
            if body_fingerprint(func) not in base_bodies:
                nodeid = f"{path}::{name}"
                found.append(
                    ChangedTest(nodeid, func.lineno, "changed" if nodeid in base_names else "added")
                )
    return found


def extract(ref: str, dest: Path, *paths: str) -> None:
    archive = subprocess.run(
        ["git", "archive", ref, *paths], capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")


def base_tree_with_head_tests(base: str, head: str, dest: Path) -> None:
    extract(base, dest, "--", ".", f":(exclude){TESTS}")
    extract(head, dest, TESTS)
    (dest / LIBRARY / "_version.py").write_text(
        '__version__ = version = "0+base"\n', encoding="utf-8"
    )


def run_on_base(index: int, test: ChangedTest, scratch: Path) -> BaseResult:
    report = scratch / f"red_green_{index}.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(scratch), str(Path(__file__).resolve().parent)]),
        TEST_ENV: test.nodeid,
        REPORT_ENV: str(report),
    }
    cmd = [sys.executable, "-m", "pytest", test.nodeid.split("::", 1)[0], *PYTEST_ARGS]
    try:
        result = subprocess.run(
            cmd, cwd=scratch, env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        return BaseResult(Outcome.TIMED_OUT, f"still running after {TIMEOUT_SECONDS} s", None)
    if result.returncode not in (0, 1, 5) or not report.exists():
        print(result.stdout[-5000:], result.stderr[-5000:], sep="\n")
        sys.exit(f"pytest exited {result.returncode} on {test.nodeid}; the gate cannot judge it")
    raw = json.loads(report.read_text(encoding="utf-8"))
    library = raw["library_file"]
    if library is not None and not Path(library).resolve().is_relative_to(scratch.resolve()):
        sys.exit(
            f"{test.nodeid} imported {LIBRARY} from {library}, not from the base copy in {scratch}"
        )
    return BaseResult(classify(raw["cases"], raw["data"]), raw["error"], raw["guard"])


def classify(cases: list[str], data: bool) -> Outcome:
    if data:
        return Outcome.DATA
    if not cases:
        return Outcome.NOT_COLLECTED
    if any(case in ("failed", "xfailed") for case in cases):
        return Outcome.RED
    if all(case == "skipped" for case in cases):
        return Outcome.SKIPPED
    return Outcome.GREEN


def judge(result: BaseResult) -> tuple[bool, str]:
    if result.guard is not None and not result.guard.strip():
        return False, f"FAIL: @pytest.mark.{MARK} needs a reason string"
    match result.outcome:
        case Outcome.GREEN if result.guard:
            return True, f"ok, guards existing behaviour: {result.guard}"
        case Outcome.GREEN:
            return False, (
                "FAIL: make it fail without the fix, or, if it guards behaviour the base "
                f'already has, mark it @pytest.mark.{MARK}(reason="...")'
            )
        case Outcome.RED | Outcome.NOT_COLLECTED | Outcome.TIMED_OUT:
            return True, f"ok: {result.detail}"
        case Outcome.SKIPPED | Outcome.DATA:
            return True, "not judged"
        case _:
            assert_never(result.outcome)


def main(base_ref: str, head_ref: str = "HEAD") -> int:
    base = git("merge-base", base_ref, head_ref).strip()
    head = git("rev-parse", head_ref).strip()
    if not git("diff", "--name-only", base, head, "--", ".", f":!{TESTS}").strip():
        print(
            f"red/green: no change outside {TESTS}/ from {base[:8]} to {head[:8]}; nothing to check"
        )
        return 0
    tests = changed_tests(base, head)
    if not tests:
        print(f"red/green: {head[:8]} adds or changes no test functions relative to {base[:8]}")
        return 0
    print(f"red/green: {len(tests)} added or changed tests of {head[:8]}, run on base {base[:8]}")
    with tempfile.TemporaryDirectory(prefix="red_green_") as tmp:
        scratch = Path(tmp)
        base_tree_with_head_tests(base, head, scratch)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            results = list(pool.map(partial(run_on_base, scratch=scratch), count(), tests))
    failed = False
    for test, result in zip(tests, results, strict=True):
        ok, verdict = judge(result)
        failed |= not ok
        print(f"  {result.outcome.value:<22} {test.change:<8} {test.nodeid}\n    {verdict}")
        if not ok and os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::error file={test.nodeid.split('::', 1)[0]},line={test.line}::{verdict}")
    print(f"red/green: {'FAILED' if failed else 'ok'}")
    return int(failed)


def function_id(nodeid: str) -> str:
    return nodeid.split("[", 1)[0]


if __name__ == "__main__":
    description = (
        f"Fail on each test that head adds or changes and that passes on the merge base's tree "
        f"with {TESTS}/ from head, unless it is marked @pytest.mark.{MARK}(reason)."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("base_ref")
    parser.add_argument("head_ref", nargs="?", default="HEAD")
    args = parser.parse_args()
    sys.exit(main(args.base_ref, args.head_ref))
