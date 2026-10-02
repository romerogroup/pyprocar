"""Fail when changed Python files gain ruff violations or lose ruff formatting.

Usage: python ruff_new_violations.py {check|format} <base-ref>

check: per changed file, count violations by rule at <base-ref> and at HEAD.
       Fail if any rule's count grew.
format: fail if a changed file was formatted at <base-ref> (or is new) and is not now.
        Files that were never formatted stay exempt until someone formats them.
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

CONFIG = ".config/.ruff.toml"


def run(*cmd: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True)


def ruff(args: list[str], path: str, source: str) -> subprocess.CompletedProcess[str]:
    flags = ["--config", CONFIG, "--force-exclude", "--stdin-filename", path, "-"]
    return run("ruff", *args, *flags, stdin=source)


def violations(path: str, source: str | None) -> list[dict]:
    if source is None:
        return []
    result = ruff(["check", "--output-format=json", "--exit-zero"], path, source)
    return json.loads(result.stdout)


def is_formatted(path: str, source: str | None) -> bool:
    return source is None or ruff(["format", "--check"], path, source).returncode == 0


def new_violations(path: str, head: str, base: str | None) -> bool:
    now = violations(path, head)
    grown = Counter(v["code"] for v in now) - Counter(v["code"] for v in violations(path, base))
    for v in now:
        if v["code"] in grown:
            loc = v["location"]
            print(f"{path}:{loc['row']}:{loc['column']}: {v['code']} {v['message']}")
    if grown:
        print(f"{path}: {dict(grown)} more than at base (all instances listed above)\n")
    return bool(grown)


def lost_formatting(path: str, head: str, base: str | None) -> bool:
    if is_formatted(path, base) and not is_formatted(path, head):
        print(f"{path}: no longer ruff-formatted. Run: ruff format --config {CONFIG} {path}")
        return True
    return False


def main(mode: str, base_ref: str) -> int:
    gate = {"check": new_violations, "format": lost_formatting}[mode]
    diff = ["diff", "--name-only", "--no-renames", "--diff-filter=AM", base_ref, "HEAD"]
    listed = run("git", *diff, "--", "*.py")
    if listed.returncode:
        sys.exit(listed.stderr)
    changed = listed.stdout.split()
    failed = False
    for path in changed:
        old = run("git", "show", f"{base_ref}:{path}")
        base = old.stdout if old.returncode == 0 else None
        failed |= gate(path, Path(path).read_text(), base)
    print(f"{mode}: {len(changed)} changed Python files, {'FAILED' if failed else 'ok'}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
