"""Every task in pixi.toml must point at things that exist.

Broken references shipped three times: the typecheck task called
run_silent_typecheck.sh while the file was run_silent_typcheck.sh (#227), `test`
depended on an undefined `test-all` and `ci-check` ran in an undefined `ci`
environment (#260, #265), and `init` and `release` ran scripts that never
existed. pixi reports these only when someone runs the task.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PIXI_RUN = re.compile(r"pixi run(?:\s+-{1,2}[\w-]+)*\s+(?:-e|--environment)\s+(\S+)\s+([\w-]+)")
PROJECT_PATH = re.compile(r"\$PIXI_PROJECT_ROOT/([^\s\"']+)")
SCRIPT_PATH = re.compile(r"\b(?:python|bash)\s+([\w./-]+\.(?:py|sh))\b")


def _manifest() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))


def _tasks_by_feature(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = {"default": manifest.get("tasks", {})}
    for name, feature in manifest.get("feature", {}).items():
        features[name] = feature.get("tasks", {})
    return features


def _features_by_environment(manifest: dict[str, Any]) -> dict[str, list[str]]:
    environments = {}
    for name, spec in manifest["environments"].items():
        features = spec if isinstance(spec, list) else spec.get("features", [])
        no_default = isinstance(spec, dict) and spec.get("no-default-feature", False)
        environments[name] = list(features) + ([] if no_default else ["default"])
    return environments


def broken_references(manifest: dict[str, Any], root: Path) -> list[str]:
    tasks_by_feature = _tasks_by_feature(manifest)
    environments = _features_by_environment(manifest)
    tasks_in = {
        env: {task for feature in features for task in tasks_by_feature.get(feature, {})}
        for env, features in environments.items()
    }
    problems = [
        f"environment {env!r} uses undefined feature {feature!r}"
        for env, features in environments.items()
        for feature in features
        if feature not in tasks_by_feature
    ]
    for feature, tasks in tasks_by_feature.items():
        visible = {t for env, fs in environments.items() if feature in fs for t in tasks_in[env]}
        for name, task in tasks.items():
            spec = task if isinstance(task, dict) else {"cmd": task}
            where = f"task {name!r} ([feature.{feature}.tasks])"
            for dep in spec.get("depends-on", []):
                dep_name = dep if isinstance(dep, str) else dep["task"]
                if dep_name not in visible:
                    problems.append(f"{where} depends on undefined task {dep_name!r}")
            cmd = spec.get("cmd", "")
            cmd = " ".join(cmd) if isinstance(cmd, list) else cmd
            for env, target in PIXI_RUN.findall(cmd):
                if env not in environments:
                    problems.append(f"{where} runs in undefined environment {env!r}")
                elif target not in tasks_in[env]:
                    problems.append(f"{where} runs {target!r}, not a task in {env!r}")
            for path in PROJECT_PATH.findall(cmd) + SCRIPT_PATH.findall(cmd):
                if not (root / path).exists():
                    problems.append(f"{where} uses missing file {path}")
    return problems


def test_pixi_tasks_reference_defined_tasks_environments_and_files():
    assert broken_references(_manifest(), ROOT) == []
