"""A message the user must see goes through warnings.warn, never the "user" logger.

The "user" logger is set to ERROR on import, so ``user_logger.warning(...)`` prints
nothing in a default call (#284): bandsplot's "fermi is not set" and FermiHandler's
"No Fermi surface found" were invisible unless something first lowered the level. The
logger carries only verbose progress (info and debug). Tests see a warning with
``pytest.warns``; a test that lowers the user logger to see one hides this defect.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOUD_METHODS = {"warning", "warn", "error", "critical", "exception", "fatal"}
REPLACEMENT = (
    "use warnings.warn (pyprocar.utils.log_utils.warn_user) for a message the user must"
    " see, or user_logger.info/debug for verbose progress"
)


def _is_user_logger_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
        and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        == "getLogger"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "user"
    )


def _shown(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def loud_user_logger_calls(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        bound = {
            ast.unparse(target)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and _is_user_logger_call(node.value)
            for target in node.targets
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LOUD_METHODS
            ):
                continue
            receiver = node.func.value
            if ast.unparse(receiver) in bound or _is_user_logger_call(receiver):
                found.append(
                    f"{_shown(path)}:{node.lineno}: user logger"
                    f" .{node.func.attr}() is hidden at its default ERROR level; {REPLACEMENT}"
                )
    return found


def user_logger_lowered_in(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"at_level", "set_level"}
                and any(
                    kw.arg == "logger"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "user"
                    for kw in node.keywords
                )
            ):
                found.append(
                    f"{_shown(path)}:{node.lineno}: caplog.{node.func.attr}"
                    "(..., logger='user') lowers the user logger so the test sees what a user"
                    " does not; assert the warning with pytest.warns"
                )
    return found


def test_library_never_warns_through_the_user_logger():
    assert loud_user_logger_calls(sorted((ROOT / "pyprocar").rglob("*.py"))) == []


def test_tests_never_lower_the_user_logger_to_see_a_warning():
    assert user_logger_lowered_in(sorted((ROOT / "tests").rglob("*.py"))) == []


def _line_number(hit: str) -> int:
    return int(hit.split(": ")[0].rsplit(":", 1)[1])


def test_the_checks_flag_each_loud_call_and_each_lowered_level(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(
        "import logging\n"
        "user_logger = logging.getLogger('user')\n"
        "user_logger.info('progress')\n"
        "user_logger.warning('hidden')\n"
        "logging.getLogger('user').error('hidden too')\n"
        "logging.getLogger(__name__).warning('package log')\n"
        "with caplog.at_level(logging.WARNING, logger='user'):\n"
        "    pass\n"
        "caplog.set_level(logging.INFO, logger='pyprocar')\n",
        encoding="utf-8",
    )

    loud = loud_user_logger_calls([module])
    lowered = user_logger_lowered_in([module])

    assert [_line_number(hit) for hit in loud] == [4, 5]
    assert "warnings.warn" in loud[0]
    assert [_line_number(hit) for hit in lowered] == [7]
