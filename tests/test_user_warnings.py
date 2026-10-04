import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WARN_USER_HOME = ROOT / "pyprocar" / "utils" / "log_utils.py"
USER_WARNING_HELPER = ROOT / "tests" / "utils" / "user_warning.py"
LOUD_METHODS = {"warning", "warn", "error", "critical", "exception", "fatal"}
LOUD_LEVEL_NAMES = {"WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"}
LOWEST_LOUD_LEVEL = 30
REPLACEMENT = (
    "use warnings.warn (pyprocar.utils.log_utils.warn_user) for a message the user must"
    " see, or user_logger.info/debug for verbose progress"
)


def _shown(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _module_parts(path: Path, root: Path) -> tuple[str, ...]:
    return (
        path.relative_to(root).with_suffix("").parts if path.is_relative_to(root) else (path.stem,)
    )


def _imported_from(node: ast.ImportFrom, parts: tuple[str, ...]) -> str:
    if node.level == 0:
        return node.module or ""
    package = parts[:-1]
    base = package[: len(package) - node.level + 1]
    return ".".join((*base, *([node.module] if node.module else [])))


class UserLoggerNames:
    """The expressions that hold the `user` logger in one module."""

    def __init__(self, tree: ast.Module, parts: tuple[str, ...], exports: dict[str, set[str]]):
        self.tree: ast.Module = tree
        self.get_logger: set[str] = {"getLogger"}
        self.bound: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                source = _imported_from(node, parts)
                for alias in node.names:
                    if source == "logging" and alias.name == "getLogger":
                        self.get_logger.add(alias.asname or alias.name)
                    elif alias.name in exports.get(source, set()):
                        self.bound.add(alias.asname or alias.name)
        while self._bind_assignments():
            pass

    def _bind_assignments(self) -> bool:
        before = len(self.bound)
        for node in ast.walk(self.tree):
            pairs: list[tuple[ast.expr, ast.expr]]
            if isinstance(node, ast.Assign):
                pairs = [(target, node.value) for target in node.targets]
            elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
                pairs = [(node.target, node.value)]
            else:
                continue
            for target, value in pairs:
                if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
                    pairs.extend(zip(target.elts, value.elts, strict=False))
                elif self.holds(value):
                    self.bound.add(ast.unparse(target))
        return len(self.bound) > before

    def holds(self, node: ast.expr) -> bool:
        if ast.unparse(node) in self.bound:
            return True
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        named = [*node.args[:1], *(kw.value for kw in node.keywords if kw.arg == "name")]
        return (
            name in self.get_logger
            and len(named) == 1
            and isinstance(named[0], ast.Constant)
            and named[0].value == "user"
        )


def user_logger_names(paths: list[Path], root: Path = ROOT) -> dict[Path, UserLoggerNames]:
    trees = {path: ast.parse(path.read_text(encoding="utf-8"), str(path)) for path in paths}
    exports: dict[str, set[str]] = {}
    while True:
        names = {
            path: UserLoggerNames(tree, _module_parts(path, root), exports)
            for path, tree in trees.items()
        }
        found = {
            ".".join(_module_parts(path, root)): {name for name in n.bound if name.isidentifier()}
            for path, n in names.items()
        }
        if found == exports:
            return names
        exports = found


def _is_loud_level(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and node.value >= LOWEST_LOUD_LEVEL
    name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
    return name in LOUD_LEVEL_NAMES


def _is_loud_call(node: ast.Call) -> bool:
    assert isinstance(node.func, ast.Attribute)
    if node.func.attr in LOUD_METHODS:
        return True
    levels = [*node.args[:1], *(kw.value for kw in node.keywords if kw.arg == "level")]
    return node.func.attr == "log" and any(_is_loud_level(level) for level in levels)


def _calls(names: UserLoggerNames):
    for node in ast.walk(names.tree):
        if isinstance(node, ast.Call):
            receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
            yield node, receiver is not None and names.holds(receiver)


def loud_user_logger_calls(paths: list[Path], root: Path = ROOT) -> list[str]:
    found = []
    for path, names in user_logger_names(paths, root).items():
        for node, on_user_logger in _calls(names):
            if on_user_logger and _is_loud_call(node):
                assert isinstance(node.func, ast.Attribute)
                found.append(
                    f"{_shown(path)}:{node.lineno}: user logger"
                    + f" .{node.func.attr}() is hidden at its default ERROR level; {REPLACEMENT}"
                )
    return found


def raw_warnings_warn_calls(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        warn_names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "warnings"
            for alias in node.names
            if alias.name == "warn"
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                ast.unparse(node.func) == "warnings.warn"
                or (isinstance(node.func, ast.Name) and node.func.id in warn_names)
            ):
                found.append(
                    f"{_shown(path)}:{node.lineno}: warnings.warn names a pyprocar line"
                    + " unless its stacklevel is exact at every depth; call"
                    + " pyprocar.utils.log_utils.warn_user, which names the user's line"
                )
    return found


def user_logger_lowered_in(paths: list[Path], root: Path = ROOT) -> list[str]:
    found = []
    for path, names in user_logger_names(paths, root).items():
        for node, on_user_logger in _calls(names):
            if not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            targets = [*node.args[1:2], *(kw.value for kw in node.keywords if kw.arg == "logger")]
            if method in {"at_level", "set_level"} and any(
                isinstance(target, ast.Constant) and target.value == "user" for target in targets
            ):
                found.append(
                    f"{_shown(path)}:{node.lineno}: caplog.{method}(..., 'user') lowers the user"
                    + " logger so the test sees what a user does not; assert the warning with"
                    + " tests.utils.user_warning"
                )
            elif method == "setLevel" and on_user_logger:
                found.append(
                    f"{_shown(path)}:{node.lineno}: setLevel on the user logger shows the test"
                    + " what a user does not see; assert the warning with tests.utils.user_warning"
                )
    return found


def bare_user_warning_asserts(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and ast.unparse(node.func) == "pytest.warns"
                and node.args
                and ast.unparse(node.args[0]) == "UserWarning"
            ):
                found.append(
                    f"{_shown(path)}:{node.lineno}: pytest.warns(UserWarning) does not check that"
                    + " the warning names the caller's line; use"
                    + " tests.utils.user_warning.user_warning(__file__, match=...)"
                )
    return found


def _library_files() -> list[Path]:
    return sorted((ROOT / "pyprocar").rglob("*.py"))


def _test_files() -> list[Path]:
    return sorted((ROOT / "tests").rglob("*.py"))


def test_library_never_warns_through_the_user_logger():
    assert loud_user_logger_calls(_library_files()) == []


def test_library_warns_only_through_warn_user():
    assert raw_warnings_warn_calls([p for p in _library_files() if p != WARN_USER_HOME]) == []


def test_tests_never_lower_the_user_logger_to_see_a_warning():
    assert user_logger_lowered_in(_test_files()) == []


def test_tests_check_where_each_user_warning_points():
    assert bare_user_warning_asserts([p for p in _test_files() if p != USER_WARNING_HELPER]) == []


def _line_numbers(hits: list[str]) -> list[int]:
    return sorted(int(hit.split(": ")[0].rsplit(":", 1)[1]) for hit in hits)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_the_checks_flag_each_loud_call_and_each_lowered_level(tmp_path):
    exporter = _write(
        tmp_path / "loggers.py", ["import logging", "shared = logging.getLogger('user')"]
    )
    module = _write(
        tmp_path / "module.py",
        [
            "import logging",
            "from logging import Logger, getLogger as gl",
            "from loggers import shared as imported",
            "user_logger = logging.getLogger('user')",
            "user_logger.info('progress')",
            "user_logger.warning('hidden')",
            "logging.getLogger('user').error('hidden too')",
            "logging.getLogger(__name__).warning('package log')",
            "annotated: Logger = logging.getLogger('user')",
            "annotated.critical('annotated alias')",
            "logging.getLogger(name='user').warning('keyword name')",
            "gl('user').warning('imported getLogger alias')",
            "alias = user_logger",
            "alias.error('plain alias')",
            "imported.warning('logger imported from another module')",
            "user_logger.log(logging.WARNING, 'log at WARNING')",
            "user_logger.log(logging.INFO, 'log at INFO')",
            "user_logger.log(40, 'log at 40')",
            "first, second = logging.getLogger('user'), logging.getLogger('pyprocar')",
            "first.warning('tuple target')",
            "second.warning('package log')",
        ],
    )
    test_module = _write(
        tmp_path / "test_module.py",
        [
            "import logging",
            "import pytest",
            "with caplog.at_level(logging.WARNING, logger='user'):",
            "    pass",
            "caplog.set_level(logging.INFO, logger='pyprocar')",
            "with caplog.at_level(logging.WARNING, 'user'):",
            "    pass",
            "logging.getLogger('user').setLevel(logging.DEBUG)",
            "logging.getLogger('pyprocar').setLevel(logging.DEBUG)",
            "with pytest.warns(UserWarning, match='x'):",
            "    pass",
            "with pytest.warns(DeprecationWarning):",
            "    pass",
        ],
    )
    raw = _write(
        tmp_path / "raw.py",
        [
            "import warnings",
            "from warnings import warn as w",
            "warnings.warn('no stacklevel')",
            "w('imported warn')",
            "warnings.catch_warnings()",
        ],
    )

    loud = loud_user_logger_calls([exporter, module], root=tmp_path)

    assert _line_numbers(loud) == [6, 7, 10, 11, 12, 14, 15, 16, 18, 20]
    assert "warnings.warn" in loud[0]
    assert _line_numbers(user_logger_lowered_in([test_module], root=tmp_path)) == [3, 6, 8]
    assert _line_numbers(bare_user_warning_asserts([test_module])) == [10]
    assert _line_numbers(raw_warnings_warn_calls([raw])) == [3, 4]
