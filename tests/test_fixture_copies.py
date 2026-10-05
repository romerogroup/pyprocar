"""Tests copy a fixture only through ``tests.utils.writable_copy``.

``verify.sh fetch`` makes fixtures read-only, and shutil, ``os.link``, ``cp`` and
``rsync`` keep those modes or share the inode, so code that writes into such a copy
fails with ``PermissionError`` on a locked fixture. CI has no fixtures and never
runs the ``data`` tests, so this check reads the source instead.

A fixture path is an expression built from ``DATA_DIR``, from the repo root
(``__file__`` or ``ROOT_DIR``) joined with ``"data"``, or from a name, attribute,
default, argument, return value, pytest fixture or ``parametrize`` value that holds
one. The result of ``writable_copy`` is a writable copy, not a fixture path.
"""

import ast
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.utils.ast_modules import imported_from, module_name, module_parts

ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "tests" / "utils" / "__init__.py"
SAFE_COPY = "writable_copy"
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
SCOPES = (ast.Module, *FUNCTIONS)
COPIES = {
    "shutil": {"copy", "copy2", "copyfile", "copytree", "copymode", "copystat"},
    "os": {"link", "system", "popen"},
    "subprocess": {"run", "call", "check_call", "check_output", "Popen"},
}
SHELLS = {(m, f) for m in ("os", "subprocess") for f in COPIES[m]} - {("os", "link")}
SHELL_COPIES = {"cp", "rsync"}
HARDLINK = ("pathlib", "hardlink_to")
REPO_ROOTS = {"__file__", "ROOT_DIR"}

type Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


def _shown(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _is_repo_data(node: ast.AST) -> bool:
    """``<repo root> / "data..."``, with the root spelled from ``__file__`` or ``ROOT_DIR``."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return False
    right = node.right
    return (
        isinstance(right, ast.Constant)
        and isinstance(right.value, str)
        and (right.value == "data" or right.value.startswith("data/"))
        and any(isinstance(n, ast.Name) and n.id in REPO_ROOTS for n in ast.walk(node.left))
    )


def _shell_program(node: ast.expr) -> str | None:
    if isinstance(node, ast.List | ast.Tuple) and node.elts:
        node = node.elts[0]
    if isinstance(node, ast.JoinedStr) and node.values:
        node = node.values[0]
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.split():
        return node.value.split()[0].rsplit("/", 1)[-1]
    return None


class FixturePaths:
    """The expressions in one module that hold a fixture path, by scope."""

    def __init__(
        self,
        tree: ast.Module,
        parts: tuple[str, ...],
        exports: dict[str, set[str]],
        fixtures: set[str],
    ):
        self.tree: ast.Module = tree
        self.exports: dict[str, set[str]] = exports
        self.parent: dict[ast.AST, ast.AST] = {
            child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        self.assigned: defaultdict[ast.AST, set[str]] = defaultdict(set)
        self.bound: defaultdict[ast.AST, set[str]] = defaultdict(set)
        self.attributes: set[str] = set()
        self.returning: set[str] = set()
        self.fixtures: set[str] = set()
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.copy_functions: dict[str, tuple[str, str]] = {}
        self.modules: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.assigned[self._scope(node)].add(node.id)
            elif isinstance(node, ast.arg):
                self.assigned[self._scope(node)].add(node.arg)
                if node.arg in fixtures:
                    self.bound[self._scope(node)].add(node.arg)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self.functions[node.name] = node
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    self.assigned[self._scope(node)].add(name)
                    if alias.name in COPIES:
                        self.modules[name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                source = imported_from(node, parts)
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.assigned[self._scope(node)].add(name)
                    if alias.name in COPIES.get(source, set()):
                        self.copy_functions[name] = (source, alias.name)
                    if alias.name in exports.get(source, set()):
                        self.bound[self._scope(node)].add(name)
        while self._bind_once():
            pass

    def _scope(self, node: ast.AST) -> ast.AST:
        node = self.parent[node]
        while not isinstance(node, SCOPES):
            node = self.parent[node]
        return node

    def function_of(self, node: ast.AST) -> Function | None:
        scope = self._scope(node)
        return scope if isinstance(scope, FUNCTIONS) else None

    def _bind(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bound[self._scope(target)].add(target.id)
        elif isinstance(target, ast.Starred):
            self._bind(target.value)
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self._bind(element)
        elif isinstance(target, ast.Attribute):
            self.attributes.add(ast.unparse(target))

    def _size(self) -> tuple[int, int, int]:
        return sum(map(len, self.bound.values())), len(self.attributes), len(self.returning)

    def _flows(self, node: ast.AST) -> Iterator[tuple[ast.expr, ast.expr | None]]:
        """Yield (source, target) pairs; a None target is a value the function returns."""
        if isinstance(node, ast.Assign):
            yield from ((node.value, target) for target in node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr) and node.value:
            yield node.value, node.target
        elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            yield node.iter, node.target
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            yield node.context_expr, node.optional_vars
        elif isinstance(node, ast.Return | ast.Yield | ast.YieldFrom) and node.value:
            yield node.value, None

    def _bind_once(self) -> bool:
        before = self._size()
        for node in ast.walk(self.tree):
            for source, target in self._flows(node):
                if not self.holds(source):
                    continue
                if target is not None:
                    self._bind(target)
                elif isinstance(function := self.function_of(node), ast.FunctionDef):
                    self.returning.add(function.name)
            if isinstance(node, FUNCTIONS):
                self._bind_defaults(node)
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in self.returning
                and any("fixture" in ast.unparse(d) for d in node.decorator_list)
            ):
                self.fixtures.add(node.name)
            if isinstance(node, ast.Call):
                self._bind_arguments(node)
        return self._size() != before

    def _bind_defaults(self, function: Function) -> None:
        args = function.args
        positional = [*args.posonlyargs, *args.args]
        defaults = [
            *zip(positional[len(positional) - len(args.defaults) :], args.defaults, strict=True),
            *zip(args.kwonlyargs, args.kw_defaults, strict=True),
        ]
        for arg, default in defaults:
            if default is not None and self.holds(default):
                self.bound[function].add(arg.arg)
        if isinstance(function, ast.Lambda):
            return
        for decorator in function.decorator_list:
            if not isinstance(decorator, ast.Call) or len(decorator.args) < 2:
                continue
            names, values = decorator.args[:2]
            if (
                ast.unparse(decorator.func).endswith("parametrize")
                and isinstance(names, ast.Constant)
                and isinstance(names.value, str)
                and self.holds(values)
            ):
                for name in names.value.split(","):
                    self.bound[function].add(name.strip())

    def _bind_arguments(self, call: ast.Call) -> None:
        if not isinstance(call.func, ast.Name) or call.func.id not in self.functions:
            return
        function = self.functions[call.func.id]
        positional = [*function.args.posonlyargs, *function.args.args]
        for param, arg in zip(positional, call.args, strict=False):
            if self.holds(arg):
                self.bound[function].add(param.arg)
        for keyword in call.keywords:
            if keyword.arg is not None and self.holds(keyword.value):
                self.bound[function].add(keyword.arg)

    def _name_holds(self, name: str, scope: ast.AST) -> bool:
        while name not in self.assigned[scope] and name not in self.bound[scope]:
            if isinstance(scope, ast.Module):
                return False
            scope = self._scope(scope)
        return name in self.bound[scope]

    def holds(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == SAFE_COPY:
                return False
            if isinstance(func, ast.Name) and func.id in self.returning:
                return True
        elif _is_repo_data(node):
            return True
        elif isinstance(node, ast.Name):
            return isinstance(node.ctx, ast.Load) and self._name_holds(node.id, self._scope(node))
        elif isinstance(node, ast.Attribute):
            if ast.unparse(node) in self.attributes:
                return True
            if node.attr in self.exports.get(ast.unparse(node.value), set()):
                return True
        return any(
            self.holds(child) for child in ast.iter_child_nodes(node) if isinstance(child, ast.expr)
        )

    def copy_call(self, node: ast.Call) -> tuple[str, str] | None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "hardlink_to":
            return HARDLINK
        if isinstance(func, ast.Name):
            found = self.copy_functions.get(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            module = self.modules.get(func.value.id)
            found = (module, func.attr) if module and func.attr in COPIES[module] else None
        else:
            found = None
        if found in SHELLS and not (node.args and _shell_program(node.args[0]) in SHELL_COPIES):
            return None
        return found


def _parse(paths: list[Path], root: Path) -> dict[Path, FixturePaths]:
    trees = {path: ast.parse(path.read_text(encoding="utf-8"), str(path)) for path in paths}
    exports: dict[str, set[str]] = {}
    fixtures: set[str] = set()
    while True:
        names = {
            path: FixturePaths(tree, module_parts(path, root), exports, fixtures)
            for path, tree in trees.items()
        }
        found = {module_name(module_parts(p, root)): n.bound[n.tree] for p, n in names.items()}
        found_fixtures = {f for n in names.values() for f in n.fixtures}
        if found == exports and found_fixtures == fixtures:
            return names
        exports, fixtures = found, found_fixtures


def raw_fixture_copies(paths: list[Path], root: Path = ROOT, helper: Path = HELPER) -> list[str]:
    found = []
    for path, names in _parse(paths, root).items():
        for node in ast.walk(names.tree):
            if not isinstance(node, ast.Call) or (call := names.copy_call(node)) is None:
                continue
            function = names.function_of(node)
            if path == helper and getattr(function, "name", None) == SAFE_COPY:
                continue
            values = [*node.args, *(k.value for k in node.keywords)]
            if call == HARDLINK and isinstance(node.func, ast.Attribute):
                values.append(node.func.value)
            if held := [v for v in values if names.holds(v)]:
                found.append(
                    f"{_shown(path)}:{node.lineno}: {ast.unparse(node.func)} on the fixture path"
                    + f" {ast.unparse(held[0])} keeps a locked fixture's read-only modes;"
                    + " copy it with tests.utils.writable_copy"
                )
    return found


@pytest.mark.guards_existing_behaviour(
    reason="it scans tests/, which this PR already converts; the self-test proves each form fails"
)
def test_tests_copy_fixtures_only_through_writable_copy():
    assert raw_fixture_copies(sorted((ROOT / "tests").rglob("*.py"))) == []


def _line_numbers(hits: list[str]) -> list[int]:
    return sorted(int(hit.split(": ")[0].rsplit(":", 1)[1]) for hit in hits)


def _write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.mark.guards_existing_behaviour(reason="it runs the check on synthetic files")
def test_the_check_flags_each_way_a_test_can_copy_a_fixture(tmp_path):
    helper = _write(
        tmp_path / "tests" / "utils" / "__init__.py",
        [
            "import shutil",
            "from pathlib import Path",
            "ROOT_DIR = Path(__file__).parent.parent.parent",
            'DATA_DIR = ROOT_DIR / "data"',
            "def writable_copy(src, dst, ignore=None):",
            "    shutil.copytree(src, dst, ignore=ignore)",
            "    return dst",
            "def other(dst):",
            '    shutil.copytree(DATA_DIR / "x", dst)',
            '    writable_copy(DATA_DIR / "y", dst)',
        ],
    )
    package = _write(
        tmp_path / "tests" / "pkg" / "__init__.py",
        ["from ..utils import ROOT_DIR", 'PKG_DIR = ROOT_DIR / "data" / "codes"'],
    )
    conftest = _write(
        tmp_path / "tests" / "conftest.py",
        [
            "import pytest",
            "from tests.utils import DATA_DIR",
            "@pytest.fixture",
            "def shared_calc():",
            '    return DATA_DIR / "codes"',
        ],
    )
    module = _write(
        tmp_path / "tests" / "test_module.py",
        [
            "import os",
            "import shutil",
            "import subprocess",
            "from os import link as hard",
            "from shutil import copytree as ct",
            "import shutil as sh",
            "from pathlib import Path",
            "import pytest",
            "import tests.utils",
            "from tests.utils import DATA_DIR, writable_copy",
            "from tests.pkg import PKG_DIR",
            'CALC = DATA_DIR / "examples/bands/x"',
            'shutil.copy(CALC / "PROCAR", "b")',
            'shutil.copy2(DATA_DIR / "a", "b")',
            'shutil.copyfile(str(CALC), "b")',
            'sh.copytree(CALC, "b")',
            'ct(CALC, "b")',
            'shutil.copymode(PKG_DIR, "b")',
            'shutil.copystat(tests.utils.DATA_DIR, "b")',
            'os.link(CALC / "PROCAR", "b")',
            'hard(CALC, "b")',
            'Path("b").hardlink_to(CALC)',
            'subprocess.run(["cp", "-a", CALC, "b"])',
            'subprocess.check_call(f"rsync -a {CALC} b", shell=True)',
            'os.system(f"/bin/cp -r {CALC} b")',
            'subprocess.run(["ls", CALC])',
            'shutil.copytree("a", "b")',
            "shutil.rmtree(CALC)",
            'copy = writable_copy(CALC, Path("b"))',
            'shutil.copy(copy / "PROCAR", "c")',
            'legacy = Path(__file__).parents[2] / "data" / "io"',
            'shutil.copytree(legacy, "b")',
            'fake = Path("tmp") / "data"',
            'shutil.copytree(fake, "b")',
            "def _copy_calc(tmp_path, src=CALC):",
            '    shutil.copy(src / "PROCAR", tmp_path)',
            "def _copy_any(src, dst):",
            "    shutil.copytree(src, dst)",
            '_copy_any(PKG_DIR, "b")',
            "def _source():",
            '    return DATA_DIR / "codes"',
            'shutil.copytree(_source(), "b")',
            "@pytest.fixture",
            "def calc_dir():",
            '    yield DATA_DIR / "codes"',
            "def test_fixture(calc_dir, tmp_path):",
            '    shutil.copytree(calc_dir, tmp_path / "c")',
            '@pytest.mark.parametrize("calc", [CALC, DATA_DIR / "y"])',
            "def test_param(calc, tmp_path):",
            "    shutil.copytree(calc, tmp_path)",
            "class Case:",
            "    def setup(self):",
            "        self.calc = CALC",
            "    def run(self):",
            '        shutil.copytree(self.calc, "b")',
            "for entry in CALC.iterdir():",
            '    shutil.copy(entry, "b")',
            "def shadow(CALC):",
            '    shutil.copy(CALC, "b")',
            "def test_conftest_fixture(shared_calc):",
            '    shutil.copy(shared_calc, "b")',
        ],
    )
    elk_304 = _write(
        tmp_path / "tests" / "test_elk_304.py",
        [
            "import shutil",
            "from tests.utils import DATA_DIR",
            'ELK_BANDS_SP = DATA_DIR / "codes" / "elk" / "6.3" / "SrVO3"'
            + ' / "spin-polarized-colinear" / "bands"',
            "def test_real_elk_6_3_bands_beside_a_task_10_elmirep_keep_their_l_m_names(tmp_path):",
            '    calc_dir = tmp_path / "bands"',
            "    shutil.copytree(",
            "        ELK_BANDS_SP,",
            "        calc_dir,",
            '        ignore=shutil.ignore_patterns("EVEC*.OUT", "STATE.OUT", "VARIABLES.OUT"),',
            "    )",
            '    shutil.copy(ELK_BANDS_SP.parent / "dos" / "ELMIREP.OUT", calc_dir)',
        ],
    )
    filtered_298 = _write(
        tmp_path / "tests" / "test_filtered_298.py",
        [
            "import shutil",
            "from pathlib import Path",
            "from tests.utils import DATA_DIR",
            'CALC = DATA_DIR / "examples/bands/non-spin-polarized"',
            'FERMI2D_CALC = DATA_DIR / "examples/fermi2d/non-spin-polarized"',
            "def _copy_calc(tmp_path: Path, src: Path = CALC, extra: tuple[str, ...] = ())"
            + " -> Path:",
            '    dst = tmp_path / "calc"',
            "    dst.mkdir()",
            '    for name in ("PROCAR", "OUTCAR", "POSCAR", "KPOINTS", *extra):',
            "        shutil.copy(src / name, dst / name)",
            "    return dst",
        ],
    )
    files = [helper, package, conftest, module, elk_304, filtered_298]

    hits = raw_fixture_copies(files, root=tmp_path, helper=helper)

    def lines(path: Path) -> list[int]:
        return _line_numbers([h for h in hits if h.startswith(f"{_shown(path)}:")])

    assert lines(helper) == [9]
    assert lines(conftest) == []
    assert lines(module) == [
        *range(13, 26), 32, 36, 38, 42, 47, 50, 55, 57, 61,
    ]  # fmt: skip
    assert lines(elk_304) == [6, 11]
    assert lines(filtered_298) == [10]
    assert "tests.utils.writable_copy" in hits[0]
