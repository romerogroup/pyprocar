"""Every ``pyprocar.<name>(...)`` call in scripts, examples and docs binds to the real signature.

A public signature change kept leaving its documented callers behind: notebooks passed
keywords fermi2D no longer took (#248), scripts/procar.py called fermi2D with ``outcar``
(#257, #264) and bandsplot without ``dirname`` (#282), and handlers kept an
``apply_symmetry`` keyword that no longer exists. This test parses every Python snippet
a user copies, including rendered ``>>>`` examples, and binds each top-level
``pyprocar`` call against ``inspect.signature``.
"""

import ast
import inspect
import json
import re
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pyprocar

ROOT = Path(__file__).resolve().parent.parent
RST_BLOCK = re.compile(r"\.\. code-block:: (?:python|ipython3?)\n((?:\n|[ \t]+.*\n)+)")
MD_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
RST_COMMENT = re.compile(r"(\s*)\.\.(?:\s+(?!\S+::|_|\[|\|).*)?")


def _rendered_lines(text: str) -> Iterator[tuple[int, str]]:
    """Number each line of an rst file, leaving out ``..`` comment blocks."""
    comment_indent: int | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        indent = len(line) - len(line.lstrip())
        if comment_indent is not None and (not line.strip() or indent > comment_indent):
            continue
        comment_indent = None
        if match := RST_COMMENT.fullmatch(line):
            comment_indent = len(match.group(1))
            continue
        yield number, line


def _doctest_blocks(text: str) -> Iterator[tuple[int, str]]:
    block: list[str] = []
    start = 0
    for number, line in [*_rendered_lines(text), (0, "")]:
        stripped = line.lstrip()
        if stripped.startswith((">>>", "...")):
            start = start or number
            block.append(stripped[4:])
        elif block:
            yield start, textwrap.dedent("\n".join(block))
            block, start = [], 0


def _notebook_snippets(path: Path) -> Iterator[tuple[str, str]]:
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            lines = source.splitlines()
            yield (
                f"#cell{index}",
                "\n".join(ln for ln in lines if not ln.lstrip().startswith(("%", "!"))),
            )
        elif cell["cell_type"] == "markdown":
            for block in MD_BLOCK.findall(source):
                yield f"#cell{index}", block


def snippets(root: Path) -> Iterator[tuple[str, str]]:
    for path in sorted([*root.glob("scripts/**/*.py"), *root.glob("examples/**/*.py")]):
        yield str(path.relative_to(root)), path.read_text(encoding="utf-8")
    for path in sorted([*root.glob("docs/**/*.rst"), root / "README.md"]):
        text = path.read_text(encoding="utf-8")
        for match in [*RST_BLOCK.finditer(text), *MD_BLOCK.finditer(text)]:
            line = text[: match.start()].count("\n") + 1
            yield f"{path.relative_to(root)}:{line}", textwrap.dedent(match.group(1))
        for line, block in _doctest_blocks(text):
            yield f"{path.relative_to(root)}:{line}", block
    for path in sorted([*root.glob("examples/**/*.ipynb"), *root.glob("docs/**/*.ipynb")]):
        for cell, source in _notebook_snippets(path):
            yield f"{path.relative_to(root)}{cell}", source


def unbound_calls(root: Path) -> list[str]:
    problems = []
    for where, source in snippets(root):
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            if "pyprocar." in source:
                problems.append(f"{where}: cannot parse a snippet that calls pyprocar: {error}")
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pyprocar"
            ):
                continue
            name = node.func.attr
            target = getattr(pyprocar, name, None)
            if target is None:
                problems.append(f"{where}:{node.lineno}: pyprocar.{name} does not exist")
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                keyword.arg is None for keyword in node.keywords
            ):
                continue
            try:
                inspect.signature(target).bind(
                    *node.args, **{str(keyword.arg): keyword.value for keyword in node.keywords}
                )
            except TypeError as error:
                problems.append(f"{where}:{node.lineno}: pyprocar.{name}: {error}")
    return problems


def test_documented_pyprocar_calls_bind_to_the_current_signatures():
    assert unbound_calls(ROOT) == []
