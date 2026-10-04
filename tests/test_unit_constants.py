"""Physical constants live in pyprocar/utils/units.py, taken from scipy.constants.

A hand-typed copy drifts: dHvA_frequency used e = 4.768e-10 statC (0.74% low),
AbinitOutput converted Hartree with 27.211396641308 and Bohr with 0.529177, and
four modules redefined HARTREE_TO_EV. This test fails on any numeric literal, or
constant-only expression such as ``1.602 * 10 ** (-19)``, close to a known constant
anywhere in pyprocar/ or scripts/, units.py included, and names the constant to use.
"""

import ast
import math
from collections.abc import Iterator
from pathlib import Path

from scipy import constants

ROOT = Path(__file__).resolve().parent.parent

HARTREE_EV = constants.physical_constants["Hartree energy in eV"][0]
BOHR_ANGSTROM = constants.physical_constants["Bohr radius"][0] / constants.angstrom
KNOWN = {
    "pyprocar.utils.units.HARTREE_TO_EV": HARTREE_EV,
    "pyprocar.utils.units.EV_TO_HARTREE": 1 / HARTREE_EV,
    "pyprocar.utils.units.AU_TO_ANG": BOHR_ANGSTROM,
    "pyprocar.utils.units.ANG_TO_AU": 1 / BOHR_ANGSTROM,
    "pyprocar.utils.units.RYDBERG_TO_EV": HARTREE_EV / 2,
    "pyprocar.utils.units.EV_TO_J": constants.e,
    "scipy.constants.e * scipy.constants.c * 10 (e in statC)": constants.e * constants.c * 10,
    "pyprocar.utils.units.HBAR_J": constants.hbar,
    "pyprocar.utils.units.HBAR_EV": constants.hbar / constants.e,
    "scipy.constants.h": constants.h,
    "pyprocar.utils.units.FREE_ELECTRON_MASS": constants.m_e,
}


def _number(node: ast.expr) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return None if isinstance(node.value, bool) else float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return None if value is None else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult | ast.Div | ast.Pow):
        left, right = _number(node.left), _number(node.right)
        if left is None or right is None:
            return None
        try:
            return float(eval(compile(ast.Expression(node), "<constant>", "eval")))
        except (ArithmeticError, ValueError):
            return None
    return None


def _numbers(tree: ast.AST) -> Iterator[tuple[int, float]]:
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.expr) and (value := _number(node)) is not None:
            yield node.lineno, value
            continue
        stack.extend(ast.iter_child_nodes(node))


def hand_typed_constants(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        for line, value in _numbers(ast.parse(path.read_text(encoding="utf-8"), str(path))):
            for name, reference in KNOWN.items():
                # SI and CGS constants are too small to collide with anything else, so a
                # loose tolerance also catches inaccurate copies like e = 4.768e-10.
                tolerance = 2e-2 if reference < 1e-5 else 1e-3
                if not value.is_integer() and math.isclose(
                    abs(value), reference, rel_tol=tolerance
                ):
                    found.append(
                        f"{path.relative_to(ROOT)}:{line}: {value!r} looks like {name}; use that"
                    )
    return found


def test_physical_constants_come_from_units_module():
    paths = sorted([*(ROOT / "pyprocar").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")])
    assert hand_typed_constants(paths) == []
