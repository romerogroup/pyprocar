import ast
from pathlib import Path


def module_parts(path: Path, root: Path) -> tuple[str, ...]:
    return (
        path.relative_to(root).with_suffix("").parts if path.is_relative_to(root) else (path.stem,)
    )


def module_name(parts: tuple[str, ...]) -> str:
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_from(node: ast.ImportFrom, parts: tuple[str, ...]) -> str:
    if node.level == 0:
        return node.module or ""
    package = parts[:-1]
    base = package[: len(package) - node.level + 1]
    return ".".join((*base, *([node.module] if node.module else [])))
