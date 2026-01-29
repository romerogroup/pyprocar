from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast, override

import dill

dill.settings["recurse"] = True

import json
import shutil

import numpy as np
import pandas as pd

ALLOWED_TYPES = (str, int, float, bool, list, dict, tuple, bytes, np.generic)


def is_python_object(
    x: object,
    check_bytes: bool = False,
    allowed_types: tuple[type, ...] = ALLOWED_TYPES,
) -> bool:
    if isinstance(x, bytes) and check_bytes:
        x = dill.loads(x)
    return isinstance(x, object) and not isinstance(x, allowed_types)


def has_python_object(values: Iterable[Any], check_bytes: bool = False) -> bool:
    """Check if a pandas Series contains Python objects (excluding simple types)."""
    is_object = False
    for value in values:
        if value is not None and is_python_object(value, check_bytes=check_bytes):
            is_object = True
            break
    return is_object


def serialize_python_objects(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    python_object_columns: list[str] = []
    for column in df.columns:
        values = df[column].values
        if has_python_object(values):
            python_object_columns.append(column)
            new_values: list[bytes | None] = []
            for value in values:
                if value is not None and not pd.isna(value):
                    new_values.append(dill.dumps(value))
                else:
                    new_values.append(None)
            df[column] = new_values

    return df, python_object_columns


def dump_python_object(value: object) -> bytes | None:
    if value is None:
        return None
    return dill.dumps(value)


def save_pickle(value: object, filepath: str) -> None:
    with open(filepath, "wb") as file:
        dill.dump(value, file)


def load_pickle(filepath: str) -> object:
    with open(filepath, "rb") as file:
        return dill.load(file)


def load_python_object(value: bytes | None) -> object:
    if value is None:
        return None
    return dill.loads(value)


def is_none(value: object) -> bool:
    return value is None


def copy_files(paths: tuple[str, str]) -> None:
    src_path, target_path = paths
    shutil.copy2(src_path, target_path)


class CompactJSONEncoder(json.JSONEncoder):
    """
    A JSONEncoder that formats lists of simple types on a single line,
    and 2D arrays with outer list on new lines and inner lists compact.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._current_indent_level: int = 0

    @override
    def encode(self, o: Any) -> str:
        """
        Override the encode method to check for compactable lists.
        """
        if isinstance(o, (list, tuple)):
            # cast: isinstance on Any narrows element types to Unknown; restore to Any
            o_seq = cast("Sequence[Any]", o)
            # Check if all items in the list are simple types
            if self._is_simple_list(o_seq):
                # If so, return a compact, single-line representation
                return f"[{', '.join(map(self.encode, o_seq))}]"
            if self._is_2d_array(o_seq):
                # Handle 2D arrays specially
                return self._encode_2d_array(o_seq)

        # For all other cases, use the default encoder
        return super().encode(o)

    @override
    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
        """
        Override iterencode to handle indentation correctly.
        """
        if isinstance(o, (list, tuple)):
            # cast: isinstance on Any narrows element types to Unknown; restore to Any
            o_seq = cast("Sequence[Any]", o)
            # Check if the list is simple
            if self._is_simple_list(o_seq):
                # If simple, yield the compact representation directly
                yield self.encode(o_seq)
            elif self._is_2d_array(o_seq):
                # Handle 2D arrays specially
                yield self._encode_2d_array(o_seq)
            else:
                # If the list is complex (contains dicts or other complex objects),
                # handle it with proper indentation
                indent_val = self._get_indent_size()
                if indent_val is None:
                    yield from super().iterencode(o, _one_shot)
                    return

                yield "["
                first = True
                item: Any
                for item in o_seq:
                    if not first:
                        yield f",\n{self._indent_str()}"
                    else:
                        yield f"\n{self._indent_str()}"
                        first = False

                    # Recursively encode each item with increased indentation
                    old_indent = self._current_indent_level
                    self._current_indent_level += 1
                    yield from self.iterencode(item)
                    self._current_indent_level = old_indent

                if not first:  # Only add closing bracket indentation if list wasn't empty
                    yield f"\n{self._parent_indent_str()}]"
                else:
                    yield "]"

        elif isinstance(o, dict):
            # cast: isinstance on Any narrows key/value types to Unknown; restore to Any
            o_dict = cast("dict[Any, Any]", o)
            # Special handling for dictionaries to ensure correct indentation
            # If we are not indenting, just use the default
            indent_val = self._get_indent_size()
            if indent_val is None:
                yield from super().iterencode(o, _one_shot)
                return

            # Custom dictionary encoding to work with our list logic
            yield "{"
            first = True
            key: Any
            value: Any
            for key, value in o_dict.items():
                if not first:
                    yield f",\n{self._indent_str()}"
                else:
                    yield f"\n{self._indent_str()}"
                    first = False

                yield f"{self.encode(key)}: "

                # Here's the key: we recursively call iterencode for the value
                # This ensures our list logic is applied at all levels
                old_indent = self._current_indent_level
                self._current_indent_level += 1
                yield from self.iterencode(value)
                self._current_indent_level = old_indent

            if not first:  # Only add closing brace indentation if dict wasn't empty
                yield f"\n{self._parent_indent_str()}}}"
            else:
                yield "}"

        else:
            # For all other primatives, use the default
            yield from super().iterencode(o, _one_shot)

    def _is_simple_list(self, o: Sequence[Any]) -> bool:
        # A list is simple if it does not contain any dicts, lists, or tuples
        return all(not isinstance(el, (dict, list, tuple)) for el in o)

    def _is_2d_array(self, o: Sequence[Any]) -> bool:
        # A 2D array is a list where all elements are lists of simple types
        return (
            isinstance(o, (list, tuple))
            and len(o) > 0
            and all(
                isinstance(el, (list, tuple)) and self._is_simple_list(cast("Sequence[Any]", el))
                for el in o
            )
        )

    def _encode_2d_array(self, o: Sequence[Any]) -> str:
        indent_val = self._get_indent_size()
        if indent_val is None:
            # No indentation, use compact format
            inner_arrays = [f"[{', '.join(map(self.encode, inner))}]" for inner in o]
            return f"[{', '.join(inner_arrays)}]"
        # With indentation, format outer list on multiple lines, inner lists compact
        indent_str = " " * (indent_val * (self._current_indent_level + 1))
        parent_indent_str = " " * (indent_val * self._current_indent_level)

        inner_arrays: list[str] = []
        for inner in o:
            compact_inner = f"[{', '.join(map(self.encode, inner))}]"
            inner_arrays.append(compact_inner)

        if len(inner_arrays) == 0:
            return "[]"

        formatted_items = f",\n{indent_str}".join(inner_arrays)
        return f"[\n{indent_str}{formatted_items}\n{parent_indent_str}]"

    def _get_indent_size(self) -> int | None:
        """Get the indent size as an int, or None if no indentation."""
        # Runtime indent can be None even though stubs type it as int | str
        indent = self.indent
        if indent is None:  # pyright: ignore[reportUnnecessaryComparison]
            return None
        if isinstance(indent, int):
            return indent
        return len(indent)

    def _indent_str(self) -> str:
        indent_val = self._get_indent_size()
        if indent_val is None:
            return ""
        return " " * (indent_val * (self._current_indent_level + 1))

    def _parent_indent_str(self) -> str:
        indent_val = self._get_indent_size()
        if indent_val is None:
            return ""
        return " " * (indent_val * self._current_indent_level)
