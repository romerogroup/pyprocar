import dill

dill.settings["recurse"] = True

import json
import shutil
import textwrap

import numpy as np
import pandas as pd

ALLOWED_TYPES = (str, int, float, bool, list, dict, tuple, bytes, np.generic)


def is_python_object(x, check_bytes=False, allowed_types=ALLOWED_TYPES):
    if isinstance(x, bytes) and check_bytes:
        x = dill.loads(x)
    return isinstance(x, object) and not isinstance(x, allowed_types)


def has_python_object(values, check_bytes=False):
    """Check if a pandas Series contains Python objects (excluding simple types)."""
    is_object = False
    for value in values:
        if value is not None and is_python_object(value, check_bytes=check_bytes):
            is_object = True
            break
    return is_object


def serialize_python_objects(df):
    python_object_columns = []
    for column in df.columns:
        values = df[column].values
        if has_python_object(values):
            python_object_columns.append(column)
            new_values = []
            for value in values:
                if value is not None and not pd.isna(value):
                    new_values.append(dill.dumps(value))
                else:
                    new_values.append(None)
            df[column] = new_values

    return df, python_object_columns


def dump_python_object(value):
    if value is None:
        return None
    else:
        return dill.dumps(value)


def save_pickle(value, filepath):
    with open(filepath, "wb") as file:
        dill.dump(value, file)


def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return dill.load(file)


def load_python_object(value):
    if value is None:
        return None
    else:
        return dill.loads(value)


def is_none(value):
    return value is None


def copy_files(paths):
    src_path, target_path = paths
    shutil.copy2(src_path, target_path)


class CompactJSONEncoder(json.JSONEncoder):
    """
    A JSONEncoder that formats lists of simple types on a single line,
    and 2D arrays with outer list on new lines and inner lists compact.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_indent_level = 0

    def encode(self, o):
        """
        Override the encode method to check for compactable lists.
        """
        if isinstance(o, (list, tuple)):
            # Check if all items in the list are simple types
            if self._is_simple_list(o):
                # If so, return a compact, single-line representation
                return f"[{', '.join(map(self.encode, o))}]"
            elif self._is_2d_array(o):
                # Handle 2D arrays specially
                return self._encode_2d_array(o)

        # For all other cases, use the default encoder
        return super().encode(o)

    def iterencode(self, o, _one_shot=False):
        """
        Override iterencode to handle indentation correctly.
        """
        if isinstance(o, (list, tuple)):
            # Check if the list is simple
            if self._is_simple_list(o):
                # If simple, yield the compact representation directly
                yield self.encode(o)
            elif self._is_2d_array(o):
                # Handle 2D arrays specially
                yield self._encode_2d_array(o)
            else:
                # If the list is complex (contains dicts or other complex objects),
                # handle it with proper indentation
                if self.indent is None:
                    yield from super().iterencode(o, _one_shot)
                    return

                yield "["
                first = True
                for item in o:
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

                if (
                    not first
                ):  # Only add closing bracket indentation if list wasn't empty
                    yield f"\n{self._parent_indent_str()}]"
                else:
                    yield "]"

        elif isinstance(o, dict):
            # Special handling for dictionaries to ensure correct indentation
            # If we are not indenting, just use the default
            if self.indent is None:
                yield from super().iterencode(o, _one_shot)
                return

            # Custom dictionary encoding to work with our list logic
            yield "{"
            first = True
            for key, value in o.items():
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

    def _is_simple_list(self, o):
        # A list is simple if it does not contain any dicts, lists, or tuples
        return all(not isinstance(el, (dict, list, tuple)) for el in o)

    def _is_2d_array(self, o):
        # A 2D array is a list where all elements are lists of simple types
        return (
            isinstance(o, (list, tuple))
            and len(o) > 0
            and all(
                isinstance(el, (list, tuple)) and self._is_simple_list(el) for el in o
            )
        )

    def _encode_2d_array(self, o):
        if self.indent is None:
            # No indentation, use compact format
            inner_arrays = [f"[{', '.join(map(self.encode, inner))}]" for inner in o]
            return f"[{', '.join(inner_arrays)}]"
        else:
            # With indentation, format outer list on multiple lines, inner lists compact
            indent_str = " " * (self.indent * (self._current_indent_level + 1))
            parent_indent_str = " " * (self.indent * self._current_indent_level)

            inner_arrays = []
            for inner in o:
                compact_inner = f"[{', '.join(map(self.encode, inner))}]"
                inner_arrays.append(compact_inner)

            if len(inner_arrays) == 0:
                return "[]"

            formatted_items = f",\n{indent_str}".join(inner_arrays)
            return f"[\n{indent_str}{formatted_items}\n{parent_indent_str}]"

    def _indent_str(self):
        if self.indent is None:
            return ""
        return " " * (self.indent * (self._current_indent_level + 1))

    def _parent_indent_str(self):
        if self.indent is None:
            return ""
        return " " * (self.indent * self._current_indent_level)
