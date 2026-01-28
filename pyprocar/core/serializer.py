from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast, override, runtime_checkable

import dill


def _json_dict_items(raw_dict: dict[object, object]) -> list[tuple[str, object]]:
    """Convert dict from JSON to list of (str key, object value) tuples."""
    result: list[tuple[str, object]] = []
    for key in raw_dict:
        str_key = str(key) if not isinstance(key, str) else key
        result.append((str_key, raw_dict[key]))
    return result


def _get_module_attr(module: ModuleType, name: str) -> object:
    """Get attribute from module, returning object type.

    Cast is used because getattr returns Any per stdlib typing.
    This is safe because all Python objects are instances of object.
    """
    return cast(object, getattr(module, name))


def _load_json(file_path: Path) -> object:
    """Load JSON file and return as object.

    Cast is used because json.load returns Any per stdlib typing.
    This is safe because all Python values are instances of object.
    """
    with open(file_path) as f:
        return cast(object, json.load(f))


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to dict."""

    def to_dict(self) -> dict[str, object]: ...


class BaseSerializer(ABC):
    """Base class for EBS writers."""

    @abstractmethod
    def save(self, obj: object, path: Path) -> None:
        """Write the EBS to a file."""

    @abstractmethod
    def load(self, path: Path) -> object:
        """Load the EBS from a file."""


class PickleSerializer(BaseSerializer):
    """Serializer for Electronic Band Structure using pickle format."""

    @override
    def save(self, obj: object, path: Path) -> None:
        """Save the EBS to a pickle file.

        Args:
            ebs: The ElectronicBandStructure object to save
            path: Path where to save the pickle file
        """
        with open(path, "wb") as file:
            dill.dump(obj, file)

    @override
    def load(self, path: Path) -> object:
        """Load an EBS from a pickle file.

        Args:
            path: Path to the pickle file

        Returns:
            The loaded ElectronicBandStructure object
        """
        with open(path, "rb") as file:
            return dill.load(file)


class JSONSerializer(BaseSerializer):
    """General purpose serializer for any 'Serializable' object using JSON."""

    @override
    def save(self, obj: object, path: Path) -> None:
        """Save the object to a JSON file with metadata.

        The object must implement the Serializable protocol (have a to_dict method).
        """
        # Verify object implements Serializable protocol using runtime check
        if not isinstance(obj, Serializable):
            raise TypeError(f"Object {obj} must implement Serializable protocol (have to_dict method)")
        # isinstance narrows obj to Serializable
        data: dict[str, object] = obj.to_dict()

        # Inject the metadata
        data["@module"] = obj.__class__.__module__
        data["@class"] = obj.__class__.__name__

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    @override
    def load(self, path: Path) -> object:
        """Load an object from a JSON file using its metadata."""
        loaded_result: object = _load_json(path)
        # Use isinstance to narrow to dict
        if not isinstance(loaded_result, dict):
            raise TypeError(f"Expected dict from JSON, got {type(loaded_result)}")
        # Post-narrowing cast: isinstance verified dict, but json.load's Any taints type params.
        raw_dict: dict[object, object] = cast(dict[object, object], loaded_result)
        # Build properly typed dict using helper to avoid unknown types in comprehension
        data: dict[str, object] = dict(_json_dict_items(raw_dict))

        # Extract metadata
        module_name_val = data.pop("@module")
        class_name_val = data.pop("@class")
        if not isinstance(module_name_val, str):
            raise TypeError(f"Expected string for @module")
        if not isinstance(class_name_val, str):
            raise TypeError(f"Expected string for @class")
        module_name: str = module_name_val
        class_name: str = class_name_val

        try:
            # Dynamically import the module and get the class
            module: ModuleType = importlib.import_module(module_name)
            # Use helper to get attribute with object return type
            cls_candidate: object = _get_module_attr(module, class_name)
            if not isinstance(cls_candidate, type):
                raise TypeError(f"Expected class, got {type(cls_candidate)}")
            cls: type[object] = cls_candidate
        except (ImportError, AttributeError) as e:
            raise TypeError(f"Could not find class {class_name} in module {module_name}") from e

        # Use the dynamically loaded class to create the object
        from_dict_method: object = getattr(cls, "from_dict", None)
        if from_dict_method is None or not callable(from_dict_method):
            raise TypeError(f"Class {class_name} must have a callable from_dict method")
        return from_dict_method(data)


SERIALIZERS = {
    "pickle": PickleSerializer(),
    "pkl": PickleSerializer(),
    "json": JSONSerializer(),
}


def get_serializer(path: Path | str) -> BaseSerializer:
    """Get the serializer for the given path."""
    if isinstance(path, str):
        path = Path(path)
    return SERIALIZERS[path.suffix[1:]]
