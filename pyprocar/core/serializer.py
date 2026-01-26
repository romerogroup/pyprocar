from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

import dill


class Serializable(Protocol):
    """Protocol for objects that can be serialized to dict."""

    def to_dict(self) -> dict[str, Any]: ...


class BaseSerializer(ABC):
    """Base class for EBS writers."""

    @abstractmethod
    def save(self, obj: object, path: Path) -> None:
        """Write the EBS to a file."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load the EBS from a file."""


class PickleSerializer(BaseSerializer):
    """Serializer for Electronic Band Structure using pickle format."""

    def save(self, obj: object, path: Path) -> None:
        """Save the EBS to a pickle file.

        Args:
            ebs: The ElectronicBandStructure object to save
            path: Path where to save the pickle file
        """
        with open(path, "wb") as file:
            dill.dump(obj, file)

    def load(self, path: Path) -> Any:
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

    def save(self, obj: object, path: Path) -> None:
        """Save the object to a JSON file with metadata.

        The object must implement the Serializable protocol (have a to_dict method).
        """
        # Get to_dict method - will raise AttributeError if not present
        to_dict_method: Any = getattr(obj, "to_dict")
        # Get the dictionary representation from the object
        data: dict[str, Any] = to_dict_method()

        # Inject the metadata
        data["@module"] = obj.__class__.__module__
        data["@class"] = obj.__class__.__name__

        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def load(self, path: Path) -> Any:
        """Load an object from a JSON file using its metadata."""
        with open(path) as file:
            data: dict[str, Any] = json.load(file)

        # Extract metadata
        module_name = data.pop("@module")
        class_name = data.pop("@class")

        try:
            # Dynamically import the module and get the class
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise TypeError(f"Could not find class {class_name} in module {module_name}") from e

        # Use the dynamically loaded class to create the object
        return cls.from_dict(data)


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
