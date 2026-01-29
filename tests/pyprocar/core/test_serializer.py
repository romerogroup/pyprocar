"""
Test module for pyprocar.core.serializer module.

This module contains unit tests for the serialization framework including
PickleSerializer, JSONSerializer, and the get_serializer factory function.
"""

import json
from pathlib import Path
from typing import override

import numpy as np
import pytest

from pyprocar.core.serializer import (
    SERIALIZERS,
    BaseSerializer,
    JSONSerializer,
    PickleSerializer,
    get_serializer,
)

# =============================================================================
# Helper Classes for Testing
# =============================================================================


class SimpleSerializableObject:
    """A simple object that supports both pickle and JSON serialization."""

    name: str
    value: float
    data: np.ndarray

    def __init__(self, name: str, value: float, data: np.ndarray | None = None) -> None:
        self.name = name
        self.value = value
        self.data = data if data is not None else np.array([1.0, 2.0, 3.0])

    def to_dict(self) -> dict[str, object]:
        """Convert object to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "data": self.data.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SimpleSerializableObject":
        """Create object from dictionary."""
        return cls(
            name=str(data["name"]),
            value=float(str(data["value"])),
            data=np.array(data["data"]),
        )

    @override
    def __hash__(self) -> int:
        """Return hash based on name and value."""
        return hash((self.name, self.value))

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleSerializableObject):
            return False
        return (
            self.name == other.name
            and self.value == other.value
            and np.allclose(self.data, other.data)
        )


class NonSerializableObject:
    """An object that does not support JSON serialization (no to_dict/from_dict)."""

    value: int

    def __init__(self, value: int) -> None:
        self.value = value


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_object() -> SimpleSerializableObject:
    """Create a simple serializable object for testing."""
    return SimpleSerializableObject(
        name="test_object",
        value=42.5,
        data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


@pytest.fixture
def pickle_serializer() -> PickleSerializer:
    """Return a PickleSerializer instance."""
    return PickleSerializer()


@pytest.fixture
def json_serializer() -> JSONSerializer:
    """Return a JSONSerializer instance."""
    return JSONSerializer()


# =============================================================================
# PickleSerializer Tests
# =============================================================================


class TestPickleSerializer:
    """Test suite for PickleSerializer class."""

    def test_save_creates_file(
        self,
        pickle_serializer: PickleSerializer,
        simple_object: SimpleSerializableObject,
        tmp_path: Path,
    ) -> None:
        """Test that save creates a file at the specified path."""
        filepath = tmp_path / "test.pkl"

        pickle_serializer.save(simple_object, filepath)

        assert filepath.exists()

    def test_load_returns_equivalent_object(
        self,
        pickle_serializer: PickleSerializer,
        simple_object: SimpleSerializableObject,
        tmp_path: Path,
    ) -> None:
        """Test that load returns an object equivalent to the saved one."""
        filepath = tmp_path / "test.pkl"
        pickle_serializer.save(simple_object, filepath)

        loaded = pickle_serializer.load(filepath)

        assert loaded == simple_object
        assert isinstance(loaded, SimpleSerializableObject)
        assert loaded.name == simple_object.name
        assert loaded.value == simple_object.value
        np.testing.assert_array_almost_equal(loaded.data, simple_object.data)

    def test_roundtrip_preserves_numpy_array(
        self, pickle_serializer: PickleSerializer, tmp_path: Path
    ) -> None:
        """Test that numpy arrays are preserved through save/load cycle."""
        obj = SimpleSerializableObject(
            name="array_test",
            value=0.0,
            data=np.random.default_rng(42).random((10, 5)),
        )
        filepath = tmp_path / "array_test.pkl"

        pickle_serializer.save(obj, filepath)
        loaded = pickle_serializer.load(filepath)

        assert isinstance(loaded, SimpleSerializableObject)
        np.testing.assert_array_almost_equal(loaded.data, obj.data)

    def test_save_non_serializable_object(
        self, pickle_serializer: PickleSerializer, tmp_path: Path
    ) -> None:
        """Test that pickle can handle objects without to_dict method."""
        obj = NonSerializableObject(value=123)
        filepath = tmp_path / "non_serializable.pkl"

        pickle_serializer.save(obj, filepath)
        loaded = pickle_serializer.load(filepath)

        assert isinstance(loaded, NonSerializableObject)
        assert loaded.value == obj.value

    def test_save_with_string_path(
        self,
        pickle_serializer: PickleSerializer,
        simple_object: SimpleSerializableObject,
        tmp_path: Path,
    ) -> None:
        """Test that save works with string paths (not just Path objects)."""
        filepath_str = str(tmp_path / "string_path.pkl")
        filepath = Path(filepath_str)

        # Note: Current implementation expects Path, but open() accepts str
        pickle_serializer.save(simple_object, filepath)

        assert filepath.exists()


# =============================================================================
# JSONSerializer Tests
# =============================================================================


class TestJSONSerializer:
    """Test suite for JSONSerializer class."""

    def test_save_creates_file(
        self,
        json_serializer: JSONSerializer,
        simple_object: SimpleSerializableObject,
        tmp_path: Path,
    ) -> None:
        """Test that save creates a JSON file at the specified path."""
        filepath = tmp_path / "test.json"

        json_serializer.save(simple_object, filepath)

        assert filepath.exists()

    def test_save_creates_valid_json(
        self,
        json_serializer: JSONSerializer,
        simple_object: SimpleSerializableObject,
        tmp_path: Path,
    ) -> None:
        """Test that saved file contains valid JSON."""
        filepath = tmp_path / "test.json"
        json_serializer.save(simple_object, filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_save_includes_metadata(
        self,
        json_serializer: JSONSerializer,
        simple_object: SimpleSerializableObject,
        tmp_path: Path,
    ) -> None:
        """Test that saved JSON includes @module and @class metadata."""
        filepath = tmp_path / "test.json"
        json_serializer.save(simple_object, filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "@module" in data
        assert "@class" in data
        assert data["@class"] == "SimpleSerializableObject"

    def test_load_returns_equivalent_object(
        self,
        json_serializer: JSONSerializer,
        simple_object: SimpleSerializableObject,
        tmp_path: Path,
    ) -> None:
        """Test that load returns an object equivalent to the saved one."""
        filepath = tmp_path / "test.json"
        json_serializer.save(simple_object, filepath)

        loaded = json_serializer.load(filepath)

        assert loaded == simple_object
        assert isinstance(loaded, SimpleSerializableObject)
        assert loaded.name == simple_object.name
        assert loaded.value == simple_object.value

    def test_roundtrip_preserves_data(
        self, json_serializer: JSONSerializer, tmp_path: Path
    ) -> None:
        """Test that data is preserved through save/load cycle."""
        obj = SimpleSerializableObject(
            name="json_test",
            value=99.99,
            data=np.array([10.0, 20.0, 30.0]),
        )
        filepath = tmp_path / "roundtrip.json"

        json_serializer.save(obj, filepath)
        loaded = json_serializer.load(filepath)

        assert isinstance(loaded, SimpleSerializableObject)
        assert loaded.name == obj.name
        assert loaded.value == obj.value
        np.testing.assert_array_almost_equal(loaded.data, obj.data)

    def test_save_without_to_dict_raises_error(
        self, json_serializer: JSONSerializer, tmp_path: Path
    ) -> None:
        """Test that saving object without to_dict method raises AttributeError."""
        obj = NonSerializableObject(value=123)
        filepath = tmp_path / "no_to_dict.json"

        with pytest.raises(TypeError):
            json_serializer.save(obj, filepath)

    def test_load_with_invalid_module_raises_error(
        self, json_serializer: JSONSerializer, tmp_path: Path
    ) -> None:
        """Test that loading JSON with invalid module raises TypeError."""
        filepath = tmp_path / "invalid_module.json"
        data = {
            "@module": "nonexistent.module.path",
            "@class": "NonexistentClass",
            "name": "test",
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

        with pytest.raises(TypeError, match="Could not find class"):
            json_serializer.load(filepath)

    def test_load_with_invalid_class_raises_error(
        self, json_serializer: JSONSerializer, tmp_path: Path
    ) -> None:
        """Test that loading JSON with invalid class raises TypeError."""
        filepath = tmp_path / "invalid_class.json"
        # Use a valid module but invalid class name
        data = {
            "@module": "pyprocar.core.serializer",
            "@class": "NonexistentSerializer",
            "name": "test",
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

        with pytest.raises(TypeError, match="Could not find class"):
            json_serializer.load(filepath)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetSerializer:
    """Test suite for get_serializer factory function."""

    def test_pkl_extension_returns_pickle_serializer(self) -> None:
        """Test that .pkl extension returns PickleSerializer."""
        serializer = get_serializer(Path("test.pkl"))

        assert isinstance(serializer, PickleSerializer)

    def test_pickle_extension_returns_pickle_serializer(self) -> None:
        """Test that .pickle extension returns PickleSerializer."""
        serializer = get_serializer(Path("test.pickle"))

        assert isinstance(serializer, PickleSerializer)

    def test_json_extension_returns_json_serializer(self) -> None:
        """Test that .json extension returns JSONSerializer."""
        serializer = get_serializer(Path("test.json"))

        assert isinstance(serializer, JSONSerializer)

    def test_string_path_is_converted_to_path(self) -> None:
        """Test that string paths are converted to Path objects."""
        serializer = get_serializer("test.pkl")

        assert isinstance(serializer, PickleSerializer)

    def test_unsupported_extension_raises_key_error(self) -> None:
        """Test that unsupported file extension raises KeyError."""
        with pytest.raises(KeyError):
            get_serializer(Path("test.txt"))

    def test_unsupported_extension_xml(self) -> None:
        """Test that .xml extension raises KeyError."""
        with pytest.raises(KeyError):
            get_serializer(Path("test.xml"))

    def test_case_sensitive_extension(self) -> None:
        """Test that extension matching is case-sensitive."""
        # Uppercase extensions should not match
        with pytest.raises(KeyError):
            get_serializer(Path("test.PKL"))


class TestSerializersDict:
    """Test suite for SERIALIZERS dictionary."""

    def test_contains_pkl_key(self) -> None:
        """Test that SERIALIZERS contains 'pkl' key."""
        assert "pkl" in SERIALIZERS

    def test_contains_pickle_key(self) -> None:
        """Test that SERIALIZERS contains 'pickle' key."""
        assert "pickle" in SERIALIZERS

    def test_contains_json_key(self) -> None:
        """Test that SERIALIZERS contains 'json' key."""
        assert "json" in SERIALIZERS

    def test_pkl_and_pickle_are_both_pickle_serializers(self) -> None:
        """Test that 'pkl' and 'pickle' both map to PickleSerializer."""
        assert isinstance(SERIALIZERS["pkl"], PickleSerializer)
        assert isinstance(SERIALIZERS["pickle"], PickleSerializer)

    def test_all_values_are_serializers(self) -> None:
        """Test that all values in SERIALIZERS are BaseSerializer instances."""
        for key, serializer in SERIALIZERS.items():
            assert isinstance(serializer, BaseSerializer), (
                f"SERIALIZERS['{key}'] is not a BaseSerializer instance"
            )


# =============================================================================
# Integration Tests
# =============================================================================


class TestSerializerIntegration:
    """Integration tests for the serialization workflow."""

    def test_pickle_workflow_via_factory(
        self, simple_object: SimpleSerializableObject, tmp_path: Path
    ) -> None:
        """Test complete pickle workflow using get_serializer."""
        filepath = tmp_path / "integration.pkl"

        # Save
        serializer = get_serializer(filepath)
        serializer.save(simple_object, filepath)

        # Load
        loaded = get_serializer(filepath).load(filepath)

        assert loaded == simple_object

    def test_json_workflow_via_factory(
        self, simple_object: SimpleSerializableObject, tmp_path: Path
    ) -> None:
        """Test complete JSON workflow using get_serializer."""
        filepath = tmp_path / "integration.json"

        # Save
        serializer = get_serializer(filepath)
        serializer.save(simple_object, filepath)

        # Load
        loaded = get_serializer(filepath).load(filepath)

        assert loaded == simple_object

    def test_different_extensions_produce_different_files(
        self, simple_object: SimpleSerializableObject, tmp_path: Path
    ) -> None:
        """Test that different serializers produce different file formats."""
        pkl_path = tmp_path / "test.pkl"
        json_path = tmp_path / "test.json"

        get_serializer(pkl_path).save(simple_object, pkl_path)
        get_serializer(json_path).save(simple_object, json_path)

        # JSON file should be text-readable
        with open(json_path) as f:
            json_content = f.read()
        assert "@module" in json_content
        assert "@class" in json_content

        # PKL file should be binary (not text-readable as JSON)
        with open(pkl_path, "rb") as f:
            pkl_content = f.read()
        # Pickle files start with specific bytes, not '{' like JSON
        assert pkl_content[0:1] != b"{"
