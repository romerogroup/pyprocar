"""
Test module for pyprocar.core.serializer module.

This module contains unit tests for the serialization framework including
PickleSerializer and the get_serializer factory function.
"""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.serializer import (
    SERIALIZERS,
    PickleSerializer,
    get_serializer,
)

# =============================================================================
# Helper Classes for Testing
# =============================================================================


class SimpleSerializableObject:
    """A simple picklable object with value equality."""

    name: str
    value: float
    data: np.ndarray

    def __init__(self, name: str, value: float, data: np.ndarray | None = None):
        self.name = name
        self.value = value
        self.data = data if data is not None else np.array([1.0, 2.0, 3.0])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleSerializableObject):
            return False
        return (
            self.name == other.name
            and self.value == other.value
            and np.allclose(self.data, other.data)
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_object():
    """Create a simple serializable object for testing."""
    return SimpleSerializableObject(
        name="test_object",
        value=42.5,
        data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


@pytest.fixture
def pickle_serializer():
    """Return a PickleSerializer instance."""
    return PickleSerializer()


# =============================================================================
# PickleSerializer Tests
# =============================================================================


class TestPickleSerializer:
    """Test suite for PickleSerializer class."""

    def test_save_creates_file(self, pickle_serializer, simple_object, tmp_path):
        """Test that save creates a file at the specified path."""
        filepath = tmp_path / "test.pkl"

        pickle_serializer.save(simple_object, filepath)

        assert filepath.exists()

    def test_load_returns_equivalent_object(self, pickle_serializer, simple_object, tmp_path):
        """Test that load returns an object equivalent to the saved one."""
        filepath = tmp_path / "test.pkl"
        pickle_serializer.save(simple_object, filepath)

        loaded = pickle_serializer.load(filepath)

        assert loaded == simple_object
        assert loaded.name == simple_object.name
        assert loaded.value == simple_object.value
        np.testing.assert_array_almost_equal(loaded.data, simple_object.data)

    def test_roundtrip_preserves_numpy_array(self, pickle_serializer, tmp_path):
        """Test that numpy arrays are preserved through save/load cycle."""
        obj = SimpleSerializableObject(
            name="array_test",
            value=0.0,
            data=np.random.default_rng(42).random((10, 5)),
        )
        filepath = tmp_path / "array_test.pkl"

        pickle_serializer.save(obj, filepath)
        loaded = pickle_serializer.load(filepath)

        np.testing.assert_array_almost_equal(loaded.data, obj.data)

    def test_save_with_string_path(self, pickle_serializer, simple_object, tmp_path):
        """Test that save works with string paths (not just Path objects)."""
        filepath = str(tmp_path / "string_path.pkl")

        # Note: Current implementation expects Path, but open() accepts str
        pickle_serializer.save(simple_object, filepath)

        assert Path(filepath).exists()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetSerializer:
    """Test suite for get_serializer factory function."""

    def test_pkl_extension_returns_pickle_serializer(self):
        """Test that .pkl extension returns PickleSerializer."""
        serializer = get_serializer(Path("test.pkl"))

        assert isinstance(serializer, PickleSerializer)

    def test_pickle_extension_returns_pickle_serializer(self):
        """Test that .pickle extension returns PickleSerializer."""
        serializer = get_serializer(Path("test.pickle"))

        assert isinstance(serializer, PickleSerializer)

    def test_string_path_is_converted_to_path(self):
        """Test that string paths are converted to Path objects."""
        serializer = get_serializer("test.pkl")

        assert isinstance(serializer, PickleSerializer)

    def test_unsupported_extension_raises_key_error(self):
        """Test that unsupported file extension raises KeyError."""
        with pytest.raises(KeyError):
            get_serializer(Path("test.txt"))

    def test_unsupported_extension_xml(self):
        """Test that .xml extension raises KeyError."""
        with pytest.raises(KeyError):
            get_serializer(Path("test.xml"))

    def test_case_sensitive_extension(self):
        """Test that extension matching is case-sensitive."""
        # Uppercase extensions should not match
        with pytest.raises(KeyError):
            get_serializer(Path("test.PKL"))


class TestSerializersDict:
    """Test suite for SERIALIZERS dictionary."""

    def test_contains_pkl_key(self):
        """Test that SERIALIZERS contains 'pkl' key."""
        assert "pkl" in SERIALIZERS

    def test_contains_pickle_key(self):
        """Test that SERIALIZERS contains 'pickle' key."""
        assert "pickle" in SERIALIZERS

    def test_pkl_and_pickle_are_both_pickle_serializers(self):
        """Test that 'pkl' and 'pickle' both map to PickleSerializer."""
        assert isinstance(SERIALIZERS["pkl"], PickleSerializer)
        assert isinstance(SERIALIZERS["pickle"], PickleSerializer)

# =============================================================================
# Integration Tests
# =============================================================================


class TestSerializerIntegration:
    """Integration tests for the serialization workflow."""

    def test_pickle_workflow_via_factory(self, simple_object, tmp_path):
        """Test complete pickle workflow using get_serializer."""
        filepath = tmp_path / "integration.pkl"

        # Save
        serializer = get_serializer(filepath)
        serializer.save(simple_object, filepath)

        # Load
        loaded = get_serializer(filepath).load(filepath)

        assert loaded == simple_object
