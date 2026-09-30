import pickle

import numpy as np
import pandas as pd
import pytest

from pyprocar.core.property_store import PointSet, Property


def generate_test_inputs(n_points=5):
    """
    Generate different types of data structures for testing
    conversion to numpy arrays.

    Args:
        n (int): number of elements (points) to generate for sequence-like inputs.

    Returns:
        dict of str -> data structure
    """
    return {
        "python_list": list(range(1, n_points + 1)),
        "python_tuple": tuple(range(10, 10 + n_points)),
        "numpy_array": np.arange(20, 20 + n_points),
        "pandas_series": pd.Series(np.arange(30, 30 + n_points)),
        "scalar_int": 42,
        "scalar_float": 3.14,
    }


@pytest.fixture
def sin_data():
    """Generate sine data for testing."""
    x = np.linspace(0, 2 * np.pi, 100)
    sin = np.sin(x)
    cos = np.cos(x)
    return {"x": x, "sin": sin, "cos": cos}


@pytest.fixture
def point_set_data(sin_data):
    """Generate point set data for testing."""
    return PointSet(points=sin_data["x"], point_data=[Property(name="sin", value=sin_data["sin"])])


class TestProperty:
    """Test suite for Property class."""

    def test_metadata_is_not_shared_between_instances(self):
        first = Property(name="first", value=np.zeros(2))
        first.metadata["source"] = "first"

        assert Property(name="second", value=np.zeros(2)).metadata == {}
        assert first.metadata == {"source": "first"}

    def test_unpickle_state_without_metadata(self):
        legacy = Property(name="legacy", value=np.arange(3.0))
        del legacy.metadata

        restored = pickle.loads(pickle.dumps(legacy))

        assert restored.metadata == {}
        assert restored.value.tolist() == [0.0, 1.0, 2.0]

    def test_init_numpy_array(self):
        test_data = generate_test_inputs(n_points=100)

        property = Property(name="property", value=test_data["numpy_array"])
        assert np.allclose(property.value, test_data["numpy_array"])

    def test_init_pandas_series(self):
        test_data = generate_test_inputs(n_points=100)
        property = Property(name="property", value=test_data["pandas_series"])
        assert np.allclose(property.value, test_data["pandas_series"])

    def test_init_python_tuple(self):
        test_data = generate_test_inputs(n_points=100)
        property = Property(name="property", value=test_data["python_tuple"])
        assert np.allclose(property.value, test_data["python_tuple"])

    def test_init_list(self):
        test_data = generate_test_inputs(n_points=100)
        property = Property(name="property", value=test_data["python_list"])
        assert np.allclose(property.value, test_data["python_list"])

    def test_init_scalar(self):
        test_data = generate_test_inputs(n_points=100)
        property = Property(name="property", value=test_data["scalar_int"])
        assert np.allclose(property.value, test_data["scalar_int"])

    def test_init_scalar_float(self):
        test_data = generate_test_inputs(n_points=100)
        property = Property(name="property", value=test_data["scalar_float"])
        assert np.allclose(property.value, test_data["scalar_float"])

    def test_init_with_gradients(self):
        """Test initialization of Property class with dictionary."""

        test_data = generate_test_inputs(n_points=100)
        value = test_data["numpy_array"]
        gradients = {1: test_data["numpy_array"], 2: test_data["numpy_array"]}

        property = Property(name="property", value=value, gradients=gradients)
        assert np.allclose(property.value, value)
        assert np.allclose(property.gradients[1], gradients[1])
        assert np.allclose(property.gradients[2], gradients[2])

    def test_init_with_invalid_type(self):
        invalid_data = {"name": "property", "invalid_key": np.array([1.0, 2.0])}
        with pytest.raises(TypeError):
            _ = Property(**invalid_data)

    def test_getitem_str(self, sin_data):
        property = Property(
            name="property",
            value=sin_data["sin"],
            gradients={1: sin_data["sin"], 2: sin_data["sin"]},
        )
        value = property["value"]
        assert np.allclose(value, sin_data["sin"])

    def test_init_with_point_set(self, sin_data):
        point_set = PointSet(points=sin_data["x"])
        property = Property(name="property", value=sin_data["sin"], point_set=point_set)
        assert np.allclose(property.point_set.points, sin_data["x"])

    def test_init_with_points_and_gradient_func(self, sin_data):
        property = Property(
            name="property",
            value=sin_data["sin"],
            points=sin_data["x"],
            gradient_func=lambda x, y: np.gradient(y, x, edge_order=2),
        )
        assert np.allclose(property.point_set.points, sin_data["x"])

    def test_init_error_with_invalid_point_set(self, sin_data):
        with pytest.raises(ValueError):
            _ = Property(name="sin", value=sin_data["sin"], points=sin_data["x"][:50])

    def test_getitem_tuple(self, sin_data):
        property = Property(
            name="property",
            value=sin_data["sin"],
            gradients={1: sin_data["sin"], 2: sin_data["sin"]},
        )
        value = property[("value", 0)]
        assert np.allclose(value, sin_data["sin"])

    def test_bind_owner(self, sin_data):
        point_set = PointSet(points=sin_data["x"])
        property = Property(name="sin", value=sin_data["sin"])
        property._bind_owner(point_set)
        assert np.allclose(property.point_set.points, sin_data["x"])

    def test_gradient(self, sin_data):
        point_set = PointSet(
            points=sin_data["x"], gradient_func=lambda x, y: np.gradient(y, x, edge_order=2)
        )
        property = Property(name="sin", value=sin_data["sin"])
        property._bind_owner(point_set)
        gradients = property.gradient(1)
        assert np.allclose(gradients, sin_data["cos"], atol=1e-2)

    def test_gradient_error_negative_order(self, sin_data):
        point_set = PointSet(
            points=sin_data["x"], gradient_func=lambda x, y: np.gradient(y, x, edge_order=2)
        )
        property = Property(name="sin", value=sin_data["sin"])
        property._bind_owner(point_set)
        with pytest.raises(ValueError):
            property.gradient(-1)
        # assert np.allclose(gradients, sin_data["cos"], atol=1e-2)

    def test_gradient_store(self, sin_data):
        point_set = PointSet(
            points=sin_data["x"], gradient_func=lambda x, y: np.gradient(y, x, edge_order=2)
        )
        property = Property(name="sin", value=sin_data["sin"], point_set=point_set)
        property.gradient(1, store=True)
        assert np.allclose(property.gradients[1], sin_data["cos"], atol=1e-2)

    def test_gradient_error_store_gradient_with_value(self, sin_data):
        point_set = PointSet(
            points=sin_data["x"], gradient_func=lambda x, y: np.gradient(y, x, edge_order=2)
        )
        property = Property(name="sin", value=sin_data["sin"], point_set=point_set)

        with pytest.raises(ValueError):
            property.gradient(1, store=True, value=sin_data["sin"])


class TestPointSet:
    """Test suite for PointSet class."""


    def test_compute_gradients_with_no_names_is_a_no_op(self, point_set_data):
        point_set_data.compute_gradients(1, names=[])

        assert point_set_data.get_property("sin").gradients[1].shape == (0,)

    def test_init(self, sin_data):
        point_set = PointSet(
            points=sin_data["x"], point_data={"sin": Property(name="sin", value=sin_data["sin"])}
        )

        assert np.allclose(point_set.points, sin_data["x"])
        assert np.allclose(point_set.point_data["sin"].value, sin_data["sin"])

    def test_init_mismatch(self, sin_data):
        with pytest.raises(ValueError):
            _ = PointSet(
                points=sin_data["x"],
                point_data={"sin": Property(name="sin", value=sin_data["sin"][:50])},
            )


class _Derived(PointSet):
    def __init__(self):
        super().__init__(points=np.zeros((2, 3)))
        self.add_property(name="source", value=np.array([1.0, 2.0]))

    def compute_property(self, name, **kwargs):
        if name != "doubled":
            return None
        return 2 * self.get_property("source").value


def test_get_property_recomputes_after_source_changes():
    ps = _Derived()
    assert ps.get_property("doubled").value.tolist() == [2.0, 4.0]

    ps.add_property(name="source", value=np.array([5.0, 6.0]))

    assert ps.get_property("doubled").value.tolist() == [10.0, 12.0]


def test_get_property_does_not_add_derived_entries_to_stored_data():
    ps = _Derived()
    ps.get_property("doubled", atoms=[0])

    assert list(ps.property_store) == ["source"]
    assert ps.get_property("source", atoms=[0]).value.tolist() == [1.0, 2.0]
