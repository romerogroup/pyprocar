import numpy as np
import pytest

from pyprocar.core.property_store import Property


class TestProperty:
    """Test suite for Property class."""
    
    def test_init(self):
        """Test initialization of Property class."""
        property = Property(name="property")
        assert property == {"name": "property", 
        "value": np.array([]), 
        "gradients": {1: np.array([]), 2: np.array([])}, 
        "divergence": np.array([]), 
        "vortex": np.array([]), 
        "laplacian": np.array([])}

    def test_init_with_dict(self):
        """Test initialization of Property class with dictionary."""
        property = Property(name="property", value=np.array([1.0, 2.0]), gradients={1: np.array([0.1, 0.2])})
        other_property = Property(name="property", value=np.array([1.0, 2.0]), gradients={1: np.array([0.1, 0.2])})
        assert property == other_property
        assert np.allclose(property.value, other_property.value)
        assert np.allclose(property.gradients[1], other_property.gradients[1])

    def test_init_with_invalid_type(self):
        invalid_data = {"name": "property", "invalid_key": np.array([1.0, 2.0])}
        with pytest.raises(TypeError):
            _ = Property(**invalid_data)

    def test_iter_values(self):
        property = Property(name="property", value=np.array([1.0, 2.0]), gradients={1: np.array([0.1, 0.2])})
        values = list(property)
        assert len(values) == 2
        assert np.allclose(values[0][0], np.array([1.0, 2.0]))
        assert values[0][1] == 0
        assert np.allclose(values[1][0], np.array([0.1, 0.2]))
        assert values[1][1] == 1

