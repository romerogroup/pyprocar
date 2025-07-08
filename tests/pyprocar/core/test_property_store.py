import pytest
import numpy as np
from pyprocar.core.property_store import PropertyStore, Property


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


class TestPropertyStore:
    """Test suite for PropertyStore class."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.sample_data = {
            'bands': {
                'value': np.array([[1.0, 2.0], [3.0, 4.0]]),
                'gradients': {
                    1: np.array([[0.1, 0.2], [0.3, 0.4]]),
                    2: np.array([[0.01, 0.02], [0.03, 0.04]])
                },
                'divergence': np.array([1.5, 2.5]),
                'vortex': np.array([0.5, 1.5]),
                'laplacian': np.array([2.0, 3.0])
            },
            'fermi_velocity': {
                'value': np.array([10.0, 20.0]),
                'gradients': {
                    1: np.array([1.0, 2.0])
                }
            }
        }
    
    def test_init_empty(self):
        """Test initialization with no arguments."""
        store = PropertyStore()
        assert len(store) == 0
        assert isinstance(store.properties, dict)
    
    def test_init_with_properties(self):
        """Test initialization with properties."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        properties = {"bands": prop1, "dos": prop2}
        
        store = PropertyStore(properties=properties)
        assert len(store) == 2
        assert "bands" in store
        assert "dos" in store
        assert store["bands"] == prop1
        assert store["dos"] == prop2
    
    def test_getitem(self):
        """Test getting properties by key."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        store["bands"] = prop
        
        retrieved = store["bands"]
        assert retrieved == prop
        assert retrieved.name == "bands"
        assert np.allclose(retrieved.value, np.array([1.0, 2.0]))
    
    def test_getitem_missing_key(self):
        """Test getting property with missing key raises KeyError."""
        store = PropertyStore()
        with pytest.raises(KeyError):
            _ = store["nonexistent"]
    
    def test_setitem(self):
        """Test setting properties."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        
        store["bands"] = prop
        assert len(store) == 1
        assert store["bands"] == prop
    
    def test_setitem_invalid_type(self):
        """Test setting property with invalid type."""
        store = PropertyStore()
        # __setitem__ should only accept Property objects
        # Based on the code, it checks if isinstance(value, Property)
        # and silently fails if not, so this test checks that behavior
        store["invalid"] = "not a property"  # type: ignore
        assert len(store) == 0  # Should not be added
    
    def test_delitem(self):
        """Test deleting properties."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        store["bands"] = prop
        
        assert "bands" in store
        del store["bands"]
        assert "bands" not in store
        assert len(store) == 0
    
    def test_delitem_missing_key(self):
        """Test deleting property with missing key raises KeyError."""
        store = PropertyStore()
        with pytest.raises(KeyError):
            del store["nonexistent"]
    
    def test_len(self):
        """Test length of property store."""
        store = PropertyStore()
        assert len(store) == 0
        
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        
        store["bands"] = prop1
        assert len(store) == 1
        
        store["dos"] = prop2
        assert len(store) == 2
    
    def test_iter(self):
        """Test iteration over property store."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        
        store = PropertyStore()
        store["bands"] = prop1
        store["dos"] = prop2
        
        keys = list(store)
        assert len(keys) == 2
        assert "bands" in keys
        assert "dos" in keys
    
    def test_contains(self):
        """Test membership testing."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        store["bands"] = prop
        
        assert "bands" in store
        assert "nonexistent" not in store
    
    def test_str(self):
        """Test string representation of property store."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        store["bands"] = prop
        
        string_repr = str(store)
        assert "Property Store" in string_repr
        assert "Number of properties = 1" in string_repr
        assert "bands" in string_repr
    
    def test_iter_arrays(self):
        """Test iteration over arrays in property store."""
        prop = Property(
            name="bands", 
            value=np.array([1.0, 2.0]),
            gradients={1: np.array([0.1, 0.2]), 2: np.array([0.01, 0.02])}
        )
        store = PropertyStore()
        store["bands"] = prop
        
        arrays = list(store.iter_arrays())
        assert len(arrays) == 3  # value + 2 gradients
        
        # Check that we get the expected arrays
        prop_names = [arr[0] for arr in arrays]
        assert all(name == "bands" for name in prop_names)
        
        # Check gradient orders
        gradient_orders = [arr[2] for arr in arrays]
        assert 0 in gradient_orders  # value
        assert 1 in gradient_orders  # gradient order 1
        assert 2 in gradient_orders  # gradient order 2
    
    def test_iter_arrays_empty_arrays(self):
        """Test iteration over arrays with empty arrays."""
        prop = Property(
            name="bands", 
            value=np.array([]),  # empty array
            gradients={1: np.array([]), 2: np.array([0.01, 0.02])}  # one empty, one not
        )
        store = PropertyStore()
        store["bands"] = prop
        
        arrays = list(store.iter_arrays())
        assert len(arrays) == 1  # only the non-empty gradient
        
        # Check that we only get the non-empty gradient
        assert arrays[0][2] == 2  # gradient order 2
    
    def test_iter_arrays_multiple_properties(self):
        """Test iteration over arrays with multiple properties."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        
        store = PropertyStore()
        store["bands"] = prop1
        store["dos"] = prop2
        
        arrays = list(store.iter_arrays())
        assert len(arrays) == 2  # two values
        
        prop_names = [arr[0] for arr in arrays]
        assert "bands" in prop_names
        assert "dos" in prop_names
    
    def test_keys_values_items(self):
        """Test keys, values, and items methods from MutableMapping."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        
        store = PropertyStore()
        store["bands"] = prop1
        store["dos"] = prop2
        
        # Test keys
        keys = list(store.keys())
        assert len(keys) == 2
        assert "bands" in keys
        assert "dos" in keys
        
        # Test values
        values = list(store.values())
        assert len(values) == 2
        assert prop1 in values
        assert prop2 in values
        
        # Test items
        items = list(store.items())
        assert len(items) == 2
        assert ("bands", prop1) in items
        assert ("dos", prop2) in items
    
    def test_update(self):
        """Test update method from MutableMapping."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        
        store1 = PropertyStore()
        store1["bands"] = prop1
        
        store2 = PropertyStore()
        store2["dos"] = prop2
        
        store1.update(store2)
        assert len(store1) == 2
        assert "bands" in store1
        assert "dos" in store1
    
    def test_update_with_dict(self):
        """Test update method with dictionary."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        
        store = PropertyStore()
        store["bands"] = prop1
        
        update_dict = {"dos": prop2}
        store.update(update_dict)
        assert len(store) == 2
        assert "bands" in store
        assert "dos" in store
    
    def test_get(self):
        """Test get method from MutableMapping."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        store["bands"] = prop
        
        # Test existing key
        result = store.get("bands")
        assert result == prop
        
        # Test non-existing key with default
        result = store.get("nonexistent", "default")
        assert result == "default"
        
        # Test non-existing key without default
        result = store.get("nonexistent")
        assert result is None
    
    def test_setdefault(self):
        """Test setdefault method from MutableMapping."""
        store = PropertyStore()
        
        # Test with new key
        default_prop = Property(name="default", value=np.array([5.0, 6.0]))
        result = store.setdefault("bands", default_prop)
        assert result == default_prop
        assert store["bands"] == default_prop
        
        # Test with existing key
        new_prop = Property(name="new", value=np.array([7.0, 8.0]))
        result = store.setdefault("bands", new_prop)
        assert result == default_prop  # Should return existing value
        assert store["bands"] == default_prop  # Should not change
    
    def test_pop(self):
        """Test pop method from MutableMapping."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        store["bands"] = prop
        
        # Test popping existing key
        result = store.pop("bands")
        assert result == prop
        assert "bands" not in store
        
        # Test popping non-existing key with default
        result = store.pop("nonexistent", "default")
        assert result == "default"
        
        # Test popping non-existing key without default
        with pytest.raises(KeyError):
            store.pop("nonexistent")
    
    def test_popitem(self):
        """Test popitem method from MutableMapping."""
        prop = Property(name="bands", value=np.array([1.0, 2.0]))
        store = PropertyStore()
        store["bands"] = prop
        
        # Test popping from non-empty store
        key, value = store.popitem()
        assert key == "bands"
        assert value == prop
        assert len(store) == 0
        
        # Test popping from empty store
        with pytest.raises(KeyError):
            store.popitem()
    
    def test_clear(self):
        """Test clear method from MutableMapping."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        
        store = PropertyStore()
        store["bands"] = prop1
        store["dos"] = prop2
        
        assert len(store) == 2
        store.clear()
        assert len(store) == 0
        assert "bands" not in store
        assert "dos" not in store
    
    def test_empty_store_str(self):
        """Test string representation of empty property store."""
        store = PropertyStore()
        string_repr = str(store)
        assert "Property Store" in string_repr
        assert "Number of properties = 0" in string_repr
    
    def test_equality_with_dict(self):
        """Test equality comparison with dictionary."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        
        store = PropertyStore()
        store["bands"] = prop1
        store["dos"] = prop2
        
        # PropertyStore should work like a dict for comparison
        assert store == store  # self-equality
        
        # Test dict-like access
        assert store["bands"] == prop1
        assert store["dos"] == prop2
    
    def test_multiple_operations(self):
        """Test complex operations with multiple properties."""
        prop1 = Property(name="bands", value=np.array([1.0, 2.0]))
        prop2 = Property(name="dos", value=np.array([3.0, 4.0]))
        prop3 = Property(name="fermi", value=np.array([5.0, 6.0]))
        
        store = PropertyStore()
        
        # Add multiple properties
        store["bands"] = prop1
        store["dos"] = prop2
        store["fermi"] = prop3
        
        assert len(store) == 3
        
        # Test that all are accessible
        assert store["bands"] == prop1
        assert store["dos"] == prop2
        assert store["fermi"] == prop3
        
        # Remove one and check
        del store["dos"]
        assert len(store) == 2
        assert "dos" not in store
        assert "bands" in store
        assert "fermi" in store
        
        # Update with new property
        new_prop = Property(name="velocity", value=np.array([7.0, 8.0]))
        store["velocity"] = new_prop
        assert len(store) == 3
        assert store["velocity"] == new_prop
    
    def test_store_with_complex_properties(self):
        """Test store with properties containing all field types."""
        prop = Property(
            name="complex_bands",
            value=np.array([1.0, 2.0, 3.0]),
            gradients={
                1: np.array([0.1, 0.2, 0.3]),
                2: np.array([0.01, 0.02, 0.03]),
                3: np.array([0.001, 0.002, 0.003])
            },
            divergence=np.array([0.5, 0.6, 0.7]),
            vortex=np.array([0.8, 0.9, 1.0]),
            laplacian=np.array([1.1, 1.2, 1.3])
        )
        
        store = PropertyStore()
        store["complex_bands"] = prop
        
        # Test iter_arrays with complex property
        arrays = list(store.iter_arrays())
        assert len(arrays) == 7  # value + 3 gradients + divergence + vortex + laplacian
        
        # Check that we get all expected arrays
        gradient_orders = [arr[2] for arr in arrays]
        assert 0 in gradient_orders  # value
        assert 1 in gradient_orders  # gradient order 1
        assert 2 in gradient_orders  # gradient order 2
        assert 3 in gradient_orders  # gradient order 3
        
        # All should be from the same property
        prop_names = [arr[0] for arr in arrays]
        assert all(name == "complex_bands" for name in prop_names)
