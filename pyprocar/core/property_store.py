

from collections.abc import Generator, MutableMapping
from typing import Callable, Literal, cast

import numpy as np
import numpy.typing as npt
import pyvista as pv
from typing_extensions import override

VALUE_ARRAY_TYPE = npt.NDArray[np.float64]
GRADIENT_TYPE =  dict[int, VALUE_ARRAY_TYPE]
PROPERTY_KEY_TYPE = str | tuple[str, int]
PROPERTY_VALUE_TYPE = VALUE_ARRAY_TYPE | GRADIENT_TYPE
PROPERTY_DICT_TYPE = dict[str, PROPERTY_VALUE_TYPE]

# pyright: reportUnknownMemberType=false

class Property:
    name: str
    value: npt.NDArray[np.float64]
    gradients: dict[int, npt.NDArray[np.float64]]
    _laplacian: npt.NDArray[np.float64] = np.array([])
    _divergence: npt.NDArray[np.float64] = np.array([])
    _curl: npt.NDArray[np.float64] = np.array([])
    _divergence_gradient: npt.NDArray[np.float64] = np.array([])
    _curl_gradient: npt.NDArray[np.float64] = np.array([])
    _magnitude: npt.NDArray[np.float64] = np.array([])

    def __init__(self, 
                name:str, 
                value: npt.NDArray[np.float64] | None = None, 
                gradients: dict[int, npt.NDArray[np.float64]] | None = None):
        self.name = name
        self.value = value if value is not None else np.array([])
        self.gradients = gradients if gradients is not None else {1: np.array([]), 2: np.array([])}

    @property
    def n_points(self) -> int:
        return self.value.shape[0]
    
    @property
    def is_vector(self) -> bool:
        return self.value.shape[-1] == 3
    
    @property
    def magnitude(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            magnitude = np.linalg.norm(self.value, axis=-1)
        else:
            magnitude = self.value
        return magnitude

    @property
    def divergence(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_1 = self.gradients[1]
            divergence = np.trace(gradient_1, axis1=-2, axis2=-1)
        else:
            divergence = np.array([])
        return divergence
        
    @property
    def curl(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_1 = self.gradients[1]
            x = gradient_1[...,2,1] - gradient_1[...,1,2]
            y = gradient_1[...,0,2] - gradient_1[...,2,0]
            z = gradient_1[...,1,0] - gradient_1[...,0,1]
            curl = np.stack([x, y, z], axis=-1)
        else:
            curl = np.array([])
        return curl
        
    @property
    def divergence_gradient(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_2 = self.gradients[2]
            x = gradient_2[...,0,0,0] + gradient_2[...,1,1,0] + gradient_2[...,2,2,0]
            y = gradient_2[...,0,0,1] + gradient_2[...,1,1,1] + gradient_2[...,2,2,1]
            z = gradient_2[...,0,0,2] + gradient_2[...,1,1,2] + gradient_2[...,2,2,2]
            divergence_gradient = np.stack([x, y, z], axis=-1)
        else:
            divergence_gradient = np.array([])
        return divergence_gradient
        
    @property
    def curl_gradient(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_2 = self.gradients[2]
            x = gradient_2[...,1,0,1] - gradient_2[...,0,1,1] - gradient_2[...,0,2,2] + gradient_2[...,2,0,2] 
            y = gradient_2[...,1,0,0] - gradient_2[...,0,1,0] - gradient_2[...,2,1,2] + gradient_2[...,1,2,2]
            z = gradient_2[...,2,0,0] - gradient_2[...,0,2,0] - gradient_2[...,2,1,1] + gradient_2[...,1,2,1]
            curl_gradient = np.stack([x, y, z], axis=-1)
        else:
            curl_gradient = np.array([])
        return curl_gradient
        
    @property
    def laplacian(self) -> npt.NDArray[np.float64]:
        gradient_2 = self.gradients[2]
        laplacian = np.trace(gradient_2, axis1=-2, axis2=-1)
        return laplacian
        
    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            other = Property(name=cast(str, other["name"]), 
                             value=cast(npt.NDArray[np.float64], other["value"]),
                             gradients=cast(dict[int, npt.NDArray[np.float64]], other["gradients"]),
                             divergence=cast(npt.NDArray[np.float64], other["divergence"]),
                             vortex=cast(npt.NDArray[np.float64], other["vortex"]),
                             laplacian=cast(npt.NDArray[np.float64], other["laplacian"]))
        if not isinstance(other, Property):
            return False

        is_equal = True
        is_equal = is_equal and self.name == other.name
        is_equal = is_equal and np.allclose(a=self.value, b=other.value)
        for gradient_order, gradient in self.gradients.items():
            is_equal = is_equal and np.allclose(a=gradient, b=other.gradients[gradient_order])
        is_equal = is_equal and np.allclose(a=self.divergence, b=other.divergence)
        is_equal = is_equal and np.allclose(a=self.vortex, b=other.vortex)
        is_equal = is_equal and np.allclose(a=self.laplacian, b=other.laplacian)
        return is_equal

    def __getitem__(self, key:  str | tuple[str, int]) -> dict[int, npt.NDArray[np.float64]] | npt.NDArray[np.float64] | str:
        calc_name, gradient_order = self._extract_key(key)
        if gradient_order == 0 and calc_name == "gradients":
            return  self.gradients
        elif gradient_order == 0 and calc_name == "name":
            return self.name
        elif gradient_order == 0 and calc_name in ["value", "divergence", "vortex", "laplacian"]:
            return cast(npt.NDArray[np.float64], getattr(self, calc_name))
        elif gradient_order > 0 and calc_name == "gradients":
            if gradient_order not in self.gradients:
                error_message = f"Gradient order {gradient_order} not found for property {calc_name}."
                error_message += f"Assigned gradients are {list(self.gradients.keys())}"
                raise ValueError(error_message)
            return self.gradients[gradient_order]
        else:
            raise ValueError(f"Invalid key: {key}. Must be a string or a tuple of two strings.")

    def __setitem__(self, 
                key:  str | tuple[str, int], 
                value: npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str) -> None:
        calc_name, gradient_order = self._extract_key(key)
        if gradient_order == 0:
            self.__dict__[calc_name] = value
        elif gradient_order > 0 and isinstance(value, np.ndarray):
            self.gradients[gradient_order] = value
        else:
            raise ValueError(f"Invalid gradient order: {gradient_order}. Must be a positive integer.")

    def __delitem__(self, key: PROPERTY_KEY_TYPE) -> None:
        self[key] = np.array([])

    @override
    def __str__(self) -> str:
        ret = f"{self.name} \n"
        ret += f" - Value: {self.value.shape}\n"
        ret += f" - Gradients:\n"
        if self.gradients[2].shape[0] != 0:
            for gradient_order, gradient in self.gradients.items():
                if gradient.shape[0] != 0:
                    ret += f"  - Gradients {gradient_order}: {gradient.shape}\n"
        return ret
    
    def iter_arrays(self) -> Generator[tuple[str, int, npt.NDArray[np.float64]], None, None]:
        for key, value in self.items():
            if isinstance(value, np.ndarray) and value.shape[0] != 0:
                yield key, 0, value
            elif isinstance(value, dict):
                for gradient_order, gradient in value.items():
                    if gradient.shape[0] != 0:
                        yield key, gradient_order, gradient

    def items(self) -> Generator[tuple[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str], None, None]:
        """Return field names and values as tuples."""
        yield from self.__dict__.items()

    def as_dict(self) -> dict[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str]:
        return self.__dict__

    def _extract_key(self, key: str | tuple[str, int]) ->  tuple[str, int]:
        if isinstance(key, str):
            return key, 0
  
        else:
            calc_name: str = key[0]
            gradient_order: int = key[1]
            if gradient_order == 0:
                calc_name = "value"
            return calc_name, gradient_order
    
class PointSet:
    _property_store: dict[str, Property]
    _gradient_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    _points: npt.NDArray[np.float64]
    
    def __init__(self, 
                 points: npt.ArrayLike, 
                 property_store: dict[str, Property] | None = None, 
                 gradient_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
                 transform_matrix: npt.NDArray[np.float64] | None = None) -> None:
        
        self._points = np.array(points)
        self._property_store = property_store if property_store is not None else {}
        self.validate_property_store()
        
        self._gradient_func = gradient_func if gradient_func is not None else lambda x, y: np.zeros_like(x)

    def __str__(self) -> str:
        ret = "\n Point Set     \n"
        ret += "============================\n"
        ret += "Points: \n"
        ret += "------------------------     \n"
        ret += f"Number of points = {self._points.shape[0]}\n"
        ret += f"Number of properties = {len(self._property_store)}\n\n"

        ret += "Properties: \n"
        ret += "------------------------     \n"
        for prop_name, prop in self._property_store.items():
            ret += f"{prop_name}: \n{prop}\n"
        return ret
    
    @property
    def points(self) -> npt.NDArray[np.float64]:
        return self._points
    
    @property
    def n_points(self) -> int:
        return self._points.shape[0]
    
    @property
    def property_store(self) -> dict[str, Property]:
        return self._property_store
    
    @property
    def n_properties(self) -> int:
        return len(self._property_store)

    @property
    def gradient_func(self) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        return self._gradient_func
    
    def validate_property_points(self, property:Property) -> None:
        if property.value.shape[0] != self._points.shape[0]:
            err_msg = f"Property ({property.name}) has {property.value.shape[0]} points. Expected {self._points.shape[0]} points."
            raise ValueError(err_msg)
        
    def validate_property_store(self, property_store:dict[str, Property] | None  = None) -> None:
        if property_store is None:
            property_store = self._property_store
        for prop_name, prop in property_store.items():
            self.validate_property_points(prop)
       
    def set_gradient_func(self, gradient_func: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> None:
        self._gradient_func = gradient_func

    def get_property(self, key = None) -> Property | None:
        
        prop_name, (calc_name, gradient_order) = self._extract_key(key)
        property = self._property_store.get(prop_name, None)
        if property is None:
            return None
        if calc_name is None:
            return property
        value = getattr(property, calc_name)
        if isinstance(value, dict) and gradient_order > 0:
            gradient = value.get(gradient_order, None)
            if gradient is None or gradient.shape[0] == 0:
                self.compute_gradients(gradient_order, names=[prop_name])
                gradient = value[gradient_order]
            return gradient
        else:
            return value
    
    def add_property(self, 
                     property: Property | None = None,
                     name:str | None = None,
                     value:npt.ArrayLike | None = None) -> None:
        if property is not None:
            self._property_store[property.name] = property
            return None
        
        if name is None or value is None:
            raise ValueError("Name and value are required to add a property.")
        
        property = self.property_store.get(name, None)
        if property is None:
            property = Property(name=name)
        
        if value is not None:
            property.value = np.array(value)
            
        self.validate_property_points(property)
        self._property_store[name] = property

    def update_property(self, 
                        property: Property | None = None,
                        name:str | None = None,
                        value:npt.ArrayLike | None = None) -> None:
        self.add_property(property=property, 
                          name=name, 
                          value=value)
        
    def update_points(self, points:npt.ArrayLike) -> None:
        self._points = np.array(points)
    
    def transform_points(self, transform_matrix:npt.NDArray[np.float64]) -> None:
        self._points = self._points @ transform_matrix
    
    def remove_property(self, name:str) -> Property | None:
        return self._property_store.pop(name, None)
    
    def compute_gradients(self, gradient_order:int, names:list[str] | None = None) -> None:
        if names is None:
            names = list(self._property_store.keys())
        if gradient_order < 0:
            raise ValueError(f"Gradient order must be greater than 0. Got {gradient_order}.")

        for name in names:
            property = self._property_store[name]

            if gradient_order == 1:
                scalars = property.value
            else:
                self.compute_gradients(gradient_order - 1, names=[name])
                scalars = property.gradients[gradient_order - 1]
            
            property.gradients[gradient_order] = self.gradient_func(self._points, scalars)
        return property.gradients[gradient_order]
    
    def iter_property_arrays(self, property_store:dict[str, Property] | None = None) -> Generator[tuple[str, str, int, npt.NDArray[np.float64]], None, None]:
        if property_store is None:
            property_store = self._property_store
        try:
            for prop_name, prop in property_store.items():
                for calc_name, gradient_order, value_array in prop.iter_arrays():
                    yield prop_name, calc_name, gradient_order, value_array
        finally:
            pass
        
    def _extract_key(self, key: str | tuple[str, int] | tuple[str,str] | tuple[str,str,int]) -> tuple[str, tuple[str | None, int]]:
        if isinstance(key, str):
            prop_name = key
            calc_name = None
            gradient_order = 0
        elif len(key) == 2 and isinstance(key[1], int):
            prop_name = key[0]
            calc_name = "gradients"
            gradient_order = key[1]
            if gradient_order == 0:
                calc_name = "value"
        elif len(key) == 2 and isinstance(key[1], str):
            prop_name = key[0]
            calc_name = key[1]
            gradient_order = 0
        elif len(key) == 3:
            prop_name = key[0]
            calc_name = key[1]
            gradient_order = key[2]
        else:
            error_message = f"Invalid key: {key}. \n"
            error_message += f"If you want to get a property, use the string key of the property. Example: 'bands' \n"
            error_message += f"If you want to get a gradient of a specific order, use the tuple of two strings." 
            error_message += f"Example: ('bands', 'gradients', 1) | ('bands', 1) | ('bands', 2) \n"
            error_message += f"If you want to get a specific calculation for a property, use the tuple of two strings. \n"
            error_message += f"Example: ('bands', 'value') | ('bands', 'gradients') | ('bands', 'vortices') | ('bands', 'divergences') | ('bands', 'laplacians') \n"
            raise ValueError(error_message)

        return prop_name, (calc_name, gradient_order)
