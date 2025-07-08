

from collections.abc import MutableMapping, Generator
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Literal, TypedDict, Any, cast, Callable
from typing_extensions import override

import numpy as np
import numpy.typing as npt

VALUE_ARRAY_TYPE = npt.NDArray[np.float64]
GRADIENT_TYPE =  dict[int, VALUE_ARRAY_TYPE]
PROPERTY_KEY_TYPE = str | tuple[str, int]
PROPERTY_VALUE_TYPE = VALUE_ARRAY_TYPE | GRADIENT_TYPE
PROPERTY_DICT_TYPE = dict[str, PROPERTY_VALUE_TYPE]

class Property:
    name: str
    value: npt.NDArray[np.float64]
    gradients: dict[int, npt.NDArray[np.float64]]
    divergence: npt.NDArray[np.float64]
    vortex: npt.NDArray[np.float64]
    laplacian: npt.NDArray[np.float64]

    def __init__(self, 
                name:str, 
                value: npt.NDArray[np.float64] | None = None, 
                gradients: dict[int, npt.NDArray[np.float64]] | None = None, 
                divergence: npt.NDArray[np.float64] | None = None, 
                vortex: npt.NDArray[np.float64] | None = None, 
                laplacian: npt.NDArray[np.float64] | None = None):
        self.name = name
        self.value = value if value is not None else np.array([])
        self.gradients = gradients if gradients is not None else {1: np.array([]), 2: np.array([])}
        self.divergence = divergence if divergence is not None else np.array([])
        self.vortex = vortex if vortex is not None else np.array([])
        self.laplacian = laplacian if laplacian is not None else np.array([])

    @property
    def n_points(self) -> int:
        return self.value.shape[0]

    @override
    def __eq__(self, other) -> bool:
        if isinstance(other, dict):
            other = Property(**other)
        is_equal = True
        is_equal = is_equal and self.name == other.name
        is_equal = is_equal and np.allclose(self.value, other.value)
        for gradient_order, gradient in self.gradients.items():
            is_equal = is_equal and np.allclose(gradient, other.gradients[gradient_order])
        is_equal = is_equal and np.allclose(self.divergence, other.divergence)
        is_equal = is_equal and np.allclose(self.vortex, other.vortex)
        is_equal = is_equal and np.allclose(self.laplacian, other.laplacian)
        return is_equal

    def __getitem__(self, key:  str | tuple[str, int]) -> dict[int, npt.NDArray[np.float64]] | npt.NDArray[np.float64] | str:
        calc_name, gradient_order = self._extract_key(key)
        calc_value = self.__dict__.get(calc_name, None)
        if calc_value is None:
            raise ValueError(f"Invalid key: {key}. Must be a string or a tuple of two strings.")
        if gradient_order == 0:
            return calc_value
        elif gradient_order > 0:
            if gradient_order not in calc_value:
                error_message = f"Gradient order {gradient_order} not found for property {calc_name}."
                error_message += f"Assigned gradients are {list(calc_value.keys())}"
                raise ValueError(error_message)
            return calc_value[gradient_order]
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
        ret = ""
        ret += f"Value: {self.value.shape}\n"
        for gradient_order, gradient in self.gradients.items():
            ret += f"Gradient {gradient_order}: {gradient.shape}\n"
        ret += f"Divergence: {self.divergence.shape}\n"
        ret += f"Vortex: {self.vortex.shape}\n"
        ret += f"Laplacian: {self.laplacian.shape}\n"
        return ret


    def items(self) -> Generator[tuple[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str], None, None]:
        """Return field names and values as tuples."""
        for key, value in self.__dict__.items():
            yield key, value

    def as_dict(self) -> dict[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str]:
        return self.__dict__

    def __iter__(self) -> Generator[tuple[npt.NDArray[np.float64], int], None, None]:
        for key, value in self.items():
            if key == "name":
                continue
            assert not isinstance(value, str)
            if isinstance(value, dict):
                for gradient_order, gradient in value.items():
                    if gradient.shape[0] != 0:
                        yield gradient, gradient_order
            else:
                if value.shape[0] != 0:
                    yield value, 0

    def _extract_key(self, key: str | tuple[str, int]) ->  tuple[str, int]:
        if isinstance(key, str):
            return key, 0
  
        else:
            calc_name = key[0]
            gradient_order = key[1]
            if gradient_order == 0:
                calc_name = "value"
            return calc_name, gradient_order

class PropertyStore(MutableMapping[str, Property]):
    properties: dict[str, Property]

    def __init__(self, properties: dict[str, Property] | None = None):
        self.properties = {}
        if properties is not None:
            for prop_name, prop in properties.items():
                self.properties[prop_name] = prop

    @override
    def __getitem__(self, key: str) -> Property:
        return self.properties[key]

    @override
    def __setitem__(self, key: str, value: Property):
        if isinstance(value, Property):
            self.properties[key] = value

    @override
    def __delitem__(self, key: str):
        del self.properties[key]
    
    @override
    def __len__(self) -> int:
        return len(self.properties)
    
    @override
    def __iter__(self) -> Generator[str, None, None]:
        yield from self.properties

    @override
    def __str__(self) -> str:
        ret = "\n Property Store     \n"
        ret += "============================\n"
        ret += "Properties: \n"
        ret += "------------------------     \n"
        ret += f"Number of properties = {len(self.properties)}\n"

        for prop_name, prop in self.items():
            ret += f"{prop_name}: \n{prop}\n"
        return ret

    def iter_arrays(self) -> Generator[tuple[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str, int], None, None]:
        for prop_name, prop in self.properties.items():
            for value_array, gradient_order in prop:
                yield prop_name, value_array, gradient_order


    def _extract_key(self, key: str | tuple[str, int] | tuple[str,str] | tuple[str,str,int]) -> tuple[str, tuple[str, int]]:
        if isinstance(key, str):
            prop_name = key
            calc_name = "value"
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


class PointPropertyStore:
    _points: npt.NDArray[np.float64]
    _point_store: PropertyStore
    _gradient_func: Callable[list[np.ndarray[tuple[int, Literal[1,2,3]], np.dtype[np.float_]], npt.NDArray[np.float64]], npt.NDArray[np.float64]]

    def __init__(self, points: npt.NDArray[np.float64], properties: dict[str, Property] | None = None):
        self._point_store = PropertyStore(properties) if properties is not None else PropertyStore()
        self._points = points

        for prop_name, prop in self._point_store.items():
            if prop.n_points != self._points.shape[0]:
                raise ValueError(f"Property arrays have to have the same shape as points.")
            self._point_store[prop_name] = prop
        
    def __getitem__(self, key: str) -> Property:
        return self._point_store[key]
    
    def __iter__(self) -> Generator[str, None, None]:
        yield from self._point_store
    
    def __len__(self) -> int:
        return len(self._point_store)

    def __contains__(self, key: str) -> bool:
        return key in self._point_store

    @property
    def gradient_func(self) -> Callable[[str, int], npt.NDArray[np.float64]]:
        return self._gradient_func

    @property
    def points(self) -> npt.NDArray[np.float64]:
        return self._points
    
    @property
    def point_store(self) -> PropertyStore:
        return self._point_store

    def items(self) -> Generator[tuple[str, Property], None, None]:
        yield from self._point_store.items()

    def iter_arrays(self) -> Generator[tuple[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str, int], None, None]:
        for prop_name, prop in self._point_store.items():
            for value_array, gradient_order in prop:
                yield prop_name, value_array, gradient_order
    
    def update(self, new_points: npt.NDArray[np.float64], new_properties: dict[str, Property]):
        if new_points.shape[0] != self.points.shape[0]:
            raise ValueError(f"New points have to have the same shape as points.")
        self._points = new_points
        for prop_name, prop in new_properties.items():
            if prop.n_points != self.points.shape[0]:
                raise ValueError(f"Property arrays have to have the same shape as points.")
            self._point_store[prop_name] = prop

    def add(self, property: Property):
        if property.n_points != self.points.shape[0]:
            raise ValueError(f"Property arrays have to have the same shape as points.")
        self._point_store[property.name] = property
    
    def __str__(self) -> str:
        ret = "\n Point Property Store     \n"
        ret += "============================\n"
        ret += "Points: \n"
        ret += "------------------------     \n"
        ret += f"Number of points = {self.points.shape[0]}\n"
        ret += "Properties: \n"
        ret += "------------------------     \n"
        ret += f"Number of properties = {len(self.point_store)}\n"
        for prop_name, prop in self.point_store.items():
            ret += f"{prop_name}: \n{prop}\n"
        return ret