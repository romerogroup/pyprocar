from __future__ import annotations

import logging
import re
import weakref
from collections.abc import Callable, Generator, Mapping, Sequence
from enum import Enum
from typing import cast, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import override

from pyprocar.utils.math import np_round_to_half

# Type alias for metadata values that can be stored
MetadataValue = str | int | float | bool | list[str] | list[int] | list[float] | None

VALUE_ARRAY_TYPE = npt.NDArray[np.float64]
GRADIENT_TYPE = dict[int, VALUE_ARRAY_TYPE]
PROPERTY_KEY_TYPE = str | tuple[str, int]
PROPERTY_VALUE_TYPE = VALUE_ARRAY_TYPE | GRADIENT_TYPE
PROPERTY_DICT_TYPE = dict[str, PROPERTY_VALUE_TYPE]

logger = logging.getLogger(__name__)


@overload
def to_numpy_array(
    data: None, dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.float64]: ...


@overload
def to_numpy_array(
    data: pd.Series[float], dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.float64]: ...


@overload
def to_numpy_array(
    data: Mapping[str, npt.ArrayLike], dtype: npt.DTypeLike | None = None
) -> dict[str, npt.NDArray[np.float64]]: ...


@overload
def to_numpy_array(
    data: npt.ArrayLike, dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.float64]: ...


def to_numpy_array(
    data: npt.ArrayLike | pd.Series[float] | Mapping[str, npt.ArrayLike] | None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.float64] | dict[str, npt.NDArray[np.float64]]:
    """
    Convert input data (list, numpy array, pandas Series) into a numpy array.

    Args:
        data: list, np.ndarray, or pd.Series
        dtype: optional datatype for the resulting array.

    Returns:
        np.ndarray
    """
    if data is None:
        return np.array([], dtype=np.float64)
    if isinstance(data, pd.Series):
        arr: npt.NDArray[np.float64] = np.asarray(data.values, dtype=dtype)
        return arr
    elif isinstance(data, dict):
        return {key: to_numpy_array(value, dtype=dtype) for key, value in data.items()}
    else:
        result: npt.NDArray[np.float64] = np.asarray(data, dtype=dtype)
        return result


class GradientOrder(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    FOURTH = 4
    OTHER = 0

    @classmethod
    def from_int(cls, order: int) -> "GradientOrder":
        if order == 1:
            return cls.FIRST
        elif order == 2:
            return cls.SECOND
        elif order == 3:
            return cls.THIRD
        elif order == 4:
            return cls.FOURTH
        else:
            return cls.OTHER

    def get_suffix(self) -> str:
        if self == self.FIRST:
            return "st"
        elif self == self.SECOND:
            return "nd"
        elif self == self.THIRD:
            return "rd"
        else:
            return "th"


GradientFuncType = Callable[
    [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
]


class Property:
    name: str
    value: npt.NDArray[np.float64]
    gradients: dict[int, npt.NDArray[np.float64]]
    units: str | None
    label: str | None
    metadata: dict[str, MetadataValue]
    _data_lim: tuple[float | None, float | None] | None
    _point_set: weakref.ReferenceType[PointSet] | PointSet | None

    def __init__(
        self,
        name: str,
        value: npt.NDArray[np.float64] | None = None,
        gradients: dict[int, npt.NDArray[np.float64]] | None = None,
        points: npt.NDArray[np.float64] | None = None,
        gradient_func: GradientFuncType | None = None,
        units: str | None = None,
        label: str | None = None,
        point_set: PointSet | None = None,
        metadata: dict[str, MetadataValue] | None = None,
        data_lim: tuple[float | None, float | None] | None = None,
    ) -> None:
        self.name = name
        self.value = to_numpy_array(value)
        self.units = units
        self.metadata = metadata if metadata is not None else {}
        self._data_lim = data_lim
        self._point_set = None

        if gradients is not None:
            for gradient_order, gradient in gradients.items():
                gradients[gradient_order] = to_numpy_array(gradient)
            self.gradients = gradients
        else:
            self.gradients = {1: to_numpy_array(None), 2: to_numpy_array(None)}

        if point_set is not None and points is None:
            self._validate_point_set(point_set)
            self.bind_owner(point_set)
        elif point_set is None and points is not None:
            new_point_set = PointSet(points=points, gradient_func=gradient_func)
            self._validate_point_set(new_point_set)
            self._point_set = new_point_set
        elif point_set is not None and points is not None:
            raise ValueError("Either point_set or points and gradient_func must be provided.")

        self.label = label if label is not None else name

    @property
    def point_set(self) -> PointSet:
        if self._point_set is None:
            raise ValueError("point_set is not set")
        if isinstance(self._point_set, weakref.ref):
            result = self._point_set()
            if result is None:
                raise ValueError("point_set weakref has been garbage collected")
            return result
        return self._point_set

    @property
    def points(self) -> npt.NDArray[np.float64]:
        return self.point_set.points

    @property
    def points_label(self) -> str | None:
        return self.point_set.points_label

    @property
    def points_units(self) -> str | None:
        return self.point_set.points_units

    def gradient(
        self, order: int, store: bool = False, value: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        if value is None:
            tmp_value = self.value
        else:
            tmp_value = value

        if store and value is not None:
            raise ValueError("Value and store cannot be used together.")

        if order < 0:
            raise ValueError(f"Gradient order must be greater than 0. Got {order}.")

        if order == 0:
            return tmp_value

        for i in range(1, order + 1):
            print(f"Calculating gradient of order {i}")
            tmp_value = self.point_set.gradient_func(self.points, tmp_value)
            if store:
                self.gradients[i] = tmp_value
        return tmp_value

    def compute_gradient_property(self, order: int) -> "Property":
        gradient = self.gradient(order=order)
        gradient_order = GradientOrder.from_int(order)
        order_suffix = gradient_order.get_suffix()
        return Property(
            name=f"{self.name}_gradient_{order}",
            value=gradient,
            units=self.create_gradient_units(order),
            label=f"{self.label} " + str(order) + "^{" + order_suffix + "}" + " Order Gradient",
            point_set=self.point_set,
        )

    @property
    def n_points(self) -> int:
        shape: tuple[int, ...] = self.value.shape
        return shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying value array.

        This property delegates to the underlying numpy array's shape,
        allowing Property objects to be used in places where array shape
        access is expected.
        """
        return self.value.shape

    @property
    def is_vector(self) -> bool:
        shape: tuple[int, ...] = self.value.shape
        return shape[-1] == 3

    @property
    def magnitude(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            # np.linalg.norm returns a floating point array; cast for type checker
            magnitude: npt.NDArray[np.float64] = np.asarray(
                np.linalg.norm(self.value, axis=-1), dtype=np.float64
            )
        else:
            magnitude = self.value
        return magnitude

    @property
    def divergence(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_1 = self.gradients[1]
            # np.trace returns an ndarray; cast for type checker
            divergence: npt.NDArray[np.float64] = np.asarray(
                np.trace(gradient_1, axis1=-2, axis2=-1), dtype=np.float64
            )
        else:
            divergence = np.array([], dtype=np.float64)
        return divergence

    @property
    def curl(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_1 = self.gradients[1]
            x = gradient_1[..., 2, 1] - gradient_1[..., 1, 2]
            y = gradient_1[..., 0, 2] - gradient_1[..., 2, 0]
            z = gradient_1[..., 1, 0] - gradient_1[..., 0, 1]
            curl = np.stack([x, y, z], axis=-1)
        else:
            curl = np.array([])
        return curl

    @property
    def divergence_gradient(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_2 = self.gradients[2]
            x = gradient_2[..., 0, 0, 0] + gradient_2[..., 1, 1, 0] + gradient_2[..., 2, 2, 0]
            y = gradient_2[..., 0, 0, 1] + gradient_2[..., 1, 1, 1] + gradient_2[..., 2, 2, 1]
            z = gradient_2[..., 0, 0, 2] + gradient_2[..., 1, 1, 2] + gradient_2[..., 2, 2, 2]
            divergence_gradient = np.stack([x, y, z], axis=-1)
        else:
            divergence_gradient = np.array([])
        return divergence_gradient

    @property
    def curl_gradient(self) -> npt.NDArray[np.float64]:
        if self.is_vector:
            gradient_2 = self.gradients[2]
            x = (
                gradient_2[..., 1, 0, 1]
                - gradient_2[..., 0, 1, 1]
                - gradient_2[..., 0, 2, 2]
                + gradient_2[..., 2, 0, 2]
            )
            y = (
                gradient_2[..., 1, 0, 0]
                - gradient_2[..., 0, 1, 0]
                - gradient_2[..., 2, 1, 2]
                + gradient_2[..., 1, 2, 2]
            )
            z = (
                gradient_2[..., 2, 0, 0]
                - gradient_2[..., 0, 2, 0]
                - gradient_2[..., 2, 1, 1]
                + gradient_2[..., 1, 2, 1]
            )
            curl_gradient = np.stack([x, y, z], axis=-1)
        else:
            curl_gradient = np.array([])
        return curl_gradient

    @property
    def laplacian(self) -> npt.NDArray[np.float64]:
        gradient_2 = self.gradients[2]
        # np.trace returns an ndarray; cast for type checker
        laplacian: npt.NDArray[np.float64] = np.asarray(
            np.trace(gradient_2, axis1=-2, axis2=-1), dtype=np.float64
        )
        return laplacian

    def bind_owner(self, point_set: "PointSet") -> None:
        self._validate_point_set(point_set)
        self._point_set = weakref.ref(point_set)

    def _validate_point_set(self, point_set: "PointSet") -> None:
        point_set_shape = point_set.points.shape
        value_shape = self.value.shape
        if point_set_shape[0] != value_shape[0]:
            raise ValueError(
                f"point set and value have different number of points. Point set has {point_set_shape[0]} points, but value has {value_shape[0]} points."
            )

    def __call__(self) -> npt.NDArray[np.float64]:
        return self.value

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            # Type narrowing: other is a dict, use get() for safe access
            other_dict = cast(dict[str, object], other)
            name_val = other_dict.get("name")
            value_val = other_dict.get("value")
            gradients_val = other_dict.get("gradients")
            if not isinstance(name_val, str):
                return False
            if not isinstance(value_val, np.ndarray):
                return False
            gradients: dict[int, npt.NDArray[np.float64]] | None = None
            if gradients_val is not None and isinstance(gradients_val, dict):
                gradients = cast(dict[int, npt.NDArray[np.float64]], gradients_val)
            other = Property(
                name=name_val,
                value=value_val,
                gradients=gradients,
            )
        if not isinstance(other, Property):
            return False

        is_equal = True
        is_equal = is_equal and self.name == other.name
        is_equal = is_equal and np.allclose(a=self.value, b=other.value)
        for gradient_order, gradient in self.gradients.items():
            is_equal = is_equal and np.allclose(a=gradient, b=other.gradients[gradient_order])
        is_equal = is_equal and np.allclose(a=self.divergence, b=other.divergence)
        is_equal = is_equal and np.allclose(a=self.curl, b=other.curl)
        is_equal = is_equal and np.allclose(a=self.laplacian, b=other.laplacian)
        return is_equal

    def __getitem__(
        self, key: str | tuple[str, int] | int | slice | npt.NDArray[np.intp]
    ) -> dict[int, npt.NDArray[np.float64]] | npt.NDArray[np.float64] | str:
        # Check if key is a string for property access
        if isinstance(key, str):
            if key == "gradients":
                return self.gradients
            elif key == "name":
                return self.name
            elif key in ["value", "divergence", "vortex", "laplacian"]:
                return cast(npt.NDArray[np.float64], getattr(self, key))
            else:
                raise ValueError(f"Invalid string key: {key}")

        # Check if key is a tuple of (str, int) for gradient access
        if isinstance(key, tuple) and len(key) == 2:
            calc_name, gradient_order = self._extract_key(key)
            if gradient_order == 0 and calc_name == "gradients":
                return self.gradients
            elif gradient_order == 0 and calc_name == "name":
                return self.name
            elif gradient_order == 0 and calc_name in ["value", "divergence", "vortex", "laplacian"]:
                return cast(npt.NDArray[np.float64], getattr(self, calc_name))
            elif gradient_order > 0 and calc_name == "gradients":
                if gradient_order not in self.gradients:
                    error_message = (
                        f"Gradient order {gradient_order} not found for property {calc_name}."
                    )
                    error_message += f"Assigned gradients are {list(self.gradients.keys())}"
                    raise ValueError(error_message)
                return self.gradients[gradient_order]
            else:
                raise ValueError(f"Invalid key: {key}. Must be a string or a tuple of (str, int).")

        # For all other key types (int, slice, array of indices),
        # delegate to the underlying value array for numpy-style indexing
        result: npt.NDArray[np.float64] = self.value[key]
        return result

    def __setitem__(
        self,
        key: str | tuple[str, int],
        value: npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str,
    ) -> None:
        calc_name, gradient_order = self._extract_key(key)
        if gradient_order == 0:
            self.__dict__[calc_name] = value
        elif gradient_order > 0 and isinstance(value, np.ndarray):
            self.gradients[gradient_order] = value
        else:
            raise ValueError(
                f"Invalid gradient order: {gradient_order}. Must be a positive integer."
            )

    def __delitem__(self, key: PROPERTY_KEY_TYPE) -> None:
        self[key] = np.array([])

    @override
    def __str__(self) -> str:
        ret = f"{self.name} \n"
        ret += f" - Value: {self.value.shape}\n"
        ret += " - Gradients:\n"
        if self.gradients[2].shape[0] != 0:
            for gradient_order, gradient in self.gradients.items():
                if gradient.shape[0] != 0:
                    ret += f"  - Gradients {gradient_order}: {gradient.shape}\n"
        return ret

    @override
    def __repr__(self) -> str:
        tmp = self.__class__.__name__
        tmp += "("
        tmp += f"name={self.name}, "
        tmp += f"value={self.value.shape}, "
        tmp += f"units={self.units}, "
        tmp += f"label={self.label}, "
        tmp += f"data_lim={self.data_lim}, "
        for gradient_order, gradient in self.gradients.items():
            tmp += f"gradient_{gradient_order}={gradient.shape}"
            if gradient_order != list(self.gradients.keys())[-1]:
                tmp += ", "
        tmp += ")"
        return tmp

    def iter_arrays(self) -> Generator[tuple[str, int, npt.NDArray[np.float64]], None, None]:
        for key, value in self.items():
            if isinstance(value, np.ndarray) and value.shape[0] != 0:
                yield key, 0, value
            elif isinstance(value, dict):
                for gradient_order, gradient in value.items():
                    if gradient.shape[0] != 0:
                        yield key, gradient_order, gradient

    @property
    def n_channels(self) -> int:
        if self.value.ndim == 1:
            return 1
        else:
            shape: tuple[int, ...] = self.value.shape
            return shape[1]

    @property
    def data_lim(self) -> npt.NDArray[np.float64] | tuple[float | None, float | None]:
        if self._data_lim is None:
            # np.min/max return ndarrays; cast for type checker
            data_mins: npt.NDArray[np.float64] = np.asarray(
                np.min(self.value, axis=0), dtype=np.float64
            )
            data_maxs: npt.NDArray[np.float64] = np.asarray(
                np.max(self.value, axis=0), dtype=np.float64
            )
            data_lims: npt.NDArray[np.float64] = np.vstack([data_mins, data_maxs]).T
            return data_lims
        return self._data_lim

    @property
    def rounded_data_lim(self) -> npt.NDArray[np.float64] | tuple[float | None, float | None]:
        if self._data_lim is None:
            data_lims = self.data_lim
            # np.vectorize result needs explicit type annotation
            rounded_lims: npt.NDArray[np.float64] = np.asarray(
                np.vectorize(np_round_to_half)(data_lims), dtype=np.float64
            )
            return rounded_lims
        return self._data_lim

    def items(
        self,
    ) -> Generator[
        tuple[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str], None, None
    ]:
        """Return field names and values as tuples."""
        yield from self.__dict__.items()

    def as_dict(
        self,
    ) -> dict[str, npt.NDArray[np.float64] | dict[int, npt.NDArray[np.float64]] | str]:
        return self.__dict__

    def _extract_key(self, key: str | tuple[str, int]) -> tuple[str, int]:
        if isinstance(key, str):
            return key, 0

        else:
            calc_name: str = key[0]
            gradient_order: int = key[1]
            if gradient_order == 0:
                calc_name = "value"
            return calc_name, gradient_order

    def to_series(self) -> pd.Series[float]:
        return pd.Series(self.value, name=self.name)

    def to_array(self) -> npt.NDArray[np.float64]:
        return self.value

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.value, columns=[self.name])

    @property
    def has_denominator_unit(self) -> bool:
        if self.units is None:
            return False
        return "\\frac{" in self.units

    @property
    def denominator_unit(self) -> str:
        _, denom = self._get_frac_units()
        return denom

    @property
    def numerator_unit(self) -> str:
        numer, _ = self._get_frac_units()
        return numer

    def _get_frac_units(self) -> tuple[str, str]:
        if self.units is None:
            return "", ""
        if self.has_denominator_unit:
            result = parse_frac(self.units)
            if result is None:
                return self.units, ""
            return result
        return self.units, ""

    def create_gradient_units(self, order: int) -> str:
        pts_units = self.points_units
        if pts_units is None:
            pts_units = ""

        denom_unit = self.denominator_unit
        numer_unit = self.numerator_unit
        if denom_unit == pts_units:
            denom_unit = pts_units + "^{" + str(order + 1) + "}"
        elif denom_unit != pts_units and order == 1:
            denom_unit = denom_unit + " " + pts_units
        elif denom_unit != pts_units and order > 1:
            denom_unit = denom_unit + " " + pts_units + "^{" + str(order) + "}"
        else:
            raise ValueError(
                f"Invalid denominator unit: {denom_unit}. Must be {pts_units}."
            )

        grad_unit = "$\\frac{" + numer_unit + "}{" + denom_unit + "}$"
        return grad_unit


class PointSet:
    _point_data: dict[str, Property]
    _gradient_func: GradientFuncType
    _points: npt.NDArray[np.float64]
    _points_label: str | None
    _points_units: str | None

    def __init__(
        self,
        points: npt.ArrayLike,
        point_data: Mapping[str, Property] | Sequence[Property] | None = None,
        gradient_func: GradientFuncType | None = None,
        transform_matrix: npt.NDArray[np.float64] | None = None,
        points_label: str | None = None,
        points_units: str | None = None,
    ) -> None:
        self._points = np.array(points)
        self._points_label = points_label
        self._points_units = points_units
        self._point_data = {}
        if isinstance(point_data, Mapping):
            for _, prop in point_data.items():
                self.add_property(property=prop)
        elif isinstance(point_data, Sequence):
            for prop in point_data:
                self.add_property(property=prop)

        self.validate_point_data()

        self._gradient_func = (
            gradient_func if gradient_func is not None else lambda x, y: np.zeros_like(x)
        )

    @override
    def __str__(self) -> str:
        ret = "\n Point Set     \n"
        ret += "============================\n"
        ret += "Points: \n"
        ret += "------------------------     \n"
        points_shape: tuple[int, ...] = self._points.shape
        ret += f"Number of points = {points_shape[0]}\n"
        ret += f"Number of properties = {len(self._point_data)}\n\n"

        ret += "Properties: \n"
        ret += "------------------------     \n"
        for prop_name, prop in self._point_data.items():
            ret += f"{prop_name}: \n{prop}\n"
        return ret

    @property
    def points(self) -> npt.NDArray[np.float64]:
        return self._points

    @property
    def points_label(self) -> str | None:
        return self._points_label

    @property
    def points_units(self) -> str | None:
        return self._points_units

    @property
    def n_points(self) -> int:
        shape: tuple[int, ...] = self._points.shape
        return shape[0]

    @property
    def point_data(self) -> dict[str, Property]:
        return self._point_data

    @property
    def property_store(self) -> dict[str, Property]:
        return self._point_data

    @property
    def n_properties(self) -> int:
        return len(self._point_data)

    @property
    def gradient_func(
        self,
    ) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        return self._gradient_func

    def validate_property_points(self, property: Property) -> None:
        if property.value.shape[0] != self._points.shape[0]:
            err_msg = f"Property ({property.name}) has {property.value.shape[0]} points. Expected {self._points.shape[0]} points."
            raise ValueError(err_msg)

    def validate_point_data(self, property_store: Mapping[str, Property] | None = None) -> None:
        if property_store is None:
            property_store = self._point_data
        for prop in property_store.values():
            self.validate_property_points(prop)

    def set_gradient_func(
        self,
        gradient_func: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
    ) -> None:
        self._gradient_func = gradient_func

    def get_property(
        self,
        key: str | tuple[str, int] | tuple[str, str] | tuple[str, str, int] | None = None,
    ) -> Property | npt.NDArray[np.float64] | None:
        if key is None:
            return None
        prop_name, (calc_name, gradient_order) = self._extract_key(key)
        prop = self._point_data.get(prop_name, None)
        if prop is None:
            return None
        if calc_name is None:
            return prop
        # calc_name can be: value, gradients, divergence, curl, laplacian, magnitude, etc.
        # Use hasattr and type narrowing rather than getattr
        if calc_name == "gradients":
            gradients_dict = prop.gradients
            if gradient_order > 0:
                gradient = gradients_dict.get(gradient_order, None)
                if gradient is None or gradient.shape[0] == 0:
                    _ = self.compute_gradients(gradient_order, names=[prop_name])
                    gradient = gradients_dict[gradient_order]
                return gradient
            return None
        # For other properties (value, divergence, curl, laplacian, magnitude)
        if calc_name == "value":
            return prop.value
        if calc_name == "divergence":
            return prop.divergence
        if calc_name == "curl":
            return prop.curl
        if calc_name == "laplacian":
            return prop.laplacian
        if calc_name == "magnitude":
            return prop.magnitude
        return None

    def add_property(
        self,
        property: Property | None = None,
        name: str | None = None,
        value: npt.ArrayLike | None = None,
        gradients: dict[int, npt.NDArray[np.float64]] | None = None,
        points: npt.NDArray[np.float64] | None = None,
        gradient_func: GradientFuncType | None = None,
        units: str | None = None,
        label: str | None = None,
        metadata: dict[str, MetadataValue] | None = None,
        data_lim: tuple[float | None, float | None] | None = None,
    ) -> None:
        if property is not None:
            logger.info("Adding property %s", property.name)
            property.bind_owner(self)
            self._point_data[property.name] = property
            return

        if name is None or value is None:
            raise ValueError("Name and value are required to add a property.")

        logger.info("Adding property %s", name)

        prop = self.point_data.get(name, None)
        if prop is None:
            prop = Property(
                name=name,
                gradients=gradients,
                points=points,
                gradient_func=gradient_func,
                units=units,
                label=label,
                metadata=metadata,
                data_lim=data_lim,
            )

        prop.value = np.array(value)

        self.validate_property_points(prop)
        prop.bind_owner(self)
        self._point_data[name] = prop

    def update_property(
        self,
        property: Property | None = None,
        name: str | None = None,
        value: npt.ArrayLike | None = None,
    ) -> None:
        self.add_property(property=property, name=name, value=value)

    def update_points(self, points: npt.ArrayLike) -> None:
        self._points = np.array(points)

    def transform_points(self, transform_matrix: npt.NDArray[np.float64]) -> None:
        self._points = self._points @ transform_matrix

    def remove_property(self, name: str) -> Property | None:
        return self._point_data.pop(name, None)

    def compute_gradients(
        self, gradient_order: int, names: Sequence[str] | None = None
    ) -> npt.NDArray[np.float64]:
        if names is None:
            names = list(self._point_data.keys())
        if gradient_order < 0:
            raise ValueError(f"Gradient order must be greater than 0. Got {gradient_order}.")

        prop: Property | None = None
        for name in names:
            prop = self._point_data[name]

            if gradient_order == 1:
                scalars = prop.value
            else:
                _ = self.compute_gradients(gradient_order - 1, names=[name])
                scalars = prop.gradients[gradient_order - 1]

            prop.gradients[gradient_order] = self.gradient_func(self._points, scalars)

        if prop is None:
            raise ValueError("No properties to compute gradients for.")
        return prop.gradients[gradient_order]

    def iter_property_arrays(
        self, property_store: dict[str, Property] | None = None
    ) -> Generator[tuple[str, str, int, npt.NDArray[np.float64]], None, None]:
        if property_store is None:
            property_store = self._point_data
        try:
            for prop_name, prop in property_store.items():
                for calc_name, gradient_order, value_array in prop.iter_arrays():
                    yield prop_name, calc_name, gradient_order, value_array
        finally:
            pass

    def select_points(
        self, indices: npt.NDArray[np.intp] | Sequence[int]
    ) -> PointSet:
        if len(indices) == 0:
            return PointSet(
                points=np.empty((0, 3)), point_data={}, gradient_func=self.gradient_func
            )

        points = self.points[indices]

        new_point_data: dict[str, Property] = {}
        for prop_name, calc_name, gradient_order, value_array in self.iter_property_arrays():
            if prop_name not in new_point_data:
                new_point_data[prop_name] = Property(name=prop_name)

            new_point_data[prop_name][(calc_name, gradient_order)] = value_array[indices]

        return PointSet(
            points=points, point_data=new_point_data, gradient_func=self.gradient_func
        )

    def _extract_key(
        self, key: str | tuple[str, int] | tuple[str, str] | tuple[str, str, int]
    ) -> tuple[str, tuple[str | None, int]]:
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
            error_message += "If you want to get a property, use the string key of the property. Example: 'bands' \n"
            error_message += (
                "If you want to get a gradient of a specific order, use the tuple of two strings."
            )
            error_message += "Example: ('bands', 'gradients', 1) | ('bands', 1) | ('bands', 2) \n"
            error_message += "If you want to get a specific calculation for a property, use the tuple of two strings. \n"
            error_message += "Example: ('bands', 'value') | ('bands', 'gradients') | ('bands', 'vortices') | ('bands', 'divergences') | ('bands', 'laplacians') \n"
            raise ValueError(error_message)

        return prop_name, (calc_name, gradient_order)


def parse_frac(latex_expr: str) -> tuple[str, str] | None:
    """
    Extract numerator and denominator from a LaTeX \\frac command.

    Parameters
    ----------
    latex_expr : str
        A string containing a LaTeX math mode expression with \\frac.

    Returns
    -------
    tuple[str, str] or None
        (numerator, denominator) if found, else None.
    """
    # This regex assumes well-formed \frac{...}{...} with no nested braces
    pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    match = re.search(pattern, latex_expr)
    if match:
        num, den = match.groups()
        return num.strip(), den.strip()
    return None
