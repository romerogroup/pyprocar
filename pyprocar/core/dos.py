"""Core density of states data object."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Any, Sequence
import copy

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline

from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer


logger = logging.getLogger(__name__)


def get_dos_from_code(
    code: str,
    dirpath: str,
    use_cache: bool = False,
    filename: str = "dos.pkl",
) -> "DensityOfStates":
    """Parse a calculation directory and return a :class:`DensityOfStates`.

    Parameters
    ----------
    code
        Identifier for the parser to use (e.g. ``"vasp"``).
    dirpath
        Calculation directory path.
    use_cache
        If ``True`` and a cached pickle exists it will be loaded instead of
        parsing the raw files.
    filename
        Name of the cache file to use when ``use_cache`` is ``True``.
    """

    from pyprocar.io import Parser

    dos_filepath = Path(dirpath) / filename

    if not use_cache or not dos_filepath.exists():
        logger.info("Parsing DOS calculation directory: %s", dirpath)
        parser = Parser(code=code, dirpath=dirpath)
        dos = parser.dos
        if use_cache:
            dos.save(dos_filepath)
    else:
        logger.info("Loading DOS from cache: %s", dos_filepath)
        dos = DensityOfStates.load(dos_filepath)

    return dos


def _finite_difference_gradient(
    points: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Simple 1D finite-difference gradient along the first axis."""
    energies = np.asarray(points, dtype=np.float64).reshape(-1)
    array = np.asarray(values, dtype=np.float64)

    if array.shape[0] != energies.shape[0]:
        raise ValueError(
            "Gradient requires the first axis of the property to match the "
            "number of sample points."
        )

    if energies.size < 2:
        return np.zeros_like(array)

    edge_order = 2 if energies.size > 2 else 1
    return np.gradient(array, energies, axis=0, edge_order=edge_order)


class NormMode(Enum):
    FRACTION = "fraction"
    RAW = "raw"
    MAX = "max"
    INTEGRAL = "integral"
    ELECTRONS = "electrons"
    TOTAL_PROJECTION = "total_projection"
    SPIN_MAGNITUDE = "spin_magnitude"
    
    @classmethod
    def from_input(cls, input: str | NormMode) -> NormMode:
        if isinstance(input, NormMode):
            return input
        if not isinstance(input, str):
            raise ValueError(f"Invalid normalization mode: {input}")
        
        lower_input = input.lower()
        if lower_input[0] == "f":
            return cls.FRACTION
        elif lower_input[0] == "r":
            return cls.RAW
        elif lower_input[0] == "m":
            return cls.MAX
        elif lower_input[0] == "i":
            return cls.INTEGRAL
        elif lower_input[0] == "e":
            return cls.ELECTRONS
        elif lower_input[0] == "t":
            return cls.TOTAL_PROJECTION
        elif lower_input[0] == "s":
            return cls.SPIN_MAGNITUDE
        
        list_modes = cls.list_modes()
        err_msg = f"Invalid normalization mode: {input}. Valid modes are:\n"
        err_msg += "\n".join([f"- {mode}" for mode in list_modes])
        raise ValueError(err_msg)
    
    @classmethod 
    def list_modes(cls) -> list[str]:
        return [mode.value for mode in cls]
    
    @classmethod
    def get_mode_prefix(cls, mode: str | NormMode) -> str:
        if mode == cls.RAW:
            return ""
        else:
            return "Normalized"
        
    @classmethod
    def get_mode_type_label(cls, mode: str | NormMode) -> str:
        if mode == cls.RAW:
            return ""
        elif mode == cls.TOTAL_PROJECTION:
            return "by Total Projection"
        elif mode == cls.SPIN_MAGNITUDE:
            return "by Spin Magnitude"
        elif mode == cls.MAX:
            return "by Max"
        elif mode == cls.INTEGRAL:
            return "by Integral"
        elif mode == cls.ELECTRONS:
            return "by Electrons"
        else:
            return ""

class DensityOfStates(PointSet):
    """Data-centric representation of a density of states calculation."""

    def __init__(
        self,
        energies: npt.ArrayLike,
        total: npt.ArrayLike,
        fermi: float = 0.0,
        projected: npt.ArrayLike | None = None,
        orbital_names: list[str] | None = None,
        gradient_func=None,
    ) -> None:
        energies_array = energies
        gradient = gradient_func or _finite_difference_gradient

        super().__init__(points=energies_array, gradient_func=gradient)

        self._fermi = float(fermi)
        self._orbital_names = orbital_names

        total_array = self._validate_total(total)
        self.add_property(name="total", 
                          value=total_array,
                          units = "states/eV",
                          label = "DOS")

        if projected is not None:
            projected_array = self._validate_projected(projected)
            self.add_property(name="projected", 
                              value=projected_array, 
                              units = "states/eV",
                              label = "Projected DOS")
            
        logger.debug(
            "Initialized DensityOfStates with %d energies, %d spin channels",
            self.n_energies,
            self.n_spin_channels,
        )

    # ------------------------------------------------------------------
    # Basic representation & comparisons
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - repr is for debugging only
        cls = self.__class__.__name__
        return (
            f"{cls}(n_energies={self.n_energies}, n_spins={self.n_spins}, "
            f"n_atoms={self.n_atoms}, n_orbitals={self.n_orbitals}, "
            f"fermi={self.fermi:.4f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DensityOfStates):
            return False
        arrays_equal = (
            np.allclose(self.energies, other.energies)
            and np.allclose(self.total, other.total)
        )
        proj_equal = True
        if self.projected is not None or other.projected is not None:
            proj_equal = np.allclose(self.projected, other.projected)
        raw_self = self.projected
        raw_other = other.projected
        if raw_self is not None or raw_other is not None:
            if raw_self is None or raw_other is None:
                return False
            proj_equal = proj_equal and np.allclose(raw_self, raw_other)
        return arrays_equal and proj_equal and np.isclose(self.fermi, other.fermi)

    @classmethod
    def from_code(
        cls,
        code: str,
        dirpath: str,
        use_cache: bool = False,
        filename: str = "dos.pkl",
    ) -> "DensityOfStates":
        return get_dos_from_code(code=code, dirpath=dirpath, use_cache=use_cache, filename=filename)

    # ------------------------------------------------------------------
    # Core data accessors
    # ------------------------------------------------------------------
    
    @property
    def points_label(self) -> str:
        return "Energy"
    
    @property
    def points_units(self) -> str:
        return "eV"
    
    @property
    def energies(self) -> npt.NDArray[np.float64]:
        return self.points
    
    
    @property
    def energy_label(self) -> str:
        return self.points_label
    
    @property
    def energy_units(self) -> str:
        return self.points_units

    @property
    def total(self) -> npt.NDArray[np.float64]:
        return self.get_property("total")

    @property
    def projected(self) -> npt.NDArray[np.float64] | None:
        return self.get_property("projected")


    @property
    def spin_texture(self) -> Property | None:
        return self.get_property("spin_texture")

    @property
    def magnetization(self) -> Property | None:
        return self.get_property("magnetization")

    @property
    def orbital_names(self) -> list[str] | None:
        return self._orbital_names

    @property
    def fermi(self) -> float:
        return self._fermi

    @property
    def n_energies(self) -> int:
        return self.points.shape[0]

    @property
    def n_spin_channels(self) -> int:
        return self.total.to_array().shape[1]

    @property
    def n_spins(self) -> int:
        if self.projected is None:
            return self.n_spin_channels
        return self.projected.to_array().shape[1]

    @property
    def n_atoms(self) -> int:
        if self.projected is None:
            return 0
        return self.projected.to_array().shape[2]

    @property
    def n_orbitals(self) -> int:
        if self.projected is None:
            return 0
        return self.projected.to_array().shape[3]

    @property
    def spin_channels(self) -> npt.NDArray[np.int_]:
        return np.arange(self.n_spin_channels, dtype=int)

    @property
    def is_spin_polarized(self) -> bool:
        return self.n_spin_channels == 2

    @property
    def is_non_collinear(self) -> bool:
        if self.projected is None:
            return False
        if self.n_spins in (3, 4):
            return True
        if self.projected.to_array().shape[-1] == 2 + 2 + 4 + 4 + 6 + 6 + 8:
            return True
        return False

    @property
    def spin_projection_names(self) -> list[str]:
        if self.is_non_collinear:
            return ["total", "x", "y", "z"]
        if self.is_spin_polarized:
            return ["Spin-up", "Spin-down"]
        return ["Spin-up"]

    @property
    def n_electrons(self) -> float:
        return self.integrate(self.total)
    
    @property
    def spin_magnitude(self) -> npt.NDArray[np.float64]:
        return self.compute_magnetization()
    
    @property
    def projected_total(self) -> npt.NDArray[np.float64]:
        return self.compute_projected_sum(normalize=False)
    
    # ------------------------------------------------------------------
    # Property store bridge
    # ------------------------------------------------------------------
    def get_property(self, key=None, **kwargs):
        prop_name, (calc_name, gradient_order) = self._extract_key(key)

        params = self._params_for_property(prop_name, kwargs)
        requested_key = self._make_property_key(prop_name, params)
        stored_key = requested_key

        if stored_key not in self.property_store:
            computed = self.compute_property(prop_name, **kwargs)
            if computed is None:
                return None
            property_obj = self._coerce_to_property(computed, stored_key)
            self.add_property(property=property_obj)
            stored_key = property_obj.name

        normalized_key = self._normalize_super_key(key, stored_key)
        return super().get_property(normalized_key)

    def _coerce_to_property(
        self,
        computed: Property | npt.ArrayLike,
        name: str,
    ) -> Property:
        if isinstance(computed, Property):
            computed.name = name
            if getattr(computed, "_point_set", None) is None:
                computed._bind_owner(self)
            return computed

        return Property(
            name=name,
            value=np.asarray(computed, dtype=np.float64),
            point_set=self,
        )

    def compute_property(self, name: str, **kwargs):
        if name in {"projected_sum", "projected_sum_total"}:
            return self.compute_projected_sum(**kwargs)

        if name == "normalized_total":
            return self.compute_normalized_total(**kwargs)

        if name == "cumulative_total":
            return self.compute_cumulative_total()

        if name == "spin_texture":
            return self.compute_spin_texture(**kwargs)

        if name == "magnetization":
            return self.compute_magnetization(**kwargs)

        return None

    def add_property(
        self,
        property: Property | None = None,
        name: str | None = None,
        value: npt.ArrayLike | None = None,
        **kwargs
    ) -> Property:
        """Attach a custom property to the DOS object.

        Users may supply an existing :class:`Property` instance or provide a
        ``name`` and array ``value`` whose first axis matches the energy grid.
        """

        if property is not None and (name is not None or value is not None):
            raise ValueError("Provide either a Property instance or name/value, not both.")

        if property is not None:
            self.validate_property_points(property)
            super().add_property(property=property, **kwargs)
            return self.property_store[property.name]

        if name is None or value is None:
            raise ValueError("Both name and value are required when not supplying a Property instance.")

        value_array = np.asarray(value, dtype=np.float64)
        if value_array.ndim == 0:
            raise ValueError("Property values must have at least one dimension aligned with energies.")
        if value_array.shape[0] != self.n_energies:
            raise ValueError(
                "Property values must share the DOS energy grid along the first axis."
            )

        super().add_property(name=name, value=value_array, **kwargs)
        return self.property_store[name]


    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def integrate(self, values_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.trapezoid(values_array, x=self.energies, axis=0)
    
    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------
    def sum_projection_components(
        self,
        values_array: npt.NDArray[np.float64],
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Sum projections over selected atoms, orbitals, and spins."""
        
        n_dims = values_array.ndim
        if n_dims <= 2:
            raise ValueError("Values array must have at least 3 dimensions")

        atoms_ndarray = self._validate_indices(atoms, self.n_atoms)
        orbitals_ndarray = self._validate_indices(orbitals, self.n_orbitals)
        spins_ndarray = self._validate_indices(spins, self.n_spins)

        tmp_array = values_array
        # Spin Channels should not be summed over only taken!
        if spins_ndarray is not None:
            tmp_array = np.take(tmp_array, spins_ndarray, axis=1)
        if atoms_ndarray is not None:
            tmp_array = np.take(tmp_array, atoms_ndarray, axis=2)
        if orbitals_ndarray is not None:
            tmp_array = np.take(tmp_array, orbitals_ndarray, axis=3)
        
        if keepdims:
            summed_array = tmp_array.sum(axis=2,keepdims=True).sum(axis=3,keepdims=True)
        else:
            summed_array = tmp_array.sum(axis=-1, keepdims=False).sum(axis=-1, keepdims=False)

        return summed_array
    
    def normalize(self, mode: str | NormMode, values_array: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        mode = NormMode.from_input(mode)
        if mode is NormMode.RAW:
            return values_array
        elif mode is NormMode.TOTAL_PROJECTION:
            return self.normalize_total_projection(values_array=values_array, **kwargs)
        elif mode is NormMode.SPIN_MAGNITUDE:
            return self.normalize_spin_magnitude(values_array=values_array, **kwargs)
        elif mode is NormMode.MAX:
            return self.normalize_max(values_array=values_array, **kwargs)
        elif mode is NormMode.INTEGRAL:
            return self.normalize_integral(values_array=values_array, **kwargs)
        elif mode is NormMode.ELECTRONS:
            return self.normalize_electrons(values_array=values_array, **kwargs)
            
            
    def normalize_max(self, values_array: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        values_array = np.asarray(values_array, dtype=np.float64)
   
        factors = np.max(np.abs(values_array), axis=0, keepdims=True)
        factors = np.asarray(factors, dtype=np.float64)
        factors = np.where(factors == 0, 1.0, factors)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_array = np.divide(values_array, 
                                factors, 
                                out=np.zeros_like(values_array), 
                                where=factors != 0)
        return normalized_array
    
    def normalize_integral(self, values_array: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        values_array = np.asarray(values_array, dtype=np.float64)
        integrals = trapezoid(values_array, x=self.energies, axis=0)
        factors = integrals[np.newaxis, :]
  
        factors = np.asarray(factors, dtype=np.float64)
        factors = np.where(factors == 0, 1.0, factors)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_array = np.divide(values_array, 
                                factors, 
                                out=np.zeros_like(values_array), 
                                where=factors != 0)
        return normalized_array
    
    
    def normalize_electrons(self, values_array: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        return values_array / self.n_electrons
            
    def normalize_spin_magnitude(self, values_array: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        if self.spin_magnitude is None:
            raise ValueError("Spin magnitude is not available for this calculation")
        normalized_array = np.zeros_like(values_array)
        for i in range(1, values_array.shape[1]):
            normalized_array[:,i,...] = np.divide(
                    values_array[:,i,...],
                    self.spin_magnitude[:,i,...],
                    out=np.zeros_like(values_array[:,i,...]),
                    where=self.spin_magnitude[:,i,...] != 0,
                )
        return normalized_array
    
    def normalize_total_projection(self,values_array: npt.NDArray[np.float64],**kwargs) -> npt.NDArray[np.float64]:
        normalized_array = np.zeros_like(values_array)
        with np.errstate(divide="ignore", invalid="ignore"):
            for ispin in range(0, values_array.shape[1]):
                normalized_array[:,ispin,...] = np.divide(
                    values_array[:,ispin,...],
                    self.projected_total[:,ispin,...],
                    out=np.zeros_like(values_array[:,ispin,...]),
                    where=self.projected_total[:,ispin,...] != 0,
                )
   
        return normalized_array

    def compute_projected_sum(
        self,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        norm_mode: str | NormMode = "raw",
        **kwargs,
    ) -> Property:
        """Return projected DOS sums as a Property instance."""
        if self.projected is None:
            raise ValueError("Projected DOS is not available for this calculation")

        norm_mode = NormMode.from_input(norm_mode)
        kwargs = dict(kwargs)
        keepdims = kwargs.pop("keepdims", False)
        values = self.sum_projection_components(
                values_array=self.projected.to_array(),
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                keepdims=keepdims,
                **kwargs,
            )
        
        values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_label = NormMode.get_mode_type_label(norm_mode)
        
        
        
        name = "projected_sum"
        label = "Projected DOS"
        data_lim = None
        units = "states/eV"
        normalize = True
        if norm_mode is NormMode.RAW:
            name = "projected_sum"
            label = "Projected DOS"
            normalize = False
        elif norm_mode is NormMode.TOTAL_PROJECTION:
            name = "normalized_projected_sum_total"
            label = "Normalized Projected DOS"
            data_lim = (0, 1.0)
            units = None
        elif norm_mode is NormMode.SPIN_MAGNITUDE:
            name = "normalized_projected_sum_spin_magnitude"
            label = "Normalized Projected DOS"
            units = None
        elif norm_mode is NormMode.MAX:
            name = "normalized_projected_sum_max"
            label = "Normalized Projected DOS"
            units = None
        elif norm_mode is NormMode.INTEGRAL:
            name = "normalized_projected_sum_integral"
            label = "Normalized Projected DOS"
            units = "1/eV"
        elif norm_mode is NormMode.ELECTRONS:
            name = "normalized_projected_sum_electrons"
            label = "Normalized Projected DOS"
            units = "1/eV"

        extra_metadata_label = self._auto_label_projected_sum(
            atoms=atoms, orbitals=orbitals, spins=spins, normalize=normalize
        )
        metadata = {
            "atoms": list(atoms) if atoms is not None else None,
            "orbitals": list(orbitals) if orbitals is not None else None,
            "spins": list(spins) if spins is not None else None,
            "norm_mode": norm_mode,
            "keepdims": keepdims,
            "label": extra_metadata_label,
        }

        return Property(
            name=name,
            value=values,
            point_set=self,
            units=units,
            metadata=metadata,
            label=label,
            data_lim=data_lim,
        )
            
    def compute_spin_texture(
        self,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        normalize: bool = False,
        normalize_by_magnitude: bool = False,
        **kwargs,
    ) -> Property:
        if not self.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )
        if self.projected is None:
            raise ValueError("Projected DOS is not available for this calculation")

        kwargs = dict(kwargs)
        keepdims = kwargs.pop("keepdims", True)
        spin_indices = [1, 2, 3]

        if normalize or normalize_by_magnitude:
            values = self.normalize_projection_components(
                values_array=self.projected.to_array(),
                atoms=atoms,
                orbitals=orbitals,
                spins=spin_indices,
                by_spin_magnitude=normalize_by_magnitude,
                keepdims=keepdims,
                **kwargs,
            )
        else:
            values = self.sum_projection_components(
                values_array=self.projected.to_array(),
                atoms=atoms,
                orbitals=orbitals,
                spins=spin_indices,
                keepdims=keepdims,
                **kwargs,
            )

        normalization_mode = "magnitude" if normalize_by_magnitude else ("fraction" if normalize else "raw")
        metadata = {
            "atoms": list(atoms) if atoms is not None else None,
            "orbitals": list(orbitals) if orbitals is not None else None,
            "normalize": normalize,
            "normalize_by_magnitude": normalize_by_magnitude,
            "normalization_mode": normalization_mode,
            "keepdims": keepdims,
        }

        return Property(
            name="Spin Texture",
            value=values,
            point_set=self,
            units="states/eV",
            metadata=metadata,
        )

    def compute_magnetization(
        self,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        keepdims: bool = False,
        **kwargs,
    ) -> Property:
        if self.projected is None:
            raise ValueError("Projected DOS is not available for this calculation")

        kwargs = dict(kwargs)
        # keepdims already explicit in signature, ensure downstream helpers receive it once
        kwargs.pop("keepdims", None)

        if self.is_non_collinear:
            components = self.sum_projection_components(
                values_array=self.projected.to_array(),
                atoms=atoms,
                orbitals=orbitals,
                spins=[1, 2, 3],
                keepdims=keepdims,
                **kwargs,
            )
            magnetization_array = np.linalg.norm(components, axis=1, keepdims=keepdims)
            mode = "non-collinear"
        elif self.is_spin_polarized:
            components = self.sum_projection_components(
                values_array=self.projected.to_array(),
                atoms=atoms,
                orbitals=orbitals,
                spins=[0, 1],
                keepdims=keepdims,
                **kwargs,
            )
            magnetization_array = components[:, 0, ...] - components[:, 1, ...]
            mode = "collinear"
        else:
            raise ValueError("DOS is not spin polarized or non-collinear")

        metadata = {
            "atoms": list(atoms) if atoms is not None else None,
            "orbitals": list(orbitals) if orbitals is not None else None,
            "keepdims": keepdims,
            "mode": mode,
        }

        return Property(
            name="Magnetization",
            value=magnetization_array,
            point_set=self,
            units="states/eV",
            metadata=metadata,
        )

    def compute_normalized_total(
        self,
        mode: str = "max",
        **kwargs,
    ) -> Property:
        _ = kwargs  # unused extra parameters for forward compatibility
        factors = self._normalization_factors(mode)
        values = self.total / factors
        metadata = {
            "mode": mode,
        }

        return Property(
            name="Normalized total DOS",
            value=values,
            point_set=self,
            units="dimensionless",
            metadata=metadata,
        )

    def compute_cumulative_total(self) -> Property:
        total = self.total.to_array()
        if total.shape[0] < 2:
            cumulative = np.zeros_like(total)
        else:
            delta_e = np.diff(self.energies)
            avg = 0.5 * (total[1:] + total[:-1])
            cumulative = np.zeros_like(total)
            cumulative[1:] = np.cumsum(avg * delta_e[:, np.newaxis], axis=0)

        metadata = {
            "integrator": "trapezoid",
        }

        return Property(
            name="Cumulative total DOS",
            value=cumulative,
            point_set=self,
            units="states",
            metadata=metadata,
        )

    def normalize_dos(self, mode: str = "max") -> "DensityOfStates":
        factors = self._normalization_factors(mode)
        total_normalized = self.total / factors

        projected_scaled = None
        projected_raw = self.projected.to_array()
        if projected_raw is not None:
            projected_scale = factors
            if projected_scale.shape[1] != self.n_spins:
                projected_scale = np.broadcast_to(
                    projected_scale[:, :1], (1, self.n_spins)
                )
            projected_scaled = projected_raw / projected_scale.reshape(
                1, projected_scale.shape[1], 1, 1
            )

        return DensityOfStates(
            energies=self.energies,
            total=total_normalized,
            fermi=self.fermi,
            projected=projected_scaled,
            orbital_names=self.orbital_names,
        )
        
    def shift_by_fermi(self) -> "DensityOfStates":
        # new_dos = copy.deepcopy(self)
        self._points -= self.fermi
        self._points_label = "E − E_F (eV)"
        return self

    def interpolate(self, factor: int = 2) -> "DensityOfStates":
        if factor in (0, 1):
            return self

        energies, total = interpolate(self.energies, self.total, factor=factor)
        projected = None
        projected_raw = self.projected_unnormalized
        if projected_raw is not None:
            _, projected = interpolate(self.energies, projected_raw, factor=factor)

        return DensityOfStates(
            energies=energies,
            total=total,
            fermi=self.fermi,
            projected=projected,
            orbital_names=self.orbital_names,
        )

    def get_current_basis(self) -> str:
        if self.projected is None:
            return "Unknown"
        n_orbitals = self.projected.shape[-1]
        if n_orbitals == 18:
            return "jm basis"
        if n_orbitals == 9:
            return "spd basis"
        if n_orbitals == 32:
            return "spdf basis"
        return "Unknown"

    def coupled_to_uncoupled_basis(self):  # pragma: no cover - legacy feature
        raise NotImplementedError(
            "Coupled-to-uncoupled basis conversion is not implemented for the "
            "new DOS representation."
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        serializer = get_serializer(path)
        serializer.save(self, path)

    @classmethod
    def load(cls, path: Path) -> "DensityOfStates":
        serializer = get_serializer(path)
        return serializer.load(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_total(self, total: npt.ArrayLike) -> npt.NDArray[np.float64]:
        total_array = np.asarray(total, dtype=np.float64)
        if total_array.ndim == 1:
            total_array = total_array[:, np.newaxis]
        if total_array.shape[0] != self.n_energies:
            raise ValueError("Total DOS must have the same number of energies")
        return total_array

    def _validate_projected(self, projected: npt.ArrayLike) -> npt.NDArray[np.float64]:
        projected_array = np.asarray(projected, dtype=np.float64)

        if projected_array.shape[0] != self.n_energies and projected_array.shape[-1] == self.n_energies:
            projected_array = np.moveaxis(projected_array, -1, 0)

        if projected_array.ndim == 3:
            projected_array = projected_array[:, np.newaxis, :, :]

        if projected_array.ndim != 4:
            raise ValueError(
                "Projected DOS must have shape (n_energies, n_spins, n_atoms, n_orbitals)"
            )

        if projected_array.shape[0] != self.n_energies:
            raise ValueError("Projected DOS must align with the energy grid")

        return projected_array
    
    
    
    def _auto_label_projected_sum(
        self,
        *,
        atoms: Sequence[int] | None,
        orbitals: Sequence[int] | None,
        spins: Sequence[int] | None,
        normalize: bool,
    ) -> str:
        def fmt_indices(ix):
            if ix is None:
                return "all"
            ix = list(ix)
            if len(ix) == 0:
                return "[]"
            # collapse simple contiguous runs like 4–8
            runs = []
            start = prev = ix[0]
            for v in ix[1:]:
                if v == prev + 1:
                    prev = v
                else:
                    runs.append((start, prev))
                    start = prev = v
            runs.append((start, prev))
            parts = [f"{a}" if a == b else f"{a}–{b}" for a, b in runs]
            return ",".join(parts)

        # Spin label
        spin_lbl = "spins="
        if spins is None:
            spin_lbl += "all"
        else:
            try:
                names = self.spin_projection_names
                parts = []
                for s in spins:
                    parts.append(names[s] if s < len(names) else str(s))
                spin_lbl = "spins=" + ",".join(parts)
            except Exception:
                spin_lbl += fmt_indices(spins)

        atoms_lbl = f"atoms={fmt_indices(atoms)}"
        orb_lbl   = f"orbitals={fmt_indices(orbitals)}"
        mode_lbl  = "fraction" if normalize else "raw"

        return f"Σ proj ({atoms_lbl}; {orb_lbl}; {spin_lbl}; {mode_lbl})"


    def _normalization_factors(self, mode: str) -> npt.NDArray[np.float64]:
        total = np.asarray(self.total, dtype=np.float64)
        if mode == "max":
            factors = np.max(np.abs(total), axis=0, keepdims=True)
        elif mode == "integral":
            integrals = trapezoid(total, x=self.energies, axis=0)
            factors = integrals[np.newaxis, :]
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")
        factors = np.asarray(factors, dtype=np.float64)
        factors = np.where(factors == 0, 1.0, factors)
        return factors
    
    

    def _validate_indices(
        self,
        indices: Sequence[int] | None,
        upper_bound: int,
    ) -> np.ndarray | None:
        if indices is None:
            return None
        arr = np.asarray(list(indices), dtype=int).ravel()
        if arr.size == 0:
            return np.array([], dtype=int)
        arr = np.where(arr < 0, upper_bound + arr, arr)
        if np.any((arr < 0) | (arr >= upper_bound)):
            raise IndexError("Index out of bounds")
        return np.unique(arr)

    def _params_for_property(self, name: str, kwargs: dict) -> dict:
        if name in {"projected_sum", "projected_sum_total"}:
            return self._projected_sum_cache_params(**kwargs)
        if name == "normalized_total":
            mode = kwargs.get("mode", "max")
            return {} if mode == "max" else {"mode": mode}
        if name == "cumulative_total":
            return {}
        return {key: value for key, value in kwargs.items() if value is not None}

    def _projected_sum_cache_params(self, **kwargs) -> dict[str, tuple[int, ...] | bool]:
        params: dict[str, tuple[int, ...] | bool] = {}
        atoms = kwargs.get("atoms")
        orbitals = kwargs.get("orbitals")
        spins = kwargs.get("spins")
        sum_noncolinear = kwargs.get("sum_noncolinear", True)

        atoms_idx = self._validate_indices(atoms, self.n_atoms)
        if atoms_idx is not None:
            params["atoms"] = tuple(int(i) for i in atoms_idx)

        orbitals_idx = self._validate_indices(orbitals, self.n_orbitals)
        if orbitals_idx is not None:
            params["orbitals"] = tuple(int(i) for i in orbitals_idx)

        spins_idx = self._validate_indices(spins, self.n_spins)
        if spins_idx is not None:
            params["spins"] = tuple(int(i) for i in spins_idx)

        if not sum_noncolinear:
            params["sum_noncolinear"] = False
        return params

    def _make_property_key(
        self,
        base_name: str,
        params: dict | None,
    ) -> str:
        if not params:
            return base_name
        tokens = [base_name]
        for key in sorted(params):
            value = params[key]
            if value is None:
                continue
            if isinstance(value, tuple):
                string = ",".join(str(v) for v in value) if value else "[]"
            else:
                string = str(value)
            tokens.append(f"{key}={string}")
        return "|".join(tokens)

    def _normalize_super_key(self, original_key, resolved_name: str):
        if original_key is None or isinstance(original_key, str):
            return resolved_name
        if isinstance(original_key, tuple):
            if len(original_key) == 2 and isinstance(original_key[1], int):
                return (resolved_name, original_key[1])
            if len(original_key) == 2 and isinstance(original_key[1], str):
                return (resolved_name, original_key[1])
            if len(original_key) == 3:
                return (resolved_name, original_key[1], original_key[2])
        return resolved_name


def interpolate(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    factor: int = 2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interpolate ``y`` over ``x`` by increasing the sample count."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    cs = CubicSpline(x_array, y_array, axis=0)
    xs = np.linspace(float(np.min(x_array)), float(np.max(x_array)), len(x_array) * factor)
    ys = cs(xs)
    return xs, ys


@dataclass(frozen=True)
class DOSSeries:
    """A single plottable DOS track on a shared energy grid."""
    energies: np.ndarray                  # (N,)
    total:    np.ndarray                  # (N,)
    values:   np.ndarray                  # (N,) or (N, n_channels)
    label:    str                         # legend label
    units:    str = "states/eV"           # y-units
    scalars:  np.ndarray | None = None    # optional parametric coloring (N, n_channels) or (N,)
    vectors:  np.ndarray | None = None    # optional (e.g., (N,3) polarization)
    energy_label: str = "E − E_F (eV)"    # x-axis label
    dos_label:    str = "DOS"             # y-axis label
    meta: dict = field(default_factory=dict)
