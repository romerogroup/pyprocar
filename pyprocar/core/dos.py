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
from scipy import integrate
from scipy.interpolate import CubicSpline

from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer
from pyprocar.utils.inspect_utils import keep_func_kwargs


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
    RAW = "raw"
    MAX = "max"
    INTEGRAL = "integral"
    ELECTRONS = "electrons"
    TOTAL_PROJECTION = "total_projection"
    SPIN_MAGNITUDE = "spin_magnitude"
    MAGNETIZATION = "magnetization"
    
    @classmethod
    def from_input(cls, input: str | NormMode) -> NormMode:
        if isinstance(input, NormMode):
            return input
        if not isinstance(input, str):
            raise ValueError(f"Invalid normalization mode: {input}")
        
        lower_input = input.lower()
        if lower_input == "raw":
            return cls.RAW
        elif lower_input == "max":
            return cls.MAX
        elif lower_input == "integral":
            return cls.INTEGRAL
        elif lower_input == "electrons":
            return cls.ELECTRONS
        elif lower_input == "total_projection":
            return cls.TOTAL_PROJECTION
        elif lower_input == "spin_magnitude":
            return cls.SPIN_MAGNITUDE
        elif lower_input == "magnetization":
            return cls.MAGNETIZATION
        
        list_modes = cls.list_modes()
        err_msg = f"Invalid normalization mode: {input}. Valid modes are:\n"
        err_msg += "\n".join([f"- {mode}" for mode in list_modes])
        raise ValueError(err_msg)
    
    @classmethod 
    def list_modes(cls) -> list[str]:
        return [mode.value for mode in cls]
        
    @classmethod
    def get_mode_type_suffix(cls, mode: str | NormMode) -> str:
        mode = cls.from_input(mode)
        if mode == cls.RAW:
            return ""
        elif mode == cls.TOTAL_PROJECTION:
            return "total"
        elif mode == cls.MAX:
            return "max"
        elif mode == cls.INTEGRAL:
            return "integral"
        elif mode == cls.ELECTRONS:
            return "electrons"
        elif mode == cls.SPIN_MAGNITUDE:
            return "spin_magnitude"
        elif mode == cls.MAGNETIZATION:
            return "magnetization"
        else:
            raise ValueError(f"Invalid normalization mode: {mode}")
    
    @classmethod
    def get_mode_prefix(cls, mode: str | NormMode) -> str:
        mode = cls.from_input(mode)
        if mode == cls.RAW:
            return ""
        elif mode == cls.TOTAL_PROJECTION:
            return "Total-Projected-Normed"
        elif mode == cls.SPIN_MAGNITUDE:
            return "Spin-Magnitude-Normed"
        elif mode == cls.MAGNETIZATION:
            return "Magnetization-Normed"
        elif mode == cls.MAX:
            return "Max-Normed"
        elif mode == cls.INTEGRAL:
            return "Integral-Normed"
        elif mode == cls.ELECTRONS:
            return "N_Electrons-Normed"
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
                          units = "$\\frac{states}{eV}$",
                          label = "DOS")

        if projected is not None:
            projected_array = self._validate_projected(projected)
            self.add_property(name="projected", 
                              value=projected_array, 
                              units = "$\\frac{states}{eV}$",
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
    
    #-------------------------------------------------------------------
    # Class methods / Constructors
    #-------------------------------------------------------------------
    
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
    
    #-------------------------------------------------------------------
    # Array Properties
    #-------------------------------------------------------------------

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
    def spin_texture_magnitude(self) -> Property | None:
        return self.get_property("spin_texture_magnitude")
    
    @property
    def spin_magnitude(self) -> Property | None:
        return self.get_property("spin_magnitude")
    
    @property
    def magnetization(self) -> Property | None:
        return self.get_property("magnetization")
    
    @property
    def cumulative_total(self) -> Property | None:
        return self.get_property("cumulative_total")
    
    @property
    def normalized_total(self) -> Property | None:
        return self.get_property("normalized_total")
    
    @property
    def projected_total(self) -> Property | None:
        return self.get_property("projected_total")
    
    #-------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------

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
    def is_non_spin_polarized(self) -> bool:
        return self.n_spin_channels == 1
    
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

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    
    def integrate(self, 
                  values_array: npt.NDArray[np.float64], 
                  energy_lim: tuple[float, float] | None = None,
                  cumulative: bool = False) -> npt.NDArray[np.float64]:
        values_to_integrate = values_array
        energies_to_integrate = self.energies
        if energy_lim is not None:
            energy_mask = (self.energies >= energy_lim[0]) & (self.energies <= energy_lim[1])
            energy_indices = np.where(energy_mask)[0]
            if len(energy_indices) == 0:
                raise ValueError(f"No energy points found in range {energy_lim}")
            
            # Select values within energy range
            values_to_integrate = values_array[energy_indices]
            energies_to_integrate = self.energies[energy_indices]
            
        integral = np.trapezoid(values_to_integrate, x=energies_to_integrate, axis=0)
        if cumulative:
            integral = np.cumsum(integral, axis=0)
        return integral
    
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
        
    def sum_projection_components(
        self,
        values_array: npt.NDArray[np.float64],
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        keepdims: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Sum projections over selected atoms, orbitals, and spins."""
        
        tmp_array = self.select_projection_components(values_array=values_array, atoms=atoms, orbitals=orbitals, spins=spins)
        
        n_dims = tmp_array.ndim
        if keepdims and n_dims == 4:
            summed_array = tmp_array.sum(axis=2,keepdims=keepdims).sum(axis=3,keepdims=keepdims)
        elif not keepdims and n_dims == 4:
            summed_array = tmp_array.sum(axis=-1).sum(axis=-1)
        elif n_dims == 3:
            summed_array = tmp_array.sum(axis=2,keepdims=keepdims)
        elif n_dims == 2:
            summed_array = tmp_array
        else:
            raise ValueError(f"An unexpected error occured. This is likely due to a bug in the code. Please report this issue.")

        logger.debug(f"summed_array: {summed_array.shape}")
        return summed_array
    
    def select_projection_components(self,
                                     values_array: npt.NDArray[np.float64],
                                     atoms: Sequence[int] | None = None,
                                     orbitals: Sequence[int] | None = None,
                                     spins: Sequence[int] | None = None) -> npt.NDArray[np.float64]:
        n_dims = values_array.ndim

        if n_dims < 2:
            raise ValueError("Values array must have at least 2 dimensions, which represent the energy and spin channels")
        
        atoms_ndarray = self._validate_indices(atoms, self.n_atoms)
        orbitals_ndarray = self._validate_indices(orbitals, self.n_orbitals)
        spins_ndarray = self._validate_indices(spins, self.n_spins)

        tmp_array = values_array
        # Spin Channels should not be summed over only taken!
        if spins_ndarray is not None:
            tmp_array = np.take(tmp_array, spins_ndarray, axis=1)
        if atoms_ndarray is not None and n_dims >= 3:
            tmp_array = np.take(tmp_array, atoms_ndarray, axis=2)
        if orbitals_ndarray is not None and n_dims >= 4:
            tmp_array = np.take(tmp_array, orbitals_ndarray, axis=3)

        logger.debug(f"selected_array: {tmp_array.shape}")
        return tmp_array
    
    def normalize(self, mode: str | NormMode, values_array: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        mode = NormMode.from_input(mode)
        if mode is NormMode.RAW:
            return values_array
        elif mode is NormMode.TOTAL_PROJECTION:
            return self.normalize_total_projection(values_array=values_array, **kwargs)
        elif mode is NormMode.SPIN_MAGNITUDE:
            return self.normalize_spin_magnitude(values_array=values_array, **kwargs)
        elif mode is NormMode.MAGNETIZATION:
            return self.normalize_magnetization(values_array=values_array, **kwargs)
        elif mode is NormMode.MAX:
            return self.normalize_max(values_array=values_array, **kwargs)
        elif mode is NormMode.INTEGRAL:
            return self.normalize_integral(values_array=values_array, **kwargs)
        elif mode is NormMode.ELECTRONS:
            return self.normalize_electrons(values_array=values_array, **kwargs)
        else:
            raise ValueError(f"Normalization mode {mode} not found. Likely forgot to add it to the normalize method.")
            
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
        integrals = integrate.trapezoid(values_array, x=self.energies, axis=0)
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
    
    def normalize_magnetization(self, 
                                values_array: npt.NDArray[np.float64], 
                                sigma: float = 1.25, 
                                fill_value: float = 0.0,
                                eps: float = 0.001,
                                **kwargs) -> npt.NDArray[np.float64]:
        if self.magnetization is None:
            raise ValueError("Magnetization is not available for this calculation")
        
        magnetization_array = self.magnetization.to_array()
        
        logger.debug(f"values_array: {values_array.shape}")
        logger.debug(f"magnetization_array: {magnetization_array.shape}")
        

        normalized_array = np.divide(values_array, 
                            magnetization_array, 
                            out=np.zeros_like(values_array), 
                            where=magnetization_array >= eps)
            
        normalized_array = filter_data_within_sigma(normalized_array, sigma=sigma, fill_value=fill_value)
        return normalized_array
            
    def normalize_spin_magnitude(self, 
                                 values_array: npt.NDArray[np.float64], 
                                 sigma: float = 1.25, 
                                 fill_value: float = 0.0,
                                 eps: float = 0.001,
                                 **kwargs) -> npt.NDArray[np.float64]:
        if self.spin_magnitude is None:
            raise ValueError("Spin magnitude is not available for this calculation")
        
        spin_magnitude_array = self.spin_magnitude.to_array()
        
        logger.debug(f"spin magnitude array: {spin_magnitude_array.shape}")
        logger.debug(f"values array: {values_array.shape}")
  
        normalized_array = np.divide(
                            values_array,
                            spin_magnitude_array,
                            out=np.zeros_like(values_array),
                            where=np.abs(spin_magnitude_array) >= eps)
        
        normalized_array = filter_data_within_sigma(normalized_array, sigma=sigma, fill_value=fill_value)
        
        return normalized_array
    
    def normalize_total_projection(self,values_array: npt.NDArray[np.float64],**kwargs) -> npt.NDArray[np.float64]:
        normalized_array = np.zeros_like(values_array)
        projected_total_array = self.projected_total.to_array()
        
        logger.debug(f"projected_total: {projected_total_array.shape}")
        logger.debug(f"values_array: {values_array.shape}")
        
    
        for ispin in range(0, values_array.shape[1]):
            normalized_array[:,ispin,...] = np.divide(
                values_array[:,ispin,...],
                projected_total_array[:,ispin,...],
                out=np.zeros_like(values_array[:,ispin,...]),
                where=projected_total_array[:,ispin,...] != 0,
            )
   
        return normalized_array
    
    # ------------------------------------------------------------------
    # Computing methods
    # ------------------------------------------------------------------

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


        # Handle normalization and metadata
        norm_mode = NormMode.from_input(norm_mode)
        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_suffix = NormMode.get_mode_type_suffix(norm_mode)
        
        name = "projected_sum"
        label = "Projected DOS"
        data_lim = None
        units = "$\\frac{states}{eV}$"
        normalize = True
        if len(mode_prefix) > 0:
            name = f"{mode_prefix.lower()} {name}"
            label = f"{mode_prefix} {label}"
        if len(mode_type_suffix) > 0:
            name = f"{name}_{mode_type_suffix}"
            
        units = "$\\frac{states}{eV}$"
        if norm_mode is NormMode.TOTAL_PROJECTION:
            data_lim = (0, 1)
        elif norm_mode is NormMode.INTEGRAL:
            units = "$\\frac{1}{eV}$"
        elif norm_mode is NormMode.ELECTRONS:
            units = "$\\frac{1}{eV}$"

        extra_metadata_label = self._auto_label_projected_sum(
            atoms=atoms, orbitals=orbitals, spins=spins, normalize=normalize
        )
        metadata = {
            "atoms": list(atoms) if atoms is not None else None,
            "orbitals": list(orbitals) if orbitals is not None else None,
            "spins": list(spins) if spins is not None else None,
            "norm_mode": norm_mode,
            "label": extra_metadata_label,
        }
        

        values = self.sum_projection_components(
                values_array=self.projected.to_array(),
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                **keep_func_kwargs(kwargs, self.sum_projection_components)
            )
        
        
        
        values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

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
        spins: Sequence[int] | None = None,
        norm_mode: str | NormMode = "raw",
        **kwargs,
    ) -> Property:
        # Handle which dos array to use
        if self.projected is None and hasattr(self, "total"):
            dos_array = self.total.to_array()
        elif self.projected is not None:
            dos_array = self.projected.to_array()
        else:
            raise ValueError("Spin texture cannot be computed, as the total or projected DOS is not provided")
        
        if not self.is_non_collinear:
            raise ValueError(
                "Spin texture is only available for non-collinear calculations"
            )

        # Handle normalization and metadata
        ALLOWED_NORM_MODES = {NormMode.TOTAL_PROJECTION, 
                              NormMode.SPIN_MAGNITUDE, 
                              NormMode.INTEGRAL, 
                              NormMode.ELECTRONS, 
                              NormMode.MAGNETIZATION,
                              NormMode.RAW}
        norm_mode = NormMode.from_input(norm_mode)
        
        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_suffix = NormMode.get_mode_type_suffix(norm_mode)
        
        
        if spins is None:
            spins = [1,2,3]
        elif not set(spins).issubset({1, 2, 3}):
            raise ValueError(f"Invalid spins for spin texture: {spins}. Must be subset of [1, 2, 3]")

        name = ""
        label = ""
        if 1 in spins:
            if len(label) > 0:
                label += " - "
                name += " - "
            name += "s_x"
            label += r"$S_x$"
        if 2 in spins:
            if len(label) > 0:
                label += " - "
                name += " - "
            name += "s_y"
            label += r"S_y"
        if 3 in spins:
            if len(label) > 0:
                label += " - "
                name += " - "
            name += "s_z"
            label += r"S_z"
        if len(spins) == 3:
            name += "spin_texture"
            label += "Spin Texture"
            
        data_lim = None
        units = "$\\frac{states}{eV}$"
        
        if len(mode_prefix) > 0:
            name = f"{mode_prefix.lower()} {name}"
            label = f"{mode_prefix} {label}"
        if len(mode_type_suffix) > 0:
            name = f"{name}_{mode_type_suffix}"
            
        units = "$\\frac{states}{eV}$"
        if norm_mode is NormMode.RAW:
            pass
        elif norm_mode is NormMode.TOTAL_PROJECTION:
            data_lim = (-1, 1)
        elif norm_mode is NormMode.SPIN_MAGNITUDE:
            data_lim = (-1, 1)
        elif norm_mode is NormMode.MAGNETIZATION:
            data_lim = None
        elif norm_mode is NormMode.INTEGRAL:
            units = "$\\frac{1}{eV}$"
        elif norm_mode is NormMode.ELECTRONS:
            units = "$\\frac{1}{eV}$"
        else:
            err_msg = f"Invalid normalization mode: {norm_mode}. Valid modes are: "
            err_msg += "\n".join([f"- {mode.value}" for mode in ALLOWED_NORM_MODES])
            raise ValueError(err_msg)

        metadata = {
            "atoms": list(atoms) if atoms is not None else None,
            "orbitals": list(orbitals) if orbitals is not None else None,
            "norm_mode": norm_mode.value,
        }
        
        
        # Compute the spin texture
        values = self.sum_projection_components(
            values_array=dos_array,
            atoms=atoms,
            orbitals=orbitals,
            spins=[1,2,3],
            **keep_func_kwargs(kwargs, self.sum_projection_components)
        )
        
        values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

        return Property(
            name=name,
            value=values,
            point_set=self,
            units=units,
            metadata=metadata,
            label=label,
            data_lim=data_lim,
        )

    def compute_magnetization(
        self,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        norm_mode: str | NormMode = "raw",
        from_total: bool = False,
        keepdims: bool = False,
        **kwargs,
    ) -> Property:
        
        # Handle which dos array to use
        if (self.projected is None and hasattr(self, "total")) or from_total:
            dos_array = self.total.to_array()
        elif self.projected is not None:
            dos_array = self.projected.to_array()
        else:
            raise ValueError("Magnetization cannot be computed, as the total or projected DOS is not provided")
        
        # Handle spin cases dependent information
        if self.is_non_collinear:
            data_lim = None
            mode = "non-collinear"
            spins = [0]
        elif self.is_spin_polarized:
            data_lim = (-1, 1)
            mode = "collinear"
            spins = [0, 1]
        else:
            raise ValueError("DOS is not non-collinear or spin polarized")
        
        
        # Handle normalization and metadata
        ALLOWED_NORM_MODES = {NormMode.RAW, NormMode.MAGNETIZATION, NormMode.INTEGRAL, NormMode.ELECTRONS}
        
        norm_mode = NormMode.from_input(norm_mode)
        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_suffix = NormMode.get_mode_type_suffix(norm_mode)
        
        name = "magnetization"
        label = "Magnetization"
        if len(mode_prefix) > 0:
            name = f"{mode_prefix.lower()} {name}"
            label = f"{mode_prefix} {label}"
        if len(mode_type_suffix) > 0:
            name = f"{name}_{mode_type_suffix}"
        
        units = "$\\frac{states}{eV}$"
        if norm_mode is NormMode.RAW:
            pass
        elif norm_mode is NormMode.MAGNETIZATION:
            units = None
        elif norm_mode is NormMode.INTEGRAL:
            units = "$\\frac{1}{eV}$"
            data_lim = None
        elif norm_mode is NormMode.ELECTRONS:
            units = "$\\frac{1}{eV}$"
            data_lim = None
        else:
            err_msg = f"Invalid normalization mode: {norm_mode}. Valid modes are: \n"
            err_msg += "\n".join([f"- {mode.value}" for mode in ALLOWED_NORM_MODES])
            raise ValueError(err_msg)
        metadata = {
            "atoms": list(atoms) if atoms is not None else None,
            "orbitals": list(orbitals) if orbitals is not None else None,
            "keepdims": keepdims,
            "norm_mode": norm_mode,
            "mode": mode,
        }
        # Compute the magnetization
        components = self.sum_projection_components(
                values_array=dos_array,
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                **keep_func_kwargs(kwargs, self.sum_projection_components)
            )
        magnetization_array = components
        if self.is_spin_polarized:
            magnetization_array = components[:, 0, ...] - components[:, 1, ...]
        
        if magnetization_array.ndim == 1:
            magnetization_array = magnetization_array[..., np.newaxis]

        values = self.normalize(mode=norm_mode, values_array=magnetization_array, **kwargs)

        return Property(
            name=name,
            value=values,
            point_set=self,
            units=units,
            label=label,
            data_lim=data_lim,
            metadata=metadata,
        )
        
    def compute_spin_texture_magnitude(
        self,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        norm_mode: str | NormMode = "raw",
        from_total: bool = False,
        keepdims: bool = False,
        **kwargs,
    ) -> Property:
        # Handle which dos array to use
        if (self.projected is None and hasattr(self, "total")) or from_total:
            dos_array = self.total.to_array()
        elif self.projected is not None:
            dos_array = self.projected.to_array()
        else:
            raise ValueError("Spin texture magnitude cannot be computed, as the total or projected DOS is not provided")
        
        if not self.is_non_collinear:
            raise ValueError("Spin texture magnitude is only available for non-collinear calculations")
  
        # Handle normalization and metadata
        norm_mode = NormMode.from_input(norm_mode)
        ALLOWED_NORM_MODES = {NormMode.INTEGRAL, NormMode.SPIN_MAGNITUDE, NormMode.ELECTRONS, NormMode.MAGNETIZATION, NormMode.RAW}
        
        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_suffix = NormMode.get_mode_type_suffix(norm_mode)
        
        name = "spin_texture_magnitude"
        label = "Spin Texture Magnitude"
        data_lim = None
        units = "$\\frac{states}{eV}$"
        if len(mode_prefix) > 0:
            name = f"{mode_prefix.lower()} {name}"
            label = f"{mode_prefix} {label}"
        if len(mode_type_suffix) > 0:
            name = f"{name}_{mode_type_suffix}"
            
        units = "$\\frac{states}{eV}$"
        if norm_mode is NormMode.RAW:
            pass
        elif norm_mode is NormMode.INTEGRAL:
            units = "$\\frac{1}{eV}$"
        elif norm_mode is NormMode.ELECTRONS:
            units = "$\\frac{1}{eV}$"
        elif norm_mode is NormMode.SPIN_MAGNITUDE:
            data_lim = (0, 1)
            units = None
        elif norm_mode is NormMode.MAGNETIZATION:
            units = None
            data_lim = None
        else:
            err_msg = f"Invalid normalization mode: {norm_mode}. Valid modes are: "
            err_msg += "\n".join([f"- {mode.value}" for mode in ALLOWED_NORM_MODES])
            raise ValueError(err_msg)

        metadata = {
            "atoms": list(atoms) if atoms is not None else None,
            "orbitals": list(orbitals) if orbitals is not None else None,
            "keepdims": keepdims,
            "norm_mode": norm_mode,
        }
        
        values = self.sum_projection_components(
            values_array=dos_array,
            atoms=atoms,
            orbitals=orbitals,
            spins=[1,2,3],
            **keep_func_kwargs(kwargs, self.sum_projection_components)
        )
        values = np.linalg.norm(values, axis=-1, keepdims=keepdims)
 
        logger.debug(f"spin texture magnitude: {values.shape}")
        logger.debug(f"spin texture magnitude (min, max): {np.min(values), np.max(values)}")
        
        if values.ndim == 1:
            values = values[..., np.newaxis]
        
        values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

        return Property(
            name=name,
            value=values,
            point_set=self,
            units=units,
            label=label,
            data_lim=data_lim,
            metadata=metadata,
        )

    def compute_normalized_total(
        self,
        norm_mode: str | NormMode = "max",
        **kwargs,
    ) -> Property:
        _ = kwargs  # unused extra parameters for forward compatibility
        metadata = {
            "norm_mode": norm_mode,
        }
        
        total_property = self.total
        values = total_property.values
        
        normed_values = self.normalize(mode=norm_mode, values_array=total_property.values, **kwargs)
        
        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_suffix = NormMode.get_mode_type_suffix(norm_mode)
        
        name = total_property.name
        label = total_property.label
        units = total_property.units
        data_lim = total_property.data_lim
        if len(mode_prefix) > 0:
            name = f"{mode_prefix.lower()} {name}"
            label = f"{mode_prefix} {label}"
        if len(mode_type_suffix) > 0:
            name = f"{name}_{mode_type_suffix}"
            
        units = "$\\frac{states}{eV}$"
        if norm_mode is NormMode.INTEGRAL:
            units = "$\\frac{1}{eV}$"
        elif norm_mode is NormMode.ELECTRONS:
            units = "$\\frac{1}{eV}$"
        
        return Property(
            name=name,
            value=normed_values,
            point_set=self,
            units=units,
            label=label,
            data_lim=data_lim,
            metadata=metadata,
        )

    def compute_cumulative_total(self, norm_mode: str | NormMode = "max") -> Property:
        total = self.total.to_array()
        
        cumlative_total = self.integrate(values_array=total, cumulative=True)

        norm_mode = NormMode.from_input(norm_mode)
        values = self.normalize(mode = norm_mode, values_array=cumlative_total)

        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_suffix = NormMode.get_mode_type_suffix(norm_mode)
        
        name = "cumlative_total_dos"
        label = "Cumulative Total DOS"
        units = None
        data_lim = None
        metadata = {
            "norm_mode": norm_mode,
        }
        if len(mode_prefix) > 0:
            name = f"{mode_prefix.lower()} {name}"
            label = f"{mode_prefix} {label}"
        if len(mode_type_suffix) > 0:
            name = f"{name}_{mode_type_suffix}"

        return Property(
            name=name,
            value=values,
            point_set=self,
            units=units,
            label=label,
            data_lim=data_lim,
            metadata=metadata,
        )
        
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
        if name in {"projected_sum", "projected_sum_total", "projected_total"}:
            return self.compute_projected_sum(**kwargs)

        if name == "normalized_total":
            return self.compute_normalized_total(**kwargs)

        if name == "cumulative_total":
            return self.compute_cumulative_total()

        if name == "spin_texture":
            return self.compute_spin_texture(**kwargs)

        if name == "magnetization":
            return self.compute_magnetization(**kwargs)

        if name in {"spin_texture_magnitude", "spin_magnitude"}:
            return self.compute_spin_texture_magnitude(**kwargs)

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
    # Basis helpers
    # ------------------------------------------------------------------
    
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


def filter_data_within_sigma(data: npt.NDArray[np.float64], sigma: float = 3, fill_value: float | None = None) -> npt.NDArray[np.float64]:
    
    indices = np.where(np.abs(data) > np.abs(data.mean()) + sigma * data.std())
    if fill_value is not None:
        data[indices] = fill_value
    else:
        plus_3_sigma = data.mean() + 3 * data.std()
        minus_3_sigma = data.mean() - 3 * data.std()
        above_3_sigma = np.where(data > plus_3_sigma)
        below_3_sigma = np.where(data < minus_3_sigma)
        data[above_3_sigma] = plus_3_sigma
        data[below_3_sigma] = minus_3_sigma
    return data