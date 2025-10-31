"""Core density of states data object."""

from __future__ import annotations

import copy
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from itertools import chain, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
from scipy import integrate
from scipy.interpolate import CubicSpline

from pyprocar.core.atomic_orbital_index import (
    AtomIndexer,
    OrbitalIndexer,
    ProjectionLabelBuilder,
    ProjectionSelectionResolver,
    ProjectionSelectionResult,
    SpinIndexer,
)
from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer
from pyprocar.utils.func_utils import expand_grouped_params_to_dicts, keep_func_kwargs
from pyprocar.utils.math import np_round_to_half

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pyprocar.core.structure import Structure




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

_FRAC_RE = re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}")
_TOKEN_RE = re.compile(r"([A-Za-z\\_^-]+?)(?:\^\{?(-?\d+)\}?)?(?=$|\s|\\cdot)")

    
def _strip_dollars(s: str) -> str:
    return s.strip().strip("$")

def _parse_product(s: str) -> Counter:
    """
    Parse a product like 'states eV^2' or 'states' into a Counter({'states':1,'eV':2})
    Accepts optional \cdot separators.
    """
    s = s.strip()
    units = Counter()
    if not s or s == "1":
        return units
    # split by whitespace or \cdot without losing tokens
    s = s.replace(r"\cdot", " ")
    for m in _TOKEN_RE.finditer(s):
        sym, exp = m.groups()
        if sym in {"", "1"}:
            continue
        k = sym.strip()
        n = int(exp) if exp is not None else 1
        units[k] += n
    # remove zeros
    for k in list(units.keys()):
        if units[k] == 0:
            del units[k]
    return units

def _parse_units(u: str) -> Counter:
    """
    Supports forms like:
      '$\\frac{states}{eV^2}$', 'states', '$\\frac{1}{eV}$'
    Returns Counter with positive exponents for numerator,
    negative for denominator.
    """
    u = _strip_dollars(u)
    if not u:
        return Counter()
    m = _FRAC_RE.search(u)
    if m:
        num, den = m.group(1), m.group(2)
        units = _parse_product(num) - _parse_product(den)  # subtract den exponents
    else:
        units = _parse_product(u)
    # drop zero exponents
    for k in list(units.keys()):
        if units[k] == 0:
            del units[k]
    return units

def _format_units(units: Counter) -> str:
    """Return a compact LaTeX string like '$\\frac{states}{eV}$' or '$1$'."""
    num = {k: v for k, v in units.items() if v > 0}
    den = {k: -v for k, v in units.items() if v < 0}

    def fmt_side(d: dict) -> str:
        if not d:
            return "1"
        # put 'states' first if present, then alphabetical for stability
        keys = sorted(d.keys(), key=lambda k: (k != "states", k))
        parts = []
        for k in keys:
            p = d[k]
            if p == 1:
                parts.append(k)
            else:
                parts.append(f"{k}^{{{p}}}")
        return r"\cdot ".join(parts)

    if den:
        num_s = fmt_side(num)
        den_s = fmt_side(den)
        # if numerator is 1 and denominator not empty → \frac{1}{...}
        return f"$\\frac{{{num_s}}}{{{den_s}}}$"
    else:
        # only numerator (or both empty)
        s = fmt_side(num)
        return "$1$" if s == "1" else f"${s}$"

def _units_divide(u_input: str, u_norm: str | None) -> str:
    """Compute simplified units = input / normalizer."""
    if not u_norm:
        # e.g. MAX: divide by a value with same units → unitless
        return None
    ui = _parse_units(u_input or "")
    un = _parse_units(u_norm or "")
    simplified = ui - un
    # drop zeros
    for k in list(simplified.keys()):
        if simplified[k] == 0:
            del simplified[k]
    # empty → dimensionless
    if not simplified:
        return None
    return _format_units(simplified)



class NormMode(Enum):
    RAW = "raw"
    MAX = "max"
    INTEGRAL = "integral"
    ELECTRONS = "electrons"
    TOTAL = "total"
    TOTAL_PROJECTION = "total_projection"
    SPIN_MAGNITUDE = "spin_magnitude"
    MAGNETIZATION = "magnetization"
    
    @classmethod
    def from_input(cls, input: str | NormMode | None) -> NormMode:
        if isinstance(input, NormMode):
            return input
        if input is None:
            return cls.RAW
        if not isinstance(input, str):
            raise ValueError(f"Invalid normalization mode: {input}")
        input_mode = None
        lower_input = input.lower()
        if lower_input == "raw":
            input_mode = cls.RAW
        elif lower_input == "max":
            input_mode = cls.MAX
        elif lower_input == "integral":
            input_mode = cls.INTEGRAL
        elif lower_input == "electrons":
            input_mode = cls.ELECTRONS
        elif lower_input == "total":
            input_mode = cls.TOTAL
        elif lower_input == "total_projection":
            input_mode = cls.TOTAL_PROJECTION
        elif lower_input == "spin_magnitude":
            input_mode = cls.SPIN_MAGNITUDE
        elif lower_input == "magnetization":
            input_mode = cls.MAGNETIZATION
        
        if input_mode is not None:
            logger.info(f"Normalization mode: {input_mode}")
            return input_mode
        
        list_modes = cls.list_modes()
        err_msg = f"Invalid normalization mode: {input}. Valid modes are:\n"
        err_msg += "\n".join([f"- {mode}" for mode in list_modes])
        raise ValueError(err_msg)
    
    @classmethod 
    def list_modes(cls) -> list[str]:
        return [mode.value for mode in cls]
        
    @classmethod
    def normalizer_units(cls, mode: NormMode, input_units: str) -> str | None:
        """
        Units of the quantity you divide by for this normalization.
        Return None for 'same-units' normalizers (e.g., MAX) to produce $1$.
        """
        mode = cls.from_input(mode)
        # Common DOS-like units
        dos_units = "$\\frac{states}{eV}$"
        if mode is cls.RAW:
            return ""  # nothing divides → keep input
        if mode is cls.MAX:
            return input_units  # divide by a max of the same quantity → unitless
        if mode is cls.INTEGRAL:
            # ∫ DOS dE → 'states' (area under curve)
            return "$states$"
        if mode is cls.ELECTRONS:
            # divide by a count of electrons (dimensionally same as 'states')
            return "$states$"
        if mode in {cls.TOTAL_PROJECTION, cls.SPIN_MAGNITUDE, cls.MAGNETIZATION, cls.TOTAL}:
            # divide by another DOS-like curve
            return dos_units
        # fallback
        return ""
    
    @classmethod
    def get_normed_units(cls, mode: NormMode, input_units: str) -> str:
        """
        Clean, simplified output units = input_units / normalizer_units(mode).
        Examples:
            input '$\\frac{states}{eV^2}$', mode=TOTAL_PROJECTION ('$states/eV$')
            → '$\\frac{1}{eV}$'
            input '$\\frac{states}{eV}$', mode=INTEGRAL ('$states$')
            → '$\\frac{1}{eV}$'
            input '$\\frac{states}{eV}$', mode=MAX (same units)
            → '$1$'
        """
        mode = cls.from_input(mode)
        if mode is cls.RAW:
            return input_units
        norm_units = cls.normalizer_units(mode, input_units)
        norm_units = _units_divide(input_units, norm_units)
        return _units_divide(input_units, norm_units)
    
    @classmethod
    def get_normed_name(cls, mode: NormMode, name: str) -> str:
        mode = cls.from_input(mode)
        prefix = cls.get_mode_prefix(mode)
        if len(prefix) > 0:
            return f"{prefix} {name}"
        else:
            return name
        
    @classmethod
    def get_normed_data_lim(cls, mode: NormMode, input_lim: tuple[float, float] | None) -> tuple[float, float] | None:
        mode = cls.from_input(mode)
        if mode == cls.RAW:
            return input_lim
        elif mode == cls.TOTAL_PROJECTION:
            return (0, 1)
        else:
            return None
        
    @classmethod
    def get_mode_type_suffix(cls, mode: str | NormMode) -> str:
        mode = cls.from_input(mode)
        if mode == cls.RAW:
            return ""
        elif mode == cls.TOTAL_PROJECTION:
            return "total_projection"
        elif mode == cls.TOTAL:
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
    def get_mode_units(cls, mode: str | NormMode, input_units: str) -> str:
        mode = cls.from_input(mode)
        if mode == cls.RAW:
            return input_units
        elif mode == cls.TOTAL:
            return "$\\frac{states}{eV}$"
        elif mode == cls.TOTAL_PROJECTION:
            return "$\\frac{states}{eV}$"
        elif mode == cls.SPIN_MAGNITUDE:
            return "$\\frac{states}{eV}$"
        elif mode == cls.MAGNETIZATION:
            return "$\\frac{states}{eV}$"
        elif mode == cls.MAX:
            return "$\\frac{states}{eV}$"
        elif mode == cls.INTEGRAL:
            return "states"
        elif mode == cls.ELECTRONS:
            return "states"
        else:
            return ""
    
    @classmethod
    def get_mode_prefix(cls, mode: str | NormMode) -> str:
        mode = cls.from_input(mode)
        if mode == cls.RAW:
            return ""
        elif mode == cls.TOTAL_PROJECTION:
            return "Total-Projected-Normed"
        elif mode == cls.TOTAL:
            return "Total-Normed"
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
        
    @classmethod
    def get_mode_footnote(cls, mode: str | NormMode) -> str:
        mode = cls.from_input(mode)
        if mode == cls.RAW:
            return ""
        elif mode == cls.TOTAL:
            return "Normalization is by the Total DoS"
        elif mode == cls.TOTAL_PROJECTION:
            return "Normalization is by the Total Projection DoS"
        elif mode == cls.SPIN_MAGNITUDE:
            return "Normalization is by the Spin Magnitude DoS"
        elif mode == cls.MAGNETIZATION:
            return "Normalization is by the Magnetization DoS"
        elif mode == cls.MAX:
            return "Normalization is by the Max DoS"
        elif mode == cls.INTEGRAL:
            return "Normalization is by the Integral DoS"
        elif mode == cls.ELECTRONS:
            return "Normalization is by the N_Electrons DoS"
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
        structure: Structure | None = None,
    ) -> None:
        energies_array = energies
        gradient = gradient_func or _finite_difference_gradient

        super().__init__(points=energies_array, gradient_func=gradient)

        self._fermi = float(fermi)
        self._orbital_names = orbital_names
        self._structure = structure
        self._projection_label_builder: ProjectionLabelBuilder | None = None
        self._projection_selection_resolver: ProjectionSelectionResolver | None = None

        total_array = self._validate_total(total)
        
        total_metadata ={}
        if total_array.shape[1] == 1:
            total_metadata["label"] = ["Total"]
        elif total_array.shape[1] == 2:
            total_metadata["label"] = ["$Total - \\uparrow$", "$Total - \\downarrow$"]
        elif total_array.shape[1] == 4:
            total_metadata["label"] = ["$Total$", "$Total - S_x$", "$Total - S_y$", "$Total - S_z$"]
        else:
            raise ValueError(f"Total array has {self.total_array.shape[1]} spin channels, which is not supported")
        
        self.add_property(name="total", 
                          value=total_array,
                          units = "$\\frac{states}{eV}$",
                          label = "DoS",
                          metadata = total_metadata)

        if projected is not None:
            projected_array = self._validate_projected(projected)
            self.add_property(name="projected", 
                              value=projected_array, 
                              units = "$\\frac{states}{eV}$",
                              label = "Projected DoS",
                              metadata = {
                              "label": ["Projected DoS"]
                          })
            
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
    
    @property
    def structure(self) -> Structure | None:
        return self._structure
    
    @property
    def atoms(self) -> npt.NDArray[np.int_]:
        return self.structure.atoms
    
    @property
    def species(self) -> list[str]:
        return self.structure.species
    
    @property
    def orbitals(self) -> list[str]:
        return self.orbital_names
    
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

    #-------------------------------------------------------------------
    # Useful getters
    #-------------------------------------------------------------------
    
    def get_species_atom_map(self, species: list[str] | str | None = None) -> dict[str, list[int]]:
        atoms_array = np.asarray(self.atoms)
        if species is None:
            species = self.species
        if isinstance(species, str):
            species = [species]
            
        species_atoms_list = {}
        for specie in species:
            species_atoms_list[specie] = tuple(np.where(atoms_array == specie)[0].tolist())
        return species_atoms_list

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
        
        return np.trapezoid(values_to_integrate, x=energies_to_integrate, axis=0)
        
    def cumsum(self, values: npt.NDArray[np.float64] | Property) -> npt.NDArray[np.float64]:
        if isinstance(values, Property):
            values = values.to_array()
        return np.cumsum(values, axis=0)
    
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
    
    def normalize(self, mode: str | NormMode | None, values_array: npt.NDArray[np.float64], **kwargs) -> npt.NDArray[np.float64]:
        mode = NormMode.from_input(mode)
        if mode is NormMode.RAW:
            return values_array
        elif mode is NormMode.TOTAL:
            return self.normalize_total(values_array=values_array, **kwargs)
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
            
    def normalize_total(self,values_array: npt.NDArray[np.float64],**kwargs) -> npt.NDArray[np.float64]:
        normalized_array = np.zeros_like(values_array)
        total_array = self.total.to_array()
        
        logger.debug(f"total: {total_array.shape}")
        logger.debug(f"values_array: {values_array.shape}")
        
        for ispin in range(0, values_array.shape[1]):
            normalized_array[:,ispin,...] = np.divide(
                values_array[:,ispin,...],
                total_array[:,ispin,...],
                out=np.zeros_like(values_array[:,ispin,...]),
                where=total_array[:,ispin,...] != 0,
            )
   
        return normalized_array
    
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
        atoms:  Iterable[int] | None = None,
        orbitals:  Iterable[int] | None = None,
        spins: Iterable[int] | None = None,
        species: Iterable[str] | None = None,
        species_orbital_map: dict[str, Iterable[int]] | None = None,
        atoms_orbital_map: dict[int, Iterable[int]] | None = None,
        norm_mode: str | NormMode | None = "raw",
        include_normal_label: bool = False,
        label: str = "Projected DoS",
        name: str = "projected_sum",
        units = "$\\frac{states}{eV}$",
        **kwargs) -> Property | list[Property]:
        """Return projected DOS sums as a Property instance."""
        if self.projected is None:
            raise ValueError("Projected DOS is not available for this calculation")

        # Build dict of grouped params
        grouped_params_dict = {
            "atoms": atoms,
            "orbitals": orbitals,
            "spins": spins,
            "species": species,
            "species_orbital_map": species_orbital_map,
            "atoms_orbital_map": atoms_orbital_map,
        }
        param_dicts = expand_grouped_params_to_dicts(grouped_params_dict)
        
        results = []
        for params in param_dicts:
            # Extract params from dict
            atoms = params["atoms"]
            orbitals = params["orbitals"]
            spins = params["spins"]
            species = params["species"]
            species_orbital_map = params["species_orbital_map"]
            atoms_orbital_map = params["atoms_orbital_map"]

            selection = self._resolve_projection_selection(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
            atoms = selection.atoms
            orbitals = selection.orbitals
            spins = selection.spins
            species = selection.species

            sum_kwargs = keep_func_kwargs(kwargs, self.sum_projection_components)
            values = self.sum_projection_components(
                values_array=self.projected.to_array(),
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                **sum_kwargs,
            )
            
            # Handle normalization and metadata
            norm_mode = NormMode.from_input(norm_mode)
            values = self.normalize(mode=norm_mode, values_array=values, **kwargs)
            
            data_min = np.min(values,axis=0)
            data_max = np.max(values,axis=0)
            data_lim = (data_min, data_max)
            rounded_data_lim = (np_round_to_half(data_min), np_round_to_half(data_max))
            
            normed_name = NormMode.get_normed_name(norm_mode, name)
            normed_units = NormMode.get_normed_units(norm_mode, units)
            footnote = NormMode.get_mode_footnote(norm_mode)
            
            
            label_plain_list, label_latex_list = self._format_selection_label(
                selection=selection,
                normalize=norm_mode is not NormMode.RAW,
                include_normal_label=include_normal_label,
            )
            metadata = {
                "atoms": list(atoms) if len(atoms) > 0 else None,
                "orbitals": list(orbitals) if orbitals is not None else None,
                "spins": list(spins) if spins is not None else None,
                "species": list(species) if len(species) > 0 else None,
                "norm_mode": norm_mode,
                "units": normed_units,
                "data_lim": data_lim,
                "rounded_data_lim": rounded_data_lim,
                "label": label_latex_list,
                "label_plain": label_plain_list,
                "footnote": footnote,
                "scalar_label": label,
                "atom_label": selection.labels.atom,
                "atom_label_latex": selection.labels.atom_latex,
                "orbital_label": selection.labels.orbital,
                "orbital_label_latex": selection.labels.orbital_latex,
                "spin_label": selection.labels.spin,
                "spin_label_latex": selection.labels.spin_latex,
                "species_label": selection.labels.species,
                "species_label_latex": selection.labels.species_latex,
                "label_prefix": selection.labels.prefix_plain,
                "label_prefix_latex": selection.labels.prefix_latex,
                "spin_component_labels": list(selection.labels.spin_components),
                "spin_component_labels_latex": list(selection.labels.spin_components_latex),
                "label_combined": selection.labels.combined,
                "label_combined_latex": selection.labels.combined_latex,
                "include_normal_label": include_normal_label,
            }
            

            
            results.append(Property(
                name=normed_name,
                value=values,
                point_set=self,
                metadata=metadata,
                label=label,
                units=normed_units,
                # data_lim=metadata.get("rounded_data_lim"),
            ))
        
        return results[0] if len(results) == 1 else results
            
    def compute_spin_texture(
        self,
        atoms: Iterable[int] | None = None,
        orbitals: Iterable[int] | None = None,
        spins: Iterable[int] | None = None,
        species: Iterable[str] | None = None,
        species_orbital_map: dict[str, Iterable[int]] | None = None,
        atoms_orbital_map: dict[int, Iterable[int]] | None = None,
        norm_mode: str | NormMode = "raw",
        **kwargs,
    ) -> Property | list[Property]:
        include_normal_label = bool(kwargs.pop("include_normal_label", False))
        scalar_label = kwargs.pop("scalar_label", "Spin Texture")

        if self.projected is None and hasattr(self, "total"):
            dos_array = self.total.to_array()
        elif self.projected is not None:
            dos_array = self.projected.to_array()
        else:
            raise ValueError("Spin texture cannot be computed, as the total or projected DOS is not provided")

        if not self.is_non_collinear:
            raise ValueError("Spin texture is only available for non-collinear calculations")

        # Build dict of grouped params
        grouped_params_dict = {
            "atoms": atoms,
            "orbitals": orbitals,
            "spins": spins,
            "species": species,
            "species_orbital_map": species_orbital_map,
            "atoms_orbital_map": atoms_orbital_map,
        }
        param_dicts = expand_grouped_params_to_dicts(grouped_params_dict)
        
        results = []
        for params in param_dicts:
            # Extract params from dict
            atoms = params["atoms"]
            orbitals = params["orbitals"]
            spins = params["spins"]
            species = params["species"]
            species_orbital_map = params["species_orbital_map"]
            atoms_orbital_map = params["atoms_orbital_map"]

            if spins is None:
                spins = (1, 2, 3)

            selection = self._resolve_projection_selection(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
            atoms = selection.atoms
            orbitals = selection.orbitals
            spins = selection.spins
            species = selection.species

            if spins is None or len(spins) == 0:
                raise ValueError("Spin texture requires at least one spin component to be selected")
            invalid_spins = set(spins) - {1, 2, 3}
            if invalid_spins:
                raise ValueError(
                    f"Invalid spins for spin texture: {sorted(invalid_spins)}. Valid components are [1, 2, 3]."
                )

            allowed_modes = {
                NormMode.TOTAL_PROJECTION,
                NormMode.SPIN_MAGNITUDE,
                NormMode.INTEGRAL,
                NormMode.ELECTRONS,
                NormMode.MAGNETIZATION,
                NormMode.RAW,
            }
            norm_mode = NormMode.from_input(norm_mode)
            if norm_mode not in allowed_modes:
                valid_modes = "\n".join(f"- {mode.value}" for mode in sorted(allowed_modes, key=lambda m: m.value))
                raise ValueError(f"Invalid normalization mode: {norm_mode}. Valid modes are:\n{valid_modes}")

            sum_kwargs = keep_func_kwargs(kwargs, self.sum_projection_components)
            values = self.sum_projection_components(
                values_array=dos_array,
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                **sum_kwargs,
            )
            values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

            data_min = np.min(values, axis=0)
            data_max = np.max(values, axis=0)
            data_lim = (data_min, data_max)
            rounded_data_lim = (np_round_to_half(data_min), np_round_to_half(data_max))

            property_name = NormMode.get_normed_name(norm_mode, "spin_texture")
            units = NormMode.get_normed_units(norm_mode, "$\\frac{states}{eV}$")
            footnote = NormMode.get_mode_footnote(norm_mode)

            label_plain_list, label_latex_list = self._format_selection_label(
                selection=selection,
                normalize=norm_mode is not NormMode.RAW,
                include_normal_label=include_normal_label,
            )

            recommended_data_lim: tuple[float, float] | None = None
            if norm_mode in {NormMode.TOTAL_PROJECTION, NormMode.SPIN_MAGNITUDE}:
                recommended_data_lim = (-1.0, 1.0)

            metadata = {
                "atoms": list(atoms) if len(atoms) > 0 else None,
                "orbitals": list(orbitals) if orbitals is not None else None,
                "spins": list(spins) if spins is not None else None,
                "species": list(species) if len(species) > 0 else None,
                "norm_mode": norm_mode,
                "units": units,
                "data_lim": data_lim,
                "rounded_data_lim": rounded_data_lim,
                "label": label_latex_list,
                "label_plain": label_plain_list,
                "footnote": footnote,
                "scalar_label": scalar_label,
                "atom_label": selection.labels.atom,
                "atom_label_latex": selection.labels.atom_latex,
                "orbital_label": selection.labels.orbital,
                "orbital_label_latex": selection.labels.orbital_latex,
                "spin_label": selection.labels.spin,
                "spin_label_latex": selection.labels.spin_latex,
                "species_label": selection.labels.species,
                "species_label_latex": selection.labels.species_latex,
                "label_prefix": selection.labels.prefix_plain,
                "label_prefix_latex": selection.labels.prefix_latex,
                "spin_component_labels": list(selection.labels.spin_components),
                "spin_component_labels_latex": list(selection.labels.spin_components_latex),
                "label_combined": selection.labels.combined,
                "label_combined_latex": selection.labels.combined_latex,
                "include_normal_label": include_normal_label,
                "recommended_data_lim": recommended_data_lim,
            }

            results.append(Property(
                name=property_name,
                value=values,
                point_set=self,
                metadata=metadata,
                label=scalar_label,
                units=units,
                data_lim=recommended_data_lim,
            ))
        
        return results[0] if len(results) == 1 else results

    def compute_magnetization(
        self,
        atoms: Iterable[int] | None = None,
        orbitals: Iterable[int] | None = None,
        spins: Iterable[int] | None = None,
        species: Iterable[str] | None = None,
        species_orbital_map: dict[str, Iterable[int]] | None = None,
        atoms_orbital_map: dict[int, Iterable[int]] | None = None,
        norm_mode: str | NormMode = "raw",
        from_total: bool = False,
        keepdims: bool = False,
        **kwargs,
    ) -> Property | list[Property]:
        include_normal_label = bool(kwargs.pop("include_normal_label", False))
        scalar_label = kwargs.pop("scalar_label", "Magnetization")

        if (self.projected is None and hasattr(self, "total")) or from_total:
            dos_array = self.total.to_array()
        elif self.projected is not None and not from_total:
            dos_array = self.projected.to_array()
        else:
            raise ValueError("Magnetization cannot be computed, as the total or projected DOS is not provided")

        if self.is_non_collinear:
            default_spins: tuple[int, ...] = (0,)
            mode = "non-collinear"
        elif self.is_spin_polarized:
            default_spins = (0, 1)
            mode = "collinear"
        else:
            raise ValueError("DOS is not non-collinear or spin polarized")

        # Build dict of grouped params
        grouped_params_dict = {
            "atoms": atoms,
            "orbitals": orbitals,
            "spins": spins,
            "species": species,
            "species_orbital_map": species_orbital_map,
            "atoms_orbital_map": atoms_orbital_map,
        }
        param_dicts = expand_grouped_params_to_dicts(grouped_params_dict)
        
        results = []
        for params in param_dicts:
            # Extract params from dict
            atoms = params["atoms"]
            orbitals = params["orbitals"]
            spins = params["spins"]
            species = params["species"]
            species_orbital_map = params["species_orbital_map"]
            atoms_orbital_map = params["atoms_orbital_map"]

            if spins is None:
                spins = default_spins

            selection = self._resolve_projection_selection(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
            atoms = selection.atoms
            orbitals = selection.orbitals
            spins = selection.spins
            species = selection.species

            if self.is_non_collinear:
                if spins is None or len(spins) != 1 or spins[0] != 0:
                    raise ValueError(
                        "Magnetization for non-collinear calculations must use the total spin channel (index 0)."
                    )
            else:
                if spins is None or len(spins) != 2:
                    raise ValueError(
                        "Magnetization for spin-polarized calculations requires exactly two spin channels (up and down)."
                    )

            allowed_modes = {NormMode.RAW, NormMode.MAGNETIZATION, NormMode.INTEGRAL, NormMode.ELECTRONS}
            norm_mode = NormMode.from_input(norm_mode)
            if norm_mode not in allowed_modes:
                valid_modes = "\n".join(f"- {mode.value}" for mode in sorted(allowed_modes, key=lambda m: m.value))
                raise ValueError(f"Invalid normalization mode: {norm_mode}. Valid modes are:\n{valid_modes}")

            sum_kwargs = keep_func_kwargs(kwargs, self.sum_projection_components)
            components = self.sum_projection_components(
                values_array=dos_array,
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                keepdims=keepdims,
                **sum_kwargs,
            )

            if self.is_spin_polarized:
                magnetization_array = components[:, 0, ...] - components[:, 1, ...]
                if keepdims:
                    magnetization_array = magnetization_array[:, np.newaxis, ...]
            else:
                magnetization_array = components

            if magnetization_array.ndim == 1:
                magnetization_array = magnetization_array[..., np.newaxis]

            values = self.normalize(mode=norm_mode, values_array=magnetization_array, **kwargs)

            data_min = np.min(values, axis=0)
            data_max = np.max(values, axis=0)
            data_lim = (data_min, data_max)
            rounded_data_lim = (np_round_to_half(data_min), np_round_to_half(data_max))

            property_name = NormMode.get_normed_name(norm_mode, "magnetization")
            units = NormMode.get_normed_units(norm_mode, "$\\frac{states}{eV}$")
            footnote = NormMode.get_mode_footnote(norm_mode)

            label_plain_list, label_latex_list = self._format_selection_label(
                selection=selection,
                normalize=norm_mode is not NormMode.RAW,
                include_normal_label=include_normal_label,
            )

            metadata = {
                "atoms": list(atoms) if len(atoms) > 0 else None,
                "orbitals": list(orbitals) if orbitals is not None else None,
                "spins": list(spins) if spins is not None else None,
                "species": list(species) if len(species) > 0 else None,
                "norm_mode": norm_mode,
                "units": units,
                "data_lim": data_lim,
                "rounded_data_lim": rounded_data_lim,
                "label": label_latex_list,
                "label_plain": label_plain_list,
                "footnote": footnote,
                "scalar_label": scalar_label,
                "atom_label": selection.labels.atom,
                "atom_label_latex": selection.labels.atom_latex,
                "orbital_label": selection.labels.orbital,
                "orbital_label_latex": selection.labels.orbital_latex,
                "spin_label": selection.labels.spin,
                "spin_label_latex": selection.labels.spin_latex,
                "species_label": selection.labels.species,
                "species_label_latex": selection.labels.species_latex,
                "label_prefix": selection.labels.prefix_plain,
                "label_prefix_latex": selection.labels.prefix_latex,
                "spin_component_labels": list(selection.labels.spin_components),
                "spin_component_labels_latex": list(selection.labels.spin_components_latex),
                "label_combined": selection.labels.combined,
                "label_combined_latex": selection.labels.combined_latex,
                "include_normal_label": include_normal_label,
                "mode": mode,
                "keepdims": keepdims,
            }

            results.append(Property(
                name=property_name,
                value=values,
                point_set=self,
                metadata=metadata,
                label=scalar_label,
                units=units,
            ))
        
        return results[0] if len(results) == 1 else results

    def compute_spin_texture_magnitude(
        self,
        atoms: Iterable[int] | None = None,
        orbitals: Iterable[int] | None = None,
        spins: Iterable[int] | None = None,
        species: Iterable[str] | None = None,
        species_orbital_map: dict[str, Iterable[int]] | None = None,
        atoms_orbital_map: dict[int, Iterable[int]] | None = None,
        norm_mode: str | NormMode = "raw",
        from_total: bool = False,
        keepdims: bool = False,
        **kwargs,
    ) -> Property | list[Property]:
        include_normal_label = bool(kwargs.pop("include_normal_label", False))
        scalar_label = kwargs.pop("scalar_label", "Spin Texture Magnitude")

        if (self.projected is None and hasattr(self, "total")) or from_total:
            dos_array = self.total.to_array()
        elif self.projected is not None and not from_total:
            dos_array = self.projected.to_array()
        else:
            raise ValueError("Spin texture magnitude cannot be computed, as the total or projected DOS is not provided")

        if not self.is_non_collinear:
            raise ValueError("Spin texture magnitude is only available for non-collinear calculations")

        # Build dict of grouped params
        grouped_params_dict = {
            "atoms": atoms,
            "orbitals": orbitals,
            "spins": spins,
            "species": species,
            "species_orbital_map": species_orbital_map,
            "atoms_orbital_map": atoms_orbital_map,
        }
        param_dicts = expand_grouped_params_to_dicts(grouped_params_dict)
        
        results = []
        for params in param_dicts:
            # Extract params from dict
            atoms = params["atoms"]
            orbitals = params["orbitals"]
            spins = params["spins"]
            species = params["species"]
            species_orbital_map = params["species_orbital_map"]
            atoms_orbital_map = params["atoms_orbital_map"]

            if spins is None:
                spins = (1, 2, 3)

            selection = self._resolve_projection_selection(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
            atoms = selection.atoms
            orbitals = selection.orbitals
            spins = selection.spins
            species = selection.species

            if spins is None or len(spins) == 0:
                raise ValueError("Spin texture magnitude requires at least one spin component to be selected")
            invalid_spins = set(spins) - {1, 2, 3}
            if invalid_spins:
                raise ValueError(
                    f"Invalid spins for spin texture magnitude: {sorted(invalid_spins)}. Valid components are [1, 2, 3]."
                )

            allowed_modes = {
                NormMode.INTEGRAL,
                NormMode.SPIN_MAGNITUDE,
                NormMode.ELECTRONS,
                NormMode.MAGNETIZATION,
                NormMode.RAW,
            }
            norm_mode = NormMode.from_input(norm_mode)
            if norm_mode not in allowed_modes:
                valid_modes = "\n".join(f"- {mode.value}" for mode in sorted(allowed_modes, key=lambda m: m.value))
                raise ValueError(f"Invalid normalization mode: {norm_mode}. Valid modes are:\n{valid_modes}")

            sum_kwargs = keep_func_kwargs(kwargs, self.sum_projection_components)
            values = self.sum_projection_components(
                values_array=dos_array,
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                **sum_kwargs,
            )
            values = np.linalg.norm(values, axis=-1, keepdims=keepdims)

            logger.debug("spin texture magnitude array shape: %s", values.shape)

            if values.ndim == 1:
                values = values[..., np.newaxis]

            values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

            data_min = np.min(values, axis=0)
            data_max = np.max(values, axis=0)
            data_lim = (data_min, data_max)
            rounded_data_lim = (np_round_to_half(data_min), np_round_to_half(data_max))

            property_name = NormMode.get_normed_name(norm_mode, "spin_texture_magnitude")
            units = NormMode.get_normed_units(norm_mode, "$\\frac{states}{eV}$")
            footnote = NormMode.get_mode_footnote(norm_mode)

            label_plain_list, label_latex_list = self._format_selection_label(
                selection=selection,
                normalize=norm_mode is not NormMode.RAW,
                include_normal_label=include_normal_label,
            )

            metadata = {
                "atoms": list(atoms) if len(atoms) > 0 else None,
                "orbitals": list(orbitals) if orbitals is not None else None,
                "spins": list(spins) if spins is not None else None,
                "species": list(species) if len(species) > 0 else None,
                "norm_mode": norm_mode,
                "units": units,
                "data_lim": data_lim,
                "rounded_data_lim": rounded_data_lim,
                "label": label_latex_list,
                "label_plain": label_plain_list,
                "footnote": footnote,
                "scalar_label": scalar_label,
                "atom_label": selection.labels.atom,
                "atom_label_latex": selection.labels.atom_latex,
                "orbital_label": selection.labels.orbital,
                "orbital_label_latex": selection.labels.orbital_latex,
                "spin_label": selection.labels.spin,
                "spin_label_latex": selection.labels.spin_latex,
                "species_label": selection.labels.species,
                "species_label_latex": selection.labels.species_latex,
                "label_prefix": selection.labels.prefix_plain,
                "label_prefix_latex": selection.labels.prefix_latex,
                "spin_component_labels": list(selection.labels.spin_components),
                "spin_component_labels_latex": list(selection.labels.spin_components_latex),
                "label_combined": selection.labels.combined,
                "label_combined_latex": selection.labels.combined_latex,
                "include_normal_label": include_normal_label,
                "keepdims": keepdims,
            }

            results.append(Property(
                name=property_name,
                value=values,
                point_set=self,
                metadata=metadata,
                label=scalar_label,
                units=units,
            ))
        
        return results[0] if len(results) == 1 else results

    def compute_normalized_total(
        self,
        norm_mode: str | NormMode = "max",
        **kwargs,
    ) -> Property:
        _ = kwargs  # unused extra parameters for forward compatibility
        metadata = {
            "norm_mode": norm_mode,
        }
        norm_mode = NormMode.from_input(norm_mode)
        
        normed_values = self.normalize(mode=norm_mode, values_array=self.total.to_array(), **kwargs)
        
        mode_prefix = NormMode.get_mode_prefix(norm_mode)
        mode_type_suffix = NormMode.get_mode_type_suffix(norm_mode)
        
        name = self.total.name
        label = self.total.label
        units = self.total.units
        data_lim = self.total.data_lim
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

    def compute_cumulative_total(self, norm_mode: str | NormMode  = None) -> Property:
        cumlative_total = self.cumsum(values=self.total)
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


        logger.info(f"Cumulative total: {values.shape}")
        logger.info(f"Cumulative total (min, max): {np.min(values), np.max(values)}")
        
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
    
    def _get_projection_label_builder(self) -> ProjectionLabelBuilder:
        if self._projection_label_builder is None:
            atom_indexer = None
            if self.structure is not None:
                atom_indexer = AtomIndexer.from_structure(self.structure)
            spin_indexer = SpinIndexer.from_projection_names(
                self.spin_projection_names
            )
            self._projection_label_builder = ProjectionLabelBuilder(
                atom_indexer=atom_indexer,
                orbital_indexer=OrbitalIndexer(),
                spin_indexer=spin_indexer,
            )
        return self._projection_label_builder

    def _get_projection_selection_resolver(self) -> ProjectionSelectionResolver:
        if self._projection_selection_resolver is None:
            label_builder = self._get_projection_label_builder()
            self._projection_selection_resolver = ProjectionSelectionResolver(
                label_builder=label_builder,
                orbital_names=self.orbital_names,
                is_non_colinear=self.is_non_collinear,
            )
        return self._projection_selection_resolver

    def _resolve_projection_selection(
        self,
        *,
        atoms: Sequence[int] | int | None = None,
        orbitals: Sequence[int] | int | None = None,
        spins: Sequence[int] | int | None = None,
        species: Sequence[str] | str | None = None,
        species_orbital_map: Sequence[Mapping[str, Iterable[int]]] | Mapping[str, Iterable[int]] | None = None,
        atoms_orbital_map: Sequence[Mapping[Iterable[int] | int, Iterable[int]]] | Mapping[Iterable[int] | int, Iterable[int]] | None = None,
    ) -> ProjectionSelectionResult:
        resolver = self._get_projection_selection_resolver()
        return resolver.resolve(
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            species=species,
            species_orbital_map=species_orbital_map,
            atoms_orbital_map=atoms_orbital_map,
        )

    @staticmethod
    def _format_selection_label(
        selection: ProjectionSelectionResult,
        *,
        normalize: bool,
        include_normal_label: bool,
    ) -> tuple[list[str], list[str]]:
        mode_plain = "fraction" if normalize else "raw"
        mode_latex = "\\mathrm{fraction}" if normalize else "\\mathrm{raw}"

        prefix_plain = selection.labels.prefix_plain
        prefix_latex = selection.labels.prefix_latex

        spin_components = selection.labels.spin_components
        spin_components_latex = selection.labels.spin_components_latex

        if not spin_components:
            spin_components = ("",)
            spin_components_latex = ("",)

        labels_plain: list[str] = []
        labels_latex: list[str] = []

        normal_suffix_plain = f" [{mode_plain}]" if include_normal_label else ""
        normal_suffix_latex = f" [{mode_latex}]" if include_normal_label else ""

        for component_plain, component_latex in zip(spin_components, spin_components_latex):
            body_plain = prefix_plain
            if component_plain:
                body_plain = f"{body_plain}[{component_plain}]" if body_plain else component_plain
            body_plain = body_plain or "all"
            labels_plain.append(f"{body_plain}{normal_suffix_plain}")

            body_latex = prefix_latex
            if component_latex:
                body_latex = (
                    f"{body_latex}[{component_latex}]"
                    if body_latex
                    else f"[{component_latex}]"
                )
            body_latex = body_latex or "\\mathrm{all}"
            labels_latex.append(f"${body_latex}{normal_suffix_latex}$")

        return labels_plain, labels_latex

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
        include_normal_label = kwargs.get("include_normal_label", True)

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
        if not include_normal_label:
            params["include_normal_label"] = False
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


    def _validate_projection_selection_params(self, 
                                              atoms: Sequence[int] | None = None, 
                                              orbitals: Sequence[int] | None = None, 
                                              spins: Sequence[int] | None = None, 
                                              species: Sequence[str] | None = None, 
                                              species_orbital_map: dict[str, Iterable[int]] | None = None, 
                                              atoms_orbital_map: dict[int, Iterable[int]] | None = None
                                              ) -> tuple[set[int], set[int], set[int], set[str]]:
 
        if species is not None and atoms is not None:
            raise ValueError("atoms and species cannot be specified together")
        selection = self._resolve_projection_selection(
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            species=species,
            species_orbital_map=species_orbital_map,
            atoms_orbital_map=atoms_orbital_map,
        )

        atoms_set = set(selection.atoms)
        orbitals_set = set(selection.orbitals) if selection.orbitals is not None else None
        spins_set = set(selection.spins) if selection.spins is not None else None
        species_set = set(selection.species)

        return atoms_set, orbitals_set, spins_set, species_set
        
        
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
