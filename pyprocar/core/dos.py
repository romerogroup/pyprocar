"""Core density of states data object."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline

from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParametricDataset:
    """Container for precomputed DOS data used by parametric plots."""

    energies: npt.NDArray[np.float64]
    totals: npt.NDArray[np.float64]
    total_projected: npt.NDArray[np.float64]
    projected: npt.NDArray[np.float64]
    spin_indices: Tuple[int, ...]
    spin_labels: Tuple[str, ...]

    def normalized(self) -> npt.NDArray[np.float64]:
        denominators = np.asarray(self.total_projected, dtype=np.float64)
        numerators = np.asarray(self.projected, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                numerators,
                denominators,
                out=np.zeros_like(numerators),
                where=denominators != 0,
            )
        return np.nan_to_num(normalized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


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
        energies_array = self._prepare_energies(energies)
        gradient = gradient_func or _finite_difference_gradient

        super().__init__(points=energies_array, gradient_func=gradient)

        self._fermi = float(fermi)
        self._orbital_names = orbital_names

        total_array = self._prepare_total(total)
        self.add_property(name="total", value=total_array)

        if projected is not None:
            projected_array = self._prepare_projected(projected)
            self.add_property(name="projected", value=projected_array)

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
    def energies(self) -> npt.NDArray[np.float64]:
        return self.points

    @property
    def total(self) -> npt.NDArray[np.float64]:
        prop = super().get_property("total")
        return prop.value if prop is not None else np.empty((0, 0))

    @property
    def projected(self) -> npt.NDArray[np.float64] | None:
        prop = super().get_property("projected")
        return prop.value if prop is not None else None

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
        return self.total.shape[1]

    @property
    def n_spins(self) -> int:
        if self.projected is None:
            return self.n_spin_channels
        return self.projected.shape[1]

    @property
    def n_atoms(self) -> int:
        if self.projected is None:
            return 0
        return self.projected.shape[2]

    @property
    def n_orbitals(self) -> int:
        if self.projected is None:
            return 0
        return self.projected.shape[3]

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
        if self.projected.shape[-1] == 2 + 2 + 4 + 4 + 6 + 6 + 8:
            return True
        return False

    @property
    def spin_projection_names(self) -> list[str]:
        if self.is_non_collinear:
            return ["total", "x", "y", "z"]
        if self.is_spin_polarized:
            return ["Spin-up", "Spin-down"]
        return ["Spin-up"]

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
            if isinstance(computed, Property):
                property_obj = computed
            else:
                property_obj = Property(name=stored_key, value=np.asarray(computed))
            self.add_property(property=property_obj)
            stored_key = property_obj.name

        normalized_key = self._normalize_super_key(key, stored_key)
        return super().get_property(normalized_key)

    def compute_property(self, name: str, **kwargs):
        if name in {"projected_sum", "projected_sum_total"}:
            total_proj, projected = self.compute_projected_sum(**kwargs)
            params = self._projected_sum_cache_params(**kwargs)
            base = "projected_sum_total" if name.endswith("_total") else "projected_sum"
            value = total_proj if name.endswith("_total") else projected
            key = self._make_property_key(base, params)
            return Property(name=key, value=value)

        if name == "normalized_total":
            mode = kwargs.get("mode", "max")
            normalized = self.compute_normalized_total(mode=mode)
            params = {} if mode == "max" else {"mode": mode}
            key = self._make_property_key(name, params)
            return Property(name=key, value=normalized)

        if name == "cumulative_total":
            cumulative = self.compute_cumulative_total()
            key = self._make_property_key(name, {})
            return Property(name=key, value=cumulative)

        return None

    def add_property(
        self,
        property: Property | None = None,
        *,
        name: str | None = None,
        value: npt.ArrayLike | None = None,
    ) -> Property:
        """Attach a custom property to the DOS object.

        Users may supply an existing :class:`Property` instance or provide a
        ``name`` and array ``value`` whose first axis matches the energy grid.
        """

        if property is not None and (name is not None or value is not None):
            raise ValueError("Provide either a Property instance or name/value, not both.")

        if property is not None:
            self.validate_property_points(property)
            super().add_property(property=property)
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

        super().add_property(name=name, value=value_array)
        return self.property_store[name]

    # ------------------------------------------------------------------
    # Core calculations
    # ------------------------------------------------------------------
    def dos_sum(
        self,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        sum_noncolinear: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Sum projections over selected atoms, orbitals, and spins."""
        if self.projected is None:
            raise ValueError("Projected DOS is not available for this calculation")

        spins_ndarray = self._sanitize_indices(spins, self.n_spins)
        if spins_ndarray is None:
            if self.n_spins == 4:
                raise ValueError(
                    "Spins must be provided for non-colinear calculations."
                )
            spins_ndarray = np.arange(self.n_spins, dtype=int)

        atoms_ndarray = self._sanitize_indices(atoms, self.n_atoms)
        orbitals_ndarray = self._sanitize_indices(orbitals, self.n_orbitals)

        zero_result = np.zeros((self.n_energies, self.n_spins), dtype=np.float64)
        if (
            atoms_ndarray is not None
            and atoms_ndarray.size == 0
            or orbitals_ndarray is not None
            and orbitals_ndarray.size == 0
        ):
            aggregated = zero_result
        else:
            projected = np.asarray(self.projected, dtype=np.float64)
            if atoms_ndarray is not None:
                projected = np.take(projected, atoms_ndarray, axis=2)
            if orbitals_ndarray is not None:
                projected = np.take(projected, orbitals_ndarray, axis=3)
            aggregated = projected.sum(axis=-1).sum(axis=-1)

        if self.is_non_collinear:
            selected = aggregated[..., spins_ndarray]
            if sum_noncolinear:
                aggregated_result = np.sum(selected, axis=-1, keepdims=True)
            else:
                aggregated_result = selected
        else:
            aggregated_result = np.zeros_like(aggregated)
            aggregated_result[..., spins_ndarray] = aggregated[..., spins_ndarray]

        return aggregated_result

    def compute_projected_sum(
        self,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        sum_noncolinear: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return total and filtered projected DOS sums."""
        if self.projected is None:
            raise ValueError("Projected DOS is not available for this calculation")

        if self.n_spins == 4:
            dos_total_projected = self.dos_sum(
                spins=spins,
                sum_noncolinear=sum_noncolinear,
            )
        else:
            dos_total_projected = self.dos_sum()

        dos_projected = self.dos_sum(
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            sum_noncolinear=sum_noncolinear,
        )

        return dos_total_projected, dos_projected

    def build_parametric_dataset(
        self,
        *,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        sum_noncolinear: bool = True,
    ) -> ParametricDataset:
        """Prepare the arrays needed for parametric DOS visualisations."""

        if self.projected is None:
            raise ValueError(
                "Projected DOS data are required to build a parametric dataset."
            )

        spin_indices = self._sanitize_indices(spins, self.n_spins)
        if spin_indices is None:
            spin_indices = np.arange(self.n_spins, dtype=int)
        if spin_indices.size == 0:
            raise ValueError("At least one spin channel must be selected.")

        total = np.asarray(self.total, dtype=np.float64)
        if total.ndim == 1:
            total = total[:, np.newaxis]

        if np.any(spin_indices >= total.shape[1]):
            raise IndexError("Spin indices exceed available DOS spin channels.")

        total_selected = total[:, spin_indices]
        if total_selected.ndim == 1:
            total_selected = total_selected[:, np.newaxis]

        dos_total_projected, dos_projected = self.compute_projected_sum(
            atoms=atoms,
            orbitals=orbitals,
            spins=spin_indices,
            sum_noncolinear=sum_noncolinear,
        )

        dos_total_projected = np.asarray(dos_total_projected, dtype=np.float64)
        dos_projected = np.asarray(dos_projected, dtype=np.float64)

        if dos_total_projected.ndim == 1:
            dos_total_projected = dos_total_projected[:, np.newaxis]
        if dos_projected.ndim == 1:
            dos_projected = dos_projected[:, np.newaxis]

        total_projected_selected = dos_total_projected[:, spin_indices]
        projected_selected = dos_projected[:, spin_indices]

        if total_projected_selected.ndim == 1:
            total_projected_selected = total_projected_selected[:, np.newaxis]
        if projected_selected.ndim == 1:
            projected_selected = projected_selected[:, np.newaxis]

        spin_labels_source = self.spin_projection_names
        spin_labels = tuple(spin_labels_source[idx] for idx in spin_indices)

        return ParametricDataset(
            energies=np.asarray(self.energies, dtype=np.float64),
            totals=total_selected.T,
            total_projected=total_projected_selected.T,
            projected=projected_selected.T,
            spin_indices=tuple(int(i) for i in spin_indices.tolist()),
            spin_labels=spin_labels,
        )

    def compute_normalized_total(self, mode: str = "max") -> npt.NDArray[np.float64]:
        factors = self._normalization_factors(mode)
        return self.total / factors

    def compute_cumulative_total(self) -> npt.NDArray[np.float64]:
        total = np.asarray(self.total, dtype=np.float64)
        if total.shape[0] < 2:
            return np.zeros_like(total)
        delta_e = np.diff(self.energies)
        avg = 0.5 * (total[1:] + total[:-1])
        cumulative = np.zeros_like(total)
        cumulative[1:] = np.cumsum(avg * delta_e[:, np.newaxis], axis=0)
        return cumulative

    def normalize_dos(self, mode: str = "max") -> "DensityOfStates":
        factors = self._normalization_factors(mode)
        total_normalized = self.total / factors

        projected_normalized = None
        if self.projected is not None:
            projected_scale = factors
            if projected_scale.shape[1] != self.n_spins:
                projected_scale = np.broadcast_to(
                    projected_scale[:, :1], (1, self.n_spins)
                )
            projected_normalized = self.projected / projected_scale.reshape(
                1, projected_scale.shape[1], 1, 1
            )

        return DensityOfStates(
            energies=self.energies,
            total=total_normalized,
            fermi=self.fermi,
            projected=projected_normalized,
            orbital_names=self.orbital_names,
        )

    def interpolate(self, factor: int = 2) -> "DensityOfStates":
        if factor in (0, 1):
            return self

        energies, total = interpolate(self.energies, self.total, factor=factor)
        projected = None
        if self.projected is not None:
            _, projected = interpolate(self.energies, self.projected, factor=factor)

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
    def _prepare_energies(self, energies: npt.ArrayLike) -> npt.NDArray[np.float64]:
        energies_array = np.asarray(energies, dtype=np.float64).reshape(-1)
        if energies_array.ndim != 1:
            raise ValueError("Energies must be a 1D array")
        return energies_array

    def _prepare_total(self, total: npt.ArrayLike) -> npt.NDArray[np.float64]:
        total_array = np.asarray(total, dtype=np.float64)
        if total_array.ndim == 1:
            total_array = total_array[:, np.newaxis]
        if total_array.shape[0] != self.n_energies:
            raise ValueError("Total DOS must have the same number of energies")
        return total_array

    def _prepare_projected(self, projected: npt.ArrayLike) -> npt.NDArray[np.float64]:
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

    def _sanitize_indices(
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

        atoms_idx = self._sanitize_indices(atoms, self.n_atoms)
        if atoms_idx is not None:
            params["atoms"] = tuple(int(i) for i in atoms_idx)

        orbitals_idx = self._sanitize_indices(orbitals, self.n_orbitals)
        if orbitals_idx is not None:
            params["orbitals"] = tuple(int(i) for i in orbitals_idx)

        spins_idx = self._sanitize_indices(spins, self.n_spins)
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
