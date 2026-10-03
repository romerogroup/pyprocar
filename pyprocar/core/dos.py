"""Core density of states data object."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import integrate
from scipy.interpolate import CubicSpline

from pyprocar.core.atomic_orbital_index import ProjectionSelectionResolver
from pyprocar.core.projection import NormMode, build_property, selection_resolver
from pyprocar.core.projection import normalize as normalize_by_mode
from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer
from pyprocar.utils.func_utils import expand_grouped_params_to_dicts, keep_func_kwargs

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pyprocar.core.structure import Structure


def get_dos_from_code(
    code: str,
    dirpath: str,
    use_cache: bool = False,
    filename: str = "dos.pkl",
) -> DensityOfStates:
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

    from pyprocar.io import get_parser

    dos_filepath = Path(dirpath) / filename

    if not use_cache or not dos_filepath.exists():
        logger.info("Parsing DOS calculation directory: %s", dirpath)
        parser = get_parser(code, dirpath)
        dos = parser.dos
        if dos is None:
            raise ValueError(f"The {code} parser found no dos in {dirpath}")
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
            "Gradient requires the first axis of the property to match the number of sample points."
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
    Accepts optional space separators.
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

        total_array = self._validate_total(total)

        total_metadata = {}
        if total_array.shape[1] == 1:
            total_metadata["label"] = ["Total"]
        elif total_array.shape[1] == 2:
            total_metadata["label"] = ["$Total - \\uparrow$", "$Total - \\downarrow$"]
        elif total_array.shape[1] == 4:
            total_metadata["label"] = ["$Total$", "$Total - S_x$", "$Total - S_y$", "$Total - S_z$"]
        else:
            raise ValueError(
                f"Total array has {self.total_array.shape[1]} spin channels, which is not supported"
            )

        self.add_property(
            name="total",
            value=total_array,
            units="$\\frac{states}{eV}$",
            label="DoS",
            metadata=total_metadata,
        )

        if projected is not None:
            projected_array = self._validate_projected(projected)
            self.add_property(
                name="projected",
                value=projected_array,
                units="$\\frac{states}{eV}$",
                label="Projected DoS",
                metadata={"label": ["Projected DoS"]},
            )

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
        arrays_equal = np.allclose(self.energies, other.energies) and np.allclose(
            self.total, other.total
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

    # -------------------------------------------------------------------
    # Class methods / Constructors
    # -------------------------------------------------------------------

    @classmethod
    def from_code(
        cls,
        code: str,
        dirpath: str,
        use_cache: bool = False,
        filename: str = "dos.pkl",
    ) -> DensityOfStates:
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
        atoms = self._require_structure().atoms
        if atoms is None:
            raise ValueError("The structure of this density of states has no atoms")
        return atoms

    @property
    def species(self) -> list[str]:
        return list(self._require_structure().species)

    def _require_structure(self) -> Structure:
        if self.structure is None:
            raise ValueError("This density of states has no structure")
        return self.structure

    @property
    def orbitals(self) -> list[str]:
        return self.orbital_names

    # -------------------------------------------------------------------
    # Array Properties
    # -------------------------------------------------------------------

    @property
    def total(self) -> Property:
        return self.get_property("total")

    @property
    def projected(self) -> Property | None:
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

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Useful getters
    # -------------------------------------------------------------------

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

    def integrate(
        self, values_array: npt.NDArray[np.float64], energy_lim: tuple[float, float] | None = None
    ) -> npt.NDArray[np.float64]:
        """Integrate the values over the energy range.

        Parameters
        ----------
        values_array: npt.NDArray[np.float64]
            The values to integrate.
        energy_lim: tuple[float, float] | None
            The energy range to integrate over.

        Returns
        -------
        npt.NDArray[np.float64]
            The integrated values.
        """
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
        """Compute the cumulative sum of the values.

        Parameters
        ----------
        values: npt.NDArray[np.float64] | Property
            The values to cumsum.

        Returns
        -------
        npt.NDArray[np.float64]
            The cumulative sum of the values.
        """
        if isinstance(values, Property):
            values = values.to_array()
        return np.cumsum(values, axis=0)

    def shift_by_fermi(self) -> DensityOfStates:
        """Shift the DOS by the Fermi energy.

        Returns
        -------
        DensityOfStates
            The shifted DOS.
        """
        # new_dos = copy.deepcopy(self)
        self._points -= self.fermi
        self._points_label = "E − E_F (eV)"
        return self

    def interpolate(self, factor: int = 2) -> DensityOfStates:
        """Interpolate the DOS by a factor.

        Parameters
        ----------
        factor: int
            The factor to interpolate by.

        Returns
        -------
        DensityOfStates
            The interpolated DOS.
        """
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
        """Sum projections over selected atoms, orbitals, and spins.

        Parameters
        ----------
        values_array: npt.NDArray[np.float64]
            The values to sum.
        atoms: Sequence[int] | None
            The atoms to sum over.
        orbitals: Sequence[int] | None
            The orbitals to sum over.
        spins: Sequence[int] | None
            The spins to sum over.
        keepdims: bool
            Whether to keep the dimensions of the summed array.

        Returns
        -------
        npt.NDArray[np.float64]
            The summed values.
        """
        tmp_array = self.select_projection_components(
            values_array=values_array, atoms=atoms, orbitals=orbitals, spins=spins
        )

        n_dims = tmp_array.ndim
        if keepdims and n_dims == 4:
            summed_array = tmp_array.sum(axis=2, keepdims=keepdims).sum(axis=3, keepdims=keepdims)
        elif not keepdims and n_dims == 4:
            summed_array = tmp_array.sum(axis=-1).sum(axis=-1)
        elif n_dims == 3:
            summed_array = tmp_array.sum(axis=2, keepdims=keepdims)
        elif n_dims == 2:
            summed_array = tmp_array
        else:
            raise ValueError(
                "An unexpected error occured. This is likely due to a bug in the code. Please report this issue."
            )

        logger.debug(f"summed_array: {summed_array.shape}")
        return summed_array

    def select_projection_components(
        self,
        values_array: npt.NDArray[np.float64],
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Select the projection components from the values array.

        Parameters
        ----------
        values_array: npt.NDArray[np.float64]
            The values to select the projection components from.
        atoms: Sequence[int] | None
            The atoms to select the projection components from.
        orbitals: Sequence[int] | None
            The orbitals to select the projection components from.
        spins: Sequence[int] | None
            The spins to select the projection components from.

        Returns
        -------
        npt.NDArray[np.float64]
            The selected projection components.
        """
        # Validate Input
        n_dims = values_array.ndim

        if n_dims < 2:
            raise ValueError(
                "Values array must have at least 2 dimensions, which represent the energy and spin channels"
            )

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

    def normalize(
        self,
        mode: str | NormMode | None,
        values_array: npt.NDArray[np.float64],
        sigma: float = 1.25,
        fill_value: float = 0.0,
        eps: float = 0.001,
        spins: Sequence[int] | None = None,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Normalize ``values_array`` (energies first, then spin channels) by ``mode``.

        For collinear data ``spins`` names the channels ``values_array`` holds, so
        the total denominators use those same channels. Non-collinear columns are
        spin components, so their denominators keep the leading columns, and a
        single component divides by the charge total.

        Spin-magnitude and magnetization modes skip denominators below ``eps`` and
        clip outliers beyond ``sigma`` standard deviations to ``fill_value``.
        """
        mode = NormMode.parse(mode)
        n_spins = np.shape(values_array)[1] if np.ndim(values_array) > 1 else None
        channels = slice(0, n_spins) if spins is None or self.is_non_collinear else list(spins)

        def above_eps(prop: Property | None, what: str, magnitude) -> npt.NDArray[np.float64]:
            if prop is None:
                raise ValueError(f"{what} is not available for this calculation")
            array = prop.to_array()
            return np.where(magnitude(array) >= eps, array, 0.0)

        def integral() -> npt.NDArray[np.float64]:
            integrals = integrate.trapezoid(values_array, x=self.energies, axis=0)[np.newaxis]
            return np.where(integrals == 0, 1.0, integrals)

        normalized = normalize_by_mode(
            values_array,
            mode,
            {
                NormMode.MAX: lambda: np.max(np.abs(values_array), axis=0, keepdims=True),
                NormMode.INTEGRAL: integral,
                NormMode.ELECTRONS: lambda: self.n_electrons,
                NormMode.TOTAL: lambda: self.total.to_array()[:, channels],
                NormMode.TOTAL_PROJECTION: lambda: self.projected_total.to_array()[:, channels],
                NormMode.MAGNETIZATION: lambda: above_eps(
                    self.magnetization, "Magnetization", lambda a: a
                ),
                NormMode.SPIN_MAGNITUDE: lambda: above_eps(
                    self.spin_magnitude, "Spin magnitude", np.abs
                ),
            },
        )
        if mode in (NormMode.MAGNETIZATION, NormMode.SPIN_MAGNITUDE):
            normalized = filter_data_within_sigma(normalized, sigma=sigma, fill_value=fill_value)
        return normalized

    def normed_units(self, mode: NormMode, units: str | None) -> str | None:
        if mode is NormMode.RAW:
            return units
        if mode is NormMode.MAX:
            normalizer = units
        elif mode in (NormMode.INTEGRAL, NormMode.ELECTRONS):
            normalizer = "$states$"
        else:
            normalizer = "$\\frac{states}{eV}$"
        return _units_divide(units, _units_divide(units, normalizer))

    # ------------------------------------------------------------------
    # Computing methods - helper functions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Computing methods
    # ------------------------------------------------------------------
    def compute_projected_sum(
        self,
        atoms: Iterable[int] | None = None,
        orbitals: Iterable[int] | None = None,
        spins: Iterable[int] | None = None,
        species: Iterable[str] | None = None,
        species_orbital_map: dict[str, Iterable[int]] | None = None,
        atoms_orbital_map: dict[int, Iterable[int]] | None = None,
        norm_mode: str | NormMode | None = "raw",
        label: str = "Projected DoS",
        name: str = "projected_sum",
        units="$\\frac{states}{eV}$",
        **kwargs,
    ) -> Property | list[Property]:
        """Compute projected DOS sums over selected atoms, orbitals, and spins.

        Sums the projected DOS components over the specified selection:
        P(E) = Σ_{atoms, orbitals} DOS(E, spins, atoms, orbitals)

        Parameters
        ----------
        atoms
            Atom indices to sum over. If None, sums over all atoms.
        orbitals
            Orbital indices to sum over. If None, sums over all orbitals.
        spins
            Spin channels to include. If None, includes all spins.
        species
            Species names to select atoms by. Cannot be used with atoms.
        species_orbital_map
            Mapping of species to orbital indices for selective summing.
        atoms_orbital_map
            Mapping of atom indices/tuples to orbital indices.
        norm_mode
            Normalization mode to apply after summing.
        label
            Scalar label for the property.
        name
            Base name for the property.
        units
            Default units string before normalization.
        **kwargs
            Additional kwargs passed to sum_projection_components.

        Returns
        -------
        Property | list[Property]
            Single Property if one result, list if multiple parameter combinations.

        """
        if self.projected is None:
            raise ValueError("Projected DOS is not available for this calculation")

        # Resolve selection parameters groups.
        param_dicts = expand_grouped_params_to_dicts(
            dict(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
        )

        results = []
        for params in param_dicts:
            # Resolve Projection Selection
            selection = self._selection_resolver.resolve(
                atoms=params["atoms"],
                orbitals=params["orbitals"],
                spins=params["spins"],
                species=params["species"],
                species_orbital_map=params["species_orbital_map"],
                atoms_orbital_map=params["atoms_orbital_map"],
            )

            # Sum Atomic Projection Components
            values = self.sum_projection_components(
                values_array=self.projected.to_array(),
                atoms=selection.atoms,
                orbitals=selection.orbitals,
                spins=selection.spins,
                **keep_func_kwargs(kwargs, self.sum_projection_components),
            )

            # Build Property
            prop = build_property(
                self,
                values=values,
                label=label,
                name=name,
                units=units,
                selection=selection,
                norm_mode=norm_mode,
                allowed_norm_modes=None,
                include_normal_label=kwargs.get("include_normal_label", False),
                normalize_kwargs={**kwargs, "spins": selection.spins},
            )

            results.append(prop)

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
        label: str = "Spin Texture",
        name: str = "spin_texture",
        units: str = "$\\frac{states}{eV}$",
        **kwargs,
    ) -> Property | list[Property]:
        """Compute spin texture (S_x, S_y, S_z components) for non-collinear calculations.

        For non-collinear calculations, the spin texture represents the vector
        components of the spin density of states:
        S(E) = [S_x(E), S_y(E), S_z(E)]

        where each component corresponds to spin channels 1, 2, and 3 respectively.
        The spin texture describes the direction and magnitude of spin polarization
        at each energy level.

        Parameters
        ----------
        atoms
            Atom indices to sum over. If None, sums over all atoms.
        orbitals
            Orbital indices to sum over. If None, sums over all orbitals.
        spins
            Spin components to include. Defaults to (1, 2, 3) for S_x, S_y, S_z.
            Must be valid non-collinear spin channels [1, 2, 3].
        species
            Species names to select atoms by.
        species_orbital_map
            Mapping of species to orbital indices.
        atoms_orbital_map
            Mapping of atom indices to orbital indices.
        norm_mode
            Normalization mode. Allowed: raw, total_projection, spin_magnitude,
            integral, electrons, magnetization.
        label
            Scalar label for the property.
        name
            Base name for the property.
        units
            Default units string before normalization.
        **kwargs
            Additional kwargs. Passed to sum_projection_components and _build_property.

        Returns
        -------
        Property | list[Property]
            Single Property if one result, list if multiple parameter combinations.
        """
        # Validate Input
        if not self.is_non_collinear:
            raise ValueError("Spin texture is only available for non-collinear calculations")

        # Resolve default values
        dos_array = (
            self.total.to_array()
            if (self.projected is None and hasattr(self, "total"))
            else self.projected.to_array()
        )
        spins = (1, 2, 3) if spins is None else spins

        for spin in spins:
            if isinstance(spin, Iterable):
                invalid_spins = set(spins) - {1, 2, 3}
                if len(invalid_spins) > 0:
                    raise ValueError(
                        f"Invalid spins for spin texture magnitude: {sorted(invalid_spins)}. Valid components are [1, 2, 3]."
                    )

        param_dicts = expand_grouped_params_to_dicts(
            dict(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
        )

        results = []
        for params in param_dicts:
            # Resolve Projection Selection
            selection = self._selection_resolver.resolve(
                atoms=params["atoms"],
                orbitals=params["orbitals"],
                spins=params["spins"],
                species=params["species"],
                species_orbital_map=params["species_orbital_map"],
                atoms_orbital_map=params["atoms_orbital_map"],
            )

            # Sum Atomic Projection Components
            values = self.sum_projection_components(
                values_array=dos_array,
                atoms=selection.atoms,
                orbitals=selection.orbitals,
                spins=selection.spins,
                **keep_func_kwargs(kwargs, self.sum_projection_components),
            )

            # Build Property
            prop = build_property(
                self,
                values=values,
                label=label,
                name=name,
                units=units,
                selection=selection,
                norm_mode=norm_mode,
                allowed_norm_modes={
                    NormMode.TOTAL_PROJECTION,
                    NormMode.SPIN_MAGNITUDE,
                    NormMode.INTEGRAL,
                    NormMode.ELECTRONS,
                    NormMode.MAGNETIZATION,
                    NormMode.RAW,
                },
                include_normal_label=kwargs.get("include_normal_label", False),
                normalize_kwargs={**kwargs, "spins": selection.spins},
            )

            results.append(prop)

        return results[0] if len(results) == 1 else results

    def compute_magnetization(
        self,
        atoms: Iterable[int] | None = None,
        orbitals: Iterable[int] | None = None,
        species: Iterable[str] | None = None,
        species_orbital_map: dict[str, Iterable[int]] | None = None,
        atoms_orbital_map: dict[int, Iterable[int]] | None = None,
        norm_mode: str | NormMode = "raw",
        from_total: bool = False,
        label: str = "Magnetization",
        name: str = "magnetization",
        units: str = "$\\frac{states}{eV}$",
        **kwargs,
    ) -> Property | list[Property]:
        """Compute magnetization density of states.

        For collinear (spin-polarized) calculations:
            M(E) = DOS_up(E) - DOS_down(E)

        For non-collinear calculations:
            M(E) = DOS_total(E)

        where DOS_total is the total spin channel (index 0). The magnetization
        represents the net spin polarization at each energy level.

        Parameters
        ----------
        atoms
            Atom indices to sum over. If None, sums over all atoms.
        orbitals
            Orbital indices to sum over. If None, sums over all orbitals.
        species
            Species names to select atoms by.
        species_orbital_map
            Mapping of species to orbital indices.
        atoms_orbital_map
            Mapping of atom indices to orbital indices.
        norm_mode
            Normalization mode. Allowed: raw, magnetization, integral, electrons.
        from_total
            If True, use total DOS even if projected is available.
        label
            Scalar label for the property.
        name
            Base name for the property.
        units
            Default units string before normalization.
        **kwargs
            Additional kwargs. Passed to sum_projection_components and _build_property.

        Returns
        -------
        Property | list[Property]
            Single Property if one result, list if multiple parameter combinations.

        Raises
        ------
        ValueError
            If calculation is not spin-polarized or non-collinear, or if invalid spins are specified.
        """

        # Validate Input
        if not (self.is_spin_polarized or self.is_non_collinear):
            raise ValueError("Magnetization requires a spin-polarized or non-collinear calculation")
        if not (hasattr(self, "total") or hasattr(self, "projected")):
            raise ValueError("Total or projected DOS is not available for this calculation")

        # Resolve default values
        dos_array = (
            self.total.to_array()
            if (self.projected is None and hasattr(self, "total")) or from_total
            else self.projected.to_array()
        )
        spins = (0,) if self.is_non_collinear else (0, 1)

        # Resolve selection parameters groups.
        param_dicts = expand_grouped_params_to_dicts(
            dict(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
        )

        results = []
        for params in param_dicts:
            # Resolve Projection Selection
            selection = self._selection_resolver.resolve(
                atoms=params["atoms"],
                orbitals=params["orbitals"],
                spins=params["spins"],
                species=params["species"],
                species_orbital_map=params["species_orbital_map"],
                atoms_orbital_map=params["atoms_orbital_map"],
            )

            # Compute Magnetization.
            # First sum over projections and then compute magnetization.

            # Sum Over Projections
            components = self.sum_projection_components(
                values_array=dos_array,
                atoms=selection.atoms,
                orbitals=selection.orbitals,
                spins=selection.spins,
                **keep_func_kwargs(kwargs, self.sum_projection_components),
            )

            # Compute Magnetization
            if self.is_spin_polarized:
                magnetization_array = components[:, 0, ...] - components[:, 1, ...]
                if kwargs.get("keepdims", False):
                    magnetization_array = magnetization_array[:, np.newaxis, ...]
            else:
                magnetization_array = components

            if magnetization_array.ndim == 1:
                magnetization_array = magnetization_array[..., np.newaxis]

            # Build Property
            prop = build_property(
                self,
                values=magnetization_array,
                label=label,
                name=name,
                units=units,
                selection=selection,
                norm_mode=norm_mode,
                allowed_norm_modes={
                    NormMode.RAW,
                    NormMode.MAGNETIZATION,
                    NormMode.INTEGRAL,
                    NormMode.ELECTRONS,
                },
                include_normal_label=kwargs.get("include_normal_label", False),
                normalize_kwargs={**kwargs, "spins": selection.spins},
            )

            results.append(prop)

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
        label: str = "Spin Texture Magnitude",
        name: str = "spin_texture_magnitude",
        units: str = "$\\frac{states}{eV}$",
        from_total: bool = False,
        **kwargs,
    ) -> Property | list[Property]:
        """Compute spin texture magnitude ||S|| for non-collinear calculations.

        The spin texture magnitude is the norm of the spin texture vector:
        ||S(E)|| = sqrt(S_x(E)^2 + S_y(E)^2 + S_z(E)^2)

        This represents the magnitude of spin polarization at each energy level,
        regardless of direction. Only available for non-collinear calculations.

        Parameters
        ----------
        atoms
            Atom indices to sum over. If None, sums over all atoms.
        orbitals
            Orbital indices to sum over. If None, sums over all orbitals.
        spins
            Spin components to include. Defaults to (1, 2, 3) for S_x, S_y, S_z.
            Must be valid non-collinear spin channels [1, 2, 3].
        species
            Species names to select atoms by.
        species_orbital_map
            Mapping of species to orbital indices.
        atoms_orbital_map
            Mapping of atom indices to orbital indices.
        norm_mode
            Normalization mode. Allowed: raw, integral, spin_magnitude, electrons, magnetization.
        from_total
            If True, use total DOS even if projected is available.
        label
            Scalar label for the property.
        name
            Base name for the property.
        units
            Default units string before normalization.
        **kwargs
            Additional kwargs. Passed to sum_projection_components and _build_property.

        Returns
        -------
        Property | list[Property]
            Single Property if one result, list if multiple parameter combinations.

        Raises
        ------
        ValueError
            If calculation is not non-collinear, or if invalid spins are specified.
        """
        # Validate Input
        if not (hasattr(self, "total") or hasattr(self, "projected")):
            raise ValueError("Total or projected DOS is not provided")
        if not self.is_non_collinear:
            raise ValueError("DOS is not non-collinear")

        # Resolve default values
        dos_array = (
            self.total.to_array()
            if (self.projected is None and hasattr(self, "total")) or from_total
            else self.projected.to_array()
        )
        spins = (1, 2, 3) if spins is None else spins

        # Validate Spin Selection
        for spin in spins:
            if isinstance(spin, Iterable):
                invalid_spins = set(spins) - {1, 2, 3}
                if len(invalid_spins) > 0:
                    raise ValueError(
                        f"Invalid spins for spin texture magnitude: {sorted(invalid_spins)}. Valid components are [1, 2, 3]."
                    )

        # Resolve selection parameters groups.
        param_dicts = expand_grouped_params_to_dicts(
            dict(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
        )

        results = []
        for params in param_dicts:
            # Resolve Projection Selection
            selection = self._selection_resolver.resolve(
                atoms=params["atoms"],
                orbitals=params["orbitals"],
                spins=params["spins"],
                species=params["species"],
                species_orbital_map=params["species_orbital_map"],
                atoms_orbital_map=params["atoms_orbital_map"],
            )

            # Sum Atomic Projection Components
            values = self.sum_projection_components(
                values_array=dos_array,
                atoms=selection.atoms,
                orbitals=selection.orbitals,
                spins=selection.spins,
                **keep_func_kwargs(kwargs, self.sum_projection_components),
            )
            values = np.linalg.norm(values, axis=-1, keepdims=kwargs.get("keepdims", False))

            values = values[..., np.newaxis] if values.ndim == 1 else values

            prop = build_property(
                self,
                values=values,
                label=label,
                name=name,
                units=units,
                selection=selection,
                norm_mode=norm_mode,
                allowed_norm_modes={
                    NormMode.INTEGRAL,
                    NormMode.SPIN_MAGNITUDE,
                    NormMode.ELECTRONS,
                    NormMode.MAGNETIZATION,
                    NormMode.RAW,
                },
                include_normal_label=kwargs.get("include_normal_label", False),
                normalize_kwargs=kwargs,
            )
            results.append(prop)

        return results[0] if len(results) == 1 else results

    def compute_normalized_total(
        self,
        norm_mode: str | NormMode = "max",
        label: str = "Normalized Total",
        name: str = "normalized_total",
        units: str = "$\\frac{states}{eV}$",
        **kwargs,
    ) -> Property:
        """Compute normalized total DOS.

        Applies the specified normalization mode to the total DOS.
        Common normalization modes include max (normalize by maximum value),
        integral (normalize by integral), and electrons (normalize by total
        electron count).

        Parameters
        ----------
        norm_mode
            Normalization mode to apply. Defaults to "max".
        label
            Scalar label for the property.
        name
            Base name for the property.
        units
            Default units string before normalization.
        **kwargs
            Additional kwargs passed to normalize method.

        Returns
        -------
        Property
            Normalized total DOS as a Property.
        """
        if not (hasattr(self, "total") and self.total is not None):
            raise ValueError("Total DOS is not provided")

        # Resolve Default Values
        name = self.total.name if name is None else name
        label = self.total.label if label is None else label
        units = self.total.units if units is None else units

        prop = build_property(
            self,
            values=self.total.to_array(),
            label=label,
            name=name,
            units=units,
            norm_mode=norm_mode,
            allowed_norm_modes={NormMode.MAX, NormMode.INTEGRAL, NormMode.ELECTRONS},
            include_normal_label=kwargs.get("include_normal_label", False),
            normalize_kwargs=kwargs,
        )

        return prop

    def compute_cumulative_total(
        self,
        norm_mode: str | NormMode = None,
        label: str = "Cumulative Total",
        name: str = "cumulative_total",
        units: str = "$\\frac{states}{eV}$",
        **kwargs,
    ) -> Property:
        """Compute cumulative total DOS.

        Computes the cumulative sum (integral) of the total DOS:
        C(E) = ∫_{-∞}^{E} DOS(E') dE'

        Optionally applies normalization after computing the cumulative sum.

        Parameters
        ----------
        norm_mode
            Normalization mode to apply after computing cumulative sum.
            If None, uses raw values.
        label
            Scalar label for the property.
        name
            Base name for the property.
        units
            Default units string before normalization.
        **kwargs
            Additional kwargs passed to normalize method.

        Returns
        -------
        Property
            Cumulative total DOS as a Property.
        """
        # Validate Input
        if not (hasattr(self, "total") and self.total is not None):
            raise ValueError("Total DOS is not provided")

        # Compute Cumulative Total
        cumlative_total = self.cumsum(values=self.total.to_array())

        # Build Property
        prop = build_property(
            self,
            values=cumlative_total,
            label=label,
            name=name,
            units=units,
            norm_mode=norm_mode,
            allowed_norm_modes={NormMode.MAX, NormMode.INTEGRAL, NormMode.ELECTRONS, NormMode.RAW},
            include_normal_label=kwargs.get("include_normal_label", False),
            normalize_kwargs=kwargs,
        )

        return prop

    def compute_property(self, name: str, **kwargs):
        """Compute a property.

        Parameters
        ----------
        name: str
            The name of the property to compute.
        **kwargs: dict[str, Any]
            Additional kwargs passed to the compute_property method.

        Returns
        -------
        Property | None
            The computed property.
        """
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
        **kwargs,
    ) -> Property:
        """Attach a custom property to the DOS object.

        Users may supply an existing :class:`Property` instance or provide a
        ``name`` and array ``value`` whose first axis matches the energy grid.

        Parameters
        ----------
        property: Property | None
            The property to add. Can be a Property object or None.
        name: str | None
            The name of the property to add.
        value: npt.ArrayLike | None
            The value of the property to add.
        **kwargs: dict[str, Any]
            Additional kwargs passed to the add_property method.

        Returns
        -------
        Property
            The added Property object.
        """
        # Validate Input
        assert property is not None or (name is not None and value is not None), (
            "Either a Property instance or name/value are required."
        )

        # Add Property if provided
        if property is not None:
            self.validate_property_points(property)
            super().add_property(property=property, **kwargs)
            return self.property_store[property.name]

        # Validate Name and Value
        assert name is not None and value is not None, (
            "Both name and value are required when not supplying a Property instance."
        )

        # Convert Value to Array
        value_array = np.asarray(value, dtype=np.float64)
        assert value_array.ndim > 0, (
            f"Property values must have at least one dimension aligned with energies. Expected 1 dimension, got {value_array.ndim} dimensions."
        )

        assert value_array.shape[0] == self.n_energies, (
            f"Property values must share the DOS energy grid along the first axis. Expected {self.n_energies} points, got {value_array.shape[0]} points."
        )

        super().add_property(name=name, value=value_array, **kwargs)
        return self.property_store[name]

    # ------------------------------------------------------------------
    # Basis helpers
    # ------------------------------------------------------------------

    def get_current_basis(self) -> str:
        """Get the current basis of the DOS.

        Returns
        -------
        str
            The current basis of the DOS.
        """
        assert hasattr(self, "projected") and self.projected is not None, (
            "Projected DOS is not provided"
        )
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

    def save(self, path: Path | str) -> None:
        """Save the DOS to a file.

        Parameters
        ----------
        path: Path
            The path to save the DOS to.
        """
        assert isinstance(path, (Path, str)), "Path must be a Path or string."
        path = Path(path)
        serializer = get_serializer(path)
        serializer.save(self, path)

    @classmethod
    def load(cls, path: Path | str) -> DensityOfStates:
        """Load the DOS from a file.

        Parameters
        ----------
        path: Path | str
            The path to load the DOS from.

        Returns
        -------
        DensityOfStates
            The loaded DOS.
        """
        assert isinstance(path, (Path, str)), "Path must be a Path or string."

        path = Path(path)
        serializer = get_serializer(path)
        return serializer.load(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_total(self, total: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Validate the total DOS.

        Parameters
        ----------
        total: npt.ArrayLike
            The total DOS to validate.
        """
        assert total is not None, "Total DOS is required"
        total_array = np.asarray(total, dtype=np.float64)
        if total_array.ndim == 1:
            total_array = total_array[:, np.newaxis]
        if total_array.shape[0] != self.n_energies:
            raise ValueError("Total DOS must have the same number of energies")
        return total_array

    def _validate_projected(self, projected: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Validate the projected DOS.

        Parameters
        ----------
        projected: npt.ArrayLike
            The projected DOS to validate.
        """
        assert projected is not None, "Projected DOS is not provided"
        projected_array = np.asarray(projected, dtype=np.float64)

        if (
            projected_array.shape[0] != self.n_energies
            and projected_array.shape[-1] == self.n_energies
        ):
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

    @cached_property
    def _selection_resolver(self) -> ProjectionSelectionResolver:
        return selection_resolver(self)

    def _validate_indices(
        self,
        indices: Sequence[int] | None,
        upper_bound: int,
    ) -> np.ndarray | None:
        """Validate the indices.

        Parameters
        ----------
        indices: Sequence[int] | None
            The indices to validate.
        upper_bound: int
            The upper bound of the indices.

        Returns
        -------
        np.ndarray | None
            The validated indices.
        """
        if indices is None:
            return None
        arr = np.asarray(list(indices), dtype=int).ravel()
        if arr.size == 0:
            return np.array([], dtype=int)
        arr = np.where(arr < 0, upper_bound + arr, arr)
        if np.any((arr < 0) | (arr >= upper_bound)):
            raise IndexError("Index out of bounds")
        return np.unique(arr)


def interpolate(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    factor: int = 2,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Interpolate ``y`` over ``x`` by increasing the sample count.

    Parameters
    ----------
    x: npt.ArrayLike
        The x values to interpolate.
    y: npt.ArrayLike
        The y values to interpolate.
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    cs = CubicSpline(x_array, y_array, axis=0)
    xs = np.linspace(float(np.min(x_array)), float(np.max(x_array)), len(x_array) * factor)
    ys = cs(xs)
    return xs, ys


def filter_data_within_sigma(
    data: npt.NDArray[np.float64], sigma: float = 3, fill_value: float | None = None
) -> npt.NDArray[np.float64]:
    """Filter the data within sigma of the mean.

    Parameters
    ----------
    data: npt.NDArray[np.float64]
        The data to filter.
    sigma: float
        The number of standard deviations to filter.
    """
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
