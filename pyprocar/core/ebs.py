"""
Created on Sat Jan 16 2021

@author: Logan Lang
@author: Pedram Tavadze
@author: Freddy Farah

"""

from __future__ import annotations

import copy
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyvista as pv
from typing_extensions import override

import pyprocar.core.kpoints as kpoints
from pyprocar.core.atomic_orbital_index import (
    PRIMARY_ORBITAL_GROUPS,
    AtomIndexer,
    OrbitalIndexer,
    ProjectionLabelBuilder,
    ProjectionSelectionResolver,
    ProjectionSelectionResult,
    SpinIndexer,
)
from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.core.property_store import PointSet, Property
from pyprocar.core.serializer import get_serializer
from pyprocar.core.structure import Structure
from pyprocar.utils import math, np_utils, physics
from pyprocar.utils.math import np_round_to_half
from pyprocar.utils.unfolder import Unfolder

pv.global_theme.allow_empty_mesh = True

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

NUMERICAL_STABILITY_FACTOR = 0.0001


BANDS_DTYPE = np.ndarray[tuple[int, int, int], np.dtype[np_utils.FLOAT_DTYPE]]
PROJECTED_DTYPE = np.ndarray[tuple[int, int, int, int, int], np.dtype[np_utils.FLOAT_DTYPE]]
PROJECTED_PHASE_DTYPE = np.ndarray[tuple[int, int, int, int, int], np.dtype[np_utils.FLOAT_DTYPE]]
WEIGHTS_DTYPE = np.ndarray[tuple[int, int], np.dtype[np_utils.FLOAT_DTYPE]]


class EBSNormMode(Enum):
    """Normalization modes for ElectronicBandStructure projections.

    Unlike DOS, band energies should NOT be normalized. These modes
    apply only to projection weights (projected, projected_sum, etc.).
    """

    RAW = "raw"
    MAX = "max"
    TOTAL = "total"
    TOTAL_PROJECTION = "total_projection"

    @classmethod
    def from_input(cls, input: str | EBSNormMode | None) -> EBSNormMode:  # noqa: A002
        if isinstance(input, EBSNormMode):
            return input
        if input is None:
            return cls.RAW
        # input must be str at this point due to the union type
        lower_input = input.lower()
        mode_map = {
            "raw": cls.RAW,
            "max": cls.MAX,
            "total": cls.TOTAL,
            "total_projection": cls.TOTAL_PROJECTION,
        }

        if lower_input in mode_map:
            return mode_map[lower_input]

        valid_modes = ", ".join(mode_map.keys())
        raise ValueError(f"Invalid normalization mode: {input}. Valid modes: {valid_modes}")

    @classmethod
    def list_modes(cls) -> list[str]:
        return [mode.value for mode in cls]

    @classmethod
    def get_mode_prefix(cls, mode: EBSNormMode) -> str:
        mode = cls.from_input(mode)
        prefixes = {
            cls.RAW: "",
            cls.MAX: "Max-Normed",
            cls.TOTAL: "Total-Normed",
            cls.TOTAL_PROJECTION: "Total-Projection-Normed",
        }
        return prefixes.get(mode, "")

    @classmethod
    def get_normed_name(cls, mode: EBSNormMode, name: str) -> str:
        mode = cls.from_input(mode)
        prefix = cls.get_mode_prefix(mode)
        if prefix:
            return f"{prefix} {name}"
        return name

    @classmethod
    def get_normed_units(cls, mode: EBSNormMode) -> str:
        """Return units after normalization.

        Projection weights are dimensionless, so normalization
        always results in dimensionless output.
        """
        mode = cls.from_input(mode)
        if mode is cls.RAW:
            return ""  # dimensionless weights
        return "$1$"  # normalized to dimensionless

    @classmethod
    def get_mode_footnote(cls, mode: EBSNormMode) -> str:
        mode = cls.from_input(mode)
        footnotes = {
            cls.RAW: "",
            cls.MAX: "Normalized by maximum projection weight",
            cls.TOTAL: "Normalized by total projection",
            cls.TOTAL_PROJECTION: "Normalized by total projected weight",
        }
        return footnotes.get(mode, "")


def get_ebs_from_data(
    kpoints: kpoints.KPOINTS_DTYPE | None = None,
    bands: BANDS_DTYPE | None = None,
    projected: PROJECTED_DTYPE | None = None,
    projected_phase: PROJECTED_PHASE_DTYPE | None = None,
    weights: WEIGHTS_DTYPE | None = None,
    fermi: float = 0.0,
    reciprocal_lattice: kpoints.RECIPROCAL_LATTICE_DTYPE | None = None,
    orbital_names: list[str] | None = None,
    structure: Structure | None = None,
    kpath: kpoints.KPath | None = None,
    kgrid_info: kpoints.KGridInfo | None = None,
    **_kwargs: Any,
) -> ElectronicBandStructure:
    if kpath is not None:
        logger.debug("Creating ElectronicBandStructurePath from EBS")
        return ElectronicBandStructurePath(
            kpoints=kpoints,
            bands=bands,
            projected=projected,
            projected_phase=projected_phase,
            weights=weights,
            fermi=fermi,
            reciprocal_lattice=reciprocal_lattice,
            orbital_names=orbital_names,
            structure=structure,
            kpath=kpath,
        )

    if kgrid_info is not None:
        logger.debug("Creating ElectronicBandStructureMesh from EBS")
        return ElectronicBandStructureMesh(
            kpoints=kpoints,
            bands=bands,
            projected=projected,
            projected_phase=projected_phase,
            weights=weights,
            fermi=fermi,
            reciprocal_lattice=reciprocal_lattice,
            orbital_names=orbital_names,
            structure=structure,
            kgrid_info=kgrid_info,
        )

    logger.debug("Creating ElectronicBandStructure from EBS")
    return ElectronicBandStructure(
        kpoints=kpoints,
        bands=bands,
        projected=projected,
        projected_phase=projected_phase,
        weights=weights,
        fermi=fermi,
        reciprocal_lattice=reciprocal_lattice,
        orbital_names=orbital_names,
        structure=structure,
    )


def get_ebs_from_code(
    code: str,
    dirpath: str,
    use_cache: bool = False,
    ebs_filename: str = "ebs.pkl",
    **_kwargs: Any,
) -> ElectronicBandStructure:
    from pyprocar.io import Parser

    ebs_filepath = Path(dirpath) / ebs_filename

    ebs: ElectronicBandStructure
    if not use_cache or not ebs_filepath.exists():
        logger.info(f"Parsing EBS calculation directory: {dirpath}")
        parser = Parser(code=code, dirpath=dirpath)
        parser_ebs = parser.ebs
        if parser_ebs is None:
            raise ValueError("Parser returned no EBS data")
        ebs = parser_ebs
        ebs.save(ebs_filepath)
    else:
        logger.info(f"Loading EBS  from picklefile: {ebs_filepath}")
        ebs = ElectronicBandStructure.load(ebs_filepath)

    return ebs


# Property key compatible with PointSet.get_property
PropertyKey = str | tuple[str, int] | tuple[str, str] | tuple[str, str, int]


class DifferentiablePropertyInterface(ABC):
    @abstractmethod
    def gradient_func(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_property(self, key: PropertyKey, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def add_property(self, label: str, value: npt.NDArray[Any], **_kwargs: Any) -> None:
        raise NotImplementedError

    def compute_band_velocity(self, **_kwargs: Any) -> npt.NDArray[np.float64]:
        logger.info("Computing band velocity")
        bands_gradient: npt.NDArray[np.float64] = self.get_property(("bands", "gradients", 1))
        return physics.calculate_band_velocity(bands_gradient)

    def compute_band_speed(self, **kwargs: Any) -> npt.NDArray[np.float64]:
        logger.info("Computing band speed")
        band_velocity = self.compute_band_velocity(**kwargs)
        return physics.calculate_band_speed(band_velocity)

    def compute_avg_inv_effective_mass(self, **_kwargs: Any) -> npt.NDArray[np.float64]:
        logger.info("Computing average inverse effective mass")
        bands_hessian: npt.NDArray[np.float64] = self.get_property(("bands", "gradients", 2))
        return physics.calculate_avg_inv_effective_mass(bands_hessian)


class PyvistaInterface(ABC):
    _mesh: pv.PolyData | pv.StructuredGrid | Any | None  # pv.PointSet not typed

    @property
    @abstractmethod
    def kpoints_cartesian(self) -> npt.NDArray[np.float64] | None:
        """Return cartesian kpoints."""
        ...

    @property
    @abstractmethod
    def kpoints(self) -> npt.NDArray[np.float64]:
        """Return kpoints."""
        ...

    def mesh(self) -> pv.PolyData | pv.StructuredGrid | Any | None:  # pv.PointSet not typed
        if self._mesh is None:
            self.to_mesh()
        return self._mesh

    def mesh_as_cart(self) -> None:
        if self._mesh is not None and self.kpoints_cartesian is not None:
            self._mesh.points = self.kpoints_cartesian

    def mesh_as_frac(self) -> None:
        if self._mesh is not None:
            self._mesh.points = self.kpoints

    @abstractmethod
    def to_mesh(
        self,
        active_scalar: tuple[str, npt.NDArray[Any]] | None = None,
        active_vector: tuple[str, npt.NDArray[Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def set_mesh_scalar(self, name: str, scalar: npt.NDArray[Any]) -> None:
        if self._mesh is not None:
            self._mesh.point_data[name] = scalar
            self._mesh.set_active_scalars(name)

    def set_mesh_vector(self, name: str, vector: npt.NDArray[Any]) -> None:
        if self._mesh is not None:
            self._mesh.point_data[name] = vector
            self._mesh.set_active_vectors(name)


class ElectronicBandStructure(PointSet, PyvistaInterface):
    """This object stores electronic band structure informomration.

    Parameters
    ----------
    kpoints : np.ndarray
        The kpoints array. Will have the shape (n_kpoints, 3)
    bands : np.ndarray
        The bands array. Will have the shape (n_kpoints, n_bands)
    fermi : float
        The fermi energy
    projected : np.ndarray, optional
        The projections array. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
    projected_phase : np.ndarray, optional
        The full projections array that incudes the complex part. Will have the shape (n_kpoints, n_bands, n_spins, norbitals,n_atoms), defaults to None
    weights : np.ndarray, optional
        The weights of the kpoints. Will have the shape (n_kpoints, 1), defaults to None
    orbital_names : list, optional
        The names of the orbitals. Defaults to None
    reciprocal_lattice : np.ndarray, optional
        The reciprocal lattice vector matrix. Will have the shape (3, 3), defaults to None
    shifted_to_fermi : bool, optional
         Boolean to determine if the fermi energy is shifted, defaults to False
    """

    _fermi: float
    _orbital_names: list[str] | None
    _reciprocal_lattice: kpoints.RECIPROCAL_LATTICE_DTYPE | None
    _shifted_to_fermi: bool
    _structure: Structure | None
    _mesh: pv.PolyData | pv.StructuredGrid | Any | None

    def __init__(
        self,
        kpoints: kpoints.KPOINTS_DTYPE | None = None,
        fermi: float = 0.0,
        bands: BANDS_DTYPE | None = None,
        projected: PROJECTED_DTYPE | None = None,
        projected_phase: PROJECTED_PHASE_DTYPE | None = None,
        weights: WEIGHTS_DTYPE | None = None,
        orbital_names: list[str] | None = None,
        reciprocal_lattice: kpoints.RECIPROCAL_LATTICE_DTYPE | None = None,
        shifted_to_fermi: bool = False,
        structure: Structure | None = None,
    ):
        # kpoints=None results in an empty array via np.array() in PointSet
        super().__init__(kpoints)  # pyright: ignore[reportArgumentType]

        logger.info("Initializing ElectronicBandStructure")

        if bands is not None:
            self.add_property(name="bands", value=bands)
        if projected is not None:
            self.add_property(name="projected", value=projected)
        if projected_phase is not None:
            self.add_property(name="projected_phase", value=projected_phase)
        if weights is not None:
            self.add_property(name="weights", value=weights)

        self._fermi = fermi
        self._orbital_names = orbital_names
        self._reciprocal_lattice = reciprocal_lattice
        self._shifted_to_fermi = shifted_to_fermi
        self._structure = structure

        # Projection selection infrastructure (lazy initialized)
        self._projection_label_builder: ProjectionLabelBuilder | None = None
        self._projection_selection_resolver: ProjectionSelectionResolver | None = None

        logger.info("___ElectronicBandStructure initialization complete___")

    @override
    def __str__(self) -> str:
        ret = "\n Electronic Band Structure     \n"
        ret += "============================\n"
        ret += f"Total number of kpoints   = {self.n_kpoints}\n"

        ret += "\nAdditional information: \n"
        if self.orbital_names is not None:
            ret += f"Orbital Names = {self.orbital_names}\n"

        if "projected" in self.property_store:
            ret += f"Spin Projection Names = {self.spin_projection_names}\n"
            ret += f"Non-colinear = {self.is_non_collinear}\n"
        else:
            ret += "Spin Projection Names = None\n"

        if self.reciprocal_lattice is not None:
            ret += f"Reciprocal Lattice = \n {self.reciprocal_lattice}\n"
        ret += f"Fermi Energy = {self.fermi}\n"
        if self.structure is not None:
            ret += "\nStructure: \n"
            ret += "------------------------     \n"
            ret += f"Structure = \n {self.structure}\n"

        return ret

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ElectronicBandStructure):
            return False

        kpoints_equal = math.compare_arrays(self.kpoints, other.kpoints)
        fermi_equal = self.fermi == other.fermi

        # Helper to extract array from Property or use directly
        def _get_array(
            prop: Property | npt.NDArray[np.float64] | None,
        ) -> npt.NDArray[np.float64] | None:
            if prop is None:
                return None
            if isinstance(prop, Property):
                return prop.to_array()
            return prop

        bands_equal = math.compare_arrays(_get_array(self.bands), _get_array(other.bands))
        projected_equal = math.compare_arrays(_get_array(self.projected), _get_array(other.projected))
        projected_phase_equal = math.compare_arrays(
            _get_array(self.projected_phase), _get_array(other.projected_phase)
        )
        weights_equal = math.compare_arrays(_get_array(self.weights), _get_array(other.weights))

        is_ebs_equal = (
            kpoints_equal
            and bands_equal
            and fermi_equal
            and projected_equal
            and projected_phase_equal
            and weights_equal
        )
        return is_ebs_equal

    @property
    @override
    def kpoints(self) -> npt.NDArray[np.float64]:
        return self._points

    @property
    def bands(self) -> Property | None:
        prop = self.get_property("bands")
        return prop if isinstance(prop, Property) else None

    @property
    def projected(self) -> Property | None:
        prop = self.get_property("projected")
        return prop if isinstance(prop, Property) else None

    @property
    def projected_phase(self) -> Property | None:
        prop = self.get_property("projected_phase")
        return prop if isinstance(prop, Property) else None

    @property
    def weights(self) -> Property | None:
        prop = self.get_property("weights")
        return prop if isinstance(prop, Property) else None

    @property
    def orbital_names(self) -> list[str] | None:
        return self._orbital_names

    @property
    def reciprocal_lattice(self) -> kpoints.RECIPROCAL_LATTICE_DTYPE | None:
        return self._reciprocal_lattice

    @property
    def brillouin_zone(self) -> BrillouinZone:
        # reciprocal_lattice may be None, but BrillouinZone handles it
        return BrillouinZone(self.reciprocal_lattice, np.array([1, 1, 1]))  # pyright: ignore[reportArgumentType]

    @property
    def fermi(self) -> float:
        return self._fermi

    @property
    def structure(self) -> Structure | None:
        return self._structure

    @property
    def n_kpoints(self) -> int:
        """The number of k points

        Returns
        -------
        int
            The number of k points
        """
        return int(self.kpoints.shape[0])

    @property
    def n_bands(self) -> int:
        """The number of bands

        Returns
        -------
        int
            The number of bands
        """
        if self.bands is None:
            raise ValueError("bands not set")
        return int(self.bands.shape[1])

    @property
    def n_spins(self) -> int:
        """The number of spin projections

        Returns
        -------
        int
            The number of spin projections
        """
        if self.projected is None:
            raise ValueError("projected not set")
        return int(self.projected.shape[2])

    @property
    def n_atoms(self) -> int:
        """The number of atoms

        Returns
        -------
        int
            The number of atoms
        """
        if self.projected is None:
            raise ValueError("projected not set")
        return int(self.projected.shape[3])

    @property
    def n_orbitals(self) -> int:
        """The number of orbitals

        Returns
        -------
        int
            The number of orbitals
        """
        if self.projected is None:
            raise ValueError("projected not set")
        return int(self.projected.shape[4])

    @property
    def n_spin_channels(self) -> int:
        """The number of spin channels

        Returns
        -------
        int
            The number of spin channels
        """
        if self.bands is None:
            raise ValueError("bands not set")
        return int(self.bands.shape[2])

    @property
    def spin_channels(self) -> npt.NDArray[np.intp]:
        """The number of spin channels

        Returns
        -------
        npt.NDArray[np.intp]
            Array of spin channel indices
        """

        return np.arange(self.n_spin_channels)

    @property
    def spin_projection_names(self) -> list[str]:
        spin_projection_names = ["Spin-up", "Spin-down"]
        if self.is_non_collinear:
            return ["total", "x", "y", "z"]
        elif self.n_spins == 2:
            return spin_projection_names
        else:
            return spin_projection_names[:1]

    @property
    @override
    def kpoints_cartesian(self) -> npt.NDArray[np.float64] | None:
        return kpoints.reduced_to_cartesian(self.kpoints, self.reciprocal_lattice)

    @property
    def inv_reciprocal_lattice(self) -> npt.NDArray[np.float64] | None:
        """Returns the inverse of the reciprocal lattice"""
        if self.reciprocal_lattice is not None:
            result = np.linalg.inv(self.reciprocal_lattice)
            return np.asarray(result, dtype=np.float64)
        else:
            print("Please provide a reciprocal lattice when initiating the Procar class")
            return None

    @property
    def is_grid(self) -> bool:
        grid_dims = math.get_grid_dims(self.kpoints)
        return bool(self.n_kpoints == np.prod(grid_dims))

    @property
    def is_non_collinear(self) -> bool:
        """Boolean to determine if this is a non-colinear calculation

        Returns
        -------
        bool
            Boolean to determine if this is a non-colinear calculation
        """
        if self.n_spins == 4:
            return True
        else:
            return False

    @property
    def is_spin_polarized(self) -> bool:
        return self.n_spin_channels == 2

    def has_spin_channels(self, property: Property) -> bool:
        property_value_shape = list(property.value.shape)
        if len(property_value_shape) >= 3:
            nspins = property_value_shape[2]
            return nspins == self.n_spin_channels
        else:
            return False

    def is_band_property(self, property: Property | npt.NDArray[Any]) -> bool:
        if isinstance(property, np.ndarray):
            property_value_shape = list(property.shape)
        else:
            property_value_shape = list(property.value.shape)
        if len(property_value_shape) >= 3:
            nbands = property_value_shape[1]
            return nbands == self.n_bands
        else:
            return False

    def is_orbital_property(self, property: Property) -> bool:
        property_value_shape = list(property.value.shape)
        if len(property_value_shape) >= 5:
            nkpoints, nbands, nspins, natoms, norbitals = (
                property_value_shape[0],
                property_value_shape[1],
                property_value_shape[2],
                property_value_shape[3],
                property_value_shape[4],
            )
            return (
                nkpoints == self.n_kpoints
                and nbands == self.n_bands
                and nspins == self.n_spin_channels
                and natoms == self.n_atoms
                and norbitals == self.n_orbitals
            )
        else:
            return False

    @property
    def has_phase(self) -> bool:
        """Boolean to determine if this is a phase calculation

        Returns
        -------
        bool
            Boolean to determine if this is a phase calculation
        """
        return self.projected_phase is not None

    @property
    def band_property_names(self) -> list[str]:
        names: list[str] = []
        for property_name in self.property_store.keys():
            property = self.get_property(property_name)
            if property is not None and self.is_band_property(property):
                names.append(property_name)
        return names

    @property
    def property_names(self) -> list[str]:
        names: list[str] = []
        for property_name in self.property_store.keys():
            names.append(property_name)
        return names

    @property
    def ebs_ipr(self) -> Property | None:
        prop = self.get_property("ebs_ipr")
        if isinstance(prop, Property):
            return prop

        return self.compute_ebs_ipr()

    @property
    def ebs_ipr_atom(self) -> Property | None:
        r"""Return the atom-resolved pIPR.

        pIPR_j = \frac{|c_j|^4}{(\sum_i |c_i^2|)^2}

        Clearly, \sum_j pIPR_j = IPR.

        Mind: c_i is the wavefunction c(n,k)_i, in pyprocar we already
        have density projections, c_i^2.

        *THIS QUANTITY IS NOT READY FOR PLOTTING*, please prefer `self.ebs_ipr()`

        Returns
        -------
        ret : Property | None
            The IPR projections as a Property object

        """
        prop = self.get_property("ebs_ipr_atom")
        if isinstance(prop, Property):
            return prop

        return self.compute_ebs_ipr_atom()

    @property
    def spin_texture(self) -> Property | None:
        prop = self.get_property("spin_texture")
        if isinstance(prop, Property):
            return prop

        return self.compute_spin_texture()

    @property
    def projected_sum(self) -> Property | None:
        prop = self.get_property("projected_sum")
        if isinstance(prop, Property):
            return prop

        return self.compute_projected_sum()

    @property
    def projected_sum_spin_texture(self) -> Property | None:
        prop = self.get_property("projected_sum_spin_texture")
        if isinstance(prop, Property):
            return prop

        return self.compute_projected_sum_spin_texture()

    @override
    def get_property(
        self, key: PropertyKey | None = None, **kwargs: Any
    ) -> Property | npt.NDArray[np.float64] | None:
        prop_name, _ = self._extract_key(key)  # pyright: ignore[reportArgumentType]
        if prop_name not in self.property_store:
            computed = self.compute_property(prop_name, **kwargs)
            if computed is not None:
                # compute_property now returns Property objects
                if isinstance(computed, Property):
                    self.property_store[prop_name] = computed
                elif isinstance(computed, np.ndarray):
                    # Fallback for any methods still returning arrays
                    self.add_property(name=prop_name, value=computed)
        return super().get_property(key)

    @override
    def to_mesh(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        scalars: tuple[str, npt.NDArray[Any]] | None = None,
        vectors: tuple[str, npt.NDArray[Any]] | None = None,
        as_cartesian: bool = True,
    ) -> Any:  # Returns pv.PointSet which is not typed
        if as_cartesian:
            mesh_points = self.kpoints_cartesian
        else:
            mesh_points = self.kpoints
        mesh = pv.PointSet(mesh_points)
        if scalars is not None:
            self.set_mesh_scalar(*scalars)
        if vectors is not None:
            self.set_mesh_vector(*vectors)
        self._mesh = mesh
        return mesh

    def compute_property(self, name: str, **kwargs: Any) -> Property | None:
        if name == "ebs_ipr":
            return self.compute_ebs_ipr(**kwargs)
        elif name == "ebs_ipr_atom":
            return self.compute_ebs_ipr_atom(**kwargs)
        elif name == "spin_texture":
            return self.compute_spin_texture(**kwargs)
        elif name == "projected_sum":
            return self.compute_projected_sum(**kwargs)
        elif name == "projected_sum_spin_texture":
            return self.compute_projected_sum_spin_texture(**kwargs)

    def compute_ebs_ipr(
        self,
        norm_mode: str | EBSNormMode | None = "raw",
        label: str = "IPR",
        name: str = "ebs_ipr",
        **_kwargs: Any,
    ) -> Property:
        """Compute Inverse Participation Ratio (IPR).

        IPR = sum(|c_i|^4) / (sum(|c_i|^2))^2

        Parameters
        ----------
        norm_mode : str | EBSNormMode | None
            Normalization mode
        label : str
            Property label
        name : str
            Property name

        Returns
        -------
        Property
            Property object with IPR values and metadata
        """
        if self.projected is None:
            raise ValueError("projected not set")

        projected_array = self.projected.to_array()
        orbitals = np.arange(self.n_orbitals, dtype=int)
        # sum over orbitals
        proj: npt.NDArray[np.float64] = np.sum(projected_array[:, :, :, :, orbitals], axis=-1)

        atoms = np.arange(self.n_atoms, dtype=int)
        # IPR = sum(|c_i|^4) / (sum(|c_i|^2))^2
        num = np.absolute(proj) ** 2
        num = np.sum(num[:, :, :, atoms], axis=-1)

        den = np.absolute(proj) ** 1 + NUMERICAL_STABILITY_FACTOR
        den = np.sum(den[:, :, :, atoms], axis=-1) ** 2

        values: npt.NDArray[np.float64] = num / den

        prop = self._build_property(
            values=values,
            label=label,
            name=name,
            norm_mode=norm_mode,
            selection=None,
            description="Inverse Participation Ratio",
            formula="IPR = sum(|c_i|^4) / (sum(|c_i|^2))^2",
        )

        return prop

    def compute_ebs_ipr_atom(
        self,
        norm_mode: str | EBSNormMode | None = "raw",
        label: str = "Partial IPR",
        name: str = "ebs_ipr_atom",
        **_kwargs: Any,
    ) -> Property:
        """Compute atom-resolved partial Inverse Participation Ratio.

        pIPR_j = |c_j|^4 / (sum(|c_i|^2))^2

        Parameters
        ----------
        norm_mode : str | EBSNormMode | None
            Normalization mode
        label : str
            Property label
        name : str
            Property name

        Returns
        -------
        Property
            Property object with partial IPR values, shape includes atom dimension
        """
        if self.projected is None:
            raise ValueError("projected not set")

        projected_array = self.projected.to_array()
        orbitals = np.arange(self.n_orbitals, dtype=int)
        proj: npt.NDArray[np.float64] = np.sum(projected_array[:, :, :, :, orbitals], axis=-1)

        atoms = np.arange(self.n_atoms, dtype=int)
        num = np.absolute(proj) ** 2
        den = np.absolute(proj)
        den = np.sum(den[..., atoms], axis=-1) ** 2

        values: npt.NDArray[np.float64] = num / den[..., np.newaxis]

        prop = self._build_property(
            values=values,
            label=label,
            name=name,
            norm_mode=norm_mode,
            selection=None,
            description="Atom-resolved partial Inverse Participation Ratio",
            formula=r"pIPR_j = |c_j|^4 / (sum(|c_i|^2))^2",
        )

        return prop

    def compute_projected_sum(
        self,
        atoms: Sequence[int] | int | None = None,
        orbitals: Sequence[int] | int | None = None,
        spins: Sequence[int] | int | None = None,
        species: Sequence[str] | str | None = None,
        species_orbital_map: Sequence[Mapping[str, Iterable[int]]]
        | Mapping[str, Iterable[int]]
        | None = None,
        atoms_orbital_map: Sequence[Mapping[Iterable[int] | int, Iterable[int]]]
        | Mapping[Iterable[int] | int, Iterable[int]]
        | None = None,
        norm_mode: str | EBSNormMode | None = "raw",
        label: str = "Projected Sum",
        name: str = "projected_sum",
    ) -> Property:
        """Compute summed projections over specified atoms/orbitals/spins.

        Parameters
        ----------
        atoms : Sequence[int] | int | None
            Atom indices to sum over
        orbitals : Sequence[int] | int | None
            Orbital indices to sum over
        spins : Sequence[int] | int | None
            Spin channels to sum over
        species : Sequence[str] | str | None
            Species names (resolved to atom indices)
        species_orbital_map : Mapping | None
            Species to orbital mapping
        atoms_orbital_map : Mapping | None
            Atoms to orbital mapping
        norm_mode : str | EBSNormMode | None
            Normalization mode
        label : str
            Property label
        name : str
            Property name

        Returns
        -------
        Property
            Property object with summed projections and metadata
        """
        # Check if any selection is specified
        has_selection = any(
            x is not None
            for x in [atoms, orbitals, spins, species, species_orbital_map, atoms_orbital_map]
        )

        selection = None
        atoms_list = None
        orbitals_list = None
        spins_list = None

        if has_selection:
            # Resolve selection
            selection = self._resolve_projection_selection(
                atoms=atoms,
                orbitals=orbitals,
                spins=spins,
                species=species,
                species_orbital_map=species_orbital_map,
                atoms_orbital_map=atoms_orbital_map,
            )
            atoms_list = list(selection.atoms) if selection.atoms else None
            orbitals_list = list(selection.orbitals) if selection.orbitals else None
            spins_list = list(selection.spins) if selection.spins else None

        # Compute sum using resolved indices
        values = self.ebs_sum(
            atoms=atoms_list,
            orbitals=orbitals_list,
            spins=spins_list,
            sum_noncolinear=True,
        )

        # Build property with metadata
        prop = self._build_property(
            values=values,
            label=label,
            name=name,
            norm_mode=norm_mode,
            selection=selection,
            atoms=atoms_list,
            orbitals=orbitals_list,
            spins=spins_list,
        )

        return prop

    def compute_spin_texture(
        self,
        label: str = "Spin Texture",
        name: str = "spin_texture",
        **_kwargs: Any,
    ) -> Property:
        """Compute spin texture for non-collinear calculations.

        Returns
        -------
        Property
            Property object with spin texture (Sx, Sy, Sz components)

        Raises
        ------
        ValueError
            If calculation is not non-collinear
        """
        if not self.is_non_collinear:
            raise ValueError("Spin texture is only available for non-collinear calculations")

        if self.projected is None:
            raise ValueError("projected not set")

        # Extract spin components (indices 1,2,3 = Sx,Sy,Sz)
        projected_array = self.projected.to_array()
        values: npt.NDArray[np.float64] = projected_array[:, :, 1:, :, :]
        values = np.moveaxis(values, 2, -1)

        prop = self._build_property(
            values=values,
            label=label,
            name=name,
            norm_mode="raw",  # Spin texture should not be normalized
            selection=None,
            description="Spin texture components (Sx, Sy, Sz)",
            is_non_collinear=True,
        )

        return prop

    def compute_projected_sum_spin_texture(
        self,
        atoms: Sequence[int] | int | None = None,
        orbitals: Sequence[int] | int | None = None,
        species: Sequence[str] | str | None = None,
        label: str = "Projected Spin Texture",
        name: str = "projected_sum_spin_texture",
        **kwargs: Any,
    ) -> Property:
        """Compute summed projections with spin texture for non-collinear calculations.

        Parameters
        ----------
        atoms : Sequence[int] | int | None
            Atom indices to sum over
        orbitals : Sequence[int] | int | None
            Orbital indices to sum over
        species : Sequence[str] | str | None
            Species names
        label : str
            Property label
        name : str
            Property name

        Returns
        -------
        Property
            Property object with projected spin texture

        Raises
        ------
        ValueError
            If calculation is not non-collinear
        """
        if not self.is_non_collinear:
            raise ValueError("Spin texture is only available for non-collinear calculations")

        # Check if any selection is specified
        has_selection = any(x is not None for x in [atoms, orbitals, species])

        selection = None
        atom_list: list[int] | None = None
        orbital_list: list[int] | None = None

        if has_selection:
            # Resolve selection
            selection = self._resolve_projection_selection(
                atoms=atoms,
                orbitals=orbitals,
                spins=None,  # All spin components needed for texture
                species=species,
            )
            atom_list = list(selection.atoms) if selection.atoms else None
            orbital_list = list(selection.orbitals) if selection.orbitals else None

        # Use all atoms/orbitals if none specified
        if atom_list is None:
            atom_list = np.arange(self.n_atoms, dtype=int).tolist()
        if orbital_list is None:
            orbital_list = np.arange(self.n_orbitals, dtype=int).tolist()

        summed_projection = self.ebs_sum(
            atoms=atom_list, orbitals=orbital_list, sum_noncolinear=False
        )

        # Extract spin texture components (exclude total at index 0)
        spin_texture_values = np.asarray(summed_projection[..., 1:], dtype=np.float64)
        temp_shape_list = list(spin_texture_values.shape)
        temp_shape_list.insert(2, 1)
        reshape_order: str = kwargs.pop("order", "F")
        spin_texture_values = np.asarray(
            spin_texture_values.reshape(
                tuple(temp_shape_list), order=reshape_order  # pyright: ignore[reportArgumentType, reportCallIssue]
            ),
            dtype=np.float64,
        )

        prop = self._build_property(
            values=spin_texture_values,
            label=label,
            name=name,
            norm_mode="raw",
            selection=selection,
            description="Projected spin texture components (Sx, Sy, Sz)",
            is_non_collinear=True,
        )

        return prop

    def ebs_sum(
        self,
        atoms: list[int] | npt.NDArray[np.intp] | None = None,
        orbitals: list[int] | npt.NDArray[np.intp] | None = None,
        spins: list[int] | npt.NDArray[np.intp] | None = None,
        sum_noncolinear: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Compute summed projections.

        Parameters
        ----------
        atoms : list[int] | npt.NDArray[np.intp] | None, optional
            list of atoms to be summed over, by default None
        orbitals : list[int] | npt.NDArray[np.intp] | None, optional
            list of orbitals to be summed over, by default None
        spins : list[int] | npt.NDArray[np.intp] | None, optional
            list of spins to be summed over, by default None
        sum_noncolinear : bool, optional
            Determines if the projection should be summed in a non-colinear calculation, by default True

        Returns
        -------
        npt.NDArray[np.float64]
            The summed projections
        """
        if self.projected is None:
            raise ValueError("projected not set")

        projected_array = self.projected.to_array()

        atoms_arr: npt.NDArray[np.intp]
        spins_arr: npt.NDArray[np.intp]
        orbitals_arr: npt.NDArray[np.intp]

        if atoms is None:
            atoms_arr = np.arange(self.n_atoms, dtype=np.intp)
        else:
            atoms_arr = np.asarray(atoms, dtype=np.intp)
        if spins is None:
            spins_arr = np.arange(self.n_spins, dtype=np.intp)
        else:
            spins_arr = np.asarray(spins, dtype=np.intp)
        if orbitals is None:
            orbitals_arr = np.arange(self.n_orbitals, dtype=np.intp)
        else:
            orbitals_arr = np.asarray(orbitals, dtype=np.intp)
        # sum over orbitals
        ret: npt.NDArray[np.float64] = np.sum(
            projected_array[:, :, :, :, orbitals_arr], axis=-1
        )
        # sum over atoms
        ret = np.sum(ret[:, :, :, atoms_arr], axis=-1)
        # sum over spins only in non collinear and reshaping for consistency (nkpoints, nbands, nspins)
        # in non-mag, non-colin nspin=1, in colin nspin=2
        if self.is_non_collinear and sum_noncolinear:
            ret = np.sum(ret[:, :, spins_arr], axis=-1).reshape(self.n_kpoints, self.n_bands, 1)

        if self.is_spin_polarized:
            # Zero out the spin channel that is not specified
            if np.allclose(np.asarray(spins_arr), np.array([0])):
                ret[..., 1] = 0
            elif np.allclose(np.asarray(spins_arr), np.array([1])):
                ret[..., 0] = 0

        return ret

    def normalize(
        self,
        mode: str | EBSNormMode | None,
        values_array: npt.NDArray[np.float64],
        **kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Normalize projection values according to the specified mode.

        Parameters
        ----------
        mode : str | EBSNormMode | None
            Normalization mode
        values_array : np.ndarray
            Array to normalize, shape (n_kpoints, n_bands, n_spins)

        Returns
        -------
        np.ndarray
            Normalized array with same shape as input
        """
        mode = EBSNormMode.from_input(mode)

        if mode is EBSNormMode.RAW:
            return values_array
        elif mode is EBSNormMode.MAX:
            return self.normalize_max(values_array, **kwargs)
        elif mode is EBSNormMode.TOTAL:
            return self.normalize_total(values_array, **kwargs)
        elif mode is EBSNormMode.TOTAL_PROJECTION:
            return self.normalize_total_projection(values_array, **kwargs)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")  # pyright: ignore[reportUnreachable]

    def normalize_max(
        self,
        values_array: npt.NDArray[np.float64],
        **_kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Normalize by maximum value across all k-points and bands."""
        max_val = np.max(np.abs(values_array))
        if max_val < NUMERICAL_STABILITY_FACTOR:
            return values_array
        return values_array / max_val

    def normalize_total(
        self,
        values_array: npt.NDArray[np.float64],
        **_kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Normalize by total projection at each k-point/band."""
        total = self.ebs_sum()  # Sum over all atoms/orbitals
        total = np.where(
            np.abs(total) < NUMERICAL_STABILITY_FACTOR,
            NUMERICAL_STABILITY_FACTOR,
            total,
        )
        return values_array / total

    def normalize_total_projection(
        self,
        values_array: npt.NDArray[np.float64],
        atoms: list[int] | None = None,
        orbitals: list[int] | None = None,
        spins: list[int] | None = None,
        **_kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        """Normalize by total projection with same selection."""
        total = self.ebs_sum(atoms=atoms, orbitals=orbitals, spins=spins)
        total = np.where(
            np.abs(total) < NUMERICAL_STABILITY_FACTOR,
            NUMERICAL_STABILITY_FACTOR,
            total,
        )
        return values_array / total

    def _get_projection_label_builder(self) -> ProjectionLabelBuilder:
        """Lazily initialize and return the projection label builder."""
        if self._projection_label_builder is None:
            atom_indexer = None
            if self._structure is not None:
                atom_indexer = AtomIndexer.from_structure(self._structure)
            spin_indexer = SpinIndexer.from_projection_names(self.spin_projection_names)
            self._projection_label_builder = ProjectionLabelBuilder(
                atom_indexer=atom_indexer,
                orbital_indexer=OrbitalIndexer(),
                spin_indexer=spin_indexer,
            )
        return self._projection_label_builder

    def _get_projection_selection_resolver(self) -> ProjectionSelectionResolver:
        """Lazily initialize and return the projection selection resolver."""
        if self._projection_selection_resolver is None:
            label_builder = self._get_projection_label_builder()
            self._projection_selection_resolver = ProjectionSelectionResolver(
                label_builder=label_builder,
                orbital_names=self._orbital_names,
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
        species_orbital_map: Sequence[Mapping[str, Iterable[int]]]
        | Mapping[str, Iterable[int]]
        | None = None,
        atoms_orbital_map: Sequence[Mapping[Iterable[int] | int, Iterable[int]]]
        | Mapping[Iterable[int] | int, Iterable[int]]
        | None = None,
    ) -> ProjectionSelectionResult:
        """Resolve projection selection parameters to canonical form with labels.

        Parameters
        ----------
        atoms : Sequence[int] | int | None
            Atom indices to select
        orbitals : Sequence[int] | int | None
            Orbital indices to select
        spins : Sequence[int] | int | None
            Spin channel indices to select
        species : Sequence[str] | str | None
            Species names to select (resolved to atom indices)
        species_orbital_map : Mapping | None
            Mapping of species to orbital indices, e.g., {"Fe": [4,5,6], "O": [0,1,2]}
        atoms_orbital_map : Mapping | None
            Mapping of atom indices to orbital indices

        Returns
        -------
        ProjectionSelectionResult
            Resolved selection with atoms, orbitals, spins, species, and labels
        """
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
        include_normal_label: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Format selection labels with optional normalization suffix.

        Parameters
        ----------
        selection : ProjectionSelectionResult
            Resolved selection with labels
        normalize : bool
            Whether normalization is applied
        include_normal_label : bool
            Whether to append [fraction]/[raw] suffix

        Returns
        -------
        tuple[list[str], list[str]]
            (plain_labels, latex_labels) for each spin component
        """
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
                    f"{body_latex}[{component_latex}]" if body_latex else f"[{component_latex}]"
                )
            body_latex = body_latex or "\\mathrm{all}"
            labels_latex.append(f"${body_latex}{normal_suffix_latex}$")

        return labels_plain, labels_latex

    def _build_property(
        self,
        values: npt.NDArray[np.float64],
        label: str,
        name: str,
        norm_mode: str | EBSNormMode | None = None,
        selection: ProjectionSelectionResult | None = None,
        include_normal_label: bool = False,
        allowed_norm_modes: set[EBSNormMode] | None = None,
        **kwargs: Any,
    ) -> Property:
        """Build a Property object with rich metadata.

        Parameters
        ----------
        values : np.ndarray
            The property values, shape (n_kpoints, n_bands, n_spins)
        label : str
            Human-readable label for the property
        name : str
            Property name for storage/retrieval
        norm_mode : str | EBSNormMode | None
            Normalization mode to apply
        selection : ProjectionSelectionResult | None
            Projection selection result with labels
        include_normal_label : bool
            Whether to include normalization info in labels
        allowed_norm_modes : set[EBSNormMode] | None
            Restrict allowed normalization modes
        **kwargs
            Additional metadata fields

        Returns
        -------
        Property
            Property object with full metadata
        """
        # Resolve and validate normalization mode
        norm_mode = EBSNormMode.from_input(norm_mode)
        if allowed_norm_modes is not None:
            if norm_mode not in allowed_norm_modes:
                valid_modes = ", ".join(m.value for m in allowed_norm_modes)
                raise ValueError(
                    f"Invalid normalization mode: {norm_mode.value}. "
                    f"Valid modes: {valid_modes}"
                )

        # Get transformed name and units
        normed_name = EBSNormMode.get_normed_name(norm_mode, name)
        normed_units = EBSNormMode.get_normed_units(norm_mode)
        footnote = EBSNormMode.get_mode_footnote(norm_mode)

        # Normalize values
        values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

        # Compute data limits
        data_min = np.min(values, axis=0)
        data_max = np.max(values, axis=0)
        data_lim = (data_min, data_max)
        rounded_data_lim = (np_round_to_half(data_min), np_round_to_half(data_max))

        # Build base metadata
        metadata: dict[str, Any] = {
            "norm_mode": norm_mode,
            "units": normed_units,
            "data_lim": data_lim,
            "rounded_data_lim": rounded_data_lim,
            "footnote": footnote,
            "scalar_label": label,
            "label": label,
            "label_plain": label,
            "include_normal_label": include_normal_label,
        }

        # Add selection metadata if available
        if selection is not None:
            label_plain_list, label_latex_list = self._format_selection_label(
                selection=selection,
                normalize=norm_mode is not EBSNormMode.RAW,
                include_normal_label=include_normal_label,
            )
            metadata.update(
                {
                    "atoms": list(selection.atoms) if len(selection.atoms) > 0 else None,
                    "orbitals": list(selection.orbitals)
                    if selection.orbitals is not None
                    else None,
                    "spins": list(selection.spins) if selection.spins is not None else None,
                    "species": list(selection.species) if len(selection.species) > 0 else None,
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
                    "label": label_latex_list,
                    "label_plain": label_plain_list,
                }
            )

        # Add any additional kwargs to metadata
        if kwargs:
            # Filter out kwargs that were used for normalization
            extra_metadata = {
                k: v for k, v in kwargs.items() if k not in ("atoms", "orbitals", "spins")
            }
            metadata.update(extra_metadata)

        # Build Property
        prop = Property(
            name=normed_name,
            value=values,
            point_set=self,
            metadata=metadata,
            label=label,
            units=normed_units,
        )

        return prop

    def iter_properties(
        self,
    ) -> Iterable[tuple[str, str, int, npt.NDArray[np.float64]]]:
        for prop_name, calc_name, gradient_order, value_array in self.iter_property_arrays():
            yield prop_name, calc_name, gradient_order, value_array

    def reduce_bands(
        self,
        bands: list[int] | None = None,
        near_fermi: bool = False,
        energy: float | None = None,
        tolerance: float = 0.7,
        inplace: bool = True,
    ) -> ElectronicBandStructure:
        if bands is not None:
            return self.reduce_bands_by_index(bands, inplace)
        elif energy is not None:
            return self.reduce_bands_near_energy(energy, tolerance, inplace)
        elif near_fermi:
            return self.reduce_bands_near_fermi(tolerance, inplace)
        else:
            raise ValueError("Either bands or energy or near_fermi must be provided")

    def reduce_bands_near_energy(
        self, energy: float, tolerance: float = 0.7, inplace: bool = True
    ) -> ElectronicBandStructure:
        """
        Reduces the bands to those near the fermi energy
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        logger.info("____Reducing bands near fermi energy____")
        full_band_index: list[int] = []
        bands_spin_index: dict[int, list[int]] = {}

        if ebs.bands is None:
            raise ValueError("bands not set")

        bands_array = ebs.bands.to_array()
        for ispin in ebs.spin_channels:
            bands_spin_index[int(ispin)] = []
            for iband in range(ebs.n_bands):
                fermi_surface_test = len(
                    np.where(
                        np.logical_and(
                            bands_array[:, iband, ispin] >= energy - tolerance,
                            bands_array[:, iband, ispin] <= energy + tolerance,
                        )
                    )[0]
                )

                if fermi_surface_test != 0:
                    bands_spin_index[int(ispin)].append(iband)

                    if iband not in full_band_index:  # Avoid duplicates
                        full_band_index.append(iband)

        band_property_names = ebs.band_property_names
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            prop = ebs.get_property(prop_name)
            if prop_name in band_property_names and prop is not None and isinstance(prop, Property):
                prop[calc_name, gradient_order] = value_array[:, full_band_index, ...]

        debug_message = f"Bands near energy {energy}. "
        debug_message += f"Spin-0 {bands_spin_index[0]} |"
        if self.n_spin_channels > 1 and not self.is_non_collinear:
            debug_message += f" Spin-1 {bands_spin_index[1]}"
        logger.debug(debug_message)
        return ebs

    def reduce_bands_near_fermi(
        self, tolerance: float = 0.7, inplace: bool = True
    ) -> ElectronicBandStructure:
        """
        Reduces the bands to those near the fermi energy
        """
        return self.reduce_bands_near_energy(self.fermi, tolerance, inplace=inplace)

    def reduce_bands_by_index(
        self, bands: list[int], inplace: bool = True
    ) -> ElectronicBandStructure:
        """
        Reduces the bands to those near the fermi energy
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        band_property_names = ebs.band_property_names
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            prop = ebs.get_property(prop_name)
            if prop_name in band_property_names and prop is not None and isinstance(prop, Property):
                prop[calc_name, gradient_order] = value_array[:, bands, ...]

        return ebs

    def fix_collinear_spin(self, inplace: bool = True) -> ElectronicBandStructure:
        """
        Converts data from two spin channels to a single channel, adjusting the spin down values to negatives. This is typically used for plotting the Density of States (DOS).

        Parameters
        ----------
        inplace : bool
            Whether to modify in place or return a copy.

        Returns
        -------
        ElectronicBandStructure
            The modified electronic band structure.
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        if ebs.n_spin_channels != 2:
            raise ValueError("Spin channels must be 2 for this function to work")

        band_property_names = ebs.band_property_names

        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            prop = ebs.get_property(prop_name)
            if (
                prop_name in band_property_names
                and prop is not None
                and isinstance(prop, Property)
                and ebs.has_spin_channels(prop)
            ):
                original_value_shape = list(value_array.shape)
                band_dim = original_value_shape[1]
                original_value_shape[1] = 2 * band_dim
                original_value_shape[2] = 1
                modified_array = value_array.reshape(original_value_shape)
                prop[calc_name, gradient_order] = modified_array

        return ebs

    def shift_bands(self, shift_value: float, inplace: bool = False) -> ElectronicBandStructure:
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        bands_prop = ebs.get_property("bands")
        if bands_prop is None or not isinstance(bands_prop, Property):
            raise ValueError("bands not set")
        bands_array = bands_prop.value
        bands_array += shift_value
        ebs.add_property(name="bands", value=bands_array)
        return ebs

    def shift_kpoints_to_fbz(self, inplace: bool = True) -> ElectronicBandStructure:
        # Shifting all kpoint to first Brillouin zone
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        # Old method
        # bound_ops = -1.0 * (ebs.kpoints > 0.5) + 1.0 * (ebs.kpoints <= -0.5)
        # new_kpoints = ebs.kpoints + bound_ops

        new_kpoints: npt.NDArray[np.float64] = -np.fmod(ebs.kpoints + 6.5, 1) + 0.5
        ebs.update_points(new_kpoints)
        return ebs

    def unfold(
        self,
        transformation_matrix: npt.NDArray[np.float64] | None = None,
        structure: Structure | None = None,
        inplace: bool = True,
    ) -> ElectronicBandStructure:
        """The method helps unfold the bands. This is done by using the unfolder to find the new kpoint weights.
        The current weights are then updated

        Parameters
        ----------
        transformation_matrix : npt.NDArray[np.float64] | None, optional
            The transformation matrix to transform the basis. Expected size is (3,3), by default None
        structure : Structure | None, optional
            The structure of a material, by default None
        inplace : bool, optional
            If True, the method will modify the current instance, by default True

        Returns
        -------
        ElectronicBandStructure
            The modified electronic band structure
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        # Provide default transformation matrix if None
        trans_matrix = transformation_matrix if transformation_matrix is not None else np.diag([1, 1, 1]).astype(np.float64)
        struct = structure if structure is not None else self.structure

        uf = Unfolder(
            ebs=ebs,
            transformation_matrix=trans_matrix,
            structure=struct,
        )

        ebs.add_property(name="weights", value=uf.weights)
        return ebs

    def save(self, path: Path) -> None:
        serializer = get_serializer(path)
        serializer.save(self, path)

    @classmethod
    def load(cls, path: Path) -> ElectronicBandStructure:
        from typing import cast

        serializer = get_serializer(path)
        ebs = cast(ElectronicBandStructure, serializer.load(path))
        return ebs

    @classmethod
    def from_code(
        cls, code: str, dirpath: str, use_cache: bool = False, ebs_filename: str = "ebs.pkl"
    ) -> ElectronicBandStructure:
        return get_ebs_from_code(code, dirpath, use_cache, ebs_filename)

    @staticmethod
    def get_atomic_orbital_label(atoms: list[int], orbitals: list[int]) -> str:
        atom_label = ElectronicBandStructure.get_atom_label(atoms)
        orbital_label = ElectronicBandStructure.get_orbital_label(orbitals)
        return f"{atom_label}|{orbital_label}"

    @staticmethod
    def get_orbital_label(orbitals: list[int]) -> str:
        orbitals_label = ",".join([str(orbital) for orbital in orbitals])
        return f"orbitals-({orbitals_label})"

    @staticmethod
    def get_atom_label(atoms: list[int]) -> str:
        atoms_label = ",".join([str(atom) for atom in atoms])
        return f"atoms-({atoms_label})"

    @staticmethod
    def get_band_label(bands: list[int] | int, spins: list[int] | int) -> str:
        bands_list: list[int]
        spins_list: list[int]
        if isinstance(bands, int):
            bands_list = [bands]
        else:
            bands_list = bands
        if isinstance(spins, int):
            spins_list = [spins]
        else:
            spins_list = spins

        bands_label = ",".join([str(band) for band in bands_list])
        spins_label = ",".join([str(spin) for spin in spins_list])
        return f"(bands|spins)-({bands_label}|{spins_label})"

    @staticmethod
    def extract_band_index(label: str) -> tuple[list[int] | int, list[int] | int]:
        raw_text = re.findall(r"\(bands\|spins\)-\((.*)\)", label)
        bands_str, spins_str = raw_text[0].split("|")
        bands_list = [int(band) for band in bands_str.split(",")]
        spins_list = [int(spin) for spin in spins_str.split(",")]

        bands_result: list[int] | int = bands_list[0] if len(bands_list) == 1 else bands_list
        spins_result: list[int] | int = spins_list[0] if len(spins_list) == 1 else spins_list

        return bands_result, spins_result

    @staticmethod
    def extract_property_label(label: str) -> str:
        property_name = label.split("__")[0]
        return property_name

    @staticmethod
    def get_spin_projection_label(spin_projections: list[int]) -> str:
        spin_projection_names_label = ",".join(
            [str(spin_projection_name) for spin_projection_name in spin_projections]
        )
        return f"spin_projections-({spin_projection_names_label})"

    @staticmethod
    def get_band_property_label(property_name: str, band_index: int, spin_index: int) -> str:
        band_label = ElectronicBandStructure.get_band_label(band_index, spin_index)
        return f"{property_name}__{band_label}"

    @staticmethod
    def get_property_gradient_label(property_name: str) -> str:
        return f"{property_name}_gradient"

    @staticmethod
    def get_property_hessian_label(property_name: str) -> str:
        return f"{property_name}_hessian"


class ElectronicBandStructurePath(  # pyright: ignore[reportIncompatibleMethodOverride]
    ElectronicBandStructure, DifferentiablePropertyInterface, PyvistaInterface
):
    _kpath: kpoints.KPath
    _mesh: pv.PolyData | pv.StructuredGrid | Any | None

    def __init__(self, kpath: kpoints.KPath, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._kpath = kpath
        self.as_cart()
        logger.debug(f"ElectronicBandStructurePath: \n {self}")
        logger.info("___ElectronicBandStructurePath initialization complete___")

    @classmethod
    @override
    def from_code(
        cls, code: str, dirpath: str, use_cache: bool = False, ebs_filename: str = "ebs.pkl"
    ) -> ElectronicBandStructure:
        return get_ebs_from_code(code, dirpath, use_cache, ebs_filename)

    @override
    def __str__(self) -> str:
        ret = super().__str__()
        ret += "\nKPath: \n"
        ret += "------------------------     \n"
        ret += f"KPath = \n {self.kpath}\n"
        return ret

    def as_cart(self) -> None:
        if self.reciprocal_lattice is None:
            raise ValueError("reciprocal_lattice not set")
        self.transform_points(self.reciprocal_lattice)

    def as_frac(self) -> None:
        if self.reciprocal_lattice is None:
            raise ValueError("reciprocal_lattice not set")
        inv_lattice = np.linalg.inv(self.reciprocal_lattice)
        self.transform_points(np.asarray(inv_lattice, dtype=np.float64))

    @property
    def kpath(self) -> kpoints.KPath:
        return self._kpath

    @property
    def knames(self) -> list[tuple[str, str]] | None:
        return self.kpath.knames

    @property
    def n_segments(self) -> int:
        return self.kpath.n_segments

    @property
    def tick_positions(self) -> list[int]:
        return self.kpath.tick_positions

    @property
    def tick_names(self) -> list[str]:
        return self.kpath.tick_names

    @property
    def tick_names_latex(self) -> list[str]:
        return self.kpath.tick_names_latex

    @property
    def special_kpoint_names(self) -> list[str]:
        return self.kpath.special_kpoint_names

    @property
    @override
    def bands(self) -> Property | None:
        """Return bands as a Property with kpath metadata.

        Overrides the base class to include pre-computed kpath information:
        - k_distances: Cumulative k-path distances for x-axis
        - tick_positions: Indices of high-symmetry points
        - tick_names: Labels for high-symmetry points
        - tick_names_latex: LaTeX-formatted tick labels

        Returns
        -------
        Property
            Property with bands data and kpath metadata
        """
        prop = self.get_property("bands")
        if prop is None or not isinstance(prop, Property):
            return None

        # Add kpath metadata to a copy of the property
        kpath_metadata = {
            "k_distances": self.kpath.get_distances(as_segments=False),
            "tick_positions": list(self.kpath.tick_positions),
            "tick_names": list(self.kpath.tick_names),
            "tick_names_latex": list(self.kpath.tick_names_latex),
        }

        # Create new Property with merged metadata
        merged_metadata = {**prop.metadata, "kpath": kpath_metadata}

        return Property(
            name=prop.name,
            value=prop.value,
            point_set=prop.point_set,
            units=prop.units,
            label=prop.label,
            metadata=merged_metadata,  # pyright: ignore[reportArgumentType]
        )

    @override
    def to_mesh(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        scalars: tuple[str, npt.NDArray[Any]] | None = None,
        vectors: tuple[str, npt.NDArray[Any]] | None = None,
        as_cartesian: bool = True,
        **kwargs: Any,
    ) -> Any:  # Returns pv.PointSet which is not typed
        if as_cartesian:
            mesh_points = self.kpoints_cartesian
        else:
            mesh_points = self.kpoints
        mesh = pv.PointSet(mesh_points)
        if scalars is not None:
            self.set_mesh_scalar(*scalars)
        if vectors is not None:
            self.set_mesh_vector(*vectors)
        self._mesh = mesh
        return mesh

    @override
    def gradient_func(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, points: npt.NDArray[np.float64], values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        continuous_segments = self.kpath.get_continuous_segments()

        gradients = np.zeros(values.shape)
        for k_indices in continuous_segments:
            kpath_segment = points[k_indices]
            delta_k = np.gradient(kpath_segment, axis=0)
            delta_k = np.linalg.norm(delta_k, axis=1)

            # Determine distance coordinates for gradient calculation.
            # cumsum does not include 0 and includes an unnecessary point at the end.
            distance_coordinates = np.cumsum(delta_k, axis=0)[:-1]
            distance_coordinates = np.insert(distance_coordinates, 0, 0, axis=0)
            gradients[k_indices, ...] = np.gradient(
                values[k_indices, ...],
                distance_coordinates,
                axis=0,
                edge_order=2,
            )
        gradients = gradients * physics.METER_ANGSTROM
        return gradients

    @override
    def compute_property(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, name: str, **kwargs: Any
    ) -> Property | npt.NDArray[np.float64] | None:
        if name == "bands_velocity":
            return self.compute_band_velocity(**kwargs)
        elif name == "bands_speed":
            return self.compute_band_speed(**kwargs)
        elif name == "avg_inv_effective_mass":
            return self.compute_avg_inv_effective_mass(**kwargs)
        else:
            return super().compute_property(name, **kwargs)

    # ------------------------------------------------------------------
    # Overlay weight builders
    # ------------------------------------------------------------------
    def build_overlay_species_weights(
        self,
        spins: Sequence[int] | int | None = None,
        orbitals: Sequence[int] | int | None = None,
        norm_mode: str | EBSNormMode | None = "raw",
    ) -> list[Property]:
        """Build per-species overlay weights for plot overlays.

        For each species present in the structure, computes summed orbital
        projections over all atoms of that species.

        Parameters
        ----------
        spins : Sequence[int] | int | None
            Spin channels to include
        orbitals : Sequence[int] | int | None
            Orbital indices to include
        norm_mode : str | EBSNormMode | None
            Normalization mode

        Returns
        -------
        list[Property]
            List of Property objects, one per species

        Raises
        ------
        ValueError
            If structure is not available
        """
        if self._structure is None:
            raise ValueError("Structure is required to build species weights")

        properties: list[Property] = []

        # Get species list - structure.species is a list of strings or we use atoms
        species_attr: Any = getattr(self._structure, "species", None)
        species_list: list[str]
        if species_attr is not None:
            species_list = list(species_attr)
        else:
            atoms_array = self._structure.atoms
            if atoms_array is not None:
                species_list = [str(s) for s in np.unique(atoms_array)]
            else:
                species_list = []

        for species_name in species_list:
            prop = self.compute_projected_sum(
                species=species_name,
                orbitals=orbitals,
                spins=spins,
                norm_mode=norm_mode,
                label=str(species_name),
                name=f"overlay_species_{species_name}",
            )
            properties.append(prop)

        return properties

    def build_overlay_orbitals_weights(
        self,
        atoms: Sequence[int] | int | None = None,
        spins: Sequence[int] | int | None = None,
        norm_mode: str | EBSNormMode | None = "raw",
    ) -> list[Property]:
        """Build per-orbital-group overlay weights for plot overlays.

        Iterates over orbital groups (s, p, d, f when available) and computes
        weights for each group.

        Parameters
        ----------
        atoms : Sequence[int] | int | None
            Atom indices to include
        spins : Sequence[int] | int | None
            Spin channels to include
        norm_mode : str | EBSNormMode | None
            Normalization mode

        Returns
        -------
        list[Property]
            List of Property objects, one per orbital group
        """
        properties: list[Property] = []

        # PRIMARY_ORBITAL_GROUPS maps "s" -> (0,), "p" -> (1,2,3), etc.
        orbital_group_map: dict[str, tuple[int, ...]] = dict(PRIMARY_ORBITAL_GROUPS)

        for orb_name in ["s", "p", "d", "f"]:
            if orb_name == "f" and self.n_orbitals <= 9:
                continue

            orb_indices = orbital_group_map.get(orb_name)
            if orb_indices is None:
                continue

            prop = self.compute_projected_sum(
                atoms=atoms,
                orbitals=list(orb_indices),
                spins=spins,
                norm_mode=norm_mode,
                label=orb_name,
                name=f"overlay_orbital_{orb_name}",
            )
            properties.append(prop)

        return properties

    def build_overlay_weights(
        self,
        items: dict[str, Sequence[str | int]] | list[dict[str, Sequence[str | int]]],
        spins: Sequence[int] | int | None = None,
        norm_mode: str | EBSNormMode | None = "raw",
    ) -> list[Property]:
        """Build overlay weights from species-orbital mappings.

        Parameters
        ----------
        items : dict | list[dict]
            Mapping like {"Fe": ["d"], "O": ["p"]} or {"Fe": [4,5,6]},
            or a list of such mappings
        spins : Sequence[int] | int | None
            Spin channels to include
        norm_mode : str | EBSNormMode | None
            Normalization mode

        Returns
        -------
        list[Property]
            List of Property objects, one per mapping entry

        Raises
        ------
        ValueError
            If structure is not available
        """
        if self._structure is None:
            raise ValueError("Structure is required to build overlay weights from items")

        items_iter: list[dict[str, Sequence[str | int]]]
        if isinstance(items, dict):
            items_iter = [items]
        else:
            items_iter = items

        properties: list[Property] = []

        # Create orbital group map for resolving names to indices
        orbital_group_map: dict[str, tuple[int, ...]] = dict(PRIMARY_ORBITAL_GROUPS)

        for mapping in items_iter:
            for species_name, orbital_spec in mapping.items():
                # Resolve orbital names to indices if needed
                orbitals: list[int]
                if len(orbital_spec) > 0 and isinstance(orbital_spec[0], str):
                    resolved_orbitals: list[int] = []
                    for orb_token in orbital_spec:
                        if isinstance(orb_token, str):
                            orb_indices = orbital_group_map.get(orb_token, ())
                            resolved_orbitals.extend(orb_indices)
                    orbitals = resolved_orbitals
                else:
                    orbitals = [int(x) for x in orbital_spec]

                # Use species_orbital_map for proper label generation
                species_orbital_map: dict[str, Iterable[int]] = {species_name: orbitals}

                prop = self.compute_projected_sum(
                    species_orbital_map=species_orbital_map,
                    spins=spins,
                    norm_mode=norm_mode,
                    label=f"{species_name}",  # Label builder will create full label
                    name=f"overlay_{species_name}_{hash(tuple(orbitals))}",
                )
                properties.append(prop)

        return properties

    def as_kdist(
        self, as_segments: bool = True
    ) -> Any:  # Returns pv.MultiBlock which is not typed
        kdistances_result = self.kpath.get_distances(as_segments=False, cumlative_across_segments=True)
        # When as_segments=False, returns NDArray not list
        kdistances = np.asarray(kdistances_result, dtype=np.float64)
        if self.bands is None:
            raise ValueError("bands not set")
        bands_array = self.bands.to_array()
        n_bands = bands_array.shape[1]
        n_spins = bands_array.shape[2]
        n_kpoints = kdistances.shape[0]
        k_indices = self.kpath.segment_indices
        blocks = pv.MultiBlock()

        if as_segments:
            for indices in k_indices:
                n_indices = indices.shape[0]
                for iband in range(n_bands):
                    for ispin in range(n_spins):
                        k_segment_distances = kdistances[indices]
                        band_values = bands_array[indices, iband, ispin]

                        band_kpoints = np.zeros(shape=(n_indices, 3))
                        band_kpoints[:, 0] = k_segment_distances
                        band_kpoints[:, 1] = band_values
                        blocks.append(pv.PolyData(band_kpoints))
        else:
            for iband in range(n_bands):
                for ispin in range(n_spins):
                    band_values = bands_array[:, iband, ispin]
                    k_distances = kdistances.copy()
                    band_kpoints = np.zeros(shape=(n_kpoints, 3))
                    band_kpoints[:, 0] = k_distances
                    band_kpoints[:, 1] = band_values
                    band_kpoints[:, 2] = ispin
                    blocks.append(pv.PolyData(band_kpoints))
        return blocks

    def plot(
        self,
        add_point_labels_args: dict[str, Any] | None = None,
        bz_add_mesh_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Plots the band structure.

        """
        self.as_cart()
        add_point_labels_args = add_point_labels_args or {}
        bz_add_mesh_args = bz_add_mesh_args or {}

        special_kpoint_names: list[str | int] = list(self.kpath.special_kpoint_names)
        special_kpoint_positions = self.kpath.get_special_kpoints(as_segments=False, cartesian=True)

        p = pv.Plotter()
        p.add_mesh(self, **kwargs)
        p.add_point_labels(special_kpoint_positions, special_kpoint_names, **add_point_labels_args)

        bz_add_mesh_args["style"] = bz_add_mesh_args.get("style", "wireframe")
        bz_add_mesh_args["line_width"] = bz_add_mesh_args.get("line_width", 2.0)
        bz_add_mesh_args["color"] = bz_add_mesh_args.get("color", "black")
        bz_add_mesh_args["opacity"] = bz_add_mesh_args.get("opacity", 1.0)

        p.add_mesh(
            self.brillouin_zone,
            **bz_add_mesh_args,
        )
        p.show()


def is_plane_aligned_with_reciprocal_lattice(
    normal: npt.NDArray[np.float64], reciprocal_lattice: npt.NDArray[np.float64] | None
) -> bool:
    """Check if the plane normal is aligned with reciprocal lattice axes"""
    if reciprocal_lattice is None:
        return False

    normal_unit: npt.NDArray[np.float64] = normal / np.linalg.norm(normal)

    # Check alignment with each reciprocal lattice vector
    for recip_vec in reciprocal_lattice:
        recip_unit = recip_vec / np.linalg.norm(recip_vec)
        dot_product = abs(np.dot(normal_unit, recip_unit))
        if dot_product > 0.99:  # Nearly parallel (within 1 degree)
            return True
    return False


def edge_diff_ramp(
    vector: npt.NDArray[np.float64],
    pad_width: tuple[int, int],
    _iaxis: int,
    _kwargs: dict[str, Any],
) -> npt.NDArray[np.float64] | None:
    if pad_width[0] == 0 or pad_width[1] == 0:
        return vector

    original_index = pad_width[0] + 1
    original_end_index = vector.shape[0] - pad_width[1] - 1

    dx = abs(vector[original_index] - vector[original_index + 1])

    # Create left padding using array operations
    left_pad = vector[original_index] - (np.arange(pad_width[0], 0, -1) + 1) * dx
    vector[: pad_width[0]] = left_pad

    # Create right padding using array operations
    right_pad = vector[original_end_index] + (np.arange(1, pad_width[1] + 1)) * dx
    vector[-pad_width[1] :] = right_pad
    return None


class ElectronicBandStructureMesh(  # pyright: ignore[reportIncompatibleMethodOverride]
    ElectronicBandStructure, DifferentiablePropertyInterface, PyvistaInterface
):
    _kgrid_info: kpoints.KGridInfo
    _mesh: pv.PolyData | pv.StructuredGrid | Any | None

    def __init__(self, kgrid_info: kpoints.KGridInfo, **kwargs: Any) -> None:
        super(ElectronicBandStructureMesh, self).__init__(**kwargs)

        self._kgrid_info = kgrid_info

        if self.n_kpoints != np.prod(self.kgrid_info.kgrid):
            if self.structure is None:
                raise ValueError("structure is required when expanding IBZ to FBZ")
            ibz2fbz(
                self,
                rotations=self.structure.rotations,
                kgrid_info=self.kgrid_info,
                decimals=4,
                inplace=True,
            )

        sort_by_kpoints(self, inplace=True)

        if self.n_kpoints != np.prod(self.kgrid_info.kgrid):
            raise ValueError("n_kpoints must be equal to np.prod(kgrid) (number of kpoints)")

    @classmethod
    @override
    def from_code(
        cls, code: str, dirpath: str, use_cache: bool = False, ebs_filename: str = "ebs.pkl"
    ) -> ElectronicBandStructure:
        ebs = get_ebs_from_code(code, dirpath, use_cache, ebs_filename)
        return ebs

    @override
    def __str__(self) -> str:
        ret = super().__str__()
        ret += "\nKGrid: \n"
        ret += "------------------------     \n"
        ret += f"(nkx, nky, nkz) = \n {self.kgrid}\n"
        return ret

    @property
    def kgrid_info(self) -> kpoints.KGridInfo:
        return self._kgrid_info

    @property
    def kgrid(self) -> tuple[int, int, int]:
        dims = self.get_kgrid()
        return (int(dims[0]), int(dims[1]), int(dims[2]))

    def get_kgrid(
        self, num_bins: int = 1000, height: int = 1, coord_tol: float = 0.01
    ) -> npt.NDArray[np.int_]:
        return math.get_grid_dims(
            self.kpoints, num_bins=num_bins, height=height, coord_tol=coord_tol
        )

    @property
    def kshift(self) -> tuple[float, float, float]:
        return self.kgrid_info.kshift

    @property
    def kgrid_mode(self) -> kpoints.KGRID_MODE:
        return self.kgrid_info.kgrid_mode

    @property
    def kbounds(self) -> npt.NDArray[np.float64]:
        return self.get_kbounds()

    @property
    def ukx(self) -> npt.NDArray[np.float64]:
        return np.asarray(np.linspace(self.kbounds[0, 0], self.kbounds[0, 1], self.n_kx), dtype=np.float64)

    @property
    def uky(self) -> npt.NDArray[np.float64]:
        return np.asarray(np.linspace(self.kbounds[1, 0], self.kbounds[1, 1], self.n_ky), dtype=np.float64)

    @property
    def ukz(self) -> npt.NDArray[np.float64]:
        return np.asarray(np.linspace(self.kbounds[2, 0], self.kbounds[2, 1], self.n_kz), dtype=np.float64)

    @property
    def n_kx(self) -> int:
        return self.kgrid[0]

    @property
    def n_ky(self) -> int:
        return self.kgrid[1]

    @property
    def n_kz(self) -> int:
        return self.kgrid[2]

    @property
    def is_ibz(self) -> bool:
        return bool(self.n_kpoints != np.prod(self.kgrid))

    @property
    def is_fbz(self) -> bool:
        return bool(self.n_kpoints == np.prod(self.kgrid))

    @property
    def is2d(self) -> bool:
        unique_coords = [self.ukx, self.uky, self.ukz]

        is_2d = False
        for dim_size in unique_coords:
            if len(dim_size) == 1:
                is_2d = True
                break

        return is_2d

    @override
    def to_mesh(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        scalars: tuple[str, npt.NDArray[Any]] | None = None,
        vectors: tuple[str, npt.NDArray[Any]] | None = None,
        as_cartesian: bool = True,
    ) -> pv.StructuredGrid:
        """Explicitly returns a new PyVista StructuredGrid."""
        # This can be the same logic as the `grid` property,
        # but without caching if a fresh object is desired.
        grid = pv.StructuredGrid()
        if as_cartesian:
            if self.kpoints_cartesian is None:
                raise ValueError("kpoints_cartesian not set")
            grid.points = self.kpoints_cartesian
        else:
            grid.points = self.kpoints
        grid.dimensions = self.kgrid
        self._mesh = grid
        if scalars is not None:
            self.set_mesh_scalar(*scalars)
        if vectors is not None:
            self.set_mesh_vector(*vectors)
        return grid

    def get_kbounds(self) -> npt.NDArray[np.float64]:
        kbounds: npt.NDArray[np.float64] = np.zeros((3, 2))
        for icoord in range(3):
            coords = self.kpoints[:, icoord]
            kx_min, kx_max = np.min(coords), np.max(coords)
            kbounds[icoord, 0] = kx_min
            kbounds[icoord, 1] = kx_max
        return kbounds

    def get_kpoints_mesh(self, **kwargs: Any) -> npt.NDArray[np.float64]:
        return math.array_to_mesh(
            array=self.kpoints,
            nkx=self.n_kx,
            nky=self.n_ky,
            nkz=self.n_kz,
            **kwargs,
        )

    def get_property_mesh(self, key: PropertyKey, **kwargs: Any) -> npt.NDArray[np.float64] | None:
        prop = self.get_property(key, **kwargs)
        if prop is None:
            return None
        # Get the value array from Property or use directly if already an array
        if isinstance(prop, Property):
            prop_value = prop.value
        else:
            prop_value = prop
        property_mesh: npt.NDArray[np.float64] = math.array_to_mesh(
            array=prop_value,
            nkx=self.n_kx,
            nky=self.n_ky,
            nkz=self.n_kz,
            **kwargs,
        )

        logger.debug(f"Property mesh shape: {prop_value.shape}")
        logger.debug(f"Nkx: {self.n_kx}, Nky: {self.n_ky}, Nkz: {self.n_kz}")
        logger.debug(f"Property mesh shape: {property_mesh.shape}")
        return property_mesh

    @override
    def compute_property(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, name: str, **kwargs: Any
    ) -> Property | npt.NDArray[np.float64] | None:
        if name == "bands_velocity":
            return self.compute_band_velocity(**kwargs)
        elif name == "bands_speed":
            return self.compute_band_speed(**kwargs)
        elif name == "avg_inv_effective_mass":
            return self.compute_avg_inv_effective_mass(**kwargs)
        else:
            return super().compute_property(name, **kwargs)

    def pad(
        self, padding: int = 10, order: str = "F", inplace: bool = True
    ) -> ElectronicBandStructureMesh:
        logger.info(f"Padding kpoints by {padding} in all directions")
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        padding_dims: list[tuple[int, int]] = []
        for _, n in enumerate(ebs.kgrid):
            if n == 1:
                padding_dims.append((0, 0))
            else:
                padding_dims.append((padding, padding))

        kpoints_padding_dims = copy.deepcopy(padding_dims)
        kpoints_padding_dims.append((0, 0))
        kpoints_mesh = ebs.get_kpoints_mesh()
        # np.pad with custom function mode is not fully typed
        padded_kpoints_mesh: npt.NDArray[np.float64] = np.pad(  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
            kpoints_mesh, kpoints_padding_dims, mode=edge_diff_ramp  # pyright: ignore[reportArgumentType]
        )
        logger.debug(f"Padded kpoints mesh shape: {padded_kpoints_mesh.shape}")  # pyright: ignore[reportUnknownMemberType]

        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            prop = ebs.get_property(prop_name)
            if prop is None or not isinstance(prop, Property):
                continue
            value_mesh = math.array_to_mesh(
                array=value_array, nkx=ebs.n_kx, nky=ebs.n_ky, nkz=ebs.n_kz
            )
            n_scalar_dims = len(value_mesh.shape[3:])
            scalar_padding_dims = copy.deepcopy(padding_dims)
            for _ in range(n_scalar_dims):
                scalar_padding_dims.append((0, 0))
            padded_mesh = np.pad(value_mesh, scalar_padding_dims, mode="wrap")
            logger.debug(f"Padded {prop_name} mesh shape: {padded_mesh.shape}")
            padded_array = math.mesh_to_array(padded_mesh, order=order)
            if padded_array is None:
                continue
            prop[calc_name, gradient_order] = padded_array

        new_kpoints = math.mesh_to_array(padded_kpoints_mesh, order=order)  # pyright: ignore[reportUnknownArgumentType]
        if new_kpoints is None:
            raise ValueError("Failed to convert padded kpoints mesh to array")
        ebs.update_points(new_kpoints)
        ebs._mesh = ebs.to_mesh()
        return ebs

    def expand_kpoints_to_supercell_by_axes(
        self, axes_to_expand: list[int] | None = None, inplace: bool = True, **_kwargs: Any
    ) -> ElectronicBandStructureMesh:
        if axes_to_expand is None:
            axes_to_expand = [0, 1, 2]
        logger.info(f"Expanding kpoints to supercell by axes: {axes_to_expand}")
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        # Validate input
        if not set(axes_to_expand).issubset({0, 1, 2}):
            raise ValueError("axes_to_expand must be a subset of [0, 1, 2]")

        # Create supercell directions based on axes to expand
        supercell_directions = list(itertools.product([1, 0, -1], repeat=len(axes_to_expand)))

        n_init_points = ebs.n_kpoints
        new_kpoints = ebs.kpoints.copy()
        for supercell_direction in supercell_directions:
            if supercell_direction == tuple([0] * len(axes_to_expand)):
                continue

            shifted_kpoints = ebs.kpoints.copy()
            for i, axis in enumerate(axes_to_expand):
                shifted_kpoints[:, axis] += supercell_direction[i]

            new_kpoints = np.concatenate([new_kpoints, shifted_kpoints], axis=0)
            for prop_name, _, gradient_order, value_array in ebs.iter_properties():
                initial_array = value_array[:n_init_points]
                new_values = np.concatenate([value_array, initial_array], axis=0)

                ebs.add_property(name=prop_name, value=new_values, return_gradient_order=gradient_order)  # pyright: ignore[reportCallIssue] - TODO: fix parameter name

        ebs.update_points(new_kpoints)
        ebs._mesh = ebs.to_mesh()
        return ebs

    def interpolate(
        self, interpolation_factor: int = 2, inplace: bool = True, _order: str = "F"
    ) -> ElectronicBandStructureMesh:
        """Interpolates the band structure meshes and properties using FFT interpolation.
        Creates and returns a new ElectronicBandStructure instance with interpolated data.

        Parameters
        ----------
        interpolation_factor : int, optional
            Factor by which to interpolate the mesh, by default 2
        inplace : bool, optional
            Whether to modify in place, by default True
        order : str, optional
            Array order, by default "F"

        Returns
        -------
        ElectronicBandStructureMesh
            New instance with interpolated data
        """
        logger.info(f"Interpolating band structure by factor {interpolation_factor}")
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        # Calculate new mesh dimensions
        kpoints_mesh = ebs.get_kpoints_mesh()
        nkx, nky, nkz = (
            kpoints_mesh.shape[0],
            kpoints_mesh.shape[1],
            kpoints_mesh.shape[2],
        )

        unique_x = kpoints_mesh[:, 0, 0, 0]
        unique_y = kpoints_mesh[0, :, 0, 1]
        unique_z = kpoints_mesh[0, 0, :, 2]

        xmin, xmax = np.min(unique_x), np.max(unique_x)
        ymin, ymax = np.min(unique_y), np.max(unique_y)
        zmin, zmax = np.min(unique_z), np.max(unique_z)

        new_x = np.linspace(xmin, xmax, nkx * interpolation_factor)
        new_y = np.linspace(ymin, ymax, nky * interpolation_factor)
        new_z = np.linspace(zmin, zmax, nkz * interpolation_factor)

        new_kpoints_mesh = np.array(np.meshgrid(new_z, new_y, new_x, indexing="ij"))
        new_kpoints = new_kpoints_mesh.reshape(-1, 3)

        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            prop = ebs.get_property(prop_name)
            if prop is None or not isinstance(prop, Property):
                continue
            value_mesh = math.array_to_mesh(
                array=value_array, nkx=ebs.n_kx, nky=ebs.n_ky, nkz=ebs.n_kz
            )
            interpolated_mesh = math.fft_interpolate_nd_3dmesh(  # pyright: ignore[reportUnknownMemberType]
                value_mesh, interpolation_factor
            )
            interpolated_value = math.mesh_to_array(interpolated_mesh)
            if interpolated_value is None:
                continue
            prop[calc_name, gradient_order] = interpolated_value

        ebs.update_points(new_kpoints)
        ebs._mesh = ebs.to_mesh()
        return ebs

    def expand_single_dimension(
        self, inplace: bool = False, fill_tol: float = 0.1
    ) -> ElectronicBandStructureMesh:
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        if not ebs.is2d:
            return ebs

        unique_coords = [ebs.ukx, ebs.uky, ebs.ukz]

        # Expand the kpoints and properties
        new_kpoints = ebs.kpoints.copy()
        for idim, dim_coords in enumerate(unique_coords):
            dim_len = len(dim_coords)
            if dim_len == 1:
                points_plus_fill = ebs.kpoints.copy()
                points_minus_fill = ebs.kpoints.copy()
                points_plus_fill[:, idim] += fill_tol
                points_minus_fill[:, idim] -= fill_tol

                new_kpoints = np.concatenate(
                    [new_kpoints, points_plus_fill, points_minus_fill], axis=0
                )

                for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
                    prop = ebs.get_property(prop_name)
                    if prop is None or not isinstance(prop, Property):
                        continue
                    initial_array = value_array[: ebs.n_kpoints]
                    new_points = np.concatenate([value_array, initial_array, initial_array], axis=0)

                    prop[calc_name, gradient_order] = new_points

        ebs.update_points(new_kpoints)
        sort_by_kpoints(ebs, inplace=True)
        ebs._mesh = ebs.to_mesh()
        return ebs

    def reduce_kpoints_to_plane(
        self, k_z_plane: float, k_z_plane_tol: float, inplace: bool = True
    ) -> None:
        """
        Reduces the kpoints to a plane
        """
        if inplace:
            ebs = self
        else:
            ebs = copy.deepcopy(self)

        if ebs.kpoints_cartesian is None:
            raise ValueError("kpoints_cartesian not set")

        i_kpoints_near_z_0 = np.where(
            np.logical_and(
                ebs.kpoints_cartesian[:, 2] < k_z_plane + k_z_plane_tol,
                ebs.kpoints_cartesian[:, 2] > k_z_plane - k_z_plane_tol,
            )
        )

        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            prop = ebs.get_property(prop_name)
            if prop is None or not isinstance(prop, Property):
                continue
            initial_array = value_array[: ebs.n_kpoints]
            new_points = initial_array[i_kpoints_near_z_0, ...][0]
            prop[calc_name, gradient_order] = new_points

        ebs.update_points(ebs.kpoints[i_kpoints_near_z_0, ...])
        ebs._mesh = ebs.to_mesh()

    def slice(
        self,
        normal: tuple[float, float, float] = (0, 0, 1),
        origin: tuple[float, float, float] = (0, 0, 0),
        scalars: tuple[str, npt.NDArray[Any]] | None = None,
        vectors: tuple[str, npt.NDArray[Any]] | None = None,
        as_cartesian: bool = True,
        **kwargs: Any,
    ) -> Any:  # Returns pv.PolyData which is not fully typed
        mesh = self.to_mesh(scalars=scalars, vectors=vectors, as_cartesian=as_cartesian)
        sliced = mesh.slice(normal=normal, origin=origin, **kwargs)

        if not as_cartesian:
            if self.reciprocal_lattice is None:
                raise ValueError("reciprocal_lattice not set")
            transform_matrix: npt.NDArray[np.float64] = np.eye(4)
            transform_matrix[:3, :3] = self.reciprocal_lattice.T
            sliced = sliced.transform(transform_matrix, inplace=False)

        n_slice_points: int = len(sliced.points)
        if n_slice_points == 0:
            error_message = "No points found in the slice."
            error_message += "Look at the coordinates in cartesian or fractional space to see if origin is right."
            raise ValueError(error_message)
        if scalars is not None:
            scalar_name, _ = scalars
            sliced.set_active_scalars(scalar_name)
        if vectors is not None:
            vector_name, _ = vectors
            sliced.set_active_vectors(vector_name)
        return sliced

    @override
    def gradient_func(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, points: npt.NDArray[np.float64], values: npt.NDArray[np.float64], **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        if self.reciprocal_lattice is None:
            raise ValueError("reciprocal_lattice not set")
        val_mesh = math.array_to_mesh(
            array=values,
            nkx=self.n_kx,
            nky=self.n_ky,
            nkz=self.n_kz,
            **kwargs,
        )
        gradients_mesh = math.calculate_3d_mesh_scalar_gradients(val_mesh, self.reciprocal_lattice)
        gradients_mesh *= physics.METER_ANGSTROM

        gradients_result = math.mesh_to_array(mesh=gradients_mesh, **kwargs)
        if gradients_result is None:
            raise ValueError("Failed to convert gradients mesh to array")
        gradients: npt.NDArray[np.float64] = gradients_result

        return gradients


def ibz2fbz(
    ebs: ElectronicBandStructure,
    rotations: npt.NDArray[np.float64] | None = None,
    kgrid_info: kpoints.KGridInfo | None = None,
    _decimals: int = 4,
    inplace: bool = True,
    **_kwargs: Any,
) -> ElectronicBandStructure:
    """Applys symmetry operations to the kpoints, bands, and projections

    Parameters
    ----------
    ebs : ElectronicBandStructure
        The electronic band structure
    rotations : npt.NDArray[np.float64] | None
        The point symmetry operations of the lattice
    kgrid_info : kpoints.KGridInfo | None
        K-grid information
    decimals : int
        The number of decimals to round the kpoints
        to when checking for uniqueness
    inplace : bool
        Whether to modify in place
    """
    if not inplace:
        ebs = copy.deepcopy(ebs)

    if ebs.is_grid:
        logger.warning("ElectronicBandStructure is already in the grid, skipping ibz2fz")
        return ebs
    logger.info("Applying symmetry operations to the kpoints")
    logger.info(f"Kgrid info: {kgrid_info}")

    rotations_list: list[npt.NDArray[np.float64]] = []
    if rotations is not None:
        rotations_list = list(rotations)
    elif ebs.structure is not None:
        rotations_list = list(ebs.structure.rotations)

    if len(rotations_list) == 0:
        logger.warning("No rotations provided, skipping ibz2fbz")
        return ebs

    n_kpoints = ebs.n_kpoints

    # Apply rotations and copy properties
    new_kpoints: npt.NDArray[np.float64] = ebs.kpoints.copy()
    for _, rotation in enumerate(rotations_list):
        # Rotate kpoints
        new_values = ebs.kpoints.dot(rotation.T)

        new_kpoints = np.concatenate([new_kpoints, new_values], axis=0)
        # Update properties
        for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
            initial_array = value_array[:n_kpoints]
            new_points = np.concatenate([value_array, initial_array], axis=0)
            prop = ebs.get_property(prop_name)
            if prop is None or not isinstance(prop, Property):
                continue
            prop[calc_name, gradient_order] = new_points

    # Apply boundary conditions to kpoints
    new_kpoints = -np.fmod(new_kpoints + 6.5, 1) + 0.5

    new_in_original_grid_indices: npt.NDArray[np.intp] | None = None
    if kgrid_info is not None:
        # Handle both KGRID_MODE enum and string for kgrid_mode
        kgrid_mode = kgrid_info.kgrid_mode
        kgrid_mode_value = kgrid_mode.value if isinstance(kgrid_mode, kpoints.KGRID_MODE) else str(kgrid_mode)
        kpoints_grid_points = kpoints.get_kpoints_from_kgrid(
            kgrid=kgrid_info.kgrid, kshift=kgrid_info.kshift, mode=kgrid_mode_value
        )

        diff = new_kpoints[:, np.newaxis, :] - kpoints_grid_points[np.newaxis, :, :]
        distances: npt.NDArray[np.float64] = np.linalg.norm(diff, axis=2)
        # Find the minimum distance for each k-point in new_kpoints to any
        # k-point in kpoints_grid_points
        min_distances = np.min(distances, axis=1)

        # Get the indices in new_kpoints where the minimum distance is within the tolerance
        new_in_original_grid_indices = np.where(min_distances < 0.000001)[0]

        new_kpoints = new_kpoints[new_in_original_grid_indices, ...]

    # # Floating point error can cause the kpoints to be off by 0.000001 or so
    # # causing the unique indices to misidentify the kpoints
    new_kpoints = new_kpoints.round(decimals=3)
    _, unique_indices = np.unique(new_kpoints, axis=0, return_index=True)

    new_kpoints = new_kpoints[unique_indices, ...]

    for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
        prop = ebs.get_property(prop_name)
        if prop is None or not isinstance(prop, Property):
            continue
        if new_in_original_grid_indices is not None:
            prop[calc_name, gradient_order] = value_array[new_in_original_grid_indices][
                unique_indices
            ]
        else:
            prop[calc_name, gradient_order] = value_array[unique_indices]

    ebs.update_points(new_kpoints)
    return sort_by_kpoints(ebs, inplace=inplace)


def sort_by_kpoints(
    ebs: ElectronicBandStructure, inplace: bool = True, order: str = "F"
) -> ElectronicBandStructure:
    """Sorts the bands and projected arrays by kpoints"""
    logger.info(f"Sorting kpoints by {order}")
    if not inplace:
        ebs = copy.deepcopy(ebs)

    if order == "C":
        sorted_indices = np.lexsort((ebs.kpoints[:, 2], ebs.kpoints[:, 1], ebs.kpoints[:, 0]))
    elif order == "F":
        sorted_indices = np.lexsort((ebs.kpoints[:, 0], ebs.kpoints[:, 1], ebs.kpoints[:, 2]))
    else:
        raise ValueError(f"Unknown order: {order}")

    ebs.update_points(ebs.kpoints[sorted_indices, ...])

    for prop_name, calc_name, gradient_order, value_array in ebs.iter_properties():
        prop = ebs.get_property(prop_name)
        if prop is None or not isinstance(prop, Property):
            continue
        prop[calc_name, gradient_order] = value_array[sorted_indices, ...]

    return ebs
