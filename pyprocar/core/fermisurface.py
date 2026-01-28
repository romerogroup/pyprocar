from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyvista as pv

from pyprocar.core.atomic_orbital_index import (
    AtomIndexer,
    OrbitalIndexer,
    ProjectionLabelBuilder,
    ProjectionSelectionResolver,
    ProjectionSelectionResult,
    SpinIndexer,
)
from pyprocar.core.brillouin_zone import BrillouinZone
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.property_store import PointSet, Property
from pyprocar.utils.physics import *

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

logger = logging.getLogger(__name__)


class FSNormMode(Enum):
    """Normalization modes for Fermi surface properties."""

    RAW = "raw"
    MAX = "max"
    TOTAL = "total"
    INTEGRAL = "integral"

    @classmethod
    def from_input(cls, input: str | FSNormMode | None) -> FSNormMode:
        """Convert string/None input to FSNormMode enum."""
        if isinstance(input, FSNormMode):
            return input
        if input is None:
            return cls.RAW
        lower_input = input.lower()
        mode_map = {
            "raw": cls.RAW,
            "max": cls.MAX,
            "total": cls.TOTAL,
            "integral": cls.INTEGRAL,
        }
        if lower_input in mode_map:
            return mode_map[lower_input]
        valid_modes = ", ".join(mode_map.keys())
        raise ValueError(f"Invalid normalization mode: {input}. Valid modes: {valid_modes}")

    @classmethod
    def list_modes(cls) -> list[str]:
        """Return list of valid mode strings."""
        return [mode.value for mode in cls]

    @staticmethod
    def get_normed_name(mode: FSNormMode, name: str) -> str:
        """Return property name with normalization prefix."""
        if mode == FSNormMode.RAW:
            return name
        return f"{mode.value}_{name}"

    @staticmethod
    def get_mode_prefix(mode: FSNormMode) -> str:
        """Return display prefix for normalization mode."""
        prefixes = {
            FSNormMode.RAW: "",
            FSNormMode.MAX: "Max-Normed",
            FSNormMode.TOTAL: "Total-Normed",
            FSNormMode.INTEGRAL: "Integral-Normed",
        }
        return prefixes.get(mode, "")


class FermiSurface(pv.PolyData):
    """
    A class to generate and manipulate Fermi surfaces from electronic band structure data.
    Extends PyVista's PolyData class for rich 3D visualization capabilities.

    Parameters
    ----------
    ebs : ElectronicBandStructure
        The electronic band structure object containing band data
    fermi : float, optional
        The Fermi energy level, by default 0.0
    fermi_shift : float, optional
        Energy shift to apply to the Fermi level, by default 0.0
    interpolation_factor : int, optional
        Factor to increase k-point density, by default 1
    projection_accuracy : str, optional
        Accuracy for projections ('high' or 'normal'), by default 'normal'
    max_distance : float, optional
        Maximum distance for point projections, by default 0.2
    """

    def __init__(
        self,
        points: npt.NDArray[np.float64],
        faces: npt.NDArray[Any],
        band_isosurfaces: dict[tuple[int, int], pv.PolyData],  # pyright: ignore[reportUnknownMemberType]
        isovalue: float,
        original_ebs: ElectronicBandStructureMesh,
        ebs: ElectronicBandStructureMesh,
        point_set: PointSet,
        point_data: dict[str, npt.NDArray[Any]] | None = None,
        cell_data: dict[str, npt.NDArray[Any]] | None = None,
        field_data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        self.points = points
        self.faces = faces
        self._band_isosurfaces = band_isosurfaces
        self._isovalue = isovalue
        self._original_ebs = original_ebs
        self._ebs = ebs
        self._point_set = point_set

        # Projection selection infrastructure (lazy initialized)
        self._projection_label_builder: ProjectionLabelBuilder | None = None
        self._projection_selection_resolver: ProjectionSelectionResolver | None = None

        # Cache invalidation tracking
        self._ebs_cache_version: int = 0
        self._cached_properties: dict[str, int] = {}  # property_name -> cache_version

        if "spin_band_index" not in point_set.property_store.keys():
            raise ValueError("spin_band_index not found in point_set.property_store")

        if "spin_index" not in point_set.property_store.keys():
            raise ValueError("spin_index not found in point_set.property_store")

        point_data_dict = point_data if point_data is not None else {}
        cell_data_dict = cell_data if cell_data is not None else {}
        field_data_dict = field_data if field_data is not None else {}

        self.point_data.update(point_data_dict)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        self.cell_data.update(cell_data_dict)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        self.field_data.update(field_data_dict)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]

        logger.debug(f"Fermi Surface: \n {self}")
        logger.debug(f"Fermi Surface Point Data: \n {self.point_data}")
        logger.debug(f"Fermi Surface Cell Data: \n {self.cell_data}")
        logger.info("___FermiSurface initialization complete___")

    @classmethod
    def from_code(
        cls,
        code: str,
        dirpath: str,
        use_cache: bool = False,
        ebs_filename: str = "ebs.pkl",
        reduce_bands_near_fermi: bool = False,
        reduce_bands_near_energy: float | None = None,
        reduce_bands_by_index: list[int] | None = None,
        padding: int = 10,
        fermi: float | None = None,
        fermi_shift: float = 0.0,
        **kwargs: Any,
    ) -> FermiSurface:
        ebs = ElectronicBandStructureMesh.from_code(
            code, dirpath, use_cache=use_cache, ebs_filename=ebs_filename
        )  # pyright: ignore[reportArgumentType]
        fermi_val = fermi if fermi is not None else ebs.fermi

        if reduce_bands_near_fermi:
            ebs.reduce_bands_near_fermi()
        elif reduce_bands_near_energy is not None:
            ebs.reduce_bands_near_energy(reduce_bands_near_energy)
        elif reduce_bands_by_index is not None:
            ebs.reduce_bands_by_index(reduce_bands_by_index)
        return cls.from_ebs(ebs, padding=padding, isovalue=fermi_val, isovalue_shift=fermi_shift)  # pyright: ignore[reportArgumentType]

    @classmethod
    def from_ebs(
        cls,
        ebs: ElectronicBandStructureMesh,
        padding: int = 10,
        isovalue: float | None = None,
        isovalue_shift: float | None = None,
        **kwargs: Any,
    ) -> FermiSurface:
        iso_val = isovalue if isovalue is not None else ebs.fermi
        if isovalue_shift is not None:
            iso_val += isovalue_shift
        results = generate_band_isosurfaces(ebs, isovalue=iso_val, padding=padding)
        combined_surface = results[0]
        band_isosurfaces = results[1]
        result_isovalue = results[2]
        original_ebs = results[3]
        padded_ebs = results[4]
        point_set = results[5]
        return cls(
            points=np.asarray(combined_surface.points, dtype=np.float64),  # pyright: ignore[reportUnknownMemberType]
            faces=np.asarray(combined_surface.faces, dtype=np.int64),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            band_isosurfaces=band_isosurfaces,
            isovalue=result_isovalue,
            original_ebs=original_ebs,
            ebs=padded_ebs,
            point_set=point_set,
        )

    @property
    def original_ebs(self) -> ElectronicBandStructureMesh:
        return self._original_ebs

    @property
    def ebs(self) -> ElectronicBandStructureMesh:
        return self._ebs

    @property
    def grid(self) -> pv.ImageData:
        x_coords = self.ebs.kpoints[:, 0]
        y_coords = self.ebs.kpoints[:, 1]
        z_coords = self.ebs.kpoints[:, 2]

        nx = self.ebs.n_kx
        ny = self.ebs.n_ky
        nz = self.ebs.n_kz

        padded_x_min = x_coords.min()
        padded_y_min = y_coords.min()
        padded_z_min = z_coords.min()

        # Must be the points on non-padded grid
        x_spacing = 1 / self.original_ebs.n_kx
        y_spacing = 1 / self.original_ebs.n_ky
        z_spacing = 1 / self.original_ebs.n_kz

        padded_grid: pv.ImageData = pv.ImageData(  # pyright: ignore[reportUnknownMemberType]
            dimensions=(nx, ny, nz),
            spacing=(x_spacing, y_spacing, z_spacing),
            origin=(padded_x_min, padded_y_min, padded_z_min),
        )
        return padded_grid  # pyright: ignore[reportReturnType]

    @property
    def transform_matrix_to_cart(self) -> npt.NDArray[np.float64]:
        transform_to_cart: npt.NDArray[np.float64] = np.eye(4)
        if self.reciprocal_lattice is not None:
            transform_to_cart[:3, :3] = self.reciprocal_lattice.T
        return transform_to_cart

    @property
    def transform_matrix_to_frac(self) -> npt.NDArray[np.float64]:
        transform_to_frac: npt.NDArray[np.float64] = np.eye(4)
        if self.reciprocal_lattice is not None:
            transform_to_frac[:3, :3] = np.linalg.inv(self.reciprocal_lattice.T)
        return transform_to_frac

    @property
    def fermi(self) -> float:
        return self.ebs.fermi

    @property
    def isovalue(self) -> float:
        return self._isovalue

    @property
    def fermi_shift(self) -> float:
        return self.isovalue - self.ebs.fermi

    @property
    def reciprocal_lattice(self) -> npt.NDArray[np.float64] | None:
        return self.ebs.reciprocal_lattice

    @property
    def n_points(self) -> int:
        return self.points.shape[0]  # pyright: ignore[reportReturnType]

    @property
    def point_set(self) -> PointSet:
        return self._point_set

    @property
    def band_isosurfaces(self) -> dict[tuple[int, int], pv.PolyData]:  # pyright: ignore[reportUnknownMemberType]
        return self._band_isosurfaces

    @property
    def band_indices(self) -> Any:
        return self.band_isosurfaces.keys()

    @property
    def band_spin_surface_map(self) -> dict[tuple[int, int], int]:
        band_spin_surface_map: dict[tuple[int, int], int] = {}
        for isurface, (iband, ispin) in enumerate(self.band_isosurfaces.keys()):
            band_spin_surface_map[(iband, ispin)] = isurface
        return band_spin_surface_map

    @property
    def surface_band_spin_map(self) -> dict[int, tuple[int, int]]:
        surface_band_spin_map: dict[int, tuple[int, int]] = {}
        for isurface, (iband, ispin) in enumerate(self.band_isosurfaces.keys()):
            surface_band_spin_map[isurface] = (iband, ispin)
        return surface_band_spin_map

    @property
    def band_spin_mask(self) -> dict[tuple[int, int], npt.NDArray[np.bool_]]:
        spin_band_index_prop = self.point_set.get_property("spin_band_index")
        if spin_band_index_prop is None or not isinstance(spin_band_index_prop, Property):
            raise ValueError("spin_band_index property not found")
        spin_band_index_array: npt.NDArray[Any] = spin_band_index_prop.value
        tmp_mask_dict: dict[tuple[int, int], npt.NDArray[np.bool_]] = {}
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            mask: npt.NDArray[np.bool_] = spin_band_index_array == surface_idx
            tmp_mask_dict[(iband, ispin)] = mask
        return tmp_mask_dict

    @property
    def spin_channel_mask(self) -> dict[Any, npt.NDArray[np.bool_]]:
        spin_index_prop = self.point_set.get_property("spin_index")
        if spin_index_prop is None or not isinstance(spin_index_prop, Property):
            raise ValueError("spin_index property not found")
        spin_index_array: npt.NDArray[Any] = spin_index_prop.value

        spin_channels: npt.NDArray[Any] = np.unique(spin_index_array)
        tmp_mask_dict: dict[Any, npt.NDArray[np.bool_]] = {}
        for ispin in spin_channels:
            mask: npt.NDArray[np.bool_] = spin_index_array == ispin
            tmp_mask_dict[ispin] = mask

        return tmp_mask_dict

    @cached_property
    def brillouin_zone(self) -> BrillouinZone:
        return self.get_brillouin_zone([1, 1, 1])

    @property
    def is2d(self) -> bool:
        return self.original_ebs.is2d

    def _invalidate_cache(self) -> None:
        """Invalidate all cached interpolated properties."""
        self._ebs_cache_version += 1
        logger.debug(f"Cache invalidated, new version: {self._ebs_cache_version}")

    def _is_cache_valid(self, property_name: str) -> bool:
        """Check if cached property is still valid."""
        if property_name not in self._cached_properties:
            return False
        return self._cached_properties[property_name] == self._ebs_cache_version

    def _mark_cached(self, property_name: str) -> None:
        """Mark property as cached at current version."""
        self._cached_properties[property_name] = self._ebs_cache_version

    def _get_projection_label_builder(self) -> ProjectionLabelBuilder:
        """Lazily initialize and return the projection label builder."""
        if self._projection_label_builder is None:
            atom_indexer = None
            if self.ebs.structure is not None:
                atom_indexer = AtomIndexer.from_structure(self.ebs.structure)

            spin_projection_names = getattr(self.ebs, "spin_projection_names", None)
            if spin_projection_names is not None:
                spin_indexer = SpinIndexer.from_projection_names(spin_projection_names)
            else:
                spin_indexer = SpinIndexer(names=["spin_up", "spin_down"])

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
            orbital_names = getattr(self.ebs, "orbital_names", None)
            is_non_colinear = getattr(self.ebs, "is_non_colinear", False)
            self._projection_selection_resolver = ProjectionSelectionResolver(
                label_builder=label_builder,
                orbital_names=orbital_names,
                is_non_colinear=is_non_colinear if is_non_colinear is not None else False,
            )
        return self._projection_selection_resolver

    def _resolve_projection_selection(
        self,
        *,
        atoms: Sequence[int] | int | None = None,
        orbitals: Sequence[int] | int | None = None,
        spins: Sequence[int] | int | None = None,
        species: Sequence[str] | str | None = None,
        species_orbital_map: Mapping[str, Iterable[int]] | None = None,
        atoms_orbital_map: Mapping[int | Iterable[int], Iterable[int]] | None = None,
    ) -> ProjectionSelectionResult:
        """
        Resolve projection selection parameters to canonical indices and labels.

        Parameters
        ----------
        atoms : Sequence[int] | int | None
            Atom indices to select
        orbitals : Sequence[int] | int | None
            Orbital indices to select
        spins : Sequence[int] | int | None
            Spin channel indices to select
        species : Sequence[str] | str | None
            Species names to select (resolves to atom indices)
        species_orbital_map : Mapping[str, Iterable[int]] | None
            Map from species to orbital indices
        atoms_orbital_map : Mapping[int | Iterable[int], Iterable[int]] | None
            Map from atom indices to orbital indices

        Returns
        -------
        ProjectionSelectionResult
            Resolved indices and labels
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

    def compute_projected_sum(
        self,
        atoms: Sequence[int] | int | None = None,
        orbitals: Sequence[int] | int | None = None,
        spins: Sequence[int] | int | None = None,
        species: Sequence[str] | str | None = None,
        species_orbital_map: Mapping[str, Iterable[int]] | None = None,
        atoms_orbital_map: Mapping[int | Iterable[int], Iterable[int]] | None = None,
        norm_mode: str | FSNormMode | None = "raw",
        label: str = "Projected Sum",
        name: str = "projected_sum",
        **kwargs: Any,
    ) -> Property:
        """
        Compute projected sum over selected atoms, orbitals, and spins on the Fermi surface.

        The projection is computed on the EBS grid and then interpolated onto the
        Fermi surface mesh.

        Parameters
        ----------
        atoms : Sequence[int] | int | None
            Atom indices to include in sum
        orbitals : Sequence[int] | int | None
            Orbital indices to include in sum
        spins : Sequence[int] | int | None
            Spin channels to include in sum
        species : Sequence[str] | str | None
            Species names to include (resolved to atom indices)
        species_orbital_map : Mapping[str, Iterable[int]] | None
            Map species to specific orbitals
        atoms_orbital_map : Mapping[int | Iterable[int], Iterable[int]] | None
            Map atoms to specific orbitals
        norm_mode : str | FSNormMode | None
            Normalization mode: 'raw', 'max', 'total', 'integral'
        label : str
            Display label for the property
        name : str
            Internal name for the property

        Returns
        -------
        Property
            Property object with projected sum values and metadata
        """
        # Check if projections are available
        if not hasattr(self.ebs, "projected") or self.ebs.projected is None:
            raise ValueError("Projected band structure data not available")

        # Resolve selection parameters
        has_selection = any(
            x is not None
            for x in [atoms, orbitals, spins, species, species_orbital_map, atoms_orbital_map]
        )

        selection = None
        if has_selection:
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
        else:
            atoms_list = None
            orbitals_list = None
            spins_list = None

        # Compute projected sum on EBS grid
        ebs_property = self.ebs.compute_projected_sum(
            atoms=atoms_list,
            orbitals=orbitals_list,
            spins=spins_list,
            norm_mode="raw",  # We'll normalize after interpolation
        )
        if isinstance(ebs_property, list):
            ebs_property = ebs_property[0]

        # Interpolate to surface
        if not isinstance(ebs_property, Property):
            raise ValueError("Expected Property object from compute_projected_sum")
        surface_values: npt.NDArray[np.float64] = self.interpolate_to_surface(ebs_property.value)

        # Build property with metadata
        prop = self._build_property(
            values=surface_values,
            label=label,
            name=name,
            norm_mode=norm_mode,
            selection=selection,
            **kwargs,
        )

        # Cache in PointSet
        self.point_set.add_property(prop)
        self._mark_cached(prop.name)

        # Sync to PyVista
        self.set_values(prop.name, prop.value)

        return prop

    def normalize(
        self,
        mode: str | FSNormMode | None,
        values_array: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Normalize values array according to specified mode.

        Parameters
        ----------
        mode : str | FSNormMode | None
            Normalization mode
        values_array : np.ndarray
            Values to normalize

        Returns
        -------
        np.ndarray
            Normalized values
        """
        mode = FSNormMode.from_input(mode)

        if mode == FSNormMode.RAW:
            return values_array
        elif mode == FSNormMode.MAX:
            return self._normalize_max(values_array, **kwargs)
        elif mode == FSNormMode.TOTAL:
            return self._normalize_total(values_array, **kwargs)
        elif mode == FSNormMode.INTEGRAL:
            return self._normalize_integral(values_array, **kwargs)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def _normalize_max(self, values_array: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Normalize by maximum absolute value."""
        values_array = np.asarray(values_array, dtype=np.float64)

        # Find max along all axes except the first (points axis)
        max_val = np.max(np.abs(values_array))
        if max_val == 0:
            return values_array

        return values_array / max_val

    def _normalize_total(self, values_array: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Normalize by sum of all values."""
        values_array = np.asarray(values_array, dtype=np.float64)

        total = np.sum(np.abs(values_array))
        if total == 0:
            return values_array

        return values_array / total

    def _normalize_integral(
        self, values_array: npt.NDArray[Any], **kwargs: Any
    ) -> npt.NDArray[np.float64]:
        """
        Normalize by surface integral approximation.

        Uses cell areas from the mesh to weight contributions.
        """
        values_array = np.asarray(values_array, dtype=np.float64)

        # Compute cell areas for weighting
        cell_sizes_result: npt.NDArray[np.float64] = np.asarray(self.compute_cell_sizes(), dtype=np.float64)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        if len(cell_sizes_result) == 0:
            # Fallback to simple sum if cell sizes unavailable
            return self._normalize_total(values_array, **kwargs)

        # Map cell areas to points (average of adjacent cells)
        point_weights: npt.NDArray[np.float64] = np.zeros(self.n_points, dtype=np.float64)
        cell_count: npt.NDArray[np.float64] = np.zeros(self.n_points, dtype=np.float64)

        for i, cell in enumerate(self.cell):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
            for point_idx in cell:  # pyright: ignore[reportUnknownVariableType]
                point_weights[point_idx] += cell_sizes_result[i]  # pyright: ignore[reportUnknownArgumentType]
                cell_count[point_idx] += 1  # pyright: ignore[reportUnknownArgumentType]

        # Average weights
        nonzero_mask = cell_count > 0
        point_weights[nonzero_mask] /= cell_count[nonzero_mask]

        # Compute weighted integral
        if values_array.ndim == 1:
            integral = np.sum(values_array * point_weights)
        else:
            integral = np.sum(values_array * point_weights[:, np.newaxis])

        if integral == 0:
            return values_array

        return values_array / integral

    def _build_property(
        self,
        values: np.ndarray,
        label: str,
        name: str,
        norm_mode: str | FSNormMode | None = None,
        selection: ProjectionSelectionResult | None = None,
        include_normal_label: bool = False,
        **kwargs: Any,
    ) -> Property:
        """
        Build a Property object with rich metadata.

        Parameters
        ----------
        values : np.ndarray
            Property values
        label : str
            Display label
        name : str
            Internal property name
        norm_mode : str | FSNormMode | None
            Normalization mode to apply
        selection : ProjectionSelectionResult | None
            Projection selection info for metadata
        include_normal_label : bool
            Whether to include normalization in label

        Returns
        -------
        Property
            Property with values and metadata
        """
        # Resolve and apply normalization
        norm_mode = FSNormMode.from_input(norm_mode)
        normed_name = FSNormMode.get_normed_name(norm_mode, name)

        values = self.normalize(mode=norm_mode, values_array=values, **kwargs)

        # Compute data limits
        data_min = float(np.min(values))
        data_max = float(np.max(values))
        data_lim = (data_min, data_max)

        # Build base metadata
        metadata: dict[str, Any] = {
            "norm_mode": norm_mode,
            "data_lim": data_lim,
            "scalar_label": label,
            "label": label,
            "label_plain": label,
            "include_normal_label": include_normal_label,
        }

        # Add normalization prefix to label if requested
        if include_normal_label and norm_mode != FSNormMode.RAW:
            prefix = FSNormMode.get_mode_prefix(norm_mode)
            metadata["label"] = f"{prefix} {label}"
            metadata["label_plain"] = f"{prefix} {label}"

        # Add selection metadata if present
        if selection is not None:
            label_plain_list, label_latex_list = self._format_selection_label(
                selection=selection,
                normalize=norm_mode != FSNormMode.RAW,
                include_normal_label=include_normal_label,
            )

            metadata.update(
                {
                    "atoms": list(selection.atoms) if selection.atoms else None,
                    "orbitals": list(selection.orbitals) if selection.orbitals else None,
                    "spins": list(selection.spins) if selection.spins else None,
                    "species": list(selection.species) if selection.species else None,
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
                    "label_combined": selection.labels.combined,
                    "label_combined_latex": selection.labels.combined_latex,
                    "label": label_latex_list,
                    "label_plain": label_plain_list,
                }
            )

        # Merge additional kwargs
        if kwargs:
            # Filter out normalization-related kwargs
            extra_metadata = {k: v for k, v in kwargs.items() if k not in ("atoms", "orbitals", "spins")}
            metadata.update(extra_metadata)

        # Construct Property
        prop = Property(
            name=normed_name,
            value=values,
            point_set=self.point_set,
            metadata=metadata,
            label=label,
        )

        return prop

    @staticmethod
    def _format_selection_label(
        selection: ProjectionSelectionResult,
        *,
        normalize: bool,
        include_normal_label: bool = False,
    ) -> tuple[list[str], list[str]]:
        """
        Format selection into plain and LaTeX labels.

        Parameters
        ----------
        selection : ProjectionSelectionResult
            Resolved selection with labels
        normalize : bool
            Whether normalization is applied
        include_normal_label : bool
            Whether to include normalization mode in label

        Returns
        -------
        tuple[list[str], list[str]]
            (plain_labels, latex_labels)
        """
        mode_plain = "fraction" if normalize else "raw"
        mode_latex = r"\mathrm{fraction}" if normalize else r"\mathrm{raw}"

        prefix_plain = selection.labels.prefix_plain
        prefix_latex = selection.labels.prefix_latex

        spin_components = selection.labels.spin_components or ("",)
        spin_components_latex = selection.labels.spin_components_latex or ("",)

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
                body_latex = f"{body_latex}[{component_latex}]" if body_latex else f"[{component_latex}]"
            body_latex = body_latex or r"\mathrm{all}"
            labels_latex.append(f"${body_latex}{normal_suffix_latex}$")

        return labels_plain, labels_latex

    def save(self, path: str) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Save FermiSurface to file.

        Parameters
        ----------
        path : str
            Output file path. Extension determines format (.pkl, .npz, etc.)

        Note
        ----
        This saves the surface geometry and point data. The EBS objects are
        also saved but may need to be reloaded from the original calculation
        for full functionality.
        """
        import copy
        import pickle
        from pathlib import Path

        path_obj = Path(path)

        # Serialize point_set properties to avoid weak reference issues
        point_set_data: dict[str, npt.NDArray[np.float64] | dict[str, dict[str, Any]]] = {
            "points": self.point_set.points.copy(),
            "properties": {},
        }
        properties_dict: dict[str, dict[str, Any]] = {}
        for prop_name, prop in self.point_set.property_store.items():
            prop_gradients = prop.gradients
            properties_dict[prop_name] = {
                "value": prop.value.copy(),
                "gradients": {k: v.copy() for k, v in prop_gradients.items() if v is not None and len(v) > 0},  # pyright: ignore[reportUnnecessaryComparison]
                "units": prop.units,
                "label": prop.label,
                "metadata": copy.deepcopy(prop.metadata) if prop.metadata else {},
                "data_lim": prop.data_lim,
            }
        point_set_data["properties"] = properties_dict

        # Serialize band_isosurfaces - just store the geometry
        band_isosurfaces_data: dict[tuple[int, int], dict[str, npt.NDArray[Any]]] = {}
        for key, surface in self.band_isosurfaces.items():
            band_isosurfaces_data[key] = {
                "points": np.asarray(surface.points, dtype=np.float64),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                "faces": np.asarray(surface.faces, dtype=np.int64),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            }

        # Collect all data needed for reconstruction
        save_data: dict[str, Any] = {
            "points": np.asarray(self.points, dtype=np.float64).copy(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            "faces": np.asarray(self.faces, dtype=np.int64).copy(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            "isovalue": self.isovalue,
            "point_data": {k: np.array(v) for k, v in self.point_data.items()},  # pyright: ignore[reportUnknownMemberType]
            "cell_data": {k: np.array(v) for k, v in self.cell_data.items()},  # pyright: ignore[reportUnknownMemberType]
            "field_data": dict(self.field_data),  # pyright: ignore[reportUnknownMemberType]
            "point_set_data": point_set_data,
            "band_isosurfaces_data": band_isosurfaces_data,
            # Note: EBS objects are not saved due to weak reference issues
            # Users should keep a reference to EBS or reload from calculation
        }

        with open(path_obj, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"FermiSurface saved to {path}")

    @classmethod
    def load(cls, path: str, ebs: ElectronicBandStructureMesh | None = None) -> FermiSurface:
        """
        Load FermiSurface from file.

        Parameters
        ----------
        path : str
            Input file path
        ebs : ElectronicBandStructureMesh | None
            Optional EBS to associate with the loaded surface. If not provided,
            property computation methods will not be available.

        Returns
        -------
        FermiSurface
            Loaded FermiSurface instance

        Note
        ----
        The loaded surface has full visualization capabilities. For property
        computation (compute_projected_sum, etc.), you need to either pass the
        original EBS or reload it from the calculation directory.
        """
        import pickle
        from pathlib import Path

        path_obj = Path(path)

        with open(path_obj, "rb") as f:
            save_data = pickle.load(f)

        # Reconstruct point_set from serialized data
        point_set_data = save_data["point_set_data"]
        point_set = PointSet(points=point_set_data["points"])

        # Reconstruct properties
        for prop_name, prop_data in point_set_data["properties"].items():
            prop = Property(
                name=prop_name,
                value=prop_data["value"],
                gradients=prop_data.get("gradients"),
                units=prop_data.get("units"),
                label=prop_data.get("label"),
                point_set=point_set,
                metadata=prop_data.get("metadata"),
                data_lim=prop_data.get("data_lim"),
            )
            point_set.add_property(prop)

        # Reconstruct band_isosurfaces from saved geometry
        band_isosurfaces: dict[tuple[int, int], pv.PolyData] = {}  # pyright: ignore[reportUnknownMemberType]
        for key, surface_data in save_data["band_isosurfaces_data"].items():
            surface: pv.PolyData = pv.PolyData(surface_data["points"], surface_data["faces"])  # pyright: ignore[reportUnknownMemberType]
            band_isosurfaces[key] = surface

        # Create a temporary PolyData with all the saved data
        temp_polydata: pv.PolyData = pv.PolyData(save_data["points"], save_data["faces"])  # pyright: ignore[reportUnknownMemberType]

        # Add point/cell/field data to temp before shallow_copy
        for k, v in save_data["point_data"].items():
            temp_polydata.point_data[k] = v  # pyright: ignore[reportUnknownMemberType]
        for k, v in save_data["cell_data"].items():
            temp_polydata.cell_data[k] = v  # pyright: ignore[reportUnknownMemberType]
        temp_polydata.field_data.update(save_data["field_data"])  # pyright: ignore[reportUnknownMemberType]

        # Create the FermiSurface by calling the parent class constructor
        # and then manually setting our attributes
        fs: FermiSurface = pv.PolyData.__new__(cls)  # pyright: ignore[reportUnknownMemberType]
        fs.shallow_copy(temp_polydata)  # pyright: ignore[reportUnknownMemberType]

        # Copy PyVista-specific internal state that isn't handled by VTK's shallow_copy
        # This is required for point_data/cell_data access to work properly
        for attr in ["_association_bitarray_names", "_association_complex_names"]:
            if hasattr(temp_polydata, attr):
                setattr(fs, attr, getattr(temp_polydata, attr).copy())

        fs._band_isosurfaces = band_isosurfaces
        fs._isovalue = save_data["isovalue"]
        fs._original_ebs = ebs  # type: ignore[assignment]
        fs._ebs = ebs  # type: ignore[assignment]
        fs._point_set = point_set

        # Projection selection infrastructure (lazy initialized)
        fs._projection_label_builder = None
        fs._projection_selection_resolver = None

        # Cache invalidation tracking
        fs._ebs_cache_version = 0
        fs._cached_properties = {}

        logger.info(f"FermiSurface loaded from {path}")
        return fs

    def get_brillouin_zone(self, supercell: list[int]):
        """Returns the BrillouinZone of the material

        Parameters
        ----------
        supercell : List[int]
            The supercell to use for the BrillouinZone

        Returns
        -------
        pyprocar.core.BrillouinZone
            The BrillouinZone of the material
        """

        return BrillouinZone(self.reciprocal_lattice, supercell)  # pyright: ignore[reportArgumentType]

    def get_property(self, key: Any, **kwargs: Any) -> npt.NDArray[np.float64]:
        prop_name, (calc_name, gradient_order) = self.ebs._extract_key(key)  # pyright: ignore[reportPrivateUsage]

        property_value: npt.NDArray[np.float64]
        if prop_name not in self.point_set.property_store:
            property_value = self.compute_property(prop_name, **kwargs)
            property_obj = Property(name=prop_name, value=property_value)
            self.point_set.add_property(property_obj)
        else:
            prop = self.point_set.get_property(prop_name)
            if prop is None or not isinstance(prop, Property):
                raise ValueError(f"Property {prop_name} not found")
            calc_name_safe = calc_name if calc_name is not None else "value"
            result = prop[calc_name_safe, gradient_order]
            if isinstance(result, np.ndarray):
                property_value = result
            else:
                raise ValueError(f"Unexpected property type for {prop_name}")

        self.set_values(prop_name, property_value)
        return property_value

    def select_bands(
        self, bands_spin_indices: list[tuple[int, int]], return_relative_indices: bool = False
    ) -> FermiSurface | tuple[FermiSurface, npt.NDArray[np.intp]]:
        """
        Select specific bands by filtering the Fermi surface to only include points
        from the specified band-spin combinations.

        Parameters
        ----------
        bands_spin_indices : List[tuple[int, int]]
            List of (band_index, spin_index) tuples to keep in the surface

        Returns
        -------
        None
            Modifies the current FermiSurface object in place
        """
        logger.info(f"Selecting bands: {bands_spin_indices}")

        # Validate that all requested band-spin combinations exist
        selected_band_surfaces = {}
        for iband, ispin in bands_spin_indices:
            if (iband, ispin) not in self.band_spin_mask:
                available_bands = list(self.band_spin_mask.keys())
                selected_band_surfaces[(iband, ispin)] = self.band_isosurfaces[(iband, ispin)]

                raise ValueError(
                    f"Band-spin combination ({iband}, {ispin}) not found in Fermi surface. "
                    f"Available combinations: {available_bands}"
                )

        # Create combined mask for all selected band-spin combinations
        combined_mask = np.zeros(self.n_points, dtype=bool)
        for iband, ispin in bands_spin_indices:
            mask = self.band_spin_mask[(iband, ispin)]
            combined_mask |= mask

        logger.debug(f"Combined mask selects {combined_mask.sum()} out of {self.n_points} points")
        logger.debug(f"Combined mask shape: {combined_mask.shape}")
        # If no points are selected, create an empty surface
        points: npt.NDArray[np.float64]
        faces: npt.NDArray[Any]
        point_data_dict: dict[str, npt.NDArray[Any]] | None
        cell_data_dict: dict[str, npt.NDArray[Any]] | None
        field_data_dict: dict[str, Any] | None
        new_point_set: PointSet
        fs_indices: npt.NDArray[np.intp] = np.array([], dtype=np.intp)
        selected_band_surfaces: dict[tuple[int, int], pv.PolyData] = {}  # pyright: ignore[reportUnknownMemberType]
        if not combined_mask.any():
            logger.warning("No points selected - creating empty surface")

            points = np.empty((0, 3), dtype=np.float64)
            faces = np.empty((0, 4), dtype=np.int32)
            point_data_dict = {}
            cell_data_dict = {}
            field_data_dict = {}
            new_point_set = self.point_set

        else:
            new_surface, fs_indices = self.remove_points(combined_mask, inplace=False)  # pyright: ignore[reportUnknownMemberType]
            points = np.asarray(new_surface.points, dtype=np.float64)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            faces = np.asarray(new_surface.faces, dtype=np.int64)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            point_data_dict = dict(new_surface.point_data)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            cell_data_dict = dict(new_surface.cell_data)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            field_data_dict = dict(new_surface.field_data)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            new_point_set = self.point_set.select_points(fs_indices)
            for iband, ispin in bands_spin_indices:
                if (iband, ispin) in self.band_isosurfaces:
                    selected_band_surfaces[(iband, ispin)] = self.band_isosurfaces[(iband, ispin)]

        fs = FermiSurface(
            points=points,
            faces=faces,
            band_isosurfaces=selected_band_surfaces,
            isovalue=self.isovalue,
            original_ebs=self.original_ebs,
            ebs=self.ebs,
            point_set=new_point_set,
            point_data=point_data_dict,
            cell_data=cell_data_dict,
            field_data=field_data_dict,
        )

        if return_relative_indices:
            return fs, fs_indices
        else:
            return fs

    def set_surface_point_data(self, name: str, values: npt.NDArray[Any]) -> None:
        if values.shape[-1] == 3:
            last_dim = 3
        else:
            last_dim = 1

        self.point_data[name] = np.zeros(shape=(self.n_points, last_dim))  # pyright: ignore[reportUnknownMemberType]
        if self.ebs.is_band_property(values):
            logger.debug(f"Adding band resolved to fermi surface point_data: {name}")
            for (iband, ispin), _surface_idx in self.band_spin_surface_map.items():
                values_band_values = values[:, iband, ispin, ...]
                mask = self.band_spin_mask[(iband, ispin)]
                values_band_values[~mask] = 0
                self.point_data[name] += values_band_values  # pyright: ignore[reportUnknownMemberType]
        else:
            logger.debug(f"Adding scalar to fermi surface point_data: {name}")
            self.point_data[name] += values  # pyright: ignore[reportUnknownMemberType]

    def set_scalars(self, name: str, value: npt.NDArray[Any]) -> None:
        self.set_surface_point_data(name, value)
        self.set_active_scalars(name, preference="point")  # pyright: ignore[reportUnknownMemberType]

    def set_vectors(self, name: str, value: npt.NDArray[Any], set_scalar: bool = True) -> None:
        if value.shape[-1] != 3:
            raise ValueError(f"Vector data must have 3 dimensions. Got {value.shape[-1]}.")

        vector_magnitude: npt.NDArray[np.float64] = np.linalg.norm(value, axis=-1)
        if set_scalar:
            self.set_surface_point_data(f"{name}-norm", vector_magnitude)
            self.set_active_scalars(f"{name}-norm", preference="point")  # pyright: ignore[reportUnknownMemberType]
        self.set_surface_point_data(name, value)
        self.set_active_vectors(name, preference="point")  # pyright: ignore[reportUnknownMemberType]

    def set_values(self, name: str, value: npt.NDArray[Any], **kwargs: Any) -> None:
        if value.shape[-1] == 3:
            self.set_vectors(name, value, **kwargs)
        else:
            self.set_scalars(name, value, **kwargs)

    def set_band_colors(
        self,
        colors: list[str | tuple[int, int, int]] | None = None,
        surface_color_map: dict[tuple[int, int], str | tuple[int, int, int]] | None = None,
        cmap: tuple[str, ...] = ("winter", "autumn"),
    ) -> None:
        bands_colors: npt.NDArray[np.float64] = np.zeros((self.n_points, 4))
        if surface_color_map is not None:
            assert len(list(surface_color_map.keys())) == len(self.band_spin_surface_map), (
                "Number of colors must match number of bands"
            )
            for (iband, ispin), color in surface_color_map.items():
                color_rgba: tuple[float, float, float, float]
                if isinstance(color, str):
                    color_rgba = tuple(mcolors.to_rgba(color))  # type: ignore[assignment] # pyright: ignore[reportArgumentType, reportUnknownArgumentType]
                else:
                    color_rgba = (float(color[0]), float(color[1]), float(color[2]), 1.0)
                if (iband, ispin) not in self.band_spin_mask:
                    raise ValueError(
                        f"Band-spin combination ({iband}, {ispin}) not found in Fermi surface. "
                        f"Available combinations: {self.band_spin_mask.keys()}"
                    )
                bands_spin_mask = self.band_spin_mask[(iband, ispin)]
                bands_colors[bands_spin_mask] = color_rgba

        elif colors is not None:
            assert len(colors) == len(self.band_spin_surface_map), (
                "Number of colors must match number of bands"
            )
            i_surface = 0
            for (_iband, _ispin), bands_spin_mask in self.band_spin_mask.items():
                color = colors[i_surface]
                color_rgba_val: tuple[float, float, float, float]
                if isinstance(color, str):
                    color_rgba_val = tuple(mcolors.to_rgba(color))  # type: ignore[assignment] # pyright: ignore[reportArgumentType, reportUnknownArgumentType]
                else:
                    color_rgba_val = (float(color[0]), float(color[1]), float(color[2]), 1.0)
                bands_colors[bands_spin_mask] = color_rgba_val
                i_surface += 1
        else:
            for (iband, ispin), _isurface in self.band_spin_surface_map.items():
                cmapper: Colormap = plt.colormaps[cmap[ispin]]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]

                bands_spin_mask = self.band_spin_mask[(iband, ispin)]
                color_arr: npt.NDArray[np.float64] = np.asarray([cmapper(iband)] * int(bands_spin_mask.sum()), dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]
                bands_colors[bands_spin_mask] = color_arr

        self.point_data["bands"] = bands_colors  # pyright: ignore[reportUnknownMemberType]
        self.set_active_scalars("bands", preference="point")  # pyright: ignore[reportUnknownMemberType]

    def set_spin_colors(self, colors: tuple[str, str] = ("red", "blue")) -> None:
        bands_colors: npt.NDArray[np.float64] = np.zeros((self.n_points, 4))

        for spin_channel, mask in self.spin_channel_mask.items():
            color_rgba: tuple[float, float, float, float] = tuple(mcolors.to_rgba(colors[spin_channel]))  # type: ignore[assignment] # pyright: ignore[reportArgumentType, reportUnknownArgumentType]
            bands_colors[mask] = color_rgba

        self.point_data["spin"] = bands_colors  # pyright: ignore[reportUnknownMemberType]
        self.set_active_scalars("spin", preference="point")  # pyright: ignore[reportUnknownMemberType]

    def compute_gradients(
        self, gradient_order: int, names: list[str] | None = None, **kwargs: Any
    ) -> None:
        if names is None:
            names = list(self.point_set.property_store.keys())
        if gradient_order < 0:
            raise ValueError(f"Gradient order must be greater than 0. Got {gradient_order}.")
        self.ebs.compute_gradients(gradient_order=gradient_order, names=names)

        for prop_name, calc_name, grad_order, value_array in self.ebs.iter_properties():
            property_obj = self.point_set.get_property(prop_name)
            if property_obj is None or not isinstance(property_obj, Property):
                continue
            surface_points = self.interpolate_to_surface(value_array)
            property_obj[calc_name, grad_order] = surface_points

        return None

    def compute_property(self, name: str, **kwargs: Any) -> npt.NDArray[np.float64]:
        if name == "fermi_velocity":
            return self.compute_fermi_velocity(**kwargs)
        elif name == "fermi_speed":
            return self.compute_fermi_speed(**kwargs)
        else:
            property_obj = self.ebs.get_property(name, **kwargs)
            if property_obj is None or not isinstance(property_obj, Property):
                raise ValueError(f"Property {name} not found")
            surface_points = self.interpolate_to_surface(property_obj.value)
            return surface_points

    def compute_fermi_speed(self, **kwargs: Any) -> npt.NDArray[np.float64]:
        band_speed = self.ebs.get_property("bands_speed", **kwargs)
        if band_speed is None or not isinstance(band_speed, Property):
            raise ValueError("bands_speed property not found")
        surface_points = self.interpolate_to_surface(band_speed.value)
        return surface_points

    def compute_fermi_velocity(self, **kwargs: Any) -> npt.NDArray[np.float64]:
        band_velocity = self.ebs.get_property("bands_velocity", **kwargs)
        if band_velocity is None or not isinstance(band_velocity, Property):
            raise ValueError("bands_velocity property not found")
        surface_points = self.interpolate_to_surface(band_velocity.value)
        return surface_points

    def interpolate_to_surface(
        self, property_value: npt.NDArray[Any]
    ) -> npt.NDArray[np.float64]:
        grid: pv.ImageData = copy.deepcopy(self.grid)  # pyright: ignore[reportUnknownMemberType]
        grid.point_data["property"] = property_value.reshape(  # pyright: ignore[reportUnknownMemberType]
            (property_value.shape[0], -1), order="F"
        )
        interpolated_surface = self._create_interpolated_surface(grid=grid)
        property_surface_points: npt.NDArray[np.float64] = np.asarray(
            interpolated_surface.point_data["property"]  # pyright: ignore[reportUnknownMemberType]
        ).reshape(
            (-1, *property_value.shape[1:]), order="F"
        )
        return property_surface_points

    def update_point_data(
        self,
        name: str,
        point_data: dict[str, npt.NDArray[Any]] | npt.NDArray[Any] | None = None,
        surface: pv.UnstructuredGrid | None = None,  # pyright: ignore[reportUnknownMemberType]
    ) -> None:
        if isinstance(point_data, dict):
            self.point_data.update(point_data)  # pyright: ignore[reportUnknownMemberType]
        elif isinstance(point_data, np.ndarray):
            self.point_data[name] = point_data  # pyright: ignore[reportUnknownMemberType]
        elif surface is not None:
            self.point_data.update(surface.point_data)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        else:
            raise ValueError(
                "Either point_data or a surface with point_data attribute must be provided"
            )

        point_data_arr: npt.NDArray[Any] = np.asarray(self.point_data[name])  # pyright: ignore[reportUnknownMemberType]
        is_point_data_vector = point_data_arr.shape[-1] == 3
        logger.debug(f"is_point_data_vector|{name}: {is_point_data_vector}")
        if is_point_data_vector:
            scalar_name = f"{name}-norm"
            self.point_data[scalar_name] = np.linalg.norm(point_data_arr, axis=-1)  # pyright: ignore[reportUnknownMemberType]
            self.set_active_scalars(scalar_name, preference="point")  # pyright: ignore[reportUnknownMemberType]
            self.set_active_vectors(name, preference="point")  # pyright: ignore[reportUnknownMemberType]
        else:
            self.set_active_scalars(name, preference="point")  # pyright: ignore[reportUnknownMemberType]

    def extend_surface(self, zone_directions: list[list[int] | tuple[int, int, int]]) -> FermiSurface:
        """
        Method to extend the surface in the direction of a reciprocal lattice vecctor

        Parameters
        ----------
        zone_directions : List[List[int] or Tuple[int,int,int]], optional
            List of directions to expand to, by default None
        """
        # The following code  creates exteneded surfaces in a given direction
        print(
            "Warning. Extending the surface will disable the further calculation of properties.\n"
            "Existing properties will still be available."
        )
        print("To enable the calculation of properties, please create a new FermiSurface object.\n")

        new_surface: FermiSurface = copy.deepcopy(self)
        initial_surface: FermiSurface = copy.deepcopy(self)
        initial_band_surfaces: dict[tuple[int, int], pv.PolyData] = copy.deepcopy(self.band_isosurfaces)  # pyright: ignore[reportUnknownMemberType]
        new_band_surfaces: dict[tuple[int, int], pv.PolyData] = copy.deepcopy(self.band_isosurfaces)  # pyright: ignore[reportUnknownMemberType]
        new_point_set: PointSet = copy.deepcopy(self.point_set)

        if self.ebs.reciprocal_lattice is None:
            raise ValueError("reciprocal_lattice is None, cannot extend surface")
        recip_lattice = self.ebs.reciprocal_lattice

        for direction in zone_directions:
            surface_copy: FermiSurface = copy.deepcopy(initial_surface)
            translation_vec: npt.NDArray[np.float64] = np.dot(direction, recip_lattice)
            translated_surface: FermiSurface = surface_copy.translate(translation_vec, inplace=True)  # pyright: ignore[reportUnknownMemberType, reportAssignmentType, reportUnknownVariableType]
            new_surface = new_surface.merge(translated_surface, merge_points=False)  # pyright: ignore[reportUnknownMemberType, reportAssignmentType, reportUnknownVariableType]

            # Copying exisiting band_isosurfaces
            for (iband, ispin), initial_band_surface in initial_band_surfaces.items():
                new_band_surface_cp: pv.PolyData = copy.deepcopy(initial_band_surface)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                new_band_surface_cp += initial_band_surface.translate(translation_vec, inplace=True)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                new_band_surfaces[(iband, ispin)] += new_band_surface_cp  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

            # Copying exisiting properties
            for (
                prop_name,
                calc_name,
                gradient_order,
                value_array,
            ) in new_point_set.iter_property_arrays():
                current_property = self.point_set.get_property(prop_name)
                if current_property is None or not isinstance(current_property, Property):
                    continue
                current_result = current_property[calc_name, gradient_order]
                if not isinstance(current_result, np.ndarray):
                    continue
                current_array: npt.NDArray[np.float64] = current_result

                new_array: npt.NDArray[Any] = np.concatenate((value_array, current_array), axis=0)
                new_property = new_point_set.get_property(prop_name)
                if new_property is not None and isinstance(new_property, Property):
                    new_property[calc_name, gradient_order] = new_array

        new_point_set.update_points(np.asarray(new_surface.points))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        fs = FermiSurface(
            points=np.asarray(new_surface.points, dtype=np.float64),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            faces=np.asarray(new_surface.faces, dtype=np.int64),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            band_isosurfaces=new_band_surfaces,
            isovalue=self.isovalue,
            original_ebs=self.original_ebs,
            ebs=self.ebs,
            point_set=new_point_set,
            point_data=dict(new_surface.point_data),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            cell_data=dict(new_surface.cell_data),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            field_data=dict(new_surface.field_data),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        return fs

    def _create_interpolated_surface(
        self,
        grid: pv.ImageData | None = None,
        meshgrids: dict[str, npt.NDArray[Any]] | None = None,
        n_points: int = 40,
        sharpness: int = 20,
        strategy: Literal["null_value", "mask_points", "closest_point"] = "null_value",
    ) -> pv.UnstructuredGrid:  # pyright: ignore[reportUnknownMemberType]
        """
        Interpolate data from a grid or meshgrids onto the Fermi surface.

        This method takes either a PyVista ImageData object or a dictionary of
        numpy meshgrids and interpolates the data onto the Fermi surface points.
        The interpolated data is stored in the point_data dictionary of the
        Fermi surface.

        Parameters
        ----------
        grid : pv.ImageData, optional
            PyVista ImageData object containing the data to interpolate.
            Must have point_data attributes to interpolate.
        meshgrids : Dict[str, np.ndarray], optional
            Dictionary mapping field names to numpy meshgrids. Each meshgrid
            should have the same shape as the grid used to define the Fermi surface.
        n_points : int, optional
            Number of points to sample the Fermi surface.
        sharpness : int, optional
            Sharpness of the interpolation.
        strategy : str, optional
            Strategy to use for interpolation.

        Returns
        -------
        pyvista.UnstructuredGrid
            The interpolated grid containing the data as point attributes.

        Notes
        -----
        The interpolated data is also stored in the point_data dictionary
        of the Fermi surface object.
        """
        logger.info("___Interpolating to surface___")

        if grid is None and meshgrids is None:
            raise ValueError("image_data, meshgrids, or unstructured_grid must be provided")

        unstructured_grid: pv.UnstructuredGrid  # pyright: ignore[reportUnknownMemberType]
        if meshgrids is not None:
            logger.info("___Interpolating to surface from meshgrids___")
            grid = copy.deepcopy(self.grid)  # pyright: ignore[reportUnknownMemberType]
            for name, meshgrid in meshgrids.items():
                grid.point_data[name] = meshgrid.reshape(-1, order="F")  # pyright: ignore[reportUnknownMemberType]
            unstructured_grid = grid.cast_to_unstructured_grid()  # pyright: ignore[reportUnknownMemberType]

        if grid is not None:
            logger.info("___Interpolating to surface from grid___")
            unstructured_grid = grid.cast_to_unstructured_grid()  # pyright: ignore[reportUnknownMemberType]

        unstructured_grid_cart: pv.UnstructuredGrid = unstructured_grid.transform(  # pyright: ignore[reportUnknownMemberType, reportPossiblyUnboundVariable, reportUnknownVariableType]
            self.transform_matrix_to_cart, transform_all_input_vectors=False, inplace=False
        )

        keys_to_interpolate = unstructured_grid_cart.point_data.keys()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        logger.debug(f"point_data to be interpolated: {keys_to_interpolate}")

        interpolated_surface: pv.UnstructuredGrid = self.interpolate(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            unstructured_grid_cart,
            n_points=n_points,
            sharpness=sharpness,
            strategy=strategy,
        )

        return interpolated_surface  # pyright: ignore[reportReturnType, reportUnknownVariableType]


def generate_band_isosurfaces(
    ebs: ElectronicBandStructureMesh, isovalue: float, padding: int = 10
) -> tuple[pv.PolyData, dict[tuple[int, int], pv.PolyData], float, ElectronicBandStructureMesh, ElectronicBandStructureMesh, PointSet]:  # pyright: ignore[reportUnknownMemberType]
    """
    Generate isosurfaces for all bands and spins that cross the Fermi level.

    This method iterates through all bands and spins in the electronic band structure,
    generating Fermi surface isosurfaces for each band-spin combination that crosses
    the Fermi level. It then merges all valid surfaces into a single combined surface.

    The method creates mappings between band-spin pairs and their corresponding surface
    indices, which are stored in the class attributes band_spin_surface_map and
    surface_band_spin_map.

    If a surface generation fails for any band-spin combination, the error is logged
    and the method continues with the next combination.

    Returns
    -------
    tuple
        Combined surface, band_isosurfaces dict, isovalue, original_ebs, padded_ebs, point_set
    """
    logger.info("___Generating all Fermi surfaces___")

    padded_ebs = ebs.pad(padding=padding, inplace=False)
    padded_ebs = padded_ebs.expand_single_dimension(inplace=False)

    transform_matrix_to_cart: npt.NDArray[np.float64] = np.eye(4)
    if ebs.reciprocal_lattice is not None:
        transform_matrix_to_cart[:3, :3] = ebs.reciprocal_lattice

    x_coords = padded_ebs.kpoints[:, 0]
    y_coords = padded_ebs.kpoints[:, 1]
    z_coords = padded_ebs.kpoints[:, 2]

    nx = padded_ebs.n_kx
    ny = padded_ebs.n_ky
    nz = padded_ebs.n_kz

    padded_x_min: float = float(x_coords.min())
    padded_y_min: float = float(y_coords.min())
    padded_z_min: float = float(z_coords.min())

    # Must be the points on non-padded grid
    x_spacing = 1 / ebs.n_kx
    y_spacing = 1 / ebs.n_kx
    z_spacing = 1 / ebs.n_kx

    grid: pv.ImageData = pv.ImageData(  # pyright: ignore[reportUnknownMemberType]
        dimensions=(nx, ny, nz),
        spacing=(x_spacing, y_spacing, z_spacing),
        origin=(padded_x_min, padded_y_min, padded_z_min),
    )
    if ebs.reciprocal_lattice is None:
        raise ValueError("reciprocal_lattice is None, cannot generate Fermi surfaces")
    brillouin_zone = BrillouinZone(
        ebs.reciprocal_lattice, transformation_matrix=[1, 1, 1]
    )

    bands_mesh = padded_ebs.get_property_mesh("bands", order="F")
    if bands_mesh is None:
        raise ValueError("bands property mesh is None")
    # Get dimensions from bands_mesh
    _, _, _, nbands, nspins = bands_mesh.shape

    band_isosurfaces: dict[tuple[int, int], pv.PolyData] = {}  # pyright: ignore[reportUnknownMemberType]
    for ispin in range(nspins):
        for iband in range(nbands):
            scalar_mesh: npt.NDArray[np.float64] = bands_mesh[..., iband, ispin]
            scalars: npt.NDArray[np.float64] = scalar_mesh.reshape(-1, order="F")
            surface: pv.PolyData = generate_isosurface(grid, scalars, isovalue=isovalue)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

            surface_points: npt.NDArray[Any] = np.asarray(surface.points)  # pyright: ignore[reportUnknownMemberType]
            if surface_points.shape[0] == 0:
                continue

            # Transform to cartesian coordinates
            surface = surface.transform(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                transform_matrix_to_cart, transform_all_input_vectors=False, inplace=False
            )

            # Clip the surface with the Brillouin zone to keep only the points inside the first Brillouin zone
            surface = clip_surface(surface, brillouin_zone)  # pyright: ignore[reportUnknownArgumentType]

            band_isosurfaces[(iband, ispin)] = surface

    if len(band_isosurfaces) == 0:
        raise ValueError("No Fermi surfaces were generated. Please check the isovalue and padding.")

    # Combine all surfaces into a single surface
    combined_surface: pv.PolyData | None = None  # pyright: ignore[reportUnknownMemberType]
    i_surface = 0
    spin_band_index: npt.NDArray[np.int32] = np.empty(0, dtype=np.int32)
    spin_index_arr: npt.NDArray[np.int32] = np.empty(0, dtype=np.int32)
    for (iband, ispin), surf in band_isosurfaces.items():
        if combined_surface is None:
            combined_surface = surf
        else:
            combined_surface = combined_surface.merge(surf, merge_points=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        surf_points: npt.NDArray[Any] = np.asarray(surf.points)  # pyright: ignore[reportUnknownMemberType]
        n_surf_points: int = int(surf_points.shape[0])
        surface_spin_band_index: npt.NDArray[np.int32] = np.full(n_surf_points, i_surface, dtype=np.int32)
        surface_spin_index: npt.NDArray[np.int32] = np.full(n_surf_points, ispin, dtype=np.int32)

        spin_band_index = np.insert(spin_band_index, 0, surface_spin_band_index, axis=0)
        spin_index_arr = np.insert(spin_index_arr, 0, surface_spin_index, axis=0)

        i_surface += 1

    if combined_surface is None:
        raise ValueError("No combined surface created")

    combined_points: npt.NDArray[np.float64] = np.asarray(combined_surface.points, dtype=np.float64)  # pyright: ignore[reportUnknownMemberType]
    point_set = PointSet(combined_points)
    point_set.add_property(name="spin_index", value=spin_index_arr)
    point_set.add_property(name="spin_band_index", value=spin_band_index)
    return combined_surface, band_isosurfaces, isovalue, ebs, padded_ebs, point_set  # pyright: ignore[reportUnknownVariableType, reportReturnType]


def generate_isosurface(
    grid: pv.ImageData,
    scalars: npt.NDArray[Any],
    isovalue: float,
    method: Literal["contour", "marching_cubes", "flying_edges"] = "marching_cubes",
) -> pv.PolyData:  # pyright: ignore[reportUnknownMemberType]
    """
    Generate a single isosurface for given band and spin.

    Parameters
    ----------
    grid : pv.ImageData
        Grid of the scalar values
    scalars : np.ndarray
        Scalars to be used for isosurface generation
    isovalue : float
        Isosurface value
    method : str, optional
        Method for isosurface generation, by default "marching_cubes"

    Returns
    -------
    pv.PolyData
        Surface for the given band and spin
    """
    # Generate isosurface
    surface: pv.PolyData = grid.contour([isovalue], scalars, method=method)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return surface  # pyright: ignore[reportUnknownVariableType, reportReturnType]


def clip_surface(surface: pv.PolyData, brillouin_zone: BrillouinZone) -> pv.PolyData:  # pyright: ignore[reportUnknownMemberType]
    """
    Clip the surface with the Brillouin zone to keep only the points inside the first Brillouin zone

    Parameters
    ----------
    surface : pv.PolyData
        Surface to be clipped
    brillouin_zone : BrillouinZone
        Brillouin zone to clip the surface with

    Returns
    -------
    pv.PolyData
        Clipped surface

    """
    # Clip surface with each face of the Brillouin zone
    for normal, center in zip(brillouin_zone.face_normals, brillouin_zone.centers):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]
        surface = surface.clip(origin=center, normal=normal, inplace=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType, reportAssignmentType]
        surf_points: npt.NDArray[Any] = np.asarray(surface.points)  # pyright: ignore[reportUnknownMemberType]
        if surf_points.shape[0] == 0:
            raise ValueError(
                f"Surface is empty after clipping with normal {normal} and center {center}"
            )
    return surface  # pyright: ignore[reportReturnType]
