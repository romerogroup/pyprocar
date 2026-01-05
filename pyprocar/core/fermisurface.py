from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import cached_property
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
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

logger = logging.getLogger(__name__)


class FSNormMode(Enum):
    """Normalization modes for Fermi surface properties."""

    RAW = "raw"
    MAX = "max"
    TOTAL = "total"
    INTEGRAL = "integral"

    @classmethod
    def from_input(cls, input: str | "FSNormMode" | None) -> "FSNormMode":
        """Convert string/None input to FSNormMode enum."""
        if isinstance(input, FSNormMode):
            return input
        if input is None:
            return cls.RAW
        if not isinstance(input, str):
            raise ValueError(f"Invalid normalization mode: {input}")

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
    def get_normed_name(mode: "FSNormMode", name: str) -> str:
        """Return property name with normalization prefix."""
        if mode == FSNormMode.RAW:
            return name
        return f"{mode.value}_{name}"

    @staticmethod
    def get_mode_prefix(mode: "FSNormMode") -> str:
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
        points: np.ndarray,
        faces: np.ndarray,
        band_isosurfaces: dict[tuple[int, int], pv.PolyData],
        isovalue: float,
        original_ebs: ElectronicBandStructureMesh,
        ebs: ElectronicBandStructureMesh,
        point_set: PointSet,
        point_data: dict[str | tuple[int, int], np.ndarray] = None,
        cell_data: dict[str | tuple[int, int], np.ndarray] = None,
        field_data: dict[str | tuple[int, int], Any] = None,
    ):
        super().__init__()

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

        point_data = point_data if point_data is not None else {}
        cell_data = cell_data if cell_data is not None else {}
        field_data = field_data if field_data is not None else {}

        self.point_data.update(point_data)
        self.cell_data.update(cell_data)
        self.field_data.update(field_data)

        logger.debug(f"Fermi Surface: \n {self}")
        logger.debug(f"Fermi Surface Point Data: \n {self.point_data}")
        logger.debug(f"Fermi Surface Cell Data: \n {self.cell_data}")
        logger.info("___FermiSurface initialization complete___")

    @classmethod
    def from_code(
        cls,
        code,
        dirpath,
        use_cache: bool = False,
        ebs_filename: str = "ebs.pkl",
        reduce_bands_near_fermi: bool = False,
        reduce_bands_near_energy: float = None,
        reduce_bands_by_index: list[int] = None,
        padding: int = 10,
        fermi: float = None,
        fermi_shift: float = 0.0,
        **kwargs,
    ):
        ebs = ElectronicBandStructureMesh.from_code(
            code, dirpath, use_cache=use_cache, ebs_filename=ebs_filename
        )
        if fermi is None:
            fermi = ebs.fermi

        if reduce_bands_near_fermi:
            ebs.reduce_bands_near_fermi()
        elif reduce_bands_near_energy is not None:
            ebs.reduce_bands_near_energy(reduce_bands_near_energy)
        elif reduce_bands_by_index is not None:
            ebs.reduce_bands_by_index(reduce_bands_by_index)
        return cls.from_ebs(ebs, padding=padding, isovalue=fermi, isovalue_shift=fermi_shift)

    @classmethod
    def from_ebs(
        cls,
        ebs: ElectronicBandStructureMesh,
        padding: int = 10,
        isovalue: float | None = None,
        isovalue_shift: float | None = None,
        **kwargs,
    ):
        if isovalue is None:
            isovalue = ebs.fermi
        if isovalue_shift is not None:
            isovalue += isovalue_shift
        results = generate_band_isosurfaces(ebs, isovalue=isovalue, padding=padding)
        combined_surface = results[0]
        band_isosurfaces = results[1]
        isovalue = results[2]
        ebs = results[3]
        padded_ebs = results[4]
        point_set = results[5]
        return cls(
            points=combined_surface.points,
            faces=combined_surface.faces,
            band_isosurfaces=band_isosurfaces,
            isovalue=isovalue,
            original_ebs=ebs,
            ebs=padded_ebs,
            point_set=point_set,
        )

    @property
    def original_ebs(self):
        return self._original_ebs

    @property
    def ebs(self):
        return self._ebs

    @property
    def grid(self):
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

        padded_grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(x_spacing, y_spacing, z_spacing),
            origin=(padded_x_min, padded_y_min, padded_z_min),
        )
        return padded_grid

    @property
    def transform_matrix_to_cart(self):
        transform_to_cart = np.eye(4)
        transform_to_cart[:3, :3] = self.reciprocal_lattice.T
        return transform_to_cart

    @property
    def transform_matrix_to_frac(self):
        transform_to_frac = np.eye(4)
        transform_to_frac[:3, :3] = np.linalg.inv(self.reciprocal_lattice.T)
        return transform_to_frac

    @property
    def fermi(self):
        return self.ebs.fermi

    @property
    def isovalue(self):
        return self._isovalue

    @property
    def fermi_shift(self):
        return self.isovalue - self.ebs.fermi

    @property
    def reciprocal_lattice(self):
        return self.ebs.reciprocal_lattice

    @property
    def n_points(self):
        return self.points.shape[0]

    @property
    def point_set(self):
        return self._point_set

    @property
    def band_isosurfaces(self):
        return self._band_isosurfaces

    @property
    def band_indices(self):
        return self.band_isosurfaces.keys()

    @property
    def band_spin_surface_map(self):
        band_spin_surface_map = {}
        for isurface, (iband, ispin) in enumerate(self.band_isosurfaces.keys()):
            band_spin_surface_map[(iband, ispin)] = isurface
        return band_spin_surface_map

    @property
    def surface_band_spin_map(self):
        surface_band_spin_map = {}
        for isurface, (iband, ispin) in enumerate(self.band_isosurfaces.keys()):
            surface_band_spin_map[isurface] = (iband, ispin)
        return surface_band_spin_map

    @property
    def band_spin_mask(self):
        spin_band_index_array = self.point_set.get_property("spin_band_index").value
        tmp_mask_dict = {}
        for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
            mask = spin_band_index_array == surface_idx
            tmp_mask_dict[(iband, ispin)] = mask
        return tmp_mask_dict

    @property
    def spin_channel_mask(self):
        spin_index_array = self.point_set.get_property("spin_index").value

        spin_channels = np.unique(spin_index_array)
        tmp_mask_dict = {}
        for ispin in spin_channels:
            mask = spin_index_array == ispin
            tmp_mask_dict[ispin] = mask

        return tmp_mask_dict

    @cached_property
    def brillouin_zone(self):
        return self.get_brillouin_zone(np.array([1, 1, 1]))

    @property
    def is2d(self):
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
                spin_indexer = SpinIndexer()

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

        # Interpolate to surface
        surface_values = self.interpolate_to_surface(ebs_property.value)

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

    def _normalize_integral(self, values_array: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Normalize by surface integral approximation.

        Uses cell areas from the mesh to weight contributions.
        """
        values_array = np.asarray(values_array, dtype=np.float64)

        # Compute cell areas for weighting
        cell_sizes = self.compute_cell_sizes()
        if cell_sizes is None or len(cell_sizes) == 0:
            # Fallback to simple sum if cell sizes unavailable
            return self._normalize_total(values_array, **kwargs)

        # Map cell areas to points (average of adjacent cells)
        point_weights = np.zeros(self.n_points)
        cell_count = np.zeros(self.n_points)

        for i, cell in enumerate(self.cell):
            for point_idx in cell:
                point_weights[point_idx] += cell_sizes[i]
                cell_count[point_idx] += 1

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

    def save(self, path: str) -> None:
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
        import pickle
        from pathlib import Path
        import copy

        path_obj = Path(path)

        # Serialize point_set properties to avoid weak reference issues
        point_set_data = {
            "points": self.point_set.points.copy(),
            "properties": {},
        }
        for prop_name, prop in self.point_set.property_store.items():
            point_set_data["properties"][prop_name] = {
                "value": prop.value.copy(),
                "gradients": {k: v.copy() for k, v in prop.gradients.items() if v is not None and len(v) > 0},
                "units": prop.units,
                "label": prop.label,
                "metadata": copy.deepcopy(prop.metadata) if prop.metadata else {},
                "data_lim": prop.data_lim,
            }

        # Serialize band_isosurfaces - just store the geometry
        band_isosurfaces_data = {}
        for key, surface in self.band_isosurfaces.items():
            band_isosurfaces_data[key] = {
                "points": np.array(surface.points),
                "faces": np.array(surface.faces),
            }

        # Collect all data needed for reconstruction
        save_data = {
            "points": self.points.copy(),
            "faces": self.faces.copy(),
            "isovalue": self.isovalue,
            "point_data": {k: np.array(v) for k, v in self.point_data.items()},
            "cell_data": {k: np.array(v) for k, v in self.cell_data.items()},
            "field_data": dict(self.field_data),
            "point_set_data": point_set_data,
            "band_isosurfaces_data": band_isosurfaces_data,
            # Note: EBS objects are not saved due to weak reference issues
            # Users should keep a reference to EBS or reload from calculation
        }

        with open(path_obj, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"FermiSurface saved to {path}")

    @classmethod
    def load(cls, path: str, ebs: ElectronicBandStructureMesh | None = None) -> "FermiSurface":
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
        band_isosurfaces = {}
        for key, surface_data in save_data["band_isosurfaces_data"].items():
            surface = pv.PolyData(surface_data["points"], surface_data["faces"])
            band_isosurfaces[key] = surface

        # Create a temporary PolyData with all the saved data
        temp_polydata = pv.PolyData(save_data["points"], save_data["faces"])

        # Add point/cell/field data to temp before shallow_copy
        for k, v in save_data["point_data"].items():
            temp_polydata.point_data[k] = v
        for k, v in save_data["cell_data"].items():
            temp_polydata.cell_data[k] = v
        temp_polydata.field_data.update(save_data["field_data"])

        # Create the FermiSurface by calling the parent class constructor
        # and then manually setting our attributes
        fs = pv.PolyData.__new__(cls)
        fs.shallow_copy(temp_polydata)

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

        return BrillouinZone(self.reciprocal_lattice, supercell)

    def get_property(self, key, **kwargs):
        prop_name, (calc_name, gradient_order) = self.ebs._extract_key(key)

        if prop_name not in self.point_set.property_store:
            property_value = self.compute_property(prop_name, **kwargs)
            property = Property(name=prop_name, value=property_value)
            self.point_set.add_property(property)
        else:
            property_value = self.point_set.get_property(prop_name)[calc_name, gradient_order]

        self.set_values(prop_name, property_value)
        return property_value

    def select_bands(
        self, bands_spin_indices: list[tuple[int, int]], return_relative_indices: bool = False
    ):
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
        if not combined_mask.any():
            logger.warning("No points selected - creating empty surface")

            points = np.empty((0, 3))
            faces = np.empty((0, 4), dtype=np.int32)
            point_data = {}
            cell_data = {}
            field_data = {}
            new_point_set = self.point_set

        else:
            new_surface, fs_indices = self.remove_points(combined_mask, inplace=False)
            points = new_surface.points
            faces = new_surface.faces
            point_data = new_surface.point_data
            cell_data = new_surface.cell_data
            field_data = new_surface.field_data
            new_point_set = self.point_set.select_points(fs_indices)

        fs = FermiSurface(
            points=points,
            faces=faces,
            band_isosurfaces=selected_band_surfaces,
            isovalue=self.isovalue,
            original_ebs=self.original_ebs,
            ebs=self.ebs,
            point_set=new_point_set,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
        )

        if return_relative_indices:
            return fs, fs_indices
        else:
            return fs

    def set_surface_point_data(self, name: str, values: np.ndarray):
        if values.shape[-1] == 3:
            last_dim = 3
        else:
            last_dim = 1

        self.point_data[name] = np.zeros(shape=(self.n_points, last_dim))
        if self.ebs.is_band_property(values):
            logger.debug(f"Adding band resolved to fermi surface point_data: {name}")
            for (iband, ispin), surface_idx in self.band_spin_surface_map.items():
                values_band_values = values[:, iband, ispin, ...]
                mask = self.band_spin_mask[(iband, ispin)]
                values_band_values[~mask] = 0
                self.point_data[name] += values_band_values
        else:
            logger.debug(f"Adding scalar to fermi surface point_data: {name}")
            self.point_data[name] += values

    def set_scalars(self, name: str, value: np.ndarray):
        self.set_surface_point_data(name, value)
        self.set_active_scalars(name, preference="point")

    def set_vectors(self, name: str, value: np.ndarray, set_scalar: bool = True):
        if value.shape[-1] != 3:
            raise ValueError(f"Vector data must have 3 dimensions. Got {value.shape[-1]}.")

        vector_magnitude = np.linalg.norm(value, axis=-1)
        if set_scalar:
            self.set_surface_point_data(f"{name}-norm", vector_magnitude)
            self.set_active_scalars(f"{name}-norm", preference="point")
        self.set_surface_point_data(name, value)
        self.set_active_vectors(name, preference="point")

    def set_values(self, name: str, value: np.ndarray, **kwargs):
        if value.shape[-1] == 3:
            self.set_vectors(name, value, **kwargs)
        else:
            self.set_scalars(name, value, **kwargs)

    def set_band_colors(
        self,
        colors: list[str | tuple[int, int, int]] = None,
        surface_color_map: dict[tuple[int, int], str | tuple[int, int, int]] = None,
        cmap: tuple[str] | tuple[str, str] = ("winter", "autumn"),
    ):
        bands_colors = np.zeros((self.n_points, 4))
        if surface_color_map is not None:
            assert len(list(surface_color_map.keys())) == len(self.band_spin_surface_map), (
                "Number of colors must match number of bands"
            )
            for (iband, ispin), color in surface_color_map.items():
                if isinstance(color, str):
                    color = mcolors.to_rgba(color)
                if (iband, ispin) not in self.band_spin_mask:
                    raise ValueError(
                        f"Band-spin combination ({iband}, {ispin}) not found in Fermi surface. "
                        f"Available combinations: {self.band_spin_mask.keys()}"
                    )
                bands_spin_mask = self.band_spin_mask[(iband, ispin)]
                bands_colors[bands_spin_mask] = color

        elif colors is not None:
            assert len(colors) == len(self.band_spin_surface_map), (
                "Number of colors must match number of bands"
            )
            i_surface = 0
            for (iband, ispin), bands_spin_mask in self.band_spin_mask.items():
                color = colors[i_surface]
                if isinstance(color, str):
                    color = mcolors.to_rgba(color)
                bands_colors[bands_spin_mask] = color
                i_surface += 1
        else:
            for (iband, ispin), isurface in self.band_spin_surface_map.items():
                cmapper = plt.get_cmap(cmap[ispin])

                bands_spin_mask = self.band_spin_mask[(iband, ispin)]
                bands_colors[bands_spin_mask] = np.array([cmapper(iband)] * bands_spin_mask.sum())

        self.point_data["bands"] = bands_colors
        self.set_active_scalars("bands", preference="point")

    def set_spin_colors(self, colors: tuple[str, str] = ("red", "blue")):
        bands_colors = np.zeros((self.n_points, 4))

        for spin_channel, mask in self.spin_channel_mask.items():
            if isinstance(colors[spin_channel], str):
                color = mcolors.to_rgba(colors[spin_channel])
            bands_colors[mask] = color

        self.point_data["spin"] = bands_colors
        self.set_active_scalars("spin", preference="point")

    def compute_gradients(
        self, gradient_order: int, names: list[str] | None = None, **kwargs
    ) -> None:
        if names is None:
            names = list(self.point_set._property_store.keys())
        if gradient_order < 0:
            raise ValueError(f"Gradient order must be greater than 0. Got {gradient_order}.")
        self.ebs.compute_gradients(gradient_order=gradient_order, names=names)

        for prop_name, calc_name, gradient_order, value_array in self.ebs.iter_properties():
            property = self.point_set.get_property(prop_name)
            surface_points = self.interpolate_to_surface(value_array)
            property[calc_name, gradient_order] = surface_points

        return None

    def compute_property(self, name: str, **kwargs):
        if name == "fermi_velocity":
            return self.compute_fermi_velocity(**kwargs)
        elif name == "fermi_speed":
            return self.compute_fermi_speed(**kwargs)
        else:
            property = self.ebs.get_property(name, **kwargs)
            surface_points = self.interpolate_to_surface(property.value)
            return surface_points

    def compute_fermi_speed(self, **kwargs):
        band_speed = self.ebs.get_property("bands_speed", **kwargs)
        surface_points = self.interpolate_to_surface(band_speed.value)
        return surface_points

    def compute_fermi_velocity(self, **kwargs):
        band_velocity = self.ebs.get_property("bands_velocity", **kwargs)
        surface_points = self.interpolate_to_surface(band_velocity.value)
        return surface_points

    def interpolate_to_surface(self, property_value: np.ndarray):
        grid = copy.deepcopy(self.grid)
        grid.point_data["property"] = property_value.reshape(
            (property_value.shape[0], -1), order="F"
        )
        interpolated_surface = self._create_interpolated_surface(grid=grid)
        property_surface_points = interpolated_surface.point_data["property"].reshape(
            (-1, *property_value.shape[1:]), order="F"
        )
        return property_surface_points

    def update_point_data(
        self,
        name: str,
        point_data: dict[str, np.ndarray] | np.ndarray = None,
        surface: pv.UnstructuredGrid = None,
    ):
        if isinstance(point_data, dict):
            self.point_data.update(point_data)
        elif isinstance(point_data, np.ndarray):
            self.point_data[name] = point_data
        elif surface is not None:
            self.point_data.update(surface.point_data)
        else:
            raise ValueError(
                "Either point_data or a surface with point_data attribute must be provided"
            )

        is_point_data_vector = self.point_data[name].shape[-1] == 3
        logger.debug(f"is_point_data_vector|{name}: {is_point_data_vector}")
        if is_point_data_vector:
            scalar_name = f"{name}-norm"
            self.point_data[scalar_name] = np.linalg.norm(self.point_data[name], axis=-1)
            self.set_active_scalars(scalar_name, preference="point")
            self.set_active_vectors(name, preference="point")
        else:
            self.set_active_scalars(name, preference="point")

    def extend_surface(self, zone_directions: list[list[int] | tuple[int, int, int]]):
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

        new_surface = copy.deepcopy(self)
        initial_surface = copy.deepcopy(self)
        initial_band_surfaces = copy.deepcopy(self.band_isosurfaces)
        new_band_surfaces = copy.deepcopy(self.band_isosurfaces)
        new_point_set = copy.deepcopy(self.point_set)
        for direction in zone_directions:
            surface = copy.deepcopy(initial_surface)
            translated_surface = surface.translate(
                np.dot(direction, self.ebs.reciprocal_lattice), inplace=True
            )
            new_surface = new_surface.merge(translated_surface, merge_points=False)

            # Copying exisiting band_isosurfaces
            for (iband, ispin), initial_band_surface in initial_band_surfaces.items():
                new_band_surface = copy.deepcopy(initial_band_surface)
                new_band_surface += initial_band_surface.translate(
                    np.dot(direction, self.ebs.reciprocal_lattice), inplace=True
                )
                new_band_surfaces[(iband, ispin)] += new_band_surface

            # Copying exisiting properties
            for (
                prop_name,
                calc_name,
                gradient_order,
                value_array,
            ) in new_point_set.iter_property_arrays():
                current_property = self.point_set.get_property(prop_name)
                current_array = current_property[calc_name, gradient_order]

                new_array = np.concatenate((value_array, current_array), axis=0)
                new_property = new_point_set.get_property(prop_name)
                new_property[calc_name, gradient_order] = new_array

        new_point_set.update_points(new_surface.points)

        fs = FermiSurface(
            points=new_surface.points,
            faces=new_surface.faces,
            band_isosurfaces=new_band_surfaces,
            isovalue=self.isovalue,
            original_ebs=self.original_ebs,
            ebs=self.ebs,
            point_set=new_point_set,
            point_data=new_surface.point_data,
            cell_data=new_surface.cell_data,
            field_data=new_surface.field_data,
        )
        return fs

    def _create_interpolated_surface(
        self,
        grid: pv.ImageData = None,
        meshgrids: dict[str, np.ndarray] = None,
        n_points: int = 40,
        sharpness: int = 20,
        strategy: str = "null_value",
    ):
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

        if meshgrids is not None:
            logger.info("___Interpolating to surface from meshgrids___")
            grid = copy.deepcopy(self.grid)
            for name, meshgrid in meshgrids.items():
                grid.point_data[name] = meshgrid.reshape(-1, order="F")
            unstructured_grid = grid.cast_to_unstructured_grid()

        if grid is not None:
            logger.info("___Interpolating to surface from grid___")
            unstructured_grid = grid.cast_to_unstructured_grid()

        unstructured_grid_cart = unstructured_grid.transform(
            self.transform_matrix_to_cart, transform_all_input_vectors=False, inplace=False
        )

        keys_to_interpolate = unstructured_grid_cart.point_data.keys()
        logger.debug(f"point_data to be interpolated: {keys_to_interpolate}")

        interpolated_surface = self.interpolate(
            unstructured_grid_cart,
            n_points=n_points,
            sharpness=sharpness,
            strategy=strategy,
        )

        return interpolated_surface


def generate_band_isosurfaces(ebs: ElectronicBandStructureMesh, isovalue: float, padding: int = 10):
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
    None
        The generated surfaces are stored in the class instance, updating its points,
        faces, point_data, and cell_data attributes.
    """
    logger.info("___Generating all Fermi surfaces___")

    padded_ebs = ebs.pad(padding=padding, inplace=False)
    padded_ebs = padded_ebs.expand_single_dimension(inplace=False)

    transform_matrix_to_cart = np.eye(4)
    transform_matrix_to_cart[:3, :3] = ebs.reciprocal_lattice

    x_coords = padded_ebs.kpoints[:, 0]
    y_coords = padded_ebs.kpoints[:, 1]
    z_coords = padded_ebs.kpoints[:, 2]

    nx = padded_ebs.n_kx
    ny = padded_ebs.n_ky
    nz = padded_ebs.n_kz

    padded_x_min = x_coords.min()
    padded_y_min = y_coords.min()
    padded_z_min = z_coords.min()

    # Must be the points on non-padded grid
    x_spacing = 1 / ebs.n_kx
    y_spacing = 1 / ebs.n_kx
    z_spacing = 1 / ebs.n_kx

    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(x_spacing, y_spacing, z_spacing),
        origin=(padded_x_min, padded_y_min, padded_z_min),
    )
    brillouin_zone = BrillouinZone(
        ebs.reciprocal_lattice, transformation_matrix=np.array([1, 1, 1])
    )

    bands_mesh = padded_ebs.get_property_mesh("bands", order="F")
    # Get dimensions from bands_mesh
    _, _, _, nbands, nspins = bands_mesh.shape

    band_isosurfaces = {}
    for ispin in range(nspins):
        for iband in range(nbands):
            scalar_mesh = bands_mesh[..., iband, ispin]
            scalars = scalar_mesh.reshape(-1, order="F")
            surface = generate_isosurface(grid, scalars, isovalue=isovalue)

            if surface.points.shape[0] == 0:
                continue

            # Transform to cartesian coordinates
            surface = surface.transform(
                transform_matrix_to_cart, transform_all_input_vectors=False, inplace=False
            )

            # Clip the surface with the Brillouin zone to keep only the points inside the first Brillouin zone
            surface = clip_surface(surface, brillouin_zone)

            band_isosurfaces[(iband, ispin)] = surface

    if len(band_isosurfaces) == 0:
        raise ValueError("No Fermi surfaces were generated. Please check the isovalue and padding.")

    # Combine all surfaces into a single surface
    combined_surface = None
    i_surface = 0
    spin_band_index = np.empty(0, dtype=np.int32)
    spin_index = np.empty(0, dtype=np.int32)
    for (iband, ispin), surface in band_isosurfaces.items():
        if combined_surface is None:
            combined_surface = surface
        else:
            combined_surface = combined_surface.merge(surface, merge_points=False)

        surface_spin_band_index = np.full(surface.points.shape[0], i_surface, dtype=np.int32)
        surface_spin_index = np.full(surface.points.shape[0], ispin, dtype=np.int32)

        spin_band_index = np.insert(spin_band_index, 0, surface_spin_band_index, axis=0)
        spin_index = np.insert(spin_index, 0, surface_spin_index, axis=0)

        i_surface += 1

    point_set = PointSet(combined_surface.points)
    point_set.add_property(name="spin_index", value=spin_index)
    point_set.add_property(name="spin_band_index", value=spin_band_index)
    return combined_surface, band_isosurfaces, isovalue, ebs, padded_ebs, point_set


def generate_isosurface(
    grid: pv.ImageData, scalars: np.ndarray, isovalue: float, method: str = "marching_cubes"
) -> pv.PolyData:
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
    surface = grid.contour([isovalue], scalars, method=method)
    return surface


def clip_surface(surface: pv.PolyData, brillouin_zone: BrillouinZone):
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
    for normal, center in zip(brillouin_zone.face_normals, brillouin_zone.centers):
        surface = surface.clip(origin=center, normal=normal, inplace=False)
        if surface.points.shape[0] == 0:
            raise ValueError(
                f"Surface is empty after clipping with normal {normal} and center {center}"
            )
    return surface
