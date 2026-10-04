from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from pyprocar.core.atomic_orbital_index import (
    AtomIndexer,
    OrbitalIndexer,
    ProjectionLabelBuilder,
    ProjectionSelectionResolver,
    ProjectionSelectionResult,
    SpinIndexer,
)
from pyprocar.core.property_store import PointSet, Property
from pyprocar.utils.math import np_round_to_half


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
    def parse(cls, mode: str | NormMode | None) -> NormMode:
        return cls(mode.lower()) if isinstance(mode, str) else cls(mode or cls.RAW)

    def normed_name(self, name: str) -> str:
        prefix = _LABELS[self][0]
        return f"{prefix} {name}" if prefix else name


_LABELS = {
    NormMode.RAW: ("", ""),
    NormMode.MAX: ("Max-Normed", "Normalized by the maximum"),
    NormMode.INTEGRAL: ("Integral-Normed", "Normalized by the integral"),
    NormMode.ELECTRONS: ("N_Electrons-Normed", "Normalized by the number of electrons"),
    NormMode.TOTAL: ("Total-Normed", "Normalized by the total"),
    NormMode.TOTAL_PROJECTION: ("Total-Projected-Normed", "Normalized by the total projection"),
    NormMode.SPIN_MAGNITUDE: ("Spin-Magnitude-Normed", "Normalized by the spin magnitude"),
    NormMode.MAGNETIZATION: ("Magnetization-Normed", "Normalized by the magnetization"),
}


@dataclass(frozen=True, slots=True)
class ProjectedLayout:
    """The axes of a projection array and the array its leading axes repeat.

    The leading axes run up to and including ``n_spins``. A non-collinear
    calculation has 4 projection channels (total, Sx, Sy, Sz) and either 4 or 1
    channels in the leading array: the VASP and Abinit parsers repeat each
    eigenvalue 4 times, the QE parser keeps one.
    """

    axes: tuple[str, ...]
    leading: str

    @property
    def n_leading(self) -> int:
        return self.axes.index("n_spins") + 1

    def __str__(self) -> str:
        return f"({', '.join(self.axes)})"


BAND_PROJECTIONS = ProjectedLayout(
    ("n_kpoints", "n_bands", "n_spins", "n_atoms", "n_orbitals"), "bands"
)
DOS_PROJECTIONS = ProjectedLayout(("n_energies", "n_spins", "n_atoms", "n_orbitals"), "total")
SPIN_CHANNELS = (1, 2, 4)


class HasAtoms(Protocol):
    @property
    def natoms(self) -> int: ...


def check_projected_layout(
    projected: npt.ArrayLike | None,
    orbital_names: Sequence[str] | None,
    layout: ProjectedLayout,
    leading: npt.ArrayLike | None = None,
    structure: HasAtoms | None = None,
    atom_groups: int | None = None,
) -> None:
    """Raise unless ``projected`` is in ``layout`` and agrees with the data it belongs to.

    Its leading axes must equal the shape of ``leading`` (the bands or the total
    DOS), its atom axis must equal ``atom_groups`` when the rows are groups of
    atoms and ``structure.natoms`` otherwise, it must have 1, 2 or 4 spin
    channels, and ``orbital_names`` must name its last axis.
    """
    if projected is None:
        return
    shape = np.shape(projected)
    if len(shape) != len(layout.axes):
        raise ValueError(
            f"projected has shape {shape}; it must have the {len(layout.axes)} axes {layout}."
        )
    leading_shape = None if leading is None else np.shape(leading)
    if leading_shape is not None and len(leading_shape) != layout.n_leading:
        raise ValueError(
            f"{layout.leading} has shape {leading_shape}; it must have the axes"
            + f" ({', '.join(layout.axes[: layout.n_leading])})."
        )
    spin_axis = layout.n_leading - 1
    n_spins = (shape if leading_shape is None else leading_shape)[spin_axis]
    if n_spins not in SPIN_CHANNELS:
        raise ValueError(
            f"{'projected' if leading_shape is None else layout.leading} has {n_spins} spin"
            + f" channels; a calculation has 1, 2 or 4 spin channels, not {n_spins}"
            + " (non-polarized, collinear up and down, or non-collinear total, Sx, Sy, Sz)."
        )
    expected: list[tuple[int, ...] | None] = [None] * len(layout.axes)
    if leading_shape is not None:
        expected[: layout.n_leading] = [(size,) for size in leading_shape]
        expected[spin_axis] = (1, 4) if n_spins == 1 else (n_spins,)
    if atom_groups is not None:
        expected[layout.n_leading] = (atom_groups,)
    elif structure is not None:
        expected[layout.n_leading] = (structure.natoms,)
    if any(
        sizes is not None and actual not in sizes
        for sizes, actual in zip(expected, shape, strict=True)
    ):
        described = ", ".join("*" if s is None else "|".join(map(str, s)) for s in expected)
        sources = [] if leading_shape is None else [f"{layout.leading} of shape {leading_shape}"]
        if atom_groups is not None:
            sources.append(f"{atom_groups} atom groups")
        elif structure is not None:
            sources.append(f"a {structure.natoms}-atom structure")
        raise ValueError(
            f"projected has shape {shape}; the layout {layout} expects ({described}) from"
            + f" {' and '.join(sources)}. Its spin channels match those of {layout.leading},"
            + " except that non-collinear projections (total, Sx, Sy, Sz) may go with one"
            + f" {layout.leading} channel."
        )
    n_orbitals = shape[-1]
    if orbital_names is not None and len(orbital_names) != n_orbitals:
        raise ValueError(
            f"{len(orbital_names)} orbital_names for {n_orbitals} projected orbitals (the last axis"
            + " of projected). Pass one name per projected orbital, for example"
            + f" OrbitalIndexer().flat_conventional[:{n_orbitals}]."
        )


def normalize(
    values: npt.ArrayLike,
    mode: str | NormMode | None,
    denominators: Mapping[NormMode, Callable[[], npt.ArrayLike]],
) -> npt.NDArray[np.float64]:
    """Divide ``values`` by the denominator registered for ``mode``.

    Denominators are thunks so only the requested one is computed. Entries
    whose denominator is zero come out as zero.
    """
    mode = NormMode.parse(mode)
    if mode is NormMode.RAW:
        return np.asarray(values)
    if mode not in denominators:
        supported = ", ".join(m.value for m in (NormMode.RAW, *denominators))
        raise ValueError(f"Invalid normalization mode: {mode.value}. Valid modes: {supported}")
    values = np.asarray(values, dtype=np.float64)
    denominator = np.asarray(denominators[mode](), dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(values.shape, denominator.shape))
    return np.divide(values, denominator, out=out, where=denominator != 0)


class ProjectionSource(Protocol):
    @property
    def structure(self) -> Any: ...
    @property
    def spin_projection_names(self) -> list[str]: ...
    @property
    def orbital_names(self) -> list[str] | None: ...
    @property
    def is_non_collinear(self) -> bool: ...
    @property
    def atom_groups(self) -> int | None: ...
    @property
    def n_atoms(self) -> int: ...


def selection_resolver(source: ProjectionSource) -> ProjectionSelectionResolver:
    atom_indexer = (
        None
        if source.structure is None or source.atom_groups is not None
        else AtomIndexer.from_structure(source.structure)
    )
    label_builder = ProjectionLabelBuilder(
        atom_indexer=atom_indexer,
        orbital_indexer=OrbitalIndexer(),
        spin_indexer=SpinIndexer.from_projection_names(source.spin_projection_names),
    )
    return ProjectionSelectionResolver(
        label_builder=label_builder,
        n_atom_rows=source.n_atoms,
        orbital_names=source.orbital_names,
        is_non_colinear=source.is_non_collinear,
    )


class ProjectionOwner(Protocol):
    def normalize(
        self, mode: str | NormMode | None, values_array: npt.NDArray[np.float64], **kwargs: Any
    ) -> npt.NDArray[np.float64]: ...

    def normed_units(self, mode: NormMode, units: str | None) -> str | None: ...


def build_property(
    owner: ProjectionOwner,
    values: npt.NDArray[np.float64],
    *,
    name: str,
    label: str,
    norm_mode: str | NormMode | None = None,
    units: str | None = None,
    selection: ProjectionSelectionResult | None = None,
    allowed_norm_modes: set[NormMode] | None = None,
    include_normal_label: bool = False,
    point_set: PointSet | None = None,
    normalize_kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Property:
    """Normalize ``values`` through ``owner`` and wrap them in a labelled Property.

    ``normalize_kwargs`` go only to ``owner.normalize``; ``metadata`` entries are
    merged last into the Property metadata. The Property belongs to ``point_set``,
    defaulting to ``owner``.
    """
    mode = NormMode.parse(norm_mode)
    if allowed_norm_modes is not None and mode not in allowed_norm_modes:
        valid_modes = ", ".join(sorted(m.value for m in allowed_norm_modes))
        raise ValueError(f"Invalid normalization mode: {mode.value}. Valid modes: {valid_modes}")

    if point_set is None:
        assert isinstance(owner, PointSet), "owner must be a PointSet unless point_set is given"
        point_set = owner

    values = owner.normalize(mode, values, **(normalize_kwargs or {}))
    normed_units = owner.normed_units(mode, units)
    if len(values) == 0:
        data_min = data_max = np.zeros(values.shape[1:])
    else:
        data_min = np.min(values, axis=0)
        data_max = np.max(values, axis=0)

    property_metadata: dict[str, Any] = {
        "norm_mode": mode,
        "units": normed_units,
        "data_lim": (data_min, data_max),
        "rounded_data_lim": (np_round_to_half(data_min), np_round_to_half(data_max)),
        "footnote": _LABELS[mode][1],
        "scalar_label": label,
        "label": label,
        "label_plain": label,
        "include_normal_label": include_normal_label,
    }
    if selection is not None:
        property_metadata.update(
            _selection_metadata(selection, mode is not NormMode.RAW, include_normal_label)
        )
    property_metadata.update(metadata or {})

    return Property(
        name=mode.normed_name(name),
        value=values,
        point_set=point_set,
        metadata=property_metadata,
        label=label,
        units=normed_units,
    )


def _selection_metadata(
    selection: ProjectionSelectionResult, normalized: bool, include_normal_label: bool
) -> dict[str, Any]:
    labels = selection.labels
    suffix_plain = suffix_latex = ""
    if include_normal_label:
        suffix_plain = " [fraction]" if normalized else " [raw]"
        suffix_latex = " [\\mathrm{fraction}]" if normalized else " [\\mathrm{raw}]"

    label_plain: list[str] = []
    label_latex: list[str] = []
    for plain, latex in zip(
        labels.spin_components or ("",), labels.spin_components_latex or ("",), strict=False
    ):
        body_plain = labels.prefix_plain
        if plain:
            body_plain = f"{body_plain}[{plain}]" if body_plain else plain
        body_plain = body_plain or "all"
        label_plain.append(f"{body_plain}{suffix_plain}")

        body_latex = labels.prefix_latex
        if latex:
            body_latex = f"{body_latex}[{latex}]" if body_latex else f"[{latex}]"
        body_latex = body_latex or "\\mathrm{all}"
        label_latex.append(f"${body_latex}{suffix_latex}$")

    return {
        "atoms": list(selection.atoms),
        "orbitals": list(selection.orbitals) if selection.orbitals is not None else None,
        "spins": list(selection.spins) if selection.spins is not None else None,
        "species": list(selection.species) if len(selection.species) > 0 else None,
        "atom_label": labels.atom,
        "atom_label_latex": labels.atom_latex,
        "orbital_label": labels.orbital,
        "orbital_label_latex": labels.orbital_latex,
        "spin_label": labels.spin,
        "spin_label_latex": labels.spin_latex,
        "species_label": labels.species,
        "species_label_latex": labels.species_latex,
        "label_prefix": labels.prefix_plain,
        "label_prefix_latex": labels.prefix_latex,
        "spin_component_labels": list(labels.spin_components),
        "spin_component_labels_latex": list(labels.spin_components_latex),
        "label_combined": labels.combined,
        "label_combined_latex": labels.combined_latex,
        "label": label_latex,
        "label_plain": label_plain,
    }
