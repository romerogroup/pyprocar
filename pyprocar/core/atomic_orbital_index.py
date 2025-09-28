"""Indexing helpers for atoms, orbitals, and spins."""

from __future__ import annotations

from collections.abc import Iterable as ABCIterable, Mapping as ABCMapping
import re

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pyprocar.core.structure import Structure

AZIMUTHAL_ORBITAL_ORDER = {
    "s": ["s"],
    "p": ["pz", "px", "py"],
    "d": ["dz2", "dzx", "dzy", "dx2-y2", "dxy"],
    "f": ["fz3", "fxz2", "fyz2", "fx( x2-3y2 )", "fxyz", "fy(3x2-y2)", "fx3"],
}

CONVENTIONAL_CUBIC_ORBITAL_ORDER = {
    "s": ["s"],
    "p": ["py", "pz", "px"],
    "d": ["dxy", "dyz", "dz2", "dxz", "x2-y2"],
    "f": ["fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"],
}

NONCOLINEAR_AZIMUTHAL_ORBITAL_ORDER = [
    {"l": 0, "j": 0.5, "m_j": -0.5},
    {"l": 0, "j": 0.5, "m_j": 0.5},
    {"l": 1, "j": 0.5, "m_j": -0.5},
    {"l": 1, "j": 0.5, "m_j": 0.5},
    {"l": 1, "j": 1.5, "m_j": -1.5},
    {"l": 1, "j": 1.5, "m_j": -0.5},
    {"l": 1, "j": 1.5, "m_j": 0.5},
    {"l": 1, "j": 1.5, "m_j": 1.5},
    {"l": 2, "j": 1.5, "m_j": -1.5},
    {"l": 2, "j": 1.5, "m_j": -0.5},
    {"l": 2, "j": 1.5, "m_j": 0.5},
    {"l": 2, "j": 1.5, "m_j": 1.5},
    {"l": 2, "j": 2.5, "m_j": -2.5},
    {"l": 2, "j": 2.5, "m_j": -1.5},
    {"l": 2, "j": 2.5, "m_j": -0.5},
    {"l": 2, "j": 2.5, "m_j": 0.5},
    {"l": 2, "j": 2.5, "m_j": 1.5},
    {"l": 2, "j": 2.5, "m_j": 2.5},
    {"l": 3, "j": 2.5, "m_j": -2.5},
    {"l": 3, "j": 2.5, "m_j": -1.5},
    {"l": 3, "j": 2.5, "m_j": -0.5},
    {"l": 3, "j": 2.5, "m_j": 0.5},
    {"l": 3, "j": 2.5, "m_j": 1.5},
    {"l": 3, "j": 2.5, "m_j": 2.5},
    {"l": 3, "j": 3.5, "m_j": -3.5},
    {"l": 3, "j": 3.5, "m_j": -2.5},
    {"l": 3, "j": 3.5, "m_j": -1.5},
    {"l": 3, "j": 3.5, "m_j": -0.5},
    {"l": 3, "j": 3.5, "m_j": 0.5},
    {"l": 3, "j": 3.5, "m_j": 1.5},
    {"l": 3, "j": 3.5, "m_j": 2.5},
    {"l": 3, "j": 3.5, "m_j": 3.5},
]

PRIMARY_ORBITAL_GROUPS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("s", (0,)),
    ("p", (1, 2, 3)),
    ("d", (4, 5, 6, 7, 8)),
    ("f", (9, 10, 11, 12, 13, 14, 15)),
)

ORBITAL_INDEX_LABEL_MAP: Dict[Union[int, Tuple[int, ...]], str] = {
    0: "s",
    1: "p_y",
    2: "p_z",
    3: "p_x",
    4: "d_{xy}",
    5: "d_{yz}",
    6: "d_{z^2}",
    7: "d_{xz}",
    8: "d_{x^2-y^2}",
    9: "f_{y^3x^2}",
    10: "f_{xyz}",
    11: "f_{yz^2}",
    12: "f_{z^3}",
    13: "f_{xz^2}",
    14: "f_{zx^2}",
    15: "f_{xx}",
    (1, 2, 3): "p",
    (4, 5, 6, 7, 8): "d",
    (9, 10, 11, 12, 13, 14, 15): "f",
    (4, 5, 7): "t_{2g}",
    (5, 6, 8): "e_g",
    tuple(range(0, 16)): "all",
}

ORBITAL_INDEX_TO_LABEL: Dict[int, str] = {
    key: value for key, value in ORBITAL_INDEX_LABEL_MAP.items() if isinstance(key, int)
}
ORBITAL_GROUP_LABELS: Dict[Tuple[int, ...], str] = {
    key: value for key, value in ORBITAL_INDEX_LABEL_MAP.items() if isinstance(key, tuple)
}

ORBITAL_LATEX_MAP: Dict[str, str] = {
    "s": "s",
    "p": "p",
    "py": "p_{y}",
    "pz": "p_{z}",
    "px": "p_{x}",
    "d": "d",
    "dxy": "d_{xy}",
    "dyz": "d_{yz}",
    "dz2": "d_{z^2}",
    "dzx": "d_{zx}",
    "dzy": "d_{zy}",
    "dxz": "d_{xz}",
    "dx2-y2": "d_{x^2-y^2}",
    "f": "f",
    "fy3x2": "f_{y^3x^2}",
    "fxyz": "f_{xyz}",
    "fyz2": "f_{yz^2}",
    "fz3": "f_{z^3}",
    "fxz2": "f_{xz^2}",
    "fzx2": "f_{zx^2}",
    "fx3": "f_{x^3}",
    "fx(x2-3y2)": "f_{x(x^2-3y^2)}",
    "fy(3x2-y2)": "f_{y(3x^2-y^2)}",
    "t2g": "t_{2g}",
    "eg": "e_g",
}

LEGACY_ORBITAL_NAMES: Dict[str, Union[int, List[int]]] = {
    "p": [1, 2, 3],
    "d": [4, 5, 6, 7, 8],
    "f": [9, 10, 11, 12, 13, 14, 15],
    "s": 0,
    "py": 1,
    "pz": 2,
    "px": 3,
    "dxy": 4,
    "dyz": 5,
    "dz2": 6,
    "dxz": 7,
    "x2-y2": 8,
    "fy3x2": 9,
    "fxyz": 10,
    "fyz2": 11,
    "fz3": 12,
    "fxz2": 13,
    "fzx2": 14,
    "fxx": 15,
}


def _normalize_indices(indices: Iterable[int] | None) -> List[int]:
    if indices is None:
        return []
    try:
        normalized = {int(idx) for idx in indices}
    except TypeError:
        normalized = {int(indices)}
    return sorted(normalized)


def format_index_ranges(indices: Sequence[int] | None) -> str:
    normalized = _normalize_indices(indices)
    if not normalized:
        return ""

    ranges: List[Tuple[int, int]] = []
    start = prev = normalized[0]
    for value in normalized[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))

    parts: List[str] = []
    for lower, upper in ranges:
        if lower == upper:
            parts.append(str(lower))
        else:
            parts.append(f"{lower}-{upper}")
    return ",".join(parts)


@dataclass
class OrbitalIndexer:
    """Provides conversions between orbital indices and naming conventions."""

    azimuthal_order: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: dict(AZIMUTHAL_ORBITAL_ORDER)
    )
    conventional_order: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: dict(CONVENTIONAL_CUBIC_ORBITAL_ORDER)
    )
    flat_soc_order: Sequence[Mapping[str, Union[int, float]]] = field(
        default_factory=lambda: list(NONCOLINEAR_AZIMUTHAL_ORBITAL_ORDER)
    )
    index_label_map: Mapping[int, str] = field(
        default_factory=lambda: dict(ORBITAL_INDEX_TO_LABEL)
    )
    group_label_map: Mapping[Tuple[int, ...], str] = field(
        default_factory=lambda: dict(ORBITAL_GROUP_LABELS)
    )

    @property
    def flat_azimuthal(self) -> List[str]:
        return self._flatten_order(self.azimuthal_order)

    @property
    def flat_conventional(self) -> List[str]:
        return self._flatten_order(self.conventional_order)

    @property
    def az_to_conv_map(self) -> Dict[int, int]:
        return self._build_index_map(self.flat_azimuthal, self.flat_conventional)

    @property
    def conv_to_az_map(self) -> Dict[int, int]:
        return {v: k for k, v in self.az_to_conv_map.items()}

    @property
    def az_to_flat_index(self) -> Dict[str, int]:
        return {orbital_name: i for i, orbital_name in enumerate(self.flat_azimuthal)}

    @property
    def conv_to_flat_index(self) -> Dict[str, int]:
        return {orbital_name: i for i, orbital_name in enumerate(self.flat_conventional)}

    @property
    def l_orbital_map(self) -> Dict[str, int]:
        return {l_orbital_name: i for i, l_orbital_name in enumerate(self.azimuthal_order.keys())}

    @property
    def az_to_lm_records(self) -> List[Dict[str, int]]:
        az_to_lm_records: List[Dict[str, int]] = []
        for l_orbital_name in self.azimuthal_order.keys():
            for i_m, _ in enumerate(self.azimuthal_order[l_orbital_name]):
                az_to_lm_records.append(
                    {
                        "l": self.l_orbital_map[l_orbital_name],
                        "m": i_m + 1,
                    }
                )
        return az_to_lm_records

    def label(
        self,
        indices: Sequence[int] | None,
        *,
        orbital_names: Sequence[str] | None = None,
        is_non_colinear: bool = False,
        prefer_groups: bool = True,
        sanitize: bool = True,
    ) -> str:
        tokens = self._build_label_tokens(
            indices=indices,
            orbital_names=orbital_names,
            is_non_colinear=is_non_colinear,
            prefer_groups=prefer_groups,
        )
        if not tokens:
            return ""
        if sanitize:
            tokens = [self._sanitize_label(token) for token in tokens]
        return ",".join(tokens)

    def label_with_latex(
        self,
        indices: Sequence[int] | None,
        *,
        orbital_names: Sequence[str] | None = None,
        is_non_colinear: bool = False,
        prefer_groups: bool = True,
        sanitize: bool = True,
    ) -> tuple[str, str]:
        tokens = self._build_label_tokens(
            indices=indices,
            orbital_names=orbital_names,
            is_non_colinear=is_non_colinear,
            prefer_groups=prefer_groups,
        )
        if not tokens:
            return "", ""

        if sanitize:
            plain_tokens = [self._sanitize_label(token) for token in tokens]
        else:
            plain_tokens = list(tokens)
        latex_tokens = [self._token_to_latex(token) for token in tokens]

        return ",".join(plain_tokens), ",".join(latex_tokens)

    def _build_label_tokens(
        self,
        *,
        indices: Sequence[int] | None,
        orbital_names: Sequence[str] | None,
        is_non_colinear: bool,
        prefer_groups: bool,
    ) -> list[str]:
        normalized = _normalize_indices(indices)
        if not normalized:
            return []

        if is_non_colinear and (orbital_names is None or len(orbital_names) == 0):
            return [self._format_soc_label(idx) for idx in normalized]

        remaining = set(normalized)
        tokens: list[str] = []
        if prefer_groups:
            for group_name, group_indices in PRIMARY_ORBITAL_GROUPS:
                group_set = set(group_indices)
                if group_set and group_set <= remaining:
                    tokens.append(group_name)
                    remaining -= group_set

        for idx in sorted(remaining):
            if orbital_names is not None and 0 <= idx < len(orbital_names):
                tokens.append(orbital_names[idx])
            else:
                tokens.append(self.index_label_map.get(idx, f"o{idx}"))

        return tokens

    def _token_to_latex(self, token: str) -> str:
        if token in ORBITAL_LATEX_MAP.values():
            return token
        if token in {"all"}:
            return token
        if token.startswith("l") and "_j" in token and "_m" in token:
            segments = token.split("_")
            latex_segments = []
            for segment in segments:
                if not segment:
                    continue
                latex_segments.append(f"{segment[0]}_{{{segment[1:]}}}")
            return "\\,".join(latex_segments)
        if "{" in token and "}" in token:
            return token
        normalized = token.replace(" ", "")
        normalized = normalized.replace("{", "").replace("}", "")
        if normalized in ORBITAL_LATEX_MAP:
            return ORBITAL_LATEX_MAP[normalized]
        normalized_no_underscore = normalized.replace("_", "")
        if normalized_no_underscore in ORBITAL_LATEX_MAP:
            return ORBITAL_LATEX_MAP[normalized_no_underscore]
        if "_" in token:
            head, tail = token.split("_", 1)
            if tail.startswith("{"):
                return f"{head}{tail}"
            return f"{head}_{{{tail}}}"
        return ORBITAL_LATEX_MAP.get(token, token)

    def get_index(self, orbital: str, convention: str = "azimuthal") -> int:
        if convention == "azimuthal":
            return self.flat_azimuthal.index(orbital)
        if convention == "conventional":
            return self.flat_conventional.index(orbital)
        raise ValueError("Convention must be 'azimuthal' or 'conventional'.")

    def map_index(self, index: int, from_convention: str, to_convention: str) -> int:
        if from_convention == "azimuthal" and to_convention == "conventional":
            return self.az_to_conv_map[index]
        if from_convention == "conventional" and to_convention == "azimuthal":
            return self.conv_to_az_map[index]
        raise ValueError("Invalid convention mapping.")

    def get_soc_index(self, l: int, j: float, m: float) -> int:
        for idx, entry in enumerate(self.flat_soc_order):
            if entry["l"] == l and entry["j"] == j and entry["m_j"] == m:
                return idx
        raise ValueError("SOC orbital not found.")

    def to_latex(self, orbital: str) -> str:
        replacements = {
            "pz": "p_z",
            "px": "p_x",
            "py": "p_y",
            "dz2": "d_{z^2}",
            "dzx": "d_{zx}",
            "dzy": "d_{zy}",
            "dx2-y2": "d_{x^2-y^2}",
            "dxy": "d_{xy}",
            "fz3": "f_{z^3}",
            "fxz2": "f_{xz^2}",
            "fyz2": "f_{yz^2}",
            "fx( x2-3y2 )": "f_{x(x^2-3y^2)}",
            "fxyz": "f_{xyz}",
            "fy(3x2-y2)": "f_{y(3x^2-y^2)}",
            "fx3": "f_{x^3}",
        }
        return f"$\\{replacements.get(orbital, orbital)}$"

    def _flatten_order(self, order_dict: Mapping[str, Sequence[str]]) -> List[str]:
        flat_list: List[str] = []
        for l_type in ("s", "p", "d", "f"):
            if l_type in order_dict:
                flat_list.extend(order_dict[l_type])
        return flat_list

    def _build_index_map(
        self, list_a: Sequence[str], list_b: Sequence[str]
    ) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for i, orb in enumerate(list_a):
            if orb in list_b:
                mapping[i] = list_b.index(orb)
        return mapping

    def _format_soc_label(self, index: int) -> str:
        if 0 <= index < len(self.flat_soc_order):
            entry = self.flat_soc_order[index]
            l = entry.get("l")
            j = entry.get("j")
            m_j = entry.get("m_j")
            j_str = f"{j:g}" if isinstance(j, float) else str(j)
            m_str = f"{m_j:g}" if isinstance(m_j, float) else str(m_j)
            return f"l{l}_j{j_str}_m{m_str}"
        return str(index)

    @staticmethod
    def _sanitize_label(label: str) -> str:
        sanitized = label.replace("\\", "")
        sanitized = sanitized.replace("_{", "_")
        sanitized = sanitized.replace("{", "")
        sanitized = sanitized.replace("}", "")
        sanitized = sanitized.replace("^", "")
        sanitized = sanitized.replace(" ", "")
        return sanitized


@dataclass
class AtomIndexer:
    """Formats atom selections grouped by species with compact ranges."""

    species_per_atom: Sequence[str] | None = None
    max_indices_for_ranges: int = 12

    def __post_init__(self) -> None:
        if self.species_per_atom is not None:
            self.species_per_atom = tuple(str(spec) for spec in self.species_per_atom)

    @classmethod
    def from_structure(
        cls, structure: Structure | None, *, max_indices_for_ranges: int = 12
    ) -> "AtomIndexer":
        if structure is None:
            return cls(species_per_atom=None, max_indices_for_ranges=max_indices_for_ranges)
        return cls(
            species_per_atom=tuple(str(atom) for atom in structure.atoms),
            max_indices_for_ranges=max_indices_for_ranges,
        )

    def label(
        self,
        indices: Sequence[int] | None,
        *,
        species: Sequence[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        normalized_species = self._normalize_species(species)
        if indices is None and normalized_species is None:
            return ""

        resolved = self._resolve_indices(indices, normalized_species)
        if not resolved:
            if normalized_species:
                return ",".join(normalized_species)
            return ""

        limit = self.max_indices_for_ranges if max_tokens is None else max_tokens
        if limit is not None and len(resolved) > limit:
            species_tokens = self._species_names_from_indices(resolved, normalized_species)
            if species_tokens:
                return ",".join(species_tokens)
            return f"{len(resolved)} atoms"

        grouped = self._group_by_species(resolved, normalized_species)
        labels: List[str] = []
        for specie, idxs in grouped:
            range_str = format_index_ranges(idxs)
            if specie:
                labels.append(f"{specie}_{{{range_str}}}" if range_str else specie)
            else:
                labels.append(range_str)
        return "".join(labels)

    def _resolve_indices(
        self, indices: Sequence[int] | None, species_list: List[str] | None
    ) -> Tuple[int, ...]:
        selected = set(_normalize_indices(indices))
        if species_list:
            species_indices = self._indices_for_species(species_list)
            if selected:
                selected &= species_indices
            else:
                selected = species_indices
        return tuple(sorted(selected))

    def _indices_for_species(self, species_list: List[str]) -> set[int]:
        if not self.species_per_atom:
            return set()
        targets = set(species_list)
        return {
            idx for idx, specie in enumerate(self.species_per_atom) if specie in targets
        }

    def _group_by_species(
        self, indices: Tuple[int, ...], species_order: List[str] | None
    ) -> List[Tuple[str, List[int]]]:
        if not indices:
            return []
        if not self.species_per_atom:
            return [("", list(indices))]

        if species_order:
            ordered_species = list(dict.fromkeys(species_order))
        else:
            ordered_species = []
            for idx in indices:
                specie = self.species_per_atom[idx]
                if specie not in ordered_species:
                    ordered_species.append(specie)

        grouped: List[Tuple[str, List[int]]] = []
        for specie in ordered_species:
            specie_indices = [
                idx for idx in indices if self.species_per_atom[idx] == specie
            ]
            if specie_indices:
                grouped.append((specie, specie_indices))
        return grouped

    def _species_names_from_indices(
        self, indices: Tuple[int, ...], species_order: List[str] | None
    ) -> List[str]:
        if self.species_per_atom:
            ordered: List[str] = []
            for idx in indices:
                specie = self.species_per_atom[idx]
                if specie not in ordered:
                    ordered.append(specie)
            return ordered
        return species_order or []

    def _normalize_species(self, species: Sequence[str] | str | None) -> List[str] | None:
        if species is None:
            return None
        if isinstance(species, str):
            return [species]
        return [str(item) for item in species]

    def species_atom_map(self, species: Sequence[str] | str | None = None) -> dict[str, tuple[int, ...]]:
        if self.species_per_atom is None:
            raise ValueError("Species information is not available for atom indexing")

        target_species = self._normalize_species(species)
        mapping: dict[str, list[int]] = {}
        for idx, specie in enumerate(self.species_per_atom):
            if target_species is None or specie in target_species:
                mapping.setdefault(specie, []).append(idx)

        if target_species is not None:
            ordered = [specie for specie in target_species if specie in mapping]
            return {specie: tuple(mapping[specie]) for specie in ordered}

        return {specie: tuple(indices) for specie, indices in mapping.items()}

    def species_from_atoms(self, atoms: Sequence[int] | None) -> list[str]:
        if self.species_per_atom is None or atoms is None:
            return []
        ordered_species: list[str] = []
        for idx in atoms:
            specie = self.species_per_atom[int(idx)]
            if specie not in ordered_species:
                ordered_species.append(specie)
        return ordered_species


@dataclass
class SpinIndexer:
    """Maps spin channel indices to descriptive labels."""

    names: Sequence[str]

    @classmethod
    def from_projection_names(cls, names: Sequence[str]) -> "SpinIndexer":
        return cls(tuple(names))

    @classmethod
    def from_counts(
        cls, n_spins: int, *, is_non_colinear: bool = False
    ) -> "SpinIndexer":
        if is_non_colinear:
            return cls(("total", "x", "y", "z"))
        if n_spins == 2:
            return cls(("Spin-up", "Spin-down"))
        return cls(("Spin-up",))

    def label(self, spins: Sequence[int] | None) -> str:
        if spins is None:
            return ""
        normalized = _normalize_indices(spins)
        if not normalized:
            return ""
        labels: List[str] = []
        for idx in normalized:
            if 0 <= idx < len(self.names):
                labels.append(self.names[idx])
            else:
                labels.append(str(idx))
        return ",".join(labels)

    def label_latex(self, spins: Sequence[int] | None) -> str:
        if spins is None:
            return ""
        normalized = _normalize_indices(spins)
        if not normalized:
            return ""
        latex_labels: list[str] = []
        for idx in normalized:
            if 0 <= idx < len(self.names):
                name = self.names[idx]
            else:
                name = str(idx)
            latex_labels.append(self._spin_name_to_latex(name))
        return ",".join(latex_labels)

    @staticmethod
    def _spin_name_to_latex(name: str) -> str:
        mapping = {
            "spin-up": "\\uparrow",
            "spin-down": "\\downarrow",
            "total": "\\mathrm{total}",
            "x": "S_x",
            "y": "S_y",
            "z": "S_z",
        }
        key = name.lower()
        return mapping.get(key, name)

    def component_labels(self, spins: Sequence[int] | None) -> list[str]:
        if spins is None:
            return []
        normalized = _normalize_indices(spins)
        labels: list[str] = []
        for idx in normalized:
            if 0 <= idx < len(self.names):
                labels.append(self.names[idx])
            else:
                labels.append(str(idx))
        return labels

    def component_labels_latex(self, spins: Sequence[int] | None) -> list[str]:
        if spins is None:
            return []
        normalized = _normalize_indices(spins)
        labels: list[str] = []
        for idx in normalized:
            if 0 <= idx < len(self.names):
                name = self.names[idx]
            else:
                name = str(idx)
            labels.append(self._spin_name_to_latex(name))
        return labels


@dataclass(frozen=True)
class ProjectionLabels:
    atom: str
    orbital: str
    spin: str
    species: str
    combined: str
    atom_latex: str
    orbital_latex: str
    spin_latex: str
    species_latex: str
    combined_latex: str
    prefix_plain: str
    prefix_latex: str
    spin_components: tuple[str, ...]
    spin_components_latex: tuple[str, ...]


@dataclass
class ProjectionLabelBuilder:
    """Compose human-readable labels for atom/orbital/spin selections."""

    atom_indexer: AtomIndexer | None = None
    orbital_indexer: OrbitalIndexer = field(default_factory=OrbitalIndexer)
    spin_indexer: SpinIndexer | None = None
    max_atom_tokens: int = 12

    def build_label(
        self,
        *,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        species: Sequence[str] | str | None = None,
        orbital_names: Sequence[str] | None = None,
        include_spins: bool = False,
        is_non_colinear: bool = False,
    ) -> str:
        return self.build_components(
            atoms=atoms,
            orbitals=orbitals,
            spins=spins,
            species=species,
            orbital_names=orbital_names,
            include_spins=include_spins,
            is_non_colinear=is_non_colinear,
        ).combined

    def build_components(
        self,
        *,
        atoms: Sequence[int] | None = None,
        orbitals: Sequence[int] | None = None,
        spins: Sequence[int] | None = None,
        species: Sequence[str] | str | None = None,
        orbital_names: Sequence[str] | None = None,
        include_spins: bool = False,
        is_non_colinear: bool = False,
    ) -> ProjectionLabels:
        species_list = self._normalize_species(species)

        atom_label = ""
        if self.atom_indexer and (atoms is not None or species_list is not None):
            atom_label = self.atom_indexer.label(
                indices=atoms, species=species_list, max_tokens=self.max_atom_tokens
            )
        elif atoms is not None:
            atom_label = format_index_ranges(atoms)
        atom_label_latex = self._atom_label_to_latex(atom_label)

        orbital_label = ""
        orbital_label_latex = ""
        if orbitals is not None:
            orbital_label, orbital_label_latex = self.orbital_indexer.label_with_latex(
                indices=orbitals,
                orbital_names=orbital_names,
                is_non_colinear=is_non_colinear,
            )

        spin_label = ""
        spin_label_latex = ""
        spin_components: tuple[str, ...] = tuple()
        spin_components_latex: tuple[str, ...] = tuple()
        if include_spins and self.spin_indexer is not None and spins is not None:
            spin_components = tuple(self.spin_indexer.component_labels(spins))
            spin_components_latex = tuple(self.spin_indexer.component_labels_latex(spins))
            spin_label = ",".join(spin_components)
            spin_label_latex = ",".join(spin_components_latex)

        prefix_plain = atom_label
        if orbital_label:
            prefix_plain = f"{prefix_plain}-({orbital_label})" if prefix_plain else f"({orbital_label})"

        prefix_latex = atom_label_latex
        if orbital_label_latex:
            prefix_latex = (
                f"{prefix_latex}-({orbital_label_latex})"
                if prefix_latex
                else f"({orbital_label_latex})"
            )

        label_plain = prefix_plain
        if spin_label:
            label_plain = f"{label_plain}[{spin_label}]" if label_plain else spin_label

        species_label = ",".join(species_list) if species_list else ""
        species_label_latex = (
            ",".join(self._species_to_latex(item) for item in species_list)
            if species_list
            else ""
        )

        combined_plain = label_plain or "all"

        combined_latex = prefix_latex
        if spin_label_latex:
            combined_latex = (
                f"{combined_latex}[{spin_label_latex}]"
                if combined_latex
                else f"[{spin_label_latex}]"
            )
        if not combined_latex:
            combined_latex = "\\mathrm{all}"

        return ProjectionLabels(
            atom=atom_label,
            orbital=orbital_label,
            spin=spin_label,
            species=species_label,
            combined=combined_plain,
            atom_latex=atom_label_latex,
            orbital_latex=orbital_label_latex,
            spin_latex=spin_label_latex,
            species_latex=species_label_latex,
            combined_latex=combined_latex,
            prefix_plain=prefix_plain,
            prefix_latex=prefix_latex,
            spin_components=spin_components,
            spin_components_latex=spin_components_latex,
        )

    @staticmethod
    def _species_to_latex(specie: str) -> str:
        return f"\\mathrm{{{specie}}}"

    def _atom_label_to_latex(self, label: str) -> str:
        if not label:
            return ""

        def replacer(match: re.Match[str]) -> str:
            return f"\\mathrm{{{match.group(1)}}}"

        return re.sub(r"([A-Z][a-z]?)", replacer, label)

    def _normalize_species(
        self, species: Sequence[str] | str | None
    ) -> List[str] | None:
        if species is None:
            return None
        if isinstance(species, str):
            return [species]
        return [str(item) for item in species]


@dataclass(frozen=True)
class ProjectionSelectionResult:
    atoms: tuple[int, ...]
    orbitals: tuple[int, ...] | None
    spins: tuple[int, ...] | None
    species: tuple[str, ...]
    labels: ProjectionLabels


class ProjectionSelectionResolver:
    """Resolve selection inputs into canonical index tuples and labels."""

    def __init__(
        self,
        *,
        label_builder: ProjectionLabelBuilder,
        orbital_names: Sequence[str] | None = None,
        is_non_colinear: bool = False,
    ) -> None:
        self.label_builder = label_builder
        self.atom_indexer = label_builder.atom_indexer
        self.orbital_names = orbital_names
        self.is_non_colinear = is_non_colinear

    def resolve(
        self,
        *,
        atoms: Sequence[int] | int | None = None,
        orbitals: Sequence[int] | int | None = None,
        spins: Sequence[int] | int | None = None,
        species: Sequence[str] | str | None = None,
        species_orbital_map: Sequence[Mapping[str, Iterable[int]]] | Mapping[str, Iterable[int]] | None = None,
        atoms_orbital_map: Sequence[Mapping[Iterable[int] | int, Iterable[int]]] | Mapping[Iterable[int] | int, Iterable[int]] | None = None,
    ) -> ProjectionSelectionResult:
        self._validate_exclusive_inputs(
            atoms=atoms,
            orbitals=orbitals,
            species=species,
            species_orbital_map=species_orbital_map,
            atoms_orbital_map=atoms_orbital_map,
        )

        atoms_set = self._normalize_indices_set(atoms)
        orbitals_set = self._normalize_indices_set(orbitals)
        spins_set = self._normalize_indices_set(spins)

        species_list = self._normalize_species_sequence(species)
        species_set = set(species_list) if species_list is not None else None

        normalized_species_maps = self._normalize_species_orbital_map(species_orbital_map)
        if normalized_species_maps is not None:
            species_list = []
            species_set = set()
            orbitals_set = set()
            for mapping in normalized_species_maps:
                for specie, orbital_indices in mapping.items():
                    specie_str = str(specie)
                    if specie_str not in species_set:
                        species_list.append(specie_str)
                        species_set.add(specie_str)
                    orbitals_set.update(self._normalize_indices_set(orbital_indices) or set())
            atoms_set = set()
            for specie in species_list:
                atoms_set.update(self._atoms_for_species(specie))

        normalized_atoms_maps = self._normalize_atoms_orbital_map(atoms_orbital_map)
        if normalized_atoms_maps is not None:
            atoms_set = set()
            orbitals_set = set()
            for mapping in normalized_atoms_maps:
                for atom_indices, orbital_indices in mapping.items():
                    atoms_set.update(self._normalize_indices_set(atom_indices) or set())
                    orbitals_set.update(self._normalize_indices_set(orbital_indices) or set())
            if atoms_set:
                species_list = self._species_from_atoms(sorted(atoms_set))
                species_set = set(species_list)
            else:
                species_list = []
                species_set = set()

        if species_list is not None:
            atoms_set = set()
            for specie in species_list:
                atoms_set.update(self._atoms_for_species(specie))
        elif atoms_set is None:
            if self.atom_indexer is None:
                raise ValueError(
                    "Atom indexer is required when atoms and species selections are omitted"
                )
            species_map = self.atom_indexer.species_atom_map()
            species_list = list(species_map.keys())
            atoms_set = {idx for indices in species_map.values() for idx in indices}
        else:
            species_list = self._species_from_atoms(sorted(atoms_set))

        atoms_tuple = tuple(sorted(atoms_set)) if atoms_set is not None else tuple()
        orbitals_tuple = (
            tuple(sorted(orbitals_set)) if orbitals_set is not None and len(orbitals_set) > 0 else None
        )
        spins_tuple = tuple(sorted(spins_set)) if spins_set is not None else None
        species_tuple = tuple(species_list) if species_list is not None else tuple()

        labels = self.label_builder.build_components(
            atoms=atoms_tuple if atoms_tuple else None,
            orbitals=orbitals_tuple,
            spins=spins_tuple,
            species=species_tuple,
            orbital_names=self.orbital_names,
            include_spins=spins_tuple is not None,
            is_non_colinear=self.is_non_colinear,
        )

        return ProjectionSelectionResult(
            atoms=atoms_tuple,
            orbitals=orbitals_tuple,
            spins=spins_tuple,
            species=species_tuple,
            labels=labels,
        )

    def _normalize_indices_set(self, values: Iterable[int] | int | None) -> set[int] | None:
        if values is None:
            return None
        return {int(item) for item in self._flatten_ints(values)}

    def _normalize_species_sequence(
        self, species: Sequence[str] | str | None
    ) -> list[str] | None:
        if species is None:
            return None
        if isinstance(species, str):
            return [species]
        return [str(item) for item in species]

    def _normalize_species_orbital_map(
        self,
        mapping: Sequence[Mapping[str, Iterable[int]]] | Mapping[str, Iterable[int]] | None,
    ) -> list[Mapping[str, Iterable[int]]] | None:
        if mapping is None:
            return None
        if isinstance(mapping, ABCMapping):
            return [mapping]
        return [self._ensure_mapping(item) for item in mapping]

    def _normalize_atoms_orbital_map(
        self,
        mapping: Sequence[Mapping[Iterable[int] | int, Iterable[int]]] | Mapping[Iterable[int] | int, Iterable[int]] | None,
    ) -> list[Mapping[Iterable[int] | int, Iterable[int]]] | None:
        if mapping is None:
            return None
        if isinstance(mapping, ABCMapping):
            return [mapping]
        return [self._ensure_mapping(item) for item in mapping]

    @staticmethod
    def _ensure_mapping(mapping: Mapping) -> Mapping:
        if not isinstance(mapping, ABCMapping):
            raise TypeError("Expected a mapping of indices to orbital selections")
        return mapping

    def _atoms_for_species(self, specie: str) -> tuple[int, ...]:
        if self.atom_indexer is None:
            raise ValueError("Species selections require atom indexing information")
        mapping = self.atom_indexer.species_atom_map([specie])
        if specie not in mapping:
            raise ValueError(f"Species '{specie}' not found in atom index")
        return mapping[specie]

    def _species_from_atoms(self, atoms: Sequence[int]) -> list[str]:
        if self.atom_indexer is None:
            return []
        return self.atom_indexer.species_from_atoms(atoms)

    @staticmethod
    def _flatten_ints(values: Iterable[int] | int) -> Iterable[int]:
        if isinstance(values, ABCIterable) and not isinstance(values, (str, bytes)):
            for value in values:
                yield from ProjectionSelectionResolver._flatten_ints(value)
            return
        yield int(values)

    @staticmethod
    def _validate_exclusive_inputs(
        *,
        atoms: Sequence[int] | int | None,
        orbitals: Sequence[int] | int | None,
        species: Sequence[str] | str | None,
        species_orbital_map: Sequence[Mapping[str, Iterable[int]]] | Mapping[str, Iterable[int]] | None,
        atoms_orbital_map: Sequence[Mapping[Iterable[int] | int, Iterable[int]]] | Mapping[Iterable[int] | int, Iterable[int]] | None,
    ) -> None:
        if species is not None and atoms is not None:
            raise ValueError("atoms and species cannot be specified together")
        if species_orbital_map is not None and (
            species is not None or atoms is not None or orbitals is not None
        ):
            raise ValueError(
                "species_orbital_map cannot be specified together with species, atoms, or orbitals"
            )
        if atoms_orbital_map is not None and (
            species is not None or atoms is not None or orbitals is not None
        ):
            raise ValueError(
                "atoms_orbital_map cannot be specified together with species, atoms, or orbitals"
            )


__all__ = [
    "AZIMUTHAL_ORBITAL_ORDER",
    "CONVENTIONAL_CUBIC_ORBITAL_ORDER",
    "NONCOLINEAR_AZIMUTHAL_ORBITAL_ORDER",
    "PRIMARY_ORBITAL_GROUPS",
    "ORBITAL_INDEX_LABEL_MAP",
    "ORBITAL_INDEX_TO_LABEL",
    "ORBITAL_GROUP_LABELS",
    "LEGACY_ORBITAL_NAMES",
    "format_index_ranges",
    "OrbitalIndexer",
    "AtomIndexer",
    "SpinIndexer",
    "ProjectionLabels",
    "ProjectionLabelBuilder",
    "ProjectionSelectionResult",
    "ProjectionSelectionResolver",
]
