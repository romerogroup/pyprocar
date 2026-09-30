"""Channel and projection selection shared by the one-call plotting functions."""

import numpy as np

from pyprocar.core.atomic_orbital_index import PRIMARY_ORBITAL_GROUPS
from pyprocar.core.property_store import Property

ORBITAL_GROUPS = {name: list(indices) for name, indices in PRIMARY_ORBITAL_GROUPS}


def orbital_indices(orbitals):
    """Resolve orbital group names ("s", "p", "d", "f") to indices; indices pass through."""
    if orbitals is None or len(orbitals) == 0 or not isinstance(orbitals[0], str):
        return orbitals
    return [i for name in orbitals for i in ORBITAL_GROUPS[name]]


def projection_components(source, kind: str, atoms=None, orbitals=None, items=None, **kwargs):
    """One projected sum per species, per orbital group, or per ``items`` entry.

    ``source`` is anything with ``compute_projected_sum``, ``structure`` and
    ``n_orbitals``: an electronic band structure or a density of states.
    """
    if kind == "species":
        selections = {s: {"species": [s], "orbitals": orbitals} for s in source.structure.species}
    elif kind == "orbitals":
        groups = ["s", "p", "d", "f"] if source.n_orbitals > 9 else ["s", "p", "d"]
        selections = {g: {"atoms": atoms, "orbitals": ORBITAL_GROUPS[g]} for g in groups}
    else:
        selections = {
            s: {"species": [s], "orbitals": orbital_indices(orbs)}
            for s, orbs in (items or {}).items()
        }
    return [
        source.compute_projected_sum(label=label, **selection, **kwargs)
        for label, selection in selections.items()
    ]


def take_channels(prop: Property, channels: list[int]) -> Property:
    """Keep only ``channels`` of the last (spin channel) axis, with their per-channel labels."""
    metadata = dict(prop.metadata)
    labels = metadata.get("label")
    if isinstance(labels, list) and len(labels) == prop.value.shape[-1]:
        metadata["label"] = [labels[c] for c in channels]
    return Property(
        name=prop.name,
        value=np.take(prop.value, channels, axis=-1),
        units=prop.units,
        label=prop.label,
        point_set=prop.point_set,
        metadata=metadata,
    )


def per_channel(value, n_channels: int):
    """Tuple config values hold one entry per spin channel; the plotters split lists by channel."""
    if not isinstance(value, (list, tuple)):
        return value
    return list(value[:n_channels]) if n_channels > 1 else value[0]
