"""Channel and projection selection shared by the one-call plotting functions."""

from typing import NamedTuple

import numpy as np

from pyprocar.core.atomic_orbital_index import orbital_shells
from pyprocar.core.property_store import Property

SHELL_LETTERS = ("s", "p", "d", "f")


def present_shells(source) -> dict[str, list[int]]:
    """The whole s, p, d and f shells among ``source``'s orbitals, by the names it carries."""
    return {
        letter: list(indices)
        for letter, indices in orbital_shells(source.orbital_names)
        if max(indices) < source.n_orbitals
    }


def orbital_indices(orbitals, source):
    """Resolve shell names (s, p, d, f) to the orbitals of ``source``; indices pass through."""
    if orbitals is None or len(orbitals) == 0:
        return orbitals
    shells = present_shells(source)
    indices = []
    for orbital in orbitals:
        if not isinstance(orbital, str):
            indices.append(orbital)
        elif orbital in shells:
            indices.extend(shells[orbital])
        elif orbital in SHELL_LETTERS:
            raise ValueError(
                f"orbitals names the {orbital} shell, but the orbitals {source.orbital_names}"
                + f" hold no whole {orbital} shell. Select orbitals by index."
            )
        else:
            raise ValueError(
                f"orbitals takes orbital indices or the shell names {', '.join(SHELL_LETTERS)},"
                + f" not {orbital!r}. Select one orbital by its index, its position in the"
                + " orbital names, for example 8 for d x2-y2 in VASP's order."
            )
    return indices


class SpinSelection(NamedTuple):
    channels: list[int]
    projection_spins: list[int]
    joined: bool = False


def resolve_spins(
    is_non_collinear: bool, n_channels: int, spins, plain: bool = True
) -> SpinSelection:
    """Return the channels to draw and the spins to project, one policy for every script.

    Collinear data draws the requested spin channels (all by default) and projects
    those same channels. ``spins=[-1, 1]`` joins both collinear channels into one,
    which only plain plots support. Non-collinear data has one channel; ``spins``
    picks one spin component (0 total, 1-3 Sx, Sy, Sz) and defaults to the total.
    """
    if spins is not None and sorted(spins) == [-1, 1]:
        if is_non_collinear or n_channels != 2:
            raise ValueError("spins=[-1, 1] joins two collinear spin channels")
        if not plain:
            raise ValueError(
                "spins=[-1, 1] joins the two spin channels and only works in plain mode"
            )
        return SpinSelection([0, 1], [0, 1], joined=True)
    if not is_non_collinear:
        channels = list(range(n_channels)) if spins is None else list(spins)
        return SpinSelection(channels, channels)
    if spins is not None and len(spins) != 1:
        raise ValueError(
            "Non-collinear calculations take one spin component"
            + f" (0 total, 1-3 Sx, Sy, Sz); got spins={list(spins)}"
        )
    return SpinSelection([0], [0] if spins is None else list(spins))


def projection_components(source, kind: str, atoms=None, orbitals=None, items=None, **kwargs):
    """One projected sum per species, per orbital group, or per ``items`` entry.

    ``source`` is anything with ``compute_projected_sum``, ``structure`` and
    ``n_orbitals``: an electronic band structure or a density of states.
    """
    if kind == "species":
        selections = [(s, {"species": [s], "orbitals": orbitals}) for s in source.structure.species]
    elif kind == "orbitals":
        shells = present_shells(source)
        if not shells:
            raise ValueError(
                f"The orbitals {source.orbital_names} hold no whole s, p, d or f shell to overlay."
            )
        selections = [(g, {"atoms": atoms, "orbitals": shells[g]}) for g in shells]
    else:
        mappings = [items] if isinstance(items, dict) else list(items or [])
        selections = [
            (s, {"species": [s], "orbitals": orbital_indices(orbs, source)})
            for mapping in mappings
            for s, orbs in mapping.items()
        ]
    return [
        source.compute_projected_sum(label=label, **selection, **kwargs)
        for label, selection in selections
    ]


def as_clim(values) -> tuple[float, float] | None:
    """A (low, high) colour range from a user-supplied two-item sequence, or None."""
    return None if values is None else (float(values[0]), float(values[1]))


def as_lim(values, current: tuple[float, float]) -> tuple[float, float] | None:
    """Axis limits from a user-supplied pair; a None bound keeps the ``current`` one."""
    if values is None:
        return None
    low, high = values
    return (
        float(current[0] if low is None else low),
        float(current[1] if high is None else high),
    )


def signed_clim(prop: Property) -> tuple[float, float] | None:
    """A colour range centred on zero for signed projections (spin components), else None."""
    values = prop.to_array()
    if np.nanmin(values) >= 0:
        return None
    bound = float(np.nanmax(np.abs(values)))
    return (-bound, bound)


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
