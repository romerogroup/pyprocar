from types import SimpleNamespace

import numpy as np
import pytest

from pyprocar.core.projection import NormMode, build_property, normalize, selection_resolver
from pyprocar.core.property_store import PointSet
from pyprocar.core.structure import Structure


def test_norm_mode_parse():
    assert NormMode.parse("MAX") is NormMode.MAX
    assert NormMode.parse(None) is NormMode.RAW
    assert NormMode.parse(NormMode.TOTAL) is NormMode.TOTAL
    with pytest.raises(ValueError):
        NormMode.parse("invalid_mode")


def test_norm_mode_normed_name():
    assert NormMode.RAW.normed_name("projected_sum") == "projected_sum"
    assert NormMode.MAX.normed_name("projected_sum") == "Max-Normed projected_sum"
    assert NormMode.TOTAL_PROJECTION.normed_name("dos") == "Total-Projected-Normed dos"


def test_normalize_divides_by_the_requested_denominator_only():
    def unused():
        raise AssertionError("only the requested denominator is computed")

    result = normalize([1.0, 2.0, 4.0], "max", {NormMode.MAX: lambda: 4.0, NormMode.TOTAL: unused})

    assert result.tolist() == [0.25, 0.5, 1.0]


def test_normalize_zero_denominator_gives_zero():
    result = normalize([[1.0, 3.0]], NormMode.TOTAL, {NormMode.TOTAL: lambda: [[0.0, 2.0]]})

    assert result.tolist() == [[0.0, 1.5]]


def test_normalize_raw_returns_values_unchanged():
    assert normalize([1.0, 2.0], None, {}).tolist() == [1.0, 2.0]


def test_normalize_rejects_mode_without_denominator():
    with pytest.raises(ValueError, match="Valid modes: raw, max"):
        normalize([1.0], "integral", {NormMode.MAX: lambda: 1.0})


class _Owner(PointSet):
    def normalize(self, mode, values_array, **kwargs):
        return normalize(values_array, mode, {NormMode.MAX: lambda: np.max(values_array)})

    def normed_units(self, mode, units):
        return units if mode is NormMode.RAW else "$1$"


def test_build_property_normalizes_and_labels():
    owner = _Owner(points=np.zeros((2, 3)))

    prop = build_property(
        owner,
        np.array([[1.0], [4.0]]),
        name="weights",
        label="Weights",
        norm_mode="max",
        units="eV",
    )

    assert prop.name == "Max-Normed weights"
    assert prop.value.tolist() == [[0.25], [1.0]]
    assert prop.units == "$1$"
    assert prop.metadata["footnote"] == "Normalized by the maximum"
    assert prop.metadata["norm_mode"] is NormMode.MAX
    assert prop.point_set is owner


def test_build_property_rejects_disallowed_mode():
    owner = _Owner(points=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="Valid modes: raw"):
        build_property(
            owner,
            np.ones((1, 1)),
            name="weights",
            label="Weights",
            norm_mode="max",
            allowed_norm_modes={NormMode.RAW},
        )


@pytest.mark.guards_existing_behaviour(
    reason="the labels are unchanged; the source only gains the atom_groups the resolver reads"
)
def test_build_property_selection_labels():
    structure = Structure(
        atoms=["Sr", "V"], fractional_coordinates=[[0, 0, 0], [0.5, 0.5, 0.5]], lattice=np.eye(3)
    )
    source = SimpleNamespace(
        structure=structure,
        spin_projection_names=["Spin-up", "Spin-down"],
        orbital_names=["s", "p"],
        is_non_collinear=False,
        atom_groups=None,
    )
    selection = selection_resolver(source).resolve(atoms=[1], orbitals=[0], spins=[1])
    owner = _Owner(points=np.zeros((1, 3)))

    prop = build_property(
        owner,
        np.ones((1, 1)),
        name="weights",
        label="Weights",
        norm_mode="max",
        selection=selection,
        include_normal_label=True,
    )

    assert prop.metadata["atoms"] == [1]
    assert prop.metadata["orbitals"] == [0]
    assert prop.metadata["spins"] == [1]
    assert prop.metadata["label_plain"] == ["V_{1}-(s)[Spin-down] [fraction]"]
    assert prop.metadata["label"] == ["$\\mathrm{V}_{1}-(s)[\\downarrow] [\\mathrm{fraction}]$"]
