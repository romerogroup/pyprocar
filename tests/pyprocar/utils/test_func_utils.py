import pytest

from pyprocar.utils.func_utils import expand_grouped_params_to_dicts


def test_expand_grouped_params_to_dicts_no_groups():
    params = {"atoms": [0, 1], "orbitals": [0, 2], "spins": None}
    result = expand_grouped_params_to_dicts(params)
    assert result == [params]


def test_expand_grouped_params_to_dicts_single_group():
    params = {"atoms": [[0, 2], [1]], "orbitals": [0, 1, 2]}
    result = expand_grouped_params_to_dicts(params)
    assert result == [
        {"atoms": [0, 2], "orbitals": [0, 1, 2]},
        {"atoms": [1], "orbitals": [0, 1, 2]},
    ]


def test_expand_grouped_params_to_dicts_aligned_groups():
    params = {"atoms": [[0], [1]], "orbitals": [[2, 3], [4]]}
    result = expand_grouped_params_to_dicts(params)
    assert result == [{"atoms": [0], "orbitals": [2, 3]}, {"atoms": [1], "orbitals": [4]}]


def test_expand_grouped_params_to_dicts_mismatched_lengths():
    params = {"atoms": [[0], [1]], "orbitals": [[2, 3]]}
    with pytest.raises(ValueError, match="same length"):
        expand_grouped_params_to_dicts(params)
