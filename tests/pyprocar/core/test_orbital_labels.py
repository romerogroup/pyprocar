"""Selection labels name orbitals the way the parser named them.

A shell letter (s, p, d, f) stands for a selection only when the parser's names
say those columns are that whole shell. The names below are the ones each code
prints: a filtered PROCAR header, projwfc.x's Lowdin labels, Elk's ELMIREP.OUT
and projwfc.x's spin-orbit pdos file names.
"""

from typing import cast

import numpy as np
import pytest

from pyprocar.core import ElectronicBandStructure, Structure, kpoints
from pyprocar.core.ebs import PROJECTED_DTYPE

QE_NAMES = ["s", "pz", "px", "py", "dz2", "dxz", "dyz", "dx2-y2", "dxy"]
ELK_NAMES = ["Y00", "Y1-1", "Y10", "Y11", "Y2-2", "Y2-1", "Y20", "Y21", "Y22"]
QE_SPIN_ORBIT_NAMES = [
    "l0_j0.5_m-0.5",
    "l0_j0.5_m0.5",
    "l1_j0.5_m-0.5",
    "l1_j0.5_m0.5",
    "l1_j1.5_m-1.5",
]


def orbital_label(orbital_names: list[str], orbitals: list[int]) -> str:
    ebs = ElectronicBandStructure(
        kpoints=cast(kpoints.KPOINTS_DTYPE, np.zeros((1, 3))),
        bands=np.zeros((1, 1, 1)),
        projected=cast(PROJECTED_DTYPE, np.ones((1, 1, 1, 1, len(orbital_names)))),
        orbital_names=orbital_names,
        structure=Structure(
            atoms=["V"], fractional_coordinates=np.zeros((1, 3)), lattice=np.eye(3)
        ),
    )
    return ebs.compute_projected_sum(orbitals=orbitals).metadata["orbital_label"]


@pytest.mark.parametrize(
    ("orbital_names", "orbitals", "expected"),
    [
        pytest.param(["o0", "o1"], [0], "o0", id="filtered-procar"),
        pytest.param(QE_SPIN_ORBIT_NAMES, [0, 1], "l0_j0.5_m-0.5,l0_j0.5_m0.5", id="qe-j-basis"),
    ],
)
def test_label_uses_the_parser_names_when_they_are_not_a_whole_shell(
    orbital_names, orbitals, expected
) -> None:
    assert orbital_label(orbital_names, orbitals) == expected


@pytest.mark.guards_existing_behaviour(
    reason="shell letters already label whole shells; the fix must keep them"
)
@pytest.mark.parametrize(
    ("orbital_names", "orbitals", "expected"),
    [
        pytest.param(QE_NAMES, [1, 2, 3], "p", id="qe-p"),
        pytest.param(QE_NAMES, [0, 4, 5, 6, 7, 8], "s,d", id="qe-s-d"),
        pytest.param(ELK_NAMES, [1, 2, 3], "p", id="elk-p"),
        pytest.param(QE_NAMES, [1, 2], "pz,px", id="qe-part-of-p"),
        pytest.param(["s", "p", "d"], [1], "p", id="filtered-procar-named-shells"),
    ],
)
def test_label_uses_the_shell_letter_for_a_whole_shell(orbital_names, orbitals, expected) -> None:
    assert orbital_label(orbital_names, orbitals) == expected


@pytest.mark.guards_existing_behaviour(
    reason="the fixed table never called py and pz a p shell; orbital_shells must not either"
)
def test_two_of_three_p_orbitals_are_not_labelled_p() -> None:
    assert orbital_label(["s", "py", "pz"], [1, 2]) == "py,pz"
