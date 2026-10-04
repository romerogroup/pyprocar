import re
from typing import cast

import numpy as np
import pytest

from pyprocar.core import DensityOfStates, ElectronicBandStructure, Structure, kpoints
from pyprocar.core.ebs import PROJECTED_DTYPE

N_K, N_BANDS, N_ATOMS, N_ORBITALS = 2, 3, 5, 9
BAND_LAYOUT = "(n_kpoints, n_bands, n_spins, n_atoms, n_orbitals)"
DOS_LAYOUT = "(n_energies, n_spins, n_atoms, n_orbitals)"


def five_atoms() -> Structure:
    return Structure(
        atoms=["Sr", "V", "O", "O", "O"],
        fractional_coordinates=np.zeros((N_ATOMS, 3)),
        lattice=np.eye(3) * 3.9,
    )


def band_structure(
    projected_shape: tuple[int, ...], n_band_spins: int = 1, structure: Structure | None = None
) -> ElectronicBandStructure:
    return ElectronicBandStructure(
        kpoints=cast(kpoints.KPOINTS_DTYPE, np.zeros((N_K, 3))),
        bands=np.zeros((N_K, N_BANDS, n_band_spins)),
        projected=cast(PROJECTED_DTYPE, np.zeros(projected_shape)),
        orbital_names=[f"o{i}" for i in range(projected_shape[-1])],
        structure=structure,
    )


@pytest.mark.parametrize(
    ("projected_shape", "n_band_spins", "with_structure", "expected"),
    [
        pytest.param(
            (N_K, N_BANDS, N_ATOMS, 1, N_ORBITALS), 1, False, "(2, 3, 1|4, *, *)", id="spin-atom-swap"
        ),
        pytest.param(
            (N_BANDS, N_K, 1, N_ATOMS, N_ORBITALS), 1, False, "(2, 3, 1|4, *, *)", id="k-band-swap"
        ),
        pytest.param(
            (N_K, N_BANDS, 1, 4, N_ORBITALS), 1, True, "(2, 3, 1|4, 5, *)", id="atoms-not-structure"
        ),
        pytest.param(
            (N_K, N_BANDS, 4, N_ATOMS, N_ORBITALS), 2, False, "(2, 3, 2, *, *)", id="collinear-4"
        ),
        pytest.param(
            (N_K, N_BANDS, 1, N_ATOMS, N_ORBITALS), 4, False, "(2, 3, 4, *, *)", id="ncl-bands-1"
        ),
    ],
)
def test_band_structure_names_the_expected_layout_for_misordered_projections(
    projected_shape, n_band_spins, with_structure, expected
):
    structure = five_atoms() if with_structure else None

    with pytest.raises(ValueError) as error:
        band_structure(projected_shape, n_band_spins, structure)

    assert BAND_LAYOUT in str(error.value)
    assert expected in str(error.value)


def test_band_structure_rejects_a_spin_count_that_is_not_1_2_or_4():
    with pytest.raises(ValueError, match=re.escape("1, 2 or 4 spin channels, not 3")):
        band_structure((N_K, N_BANDS, 3, N_ATOMS, N_ORBITALS), n_band_spins=3)


@pytest.mark.parametrize(
    ("n_band_spins", "n_spins"), [(1, 1), (2, 2), (4, 4), (1, 4)], ids=["1", "2", "4", "1-4"]
)
def test_band_structure_accepts_each_spin_layout_with_a_matching_structure(
    n_band_spins, n_spins
):
    ebs = band_structure((N_K, N_BANDS, n_spins, N_ATOMS, N_ORBITALS), n_band_spins, five_atoms())

    assert ebs.projected is not None
    assert ebs.projected.to_array().shape == (N_K, N_BANDS, n_spins, N_ATOMS, N_ORBITALS)


@pytest.mark.parametrize(
    ("total_shape", "projected_shape", "with_structure", "expected"),
    [
        pytest.param((7, 1), (7, N_ATOMS, 1, N_ORBITALS), False, "(7, 1|4, *, *)", id="spin-atom-swap"),
        pytest.param((7, 2), (7, 1, N_ATOMS, N_ORBITALS), False, "(7, 2, *, *)", id="spins-not-total"),
        pytest.param((7, 1), (6, 1, N_ATOMS, N_ORBITALS), False, "(7, 1|4, *, *)", id="energies"),
        pytest.param((7, 4), (7, 4, 2, N_ORBITALS), True, "(7, 4, 5, *)", id="atoms-not-structure"),
    ],
)
def test_dos_names_the_expected_layout_for_misordered_projections(
    total_shape, projected_shape, with_structure, expected
):
    with pytest.raises(ValueError) as error:
        DensityOfStates(
            energies=np.linspace(-1.0, 1.0, total_shape[0]),
            total=np.ones(total_shape),
            projected=np.ones(projected_shape),
            orbital_names=[f"o{i}" for i in range(projected_shape[-1])],
            structure=five_atoms() if with_structure else None,
        )

    assert DOS_LAYOUT in str(error.value)
    assert expected in str(error.value)
