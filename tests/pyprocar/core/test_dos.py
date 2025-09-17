import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from pyprocar.core.dos import DensityOfStates
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)
user_logger = logging.getLogger("user")


@pytest.fixture
def non_spin_polarized_dir():
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return DATA_DIR / "examples" / "dos" / "non-spin-polarized"


def _make_random_dos(
    n_energies: int = 50,
    n_spins: int = 2,
    n_atoms: int = 3,
    n_orbitals: int = 3,
):
    rng = np.random.default_rng(42)
    energies = np.linspace(-5.0, 5.0, n_energies)
    total = rng.random((n_energies,n_spins))

    projected = rng.random((n_energies, n_spins, n_atoms, n_orbitals))
    dos = DensityOfStates(
        energies=energies,
        total=total,
        projected=projected,
        orbital_names=["s", "p_y", "p_z", "p_x", "d_xy", "d_yz", "d_z2", "d_xz", "d_x2-y2"],
    )
    return dos

@pytest.fixture
def dos_non_spin_polarized():
    dos = _make_random_dos(
        n_energies=16, n_spins=1, n_atoms=3, n_orbitals=9
    )
    return dos


@pytest.fixture
def dos_spin_polarized():
    dos = _make_random_dos(
        n_energies=16, n_spins=2, n_atoms=3, n_orbitals=9
    )
    return dos

@pytest.fixture
def dos_non_collinear():
    dos = _make_random_dos(
        n_energies=16, n_spins=4, n_atoms=3, n_orbitals=9
    )
    return dos




def test_dos_from_code(non_spin_polarized_dir):
    dos = DensityOfStates.from_code(code="vasp", dirpath=non_spin_polarized_dir)
    assert dos is not None, "DOS is None"
    assert dos.projected is not None, "Projected DOS is None"
    assert dos.energies is not None, "Energies are None"
    assert dos.total is not None, "Total DOS is None"

# def test_dos_sum_random_initialization():
#     dos, projected = _make_random_dos(
#         n_energies=20, n_spins=2, n_atoms=3, n_orbitals=4
#     )

#     atoms_sel = [0, 2]
#     orbitals_sel = [1, 3]
#     spins_sel = [0]

#     result = dos.dos_sum(atoms=atoms_sel, orbitals=orbitals_sel, spins=spins_sel)

#     # Manual expected following dos_sum implementation
#     expected = np.sum(projected[..., orbitals_sel], axis=-1)
#     expected = np.sum(expected[..., atoms_sel], axis=-1)
#     # Spin-polarized zeroing behavior when a single spin is requested
#     expected[..., 1] = 0
#     assert np.allclose(result, expected)


# def test_compute_projected_sum_random_initialization():
#     # Choose sizes to avoid index errors due to property shapes in dos_sum defaults
#     dos, projected = _make_random_dos(
#         n_energies=16, n_spins=2, n_atoms=3, n_orbitals=3
#     )

#     atoms_sel = [1, 2]
#     orbitals_sel = [0, 2]
#     spin_projections = [0]

#     total_proj, proj = dos.compute_projected_sum(
#         atoms=atoms_sel, orbitals=orbitals_sel, spin_projections=spin_projections
#     )

#     # Expected for proj (explicit atoms/orbitals/spins)
#     expected_proj = np.sum(projected[..., orbitals_sel], axis=-1)
#     expected_proj = np.sum(expected_proj[..., atoms_sel], axis=-1)
#     expected_proj[..., 1] = 0

#     # Expected for total_proj (dos_sum called with no args inside compute_projected_sum)
#     # Defaults in dos_sum: orbitals = range(self.n_orbitals) -> shape[3] == n_atoms
#     # atoms = range(self.n_atoms) -> shape[2] == n_spins
#     orbs_default = list(range(projected.shape[3]))  # equals n_atoms by class property
#     atoms_default = list(range(projected.shape[2]))  # equals n_spins by class property
#     expected_total = np.sum(projected[..., orbs_default], axis=-1)
#     expected_total = np.sum(expected_total[..., atoms_default], axis=-1)

#     assert proj.shape == expected_proj.shape == (
#         projected.shape[0],
#         projected.shape[1],
#     )
#     assert total_proj.shape == expected_total.shape == (
#         projected.shape[0],
#         projected.shape[1],
#     )
#     assert np.allclose(proj, expected_proj)
#     assert np.allclose(total_proj, expected_total)


def test_non_spin_polarized_dos(dos_non_spin_polarized):
    dos = dos_non_spin_polarized
    assert not dos.is_spin_polarized, "DOS isspin polarized"
    assert dos.is_non_collinear == False, "DOS is non-collinear"
    assert dos.n_spins == 1, "DOS has the wrong number of spins"
    assert dos.spin_projection_names == ["Spin-up"], "DOS has the wrong spin projection names"
    assert np.allclose(dos.spin_channels, [0]), "DOS has the wrong spin channels"

def test_non_spin_polarized_dos_sum(dos_non_spin_polarized):
    dos = dos_non_spin_polarized
    
    atoms_sel = [0, 2]
    orbitals_sel = [1, 3]

    result = dos.dos_sum(atoms=atoms_sel, orbitals=orbitals_sel)
    expected = np.sum(dos.projected[..., orbitals_sel], axis=-1)
    expected = np.sum(expected[..., atoms_sel], axis=-1)
 
    assert result.shape == (16, 1), "Non-spin polarized DOS sum has the wrong shape"
    assert np.allclose(result, expected), "DOS sum is not correct"


def test_spin_polarized_dos(dos_spin_polarized):
    dos = dos_spin_polarized
    assert dos.is_spin_polarized, "DOS is not spin polarized"
    assert dos.is_non_collinear == False, "DOS is non-collinear"
    assert dos.n_spins == 2, "DOS has the wrong number of spins"
    assert dos.spin_projection_names == ["Spin-up", "Spin-down"], "DOS has the wrong spin projection names"
    assert np.allclose(dos.spin_channels, [0, 1]), "DOS has the wrong spin channels"


def test_spin_polarized_dos_sum(dos_spin_polarized):
    dos = dos_spin_polarized
    
    atoms_sel = [0, 2]
    orbitals_sel = [1, 3]
    spins_sel = [0,1]

    result = dos.dos_sum(atoms=atoms_sel, orbitals=orbitals_sel, spins=spins_sel)
    expected = np.sum(dos.projected[..., orbitals_sel], axis=-1)
    expected = np.sum(expected[..., atoms_sel], axis=-1)
 
    assert result.shape == (16, 2), "Spin polarized DOS sum has the wrong shape"
    assert np.allclose(result, expected), "DOS sum is not correct"
    
    atoms_sel = [0, 2]
    orbitals_sel = [1, 3]
    spins_sel = [1]

    result = dos.dos_sum(atoms=atoms_sel, orbitals=orbitals_sel, spins=spins_sel)
    expected = np.sum(dos.projected[..., orbitals_sel], axis=-1)
    expected = np.sum(expected[..., atoms_sel], axis=-1)
    expected[..., 0] = 0
    

    assert result.shape == (16, 2), "Spin polarized DOS sum has the wrong shape"
    assert np.allclose(result, expected), "DOS sum is not correct"
    
    

def test_non_collinear_dos(dos_non_collinear):
    dos = dos_non_collinear
    assert dos.is_non_collinear, "DOS is not non-collinear"
    assert dos.is_spin_polarized == False, "DOS is spin polarized"
    assert dos.n_spins == 4, "DOS has the wrong number of spins"
    assert dos.spin_projection_names == ["total", "x", "y", "z"], "DOS has the wrong spin projection names"
    assert np.allclose(dos.spin_channels, [0, 1, 2, 3]), "DOS has the wrong spin channels"

def test_non_collinear_dos_sum(dos_non_collinear):
    dos = dos_non_collinear
    
    atoms_sel = [0, 2]
    orbitals_sel = [1, 3]
    spins_sel = [1, 2, 3]

    result = dos.dos_sum(atoms=atoms_sel, orbitals=orbitals_sel, spins=spins_sel)
    expected = np.sum(dos.projected[..., orbitals_sel], axis=-1)
    expected = np.sum(expected[..., atoms_sel], axis=-1)
    expected = np.sum(expected[..., spins_sel], axis=-1)
    expected = expected[..., np.newaxis]
    print(result[:5])
    print(expected[:5])
    
    assert result.shape == (16, 1), "Non-collinear DOS sum has the wrong shape"
    assert np.allclose(result, expected), "DOS sum is not correct"
    
    result = dos.dos_sum(atoms=atoms_sel, orbitals=orbitals_sel, spins=spins_sel, sum_noncolinear=False)
    expected = np.sum(dos.projected[..., orbitals_sel], axis=-1)
    expected = np.sum(expected[..., atoms_sel], axis=-1)
    expected = expected[..., spins_sel]
    
    assert result.shape == (16, 3), "Non-collinear DOS sum has the wrong shape"
    assert np.allclose(result, expected), "DOS sum is not correct"


def test_normalize_max_dos(dos_non_spin_polarized):
    dos = dos_non_spin_polarized
    dos_normalized = dos.normalize_dos(mode="max")
    
    max_total = np.max(dos_normalized.total)
    assert max_total == 1, "DOS is not normalized by max"

def test_normalize_integral_dos(dos_non_spin_polarized):
    dos = dos_non_spin_polarized
    dos_normalized = dos.normalize_dos(mode="integral")
    
    integral = np.trapezoid(dos_normalized.total[:,0], x=dos_normalized.energies)
    assert integral == 1, "DOS is not normalized by integral"

def test_interpolate_dos(dos_non_spin_polarized):
    dos = dos_non_spin_polarized
    dos_interpolated = dos.interpolate(factor=2)
    assert dos_interpolated.energies.shape[0] == 2 * dos.energies.shape[0], "DOS is not interpolated"
    assert dos_interpolated.projected.shape[0] == 2 * dos.projected.shape[0], "DOS is not interpolated"
    assert dos_interpolated.total.shape[0] == 2 * dos.total.shape[0], "DOS is not interpolated"
    

# def test_spin_projection_names():
#     dos, projected = _make_random_dos(
#         n_energies=16, n_spins=1, n_atoms=3, n_orbitals=3
#     )
#     assert dos.spin_projection_names == ["Spin-up"]
    
#     dos, projected = _make_random_dos(
#         n_energies=16,  n_spins=2, n_atoms=3, n_orbitals=3
#     )
#     assert dos.spin_projection_names == ["Spin-up", "Spin-down"]
    
#     dos, projected = _make_random_dos(
#         n_energies=16,  n_spins=4, n_atoms=3, n_orbitals=3
#     )
    