"""shift_kpoints_to_fbz moves k-points into the Wigner-Seitz first Brillouin zone (#302, item E)."""

import itertools

import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructure, ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo, get_kpoints_from_kgrid

HEXAGONAL_REAL = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.6]])
FCC_REAL = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
RECIPROCAL_LATTICES = {
    "hexagonal": np.linalg.inv(HEXAGONAL_REAL).T,
    "fcc": np.linalg.inv(FCC_REAL).T,
    "cubic, b2'=(3,1,0)": np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
}
# Every lattice vector within 4 cells: a superset of the zone's face vectors for these lattices.
LATTICE_STEPS = np.array([s for s in itertools.product(range(-4, 5), repeat=3) if any(s)])


def plain_ebs(
    kpoints: np.ndarray, reciprocal_lattice: np.ndarray | None
) -> ElectronicBandStructure:
    return ElectronicBandStructure(
        kpoints=kpoints,
        bands=np.zeros((len(kpoints), 1, 1)),
        fermi=0.0,
        reciprocal_lattice=reciprocal_lattice,
    )


@pytest.mark.parametrize("lattice", RECIPROCAL_LATTICES.values(), ids=RECIPROCAL_LATTICES.keys())
def test_shifted_points_lie_in_the_first_zone(lattice: np.ndarray):
    """No lattice vector G brings a shifted k closer to Gamma: |k|^2 <= |k - G|^2."""
    kpoints = np.random.default_rng(0).uniform(-3, 3, (2000, 3))

    shifted = plain_ebs(kpoints, lattice).shift_kpoints_to_fbz(inplace=False).kpoints

    # |k|^2 > |k - G|^2 exactly when 2 k.G > |G|^2.
    cartesian = shifted @ lattice
    translates = LATTICE_STEPS @ lattice
    closer = 2 * cartesian @ translates.T > (translates**2).sum(axis=1) + 1e-9
    outside = closer.any(axis=1)
    assert not outside.any(), f"{outside.mean():.1%} of the shifted points lie outside the zone"
    moved_by = shifted - kpoints
    np.testing.assert_allclose(moved_by, np.rint(moved_by), rtol=0, atol=1e-9)


def test_shift_needs_the_reciprocal_lattice():
    ebs = plain_ebs(np.array([[0.7, 0.0, 0.0]]), None)

    with pytest.raises(ValueError, match="reciprocal lattice"):
        ebs.shift_kpoints_to_fbz()


def test_shift_refuses_a_mesh():
    """Zone-reduced points are no box grid, so a Mesh would misread its own size."""
    kgrid = (6, 6, 1)
    kpoints = get_kpoints_from_kgrid(kgrid, (0.0, 0.0, 0.0), KGRID_MODE.GAMMA)
    ebs = ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=kgrid, kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)),
        kpoints=kpoints,
        bands=np.zeros((len(kpoints), 1, 1)),
        fermi=0.0,
        reciprocal_lattice=RECIPROCAL_LATTICES["hexagonal"],
    )

    with pytest.raises(ValueError, match="Mesh"):
        ebs.shift_kpoints_to_fbz()
