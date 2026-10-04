import itertools

import numpy as np

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.core.structure import Structure

N = 61
FERMI = 0.1
CUBIC_OPERATIONS = [
    np.diag(signs)[list(order)]
    for order in itertools.permutations(range(3))
    for signs in itertools.product([1, -1], repeat=3)
]


def irreducible_square_mesh() -> ElectronicBandStructureMesh:
    """E = kx^2 + ky^2 on the 0 <= ky <= kx <= 1/2 wedge of a 61x61x1 Gamma grid.

    The k-points are rounded to 8 decimals, as VASP writes them.
    """
    wedge = np.array([(i, j, 0) for i in range(N // 2 + 1) for j in range(i + 1)]) / N
    wedge = wedge.round(8)
    energies = np.sum(wedge[:, :2] ** 2, axis=1)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=(N, N, 1), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0, 0, 0)),
        kpoints=wedge,
        bands=np.stack([energies, np.full(len(wedge), 5.0)], axis=1)[..., np.newaxis],
        projected=np.ones((len(wedge), 2, 1, 1, 1)),
        fermi=FERMI,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
        structure=Structure(
            atoms=["X"],
            fractional_coordinates=[[0.0, 0.0, 0.0]],
            lattice=np.eye(3),
            rotations=CUBIC_OPERATIONS,
        ),
    )


def test_unfolded_kpoints_are_the_exact_grid_points():
    kpoints = irreducible_square_mesh().kpoints

    assert kpoints.shape == (N * N, 3)
    np.testing.assert_allclose(kpoints * N, np.rint(kpoints * N), atol=1e-12)


def test_fermi_contour_of_an_unfolded_mesh_is_centred_on_gamma():
    fs = FermiSurface.from_ebs(irreducible_square_mesh())

    contour = np.asarray(fs.slice(normal=(0, 0, 1), origin=(0, 0, 0)).points)[:, :2]

    np.testing.assert_allclose(contour.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(contour, axis=1), np.sqrt(FERMI), atol=2e-3)
