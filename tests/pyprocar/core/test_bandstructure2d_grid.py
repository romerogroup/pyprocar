import numpy as np
import pytest

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

# Graphene's reciprocal lattice (1/Angstrom, rows): a1* and a2* are 60 degrees apart.
HEXAGONAL = np.array(
    [[-0.405231703, -0.233960597, 0.0], [-0.405231703, 0.233960597, 0.0], [0.0, 0.0, -0.115140544]]
)
SQUARE = np.diag([0.25, 0.25, 0.1])
N_K, PADDING = 24, 3


def tight_binding_graphene(frac: np.ndarray) -> np.ndarray:
    return np.abs(1 + np.exp(2j * np.pi * frac[:, 0]) + np.exp(2j * np.pi * frac[:, 1]))


def square_band(frac: np.ndarray) -> np.ndarray:
    return np.cos(2 * np.pi * frac[:, 0]) + 2 * np.cos(2 * np.pi * frac[:, 1])


def two_band_mesh(lattice: np.ndarray, band) -> ElectronicBandStructureMesh:
    axis = np.arange(N_K) / N_K
    frac = np.stack(np.meshgrid(axis, axis, [0.0], indexing="ij"), axis=-1).reshape(-1, 3)
    energies = band(frac)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, 1), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=frac,
        bands=np.stack([-energies, energies], axis=1)[:, :, None],
        projected=np.ones((len(frac), 2, 1, 1, 1)),
        fermi=0.0,
        reciprocal_lattice=lattice,
        orbital_names=["s"],
    )


def test_hexagonal_bs2d_surface_covers_the_k_patch_with_the_analytic_bands():
    grid = N_K + 2 * PADDING
    bs = BandStructure2D.from_ebs(
        two_band_mesh(HEXAGONAL, tight_binding_graphene),
        grid_interpolation=(grid, grid),
        padding=PADDING,
        as_cartesian=False,
    )

    frac = np.column_stack([bs.points[:, :2] / (2 * np.pi), np.zeros(bs.n_points)]) @ np.linalg.inv(
        HEXAGONAL
    )
    sign = np.where(bs.point_data["spin_band_index"] == 1, 1.0, -1.0)
    assert np.isfinite(bs.points).all()
    np.testing.assert_allclose(bs.points[:, 2], sign * tight_binding_graphene(frac), atol=1e-9)


@pytest.mark.parametrize("grid", [(30, 30), (30, 20)])
def test_bs2d_band_values_belong_to_their_own_surface_points(grid):
    bs = BandStructure2D.from_ebs(
        two_band_mesh(SQUARE, square_band),
        grid_interpolation=grid,
        padding=PADDING,
        as_cartesian=False,
    )

    assert bs.n_points == 2 * grid[0] * grid[1]
    np.testing.assert_array_equal(bs.get_property("bands").value, bs.points[:, 2])
