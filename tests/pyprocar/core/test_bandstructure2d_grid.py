import subprocess
import sys

import numpy as np

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from tests.utils import ROOT_DIR

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


def bs2d_arrays(lattice: np.ndarray, band, grid: tuple[int, int]):
    """Points, sheet index and band values as plain arrays, so no VTK object outlives the call."""
    bs = BandStructure2D.from_ebs(
        two_band_mesh(lattice, band), grid_interpolation=grid, padding=PADDING, as_cartesian=False
    )
    return (
        np.array(bs.points),
        np.array(bs.point_data["spin_band_index"]),
        np.array(bs.get_property("bands").value),
    )


def test_hexagonal_bs2d_surface_covers_the_k_patch_with_the_analytic_bands():
    grid = N_K + 2 * PADDING
    points, sheet, _ = bs2d_arrays(HEXAGONAL, tight_binding_graphene, (grid, grid))

    frac = np.column_stack([points[:, :2] / (2 * np.pi), np.zeros(len(points))]) @ np.linalg.inv(
        HEXAGONAL
    )
    sign = np.where(sheet == 1, 1.0, -1.0)
    assert np.isfinite(points).all()
    np.testing.assert_allclose(points[:, 2], sign * tight_binding_graphene(frac), atol=1e-9)


def test_bs2d_band_values_belong_to_their_own_surface_points():
    points, _, band_values = bs2d_arrays(SQUARE, square_band, (30, 30))

    np.testing.assert_array_equal(band_values, points[:, 2])


def test_bs2d_on_a_non_square_grid_builds_every_point():
    # Before #285 a non-square grid gave VTK a grid of the wrong size and crashed the
    # interpreter, so the build runs in its own process.
    script = (
        "import numpy as np\n"
        "import tests.pyprocar.core.test_bandstructure2d_grid as t\n"
        "points, _, values = t.bs2d_arrays(t.SQUARE, t.square_band, (30, 20))\n"
        "print(len(points), int(np.sum(values != points[:, 2])))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT_DIR, capture_output=True, text=True, timeout=300
    )

    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split() == [str(2 * 30 * 20), "0"]
