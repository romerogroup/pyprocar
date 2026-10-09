import subprocess
import sys

import numpy as np
import pytest

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from tests.utils import ROOT_DIR

# Graphene's reciprocal lattice (1/Angstrom, rows): a1* and a2* are 60 degrees apart.
HEXAGONAL = np.array(
    [[-0.405231703, -0.233960597, 0.0], [-0.405231703, 0.233960597, 0.0], [0.0, 0.0, -0.115140544]]
)
SQUARE = np.diag([0.25, 0.25, 0.1])
# Monolayer BiSb's reciprocal lattice: c* is off the z axis by parts in 10^7.
NEAR_SQUARE = np.array(
    [
        [0.235005607, 0.13568153, 1.89e-07],
        [1.54e-07, 0.271361575, -1.35e-07],
        [7e-09, -5e-09, 0.061],
    ]
)
# c* tilted towards a* and b*, so planes at different kz sit at different (kx, ky). c* is tall
# enough for the 24 x 24 x n_kz k-grids below to stay reduced, so they are drawn as padded.
SKEWED = np.array([[0.25, 0.0, 0.0], [0.1, 0.22, 0.0], [0.03, -0.04, 0.2]])
# SKEWED with a* tilted out of the kz = 0 plane, so a Cartesian kz plane crosses the kz layers.
TILT = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
TILTED = SKEWED + TILT
# SKEWED before #302. Its k-step c*/n_kz - a*/24 + b*/24 is shorter than c*/n_kz, so these
# k-grids are not reduced and are drawn in another basis.
SKEWED_SHORT_C = np.array([[0.25, 0.0, 0.0], [0.1, 0.22, 0.0], [0.03, -0.04, 0.12]])
# The pad covers the first zone of each lattice here (a [0, 1) grid reaches fractional -2/3 on
# the hexagonal ones), so the drawn mesh is the padded grid and the patch below.
N_K, PADDING = 24, 20
PATCH = (N_K - 1 + 2 * PADDING) / N_K


def tight_binding_graphene(frac: np.ndarray) -> np.ndarray:
    return np.abs(1 + np.exp(2j * np.pi * frac[:, 0]) + np.exp(2j * np.pi * frac[:, 1]))


def square_band(frac: np.ndarray) -> np.ndarray:
    return np.cos(2 * np.pi * frac[:, 0]) + 2 * np.cos(2 * np.pi * frac[:, 1])


def layered_band(frac: np.ndarray) -> np.ndarray:
    return square_band(frac) + 0.5 * np.sin(2 * np.pi * frac[:, 2])


def two_band_mesh(
    lattice: np.ndarray, band, n_kz: int = 1, offset: float = 0.0
) -> ElectronicBandStructureMesh:
    axis = np.arange(N_K) / N_K
    frac = np.stack(
        np.meshgrid(axis, axis, np.arange(n_kz) / n_kz, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    energies = band(frac)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, n_kz), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=frac,
        bands=np.stack([offset - energies, offset + energies], axis=1)[:, :, None],
        projected=np.ones((len(frac), 2, 1, 1, 1)),
        fermi=0.0,
        reciprocal_lattice=lattice,
        orbital_names=["s"],
    )


def bs2d_arrays(
    lattice: np.ndarray,
    band,
    grid: tuple[int, int],
    as_cartesian: bool = False,
    n_kz: int = 1,
    origin: tuple[float, float, float] = (0, 0, 0),
    padding: int = PADDING,
):
    """Points, sheet index, band values and the first sheet's quads as plain arrays, so no VTK
    object outlives the call."""
    bs = BandStructure2D.from_ebs(
        two_band_mesh(lattice, band, n_kz),
        origin=origin,
        grid_interpolation=grid,
        padding=padding,
        as_cartesian=as_cartesian,
    )
    sheet = bs.band_surfaces[(0, 0)]
    return (
        np.array(bs.points),
        np.array(bs.point_data["spin_band_index"]),
        np.array(bs.get_property("bands").value),
        np.array(sheet.points)[np.array(sheet.faces).reshape(-1, 5)[:, 1:]],
    )


def quad_area(quads: np.ndarray) -> float:
    """Total area of the quads' projections onto the k-plane, in (1/Angstrom)^2 without 2 pi."""
    x, y = quads[..., 0] / (2 * np.pi), quads[..., 1] / (2 * np.pi)
    shoelace = x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y
    return float(np.abs(shoelace.sum(axis=1)).sum() / 2)


@pytest.mark.parametrize("as_cartesian", [True, False], ids=["cartesian", "fractional"])
def test_hexagonal_bs2d_surface_covers_the_k_patch_with_the_analytic_bands(as_cartesian):
    grid = N_K + 2 * PADDING
    points, sheet, _, quads = bs2d_arrays(
        HEXAGONAL, tight_binding_graphene, (grid, grid), as_cartesian
    )

    frac = np.column_stack([points[:, :2] / (2 * np.pi), np.zeros(len(points))]) @ np.linalg.inv(
        HEXAGONAL
    )
    sign = np.where(sheet == 1, 1.0, -1.0)
    assert np.isfinite(points).all()
    np.testing.assert_allclose(points[:, 2], sign * tight_binding_graphene(frac), atol=1e-9)
    hexagon_cell = 0.405231703 * 0.233960597 * 2
    assert quad_area(quads) == pytest.approx(PATCH**2 * hexagon_cell, rel=1e-9)


@pytest.mark.parametrize("as_cartesian", [True, False], ids=["cartesian", "fractional"])
def test_bs2d_on_a_lattice_slightly_off_square_leaves_no_grid_point_empty(as_cartesian):
    # Padding 10 puts a grid corner where rounding once left it just outside the data.
    points, _, _, _ = bs2d_arrays(NEAR_SQUARE, square_band, (30, 30), as_cartesian, padding=10)

    assert np.isfinite(points).all()


def test_bs2d_plane_off_the_origin_on_a_skewed_lattice_has_the_bands_of_that_plane():
    grid = N_K + 2 * PADDING
    points, sheet, _, quads = bs2d_arrays(
        SKEWED, layered_band, (grid, grid), n_kz=4, origin=(0, 0, 0.25)
    )

    k_in_plane = points[:, :2] / (2 * np.pi) - 0.25 * SKEWED[2, :2]
    frac = np.column_stack(
        [np.linalg.solve(SKEWED[:2, :2].T, k_in_plane.T).T, np.full(len(points), 0.25)]
    )
    sign = np.where(sheet == 1, 1.0, -1.0)
    assert np.isfinite(points).all()
    np.testing.assert_allclose(points[:, 2], sign * layered_band(frac), atol=1e-9)
    assert quad_area(quads) == pytest.approx(PATCH**2 * 0.25 * 0.22, rel=1e-9)


def test_cartesian_bs2d_plane_across_the_kz_layers_covers_its_patch():
    # kz = 0.036 keeps the plane inside the padded mesh, which spans fractional kz -20/8 to 27/8.
    points, sheet, _, quads = bs2d_arrays(
        TILTED, layered_band, (30, 30), True, n_kz=8, origin=(0, 0, 0.036)
    )

    k = np.column_stack([points[:, :2] / (2 * np.pi), np.full(len(points), 0.036)])
    frac = np.linalg.solve(TILTED.T, k.T).T
    sign = np.where(sheet == 1, 1.0, -1.0)
    assert np.isfinite(points).all()
    # Linear interpolation on this 24x24x8 mesh is within |f''| h^2 / 8 = 0.065 eV of the band.
    np.testing.assert_allclose(points[:, 2], sign * layered_band(frac), atol=0.07)
    # The in-plane fractional coordinates span the padded mesh; kz follows from them.
    edges = TILTED[:2, :2] - np.outer(TILTED[:2, 2] / TILTED[2, 2], TILTED[2, :2])
    assert quad_area(quads) == pytest.approx(PATCH**2 * abs(np.linalg.det(edges)), rel=1e-9)


def test_bs2d_band_values_belong_to_their_own_surface_points():
    points, _, band_values, _ = bs2d_arrays(SQUARE, square_band, (30, 30))

    np.testing.assert_array_equal(band_values, points[:, 2])


def test_bs2d_on_a_non_square_grid_builds_every_point():
    # Before #285 a non-square grid gave VTK a grid of the wrong size and crashed the
    # interpreter, so the build runs in its own process.
    script = (
        "import numpy as np\n"
        "import tests.pyprocar.core.test_bandstructure2d_grid as t\n"
        "points, _, values, quads = t.bs2d_arrays(t.SQUARE, t.square_band, (30, 20))\n"
        "print(len(points), int(np.sum(values != points[:, 2])), len(quads), t.quad_area(quads))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT_DIR, capture_output=True, text=True, timeout=300
    )

    assert run.returncode == 0, run.stderr[-2000:]
    n_points, n_off, n_quads, area = run.stdout.split()
    assert [n_points, n_off, n_quads] == [str(2 * 30 * 20), "0", str(29 * 19)]
    assert float(area) == pytest.approx(PATCH**2 * 0.25 * 0.25, rel=1e-9)


SHORT_PADDING = 3
"""The pad before #302. On the [0, 1) grids above it misses the first zone."""


@pytest.mark.parametrize(
    ("lattice", "band", "first", "as_cartesian"),
    [
        (HEXAGONAL, tight_binding_graphene, -17, True),
        (HEXAGONAL, tight_binding_graphene, -17, False),
        (SQUARE, square_band, -13, False),
    ],
    ids=["hexagonal-cartesian", "hexagonal-fractional", "square"],
)
def test_a_pad_short_of_the_zone_draws_the_zone_and_the_pad_with_the_analytic_bands(
    lattice, band, first, as_cartesian
):
    """The drawn box runs from the zone's box, one point past the zone corner at fractional
    -2/3 (16 points) on graphene and the edge at -1/2 (12 points) on the square, to the pad's
    end at 23 + 3. A (u, v) grid of one sample per k-point then lands on the k-points."""
    last = N_K - 1 + SHORT_PADDING
    grid = last - first + 1
    points, sheet, _, quads = bs2d_arrays(
        lattice, band, (grid, grid), as_cartesian, padding=SHORT_PADDING
    )

    frac = np.column_stack([points[:, :2] / (2 * np.pi), np.zeros(len(points))]) @ np.linalg.inv(
        lattice
    )
    sign = np.where(sheet == 1, 1.0, -1.0)
    assert np.isfinite(points).all()
    np.testing.assert_allclose(points[:, 2], sign * band(frac), atol=1e-9)
    span = (last - first) / N_K
    cell = abs(np.linalg.det(lattice[:2, :2]))
    assert quad_area(quads) == pytest.approx(span**2 * cell, rel=1e-9)


def linear_cell_band(frac: np.ndarray) -> np.ndarray:
    """Linear inside the cell |f_i| < 1/2 and periodic."""
    saw = frac - np.round(frac)
    return saw[:, 0] + 2 * saw[:, 1] + 3 * saw[:, 2]


@pytest.mark.parametrize(
    ("lattice", "n_kz", "origin", "kz", "as_cartesian"),
    [
        (SKEWED_SHORT_C, 4, (0, 0, 0.25), 0.25 * 0.12, False),
        (SKEWED_SHORT_C + TILT, 8, (0, 0, 0.036), 0.036, True),
    ],
    ids=["skewed-plane-off-the-origin", "tilted-plane-across-the-kz-layers"],
)
def test_a_k_grid_that_is_not_reduced_draws_the_band_wherever_its_cells_keep_it_linear(
    lattice, n_kz, origin, kz, as_cartesian
):
    """A drawn cell spans at most 1/12 of a* and b* and 1/n_kz of c*, so a sample within 0.3 of
    the cell centre interpolates only k-points where the band is linear, and must be exact.
    The sheet reaches -0.3 along a* and b*, which today's pad from -3/24 does not."""
    points, sheet, _, _ = bs2d_arrays(
        lattice, linear_cell_band, (30, 30), as_cartesian, n_kz, origin, padding=SHORT_PADDING
    )

    k = np.column_stack([points[:, :2] / (2 * np.pi), np.full(len(points), kz)])
    frac = np.linalg.solve(lattice.T, k.T).T
    inner = (np.abs(frac) < 0.3).all(axis=1)
    sign = np.where(sheet == 1, 1.0, -1.0)
    assert np.isfinite(points).all()
    assert (frac[:, :2].min(axis=0) <= -0.3).all()
    assert (frac[:, :2].max(axis=0) >= 0.3).all()
    np.testing.assert_allclose(
        points[inner, 2], sign[inner] * linear_cell_band(frac[inner]), atol=1e-9
    )
