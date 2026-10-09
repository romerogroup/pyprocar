"""The drawn Fermi surface covers the first zone on any basis and any full grid (#302, A and D).

Each case builds a synthetic Mesh whose band is periodic over the reciprocal lattice. A sheared
case gives the same k-point set in a sheared basis of the lattice, so its drawn area must equal the
area today's code draws on the unsheared basis (the literal). Area tolerances are 2e-8: pyvista
contours in float32, and moving only the box origin moves a clipped area by up to 1.1e-8.
"""

import hashlib
import itertools
import math
import warnings

import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from tests.utils.user_warning import user_warning

AREA_TOL = 2e-8
SPHERE_AREA = 1.2365362569504919
"""Today's area of |k|^2 = 0.1 on the cubic 16^3 [0,1) grid."""
ANALYTIC_SPHERE = 4 * math.pi * 0.1
CUBIC = np.eye(3)
U1 = np.array([[0, -1, 4], [2, -2, -5], [1, -2, 2]], dtype=float)
HEX_REAL = np.array([[1.0, 0, 0], [-0.5, math.sqrt(3) / 2, 0], [0, 0, 1.6]])
HEX = np.linalg.inv(HEX_REAL).T
FCC = np.linalg.inv(np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])).T
BCC = np.linalg.inv(0.5 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])).T


def sheared(k: int) -> np.ndarray:
    """The cubic reciprocal lattice in the basis b1, b2 + k b1, b3."""
    return np.array([[1.0, 0, 0], [k, 1, 0], [0, 0, 1]])


def grid_fracs(kgrid, kshift=(0.0, 0.0, 0.0), centred: bool = False) -> np.ndarray:
    axes = []
    for n, s in zip(kgrid, kshift, strict=True):
        axis = (np.arange(n) + s) / n
        if centred:
            axis = np.where(axis > 0.5 + 1e-12, axis - 1.0, axis)
        axes.append(axis)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def mesh(
    kpoints, bands, lattice, kgrid, fermi, kshift=(0.0, 0.0, 0.0)
) -> ElectronicBandStructureMesh:
    """A one-spin Mesh; ``bands`` is (n_k,) or (n_k, n_bands)."""
    bands = np.asarray(bands, dtype=float)
    if bands.ndim == 1:
        bands = bands[:, None]
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=tuple(kgrid), kgrid_mode=KGRID_MODE.GAMMA, kshift=tuple(kshift)),
        kpoints=np.asarray(kpoints, dtype=float),
        bands=bands[..., None],
        projected=np.ones((len(kpoints), bands.shape[1], 1, 1, 1)),
        fermi=fermi,
        reciprocal_lattice=np.asarray(lattice, dtype=float),
        orbital_names=["s"],
    )


def sphere(cart: np.ndarray) -> np.ndarray:
    """|k|^2 around the nearest cubic lattice point."""
    return np.sum(((cart + 0.5) % 1.0 - 0.5) ** 2, axis=1)


def r_pocket(cart: np.ndarray) -> np.ndarray:
    """5 |k - R|^2 around the nearest cubic zone corner R."""
    return 5 * np.sum((cart % 1.0 - 0.5) ** 2, axis=1)


def k_cylinders(cart: np.ndarray) -> np.ndarray:
    """Squared in-plane distance to the nearest hexagonal K or K' column."""
    k_point = (HEX[0] + HEX[1]) / 3 if np.dot(HEX[0], HEX[1]) < 0 else (2 * HEX[0] - HEX[1]) / 3
    images = np.array(list(itertools.product(range(-3, 4), range(-3, 4), [0]))) @ HEX
    best = np.full(len(cart), np.inf)
    for centre in (k_point, -k_point):
        d = cart[:, None, :2] - (centre + images)[None, :, :2]
        best = np.minimum(best, (d**2).sum(axis=2).min(axis=1))
    return best


HEX_RADIUS_SQ = (0.15 * np.linalg.norm(HEX[0])) ** 2


def area(fs: FermiSurface) -> float:
    return float(sum(s.area for s in fs.band_isosurfaces.values()))


def drawn(lattice, kgrid, band, iso, kshift=(0.0, 0.0, 0.0)) -> FermiSurface:
    """The surface of ``band`` sampled on the [0,1) grid ``kgrid`` given in ``lattice``."""
    f = grid_fracs(kgrid, kshift)
    return FermiSurface.from_ebs(mesh(f, band(f @ lattice), lattice, kgrid, iso, kshift))


@pytest.mark.parametrize("k", [2, 3, 4, 10])
def test_sphere_on_a_sheared_basis_has_the_reduced_area(k):
    """A1: today 1.1941, 1.1307, 1.2355 and 3.9434: the pad misses the zone."""
    fs = drawn(sheared(k), (16, 16, 16), sphere, 0.1)

    assert area(fs) == pytest.approx(SPHERE_AREA, abs=AREA_TOL)


@pytest.mark.parametrize("k", [2, 3, 4, 10])
def test_corner_pocket_on_a_sheared_basis_has_the_reduced_area(k):
    """A2: today 0.1846, 0.0804, 0.0855 and an empty-clip ValueError."""
    fs = drawn(sheared(k), (16, 16, 16), r_pocket, 0.1)

    assert area(fs) == pytest.approx(0.23219002292958507, abs=AREA_TOL)


@pytest.mark.parametrize(
    ("k", "expected"),
    [(2, 1.2415255063409503), (3, 1.2424330621365802), (4, 1.2427523743479532)],
)
def test_sphere_on_a_sheared_unequal_grid_has_the_reduced_area(k, expected):
    """A3: (16, 16k, 16) on b2 = (k,1,0) is the cubic (16, 16k, 16) grid; today 1.18, 0.83, 0.64."""
    fs = drawn(sheared(k), (16, 16 * k, 16), sphere, 0.1)

    assert area(fs) == pytest.approx(expected, abs=AREA_TOL)


@pytest.mark.parametrize(
    "lattice", [U1, np.array([[1.0, 0, 0], [3, 1, 0], [-2, 2, 1]])], ids=["U1", "b2-b3"]
)
def test_sphere_on_a_general_shear_has_the_reduced_area(lattice):
    """A4: today 5.3422 and 0.9854."""
    fs = drawn(lattice, (16, 16, 16), sphere, 0.1)

    assert area(fs) == pytest.approx(SPHERE_AREA, abs=AREA_TOL)


@pytest.mark.parametrize(
    ("nk", "expected"),
    [(18, 1.3432284014540712), (30, 1.353383925147638), (60, 1.3583545363774165)],
)
def test_hexagonal_k_columns_on_a_unit_cell_grid_match_the_centred_grid(nk, expected):
    """D1: the hexagonal zone reaches -2/3; today the [0,1) grid draws 1.2677, 0.6767, 0.2849.
    The literals are today's areas on the centred (-1/2, 1/2] grid. Clipped through the zone
    corners, these columns sit on a wider float32 noise floor: today's own centred NK=18 area
    moves by 1.4e-7 as its padding goes from 10 to 16."""
    fs = drawn(HEX, (nk, nk, max(6, nk // 3)), k_cylinders, HEX_RADIUS_SQ)

    assert area(fs) == pytest.approx(expected, abs=1.5e-7)


def wrapped_sq(cart: np.ndarray, centre: np.ndarray) -> np.ndarray:
    frac = (cart - centre) @ np.linalg.inv(FCC)
    return np.sum(((frac - np.rint(frac)) @ FCC) ** 2, axis=1)


@pytest.fixture(scope="module")
def fcc_dense():
    """Centred fcc 60^3: a sphere r=0.3 at Gamma and a pocket r=0.1 at the W corner (1, 1/2, 0)."""
    f = grid_fracs((60, 60, 60), centred=True)
    cart = f @ FCC
    w_corner = np.array([-1.0, -0.5, 0.0])
    bands = np.stack(
        [
            wrapped_sq(cart, np.zeros(3)) - 0.09,
            wrapped_sq(cart, w_corner) - 0.01,
            np.full(len(f), 50.0),
        ],
        axis=1,
    )
    return FermiSurface.from_ebs(mesh(f, bands, FCC, (60, 60, 60), 0.0))


def test_dense_fcc_grid_draws_the_whole_w_corner_pocket(fcc_dense):
    """D2: today's 80^3 box misses the W corners and draws 0.0613. The reference is the same
    pocket at Gamma, which today draws in full: 0.1224768637985816."""
    areas = {key: float(s.area) for key, s in fcc_dense.band_isosurfaces.items()}

    assert areas[(1, 0)] == pytest.approx(0.1224768637985816, abs=1e-4)
    assert areas[(0, 0)] == pytest.approx(1.1274197671360788, abs=AREA_TOL)


def test_shifted_grid_on_a_sheared_basis_has_the_reduced_area():
    """S1: Monkhorst-Pack 16^3 shifted by 1/2 on b2 = (3,1,0) is the cubic grid shifted by
    (0, 1/2, 1/2); today 1.0983. The literal is today's area on that cubic grid."""
    fs = drawn(sheared(3), (16, 16, 16), sphere, 0.1, kshift=(0.5, 0.5, 0.5))

    assert area(fs) == pytest.approx(1.236140263668144, abs=AREA_TOL)


@pytest.mark.parametrize(
    ("kgrid", "low"), [((16, 32, 16), SPHERE_AREA), ((15, 16, 16), 1.2300)], ids=["U1", "U2"]
)
def test_unequal_grid_on_a_sheared_basis_draws_the_whole_sphere(kgrid, low):
    """U1, U2: no reciprocal basis makes these grids diagonal; today 1.0489 and 1.1507.
    The bounds are the 16^3 area (U2: 1.23) and the analytic sphere 4 pi 0.1."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fs = drawn(sheared(3), kgrid, sphere, 0.1)

    assert low < area(fs) < ANALYTIC_SPHERE
    assert [w for w in caught if issubclass(w.category, UserWarning)] == []


@pytest.mark.parametrize(
    ("nk", "expected"),
    [(18, 0.26864568026830254), (30, 0.27067678499971626), (60, 0.27167090726616994)],
)
def test_two_dimensional_grid_on_a_sheared_plane_has_the_unsheared_area(nk, expected):
    """T1: (NK, NK, 1) hexagonal K columns given in b2' = b2 + 2 b1; the literals are today's
    areas on the unsheared centred grid. Energies are copied from the unsheared grid."""
    kgrid = (nk, nk, 1)
    n = np.array(kgrid)
    f = grid_fracs(kgrid)
    values = k_cylinders(f @ HEX).reshape(kgrid)
    lattice = np.array([[1, 0, 0], [2, 1, 0], [0, 0, 1]]) @ HEX
    index = np.rint(f @ lattice @ np.linalg.inv(HEX) * n).astype(int) % n
    energies = values[index[:, 0], index[:, 1], index[:, 2]]

    fs = FermiSurface.from_ebs(mesh(f, energies, lattice, kgrid, HEX_RADIUS_SQ))

    assert area(fs) == pytest.approx(expected, abs=AREA_TOL)


def interior_gradients(ebs: ElectronicBandStructureMesh) -> dict[tuple[int, ...], np.ndarray]:
    """Band 0's gradient at each 1/16 cubic grid point whose six cubic neighbours lie inside the
    cube |k|_inf <= 1/2, where E = |k|^2 holds without a wrap, keyed by 16 k."""
    cart = np.asarray(ebs.kpoints_cartesian)
    gradient = np.asarray(ebs.get_property(("bands", "gradients", 1)))[:, 0, 0, :]
    inside = (np.abs(cart).max(axis=1) <= 0.5 - 1 / 16 + 1e-9) & (
        np.linalg.norm(cart, axis=1) > 1e-9
    )
    keys = np.rint(cart[inside] * 16).astype(int)
    return {tuple(int(x) for x in key): g for key, g in zip(keys, gradient[inside], strict=True)}


@pytest.mark.parametrize("lattice", [U1, sheared(3)], ids=["U1", "b2=(3,1,0)"])
def test_band_gradient_on_the_drawn_mesh_equals_the_reduced_build(lattice):
    """V1: the drawn mesh holds every interior zone point with the reduced build's gradient."""
    reference = interior_gradients(drawn(CUBIC, (16, 16, 16), sphere, 0.1).ebs)
    found = interior_gradients(drawn(lattice, (16, 16, 16), sphere, 0.1).ebs)

    assert sorted(set(reference) - set(found)) == []
    for key, expected in reference.items():
        np.testing.assert_allclose(found[key], expected, rtol=1e-12, atol=1e-12)


def two_band(lattice) -> FermiSurface:
    """A sphere |k| = 0.25 at Gamma and a pocket r = 0.08 at (0.4, 0.4, 0) on 24^3."""
    f = grid_fracs((24, 24, 24))
    cart = f @ lattice
    d0 = (cart + 0.5) % 1.0 - 0.5
    d1 = (cart - np.array([0.4, 0.4, 0.0]) + 0.5) % 1.0 - 0.5
    bands = np.stack(
        [np.sum(d0**2, 1) - 0.25**2, np.sum(d1**2, 1) - 0.08**2, np.full(len(f), 5.0)], 1
    )
    return FermiSurface.from_ebs(mesh(f, bands, lattice, (24, 24, 24), 0.0))


@pytest.mark.parametrize("k", [3, 4])
def test_both_bands_survive_the_clip_on_a_sheared_basis(k):
    """H1: today the pocket clips to empty and the build raises; the literals are the cubic
    build's."""
    areas = {key: float(s.area) for key, s in two_band(sheared(k)).band_isosurfaces.items()}

    assert sorted(areas) == [(0, 0), (1, 0)]
    assert areas[(0, 0)] == pytest.approx(0.7762758114909112, abs=AREA_TOL)
    assert areas[(1, 0)] == pytest.approx(0.07144548061251052, abs=AREA_TOL)


def surface_digest(fs: FermiSurface) -> dict[tuple[int, int], str]:
    return {
        key: hashlib.sha256(
            np.ascontiguousarray(s.points).tobytes() + np.ascontiguousarray(s.faces).tobytes()
        ).hexdigest()[:16]
        for key, s in fs.band_isosurfaces.items()
    }


def hex_band(cart: np.ndarray) -> np.ndarray:
    return 4 * sphere(cart / np.linalg.norm(HEX[0]))


R1_DIGESTS = {
    ("cubic", False): "8d5adcd605464988",
    ("cubic", True): "52ea96678876f717",
    ("hex", False): "4c6658cf642e6c96",
    ("hex", True): "c43c31ff19103f17",
}


@pytest.mark.guards_existing_behaviour(
    reason="a reduced basis whose box covers the zone keeps today's pad, bit for bit"
)
@pytest.mark.parametrize(("lattice_name", "centred"), list(R1_DIGESTS))
def test_reduced_basis_draws_todays_surface_bit_for_bit(lattice_name, centred):
    """R1: points and faces hash to the values the base code (2bbab10b) draws."""
    if lattice_name == "cubic":
        lattice, kgrid, band, iso = CUBIC, (16, 16, 16), sphere, 0.1
    else:
        lattice, kgrid, band, iso = HEX, (12, 12, 4), hex_band, 0.4
    f = grid_fracs(kgrid, centred=centred)
    fs = FermiSurface.from_ebs(mesh(f, band(f @ lattice), lattice, kgrid, iso))

    assert surface_digest(fs) == {(0, 0): R1_DIGESTS[(lattice_name, centred)]}


@pytest.mark.guards_existing_behaviour(
    reason="zone directions are integers in the user's basis, before and after the drawn mesh"
)
def test_extend_surface_translates_by_the_given_basis():
    """E1: [1, 0, 0] on b2 = (3,1,0) is b1 = (1, 0, 0), whatever basis the surface is drawn in."""
    fs = drawn(sheared(3), (16, 16, 16), sphere, 0.1)

    extended = fs.extend_surface([[1, 0, 0]])

    np.testing.assert_array_equal(fs.reciprocal_lattice, sheared(3))
    n = fs.n_points
    np.testing.assert_allclose(extended.points[n:], fs.points + [1.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize(
    ("lattice", "points"),
    [(HEX, 83 * 83 * 63), (BCC, 93**3), (FCC, 93**3)],
    ids=["hex", "bcc", "fcc"],
)
def test_drawn_box_on_a_dense_centred_grid_is_the_zone_box(lattice, points):
    """G1: the drawn box is the zone's bounding box plus one point each side; today 80^3 =
    512000 each. Zone corners reach 2/3 in-plane and 1/2 along c for hex (40 and 30 of 60
    points), and 3/4 for fcc and bcc (45)."""
    f = grid_fracs((60, 60, 60), centred=True)
    fs = FermiSurface.from_ebs(mesh(f, sphere(f @ lattice), lattice, (60, 60, 60), 0.05))

    assert fs.ebs.n_kpoints == points


@pytest.mark.guards_existing_behaviour(reason="a cubic box already covers the zone: today's pad")
def test_drawn_box_on_a_dense_centred_cubic_grid_is_todays_pad():
    """G1, cubic: 80^3 = 512000, today's box."""
    f = grid_fracs((60, 60, 60), centred=True)
    fs = FermiSurface.from_ebs(mesh(f, sphere(f), CUBIC, (60, 60, 60), 0.1))

    assert fs.ebs.n_kpoints == 512000


def test_stretched_grid_on_a_sheared_basis_warns_that_it_draws_the_given_basis():
    """Points that are not one uniform grid keep today's pad; on a sheared basis that can miss
    part of the zone, so the user is told."""
    axis = (np.arange(16) / 16) ** 1.05
    f = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    ebs = mesh(f, sphere(f @ sheared(3)), sheared(3), (16, 16, 16), 0.1)

    with user_warning(__file__, match="not one uniform grid"):
        FermiSurface.from_ebs(ebs)


def test_tile_holds_one_period_of_the_drawn_grid():
    """(16, 32, 16) on b2 = (3,1,0) is diagonal only on a 32 x 32 x 16 tile, twice the grid;
    each tile point holds the band at its own Cartesian k."""
    fs = drawn(sheared(3), (16, 32, 16), sphere, 0.1)
    grid = fs._periodic_grid
    bands = fs.original_ebs.bands
    assert grid is not None and bands is not None

    tile = grid.tile(bands.value[:, 0, 0])

    assert grid.n == (32, 32, 16)
    assert grid.tile_multiple == 2
    m = np.stack(np.meshgrid(*[np.arange(n) for n in grid.n], indexing="ij"), axis=-1)
    cart = ((m + grid.shift) / np.array(grid.n)) @ grid.lattice
    np.testing.assert_allclose(tile, sphere(cart.reshape(-1, 3)).reshape(grid.n), atol=1e-12)


BISB = np.array(
    [
        [0.235005607, 0.13568153, 1.89e-07],
        [1.54e-07, 0.271361575, -1.35e-07],
        [7e-09, -5e-09, 0.061080856],
    ]
)
"""The BiSb monolayer fixture's reciprocal lattice: hexagonal, but |b1 - b2| is 3.2e-6 shorter
than |b1| relative, from the rounding of its POSCAR."""


@pytest.mark.guards_existing_behaviour(
    reason="today every basis is drawn as given; a reduced one with rounded lengths stays so"
)
def test_hexagonal_basis_with_rounded_lengths_is_drawn_as_given():
    """Both 60 and 120 degree settings of a hexagonal plane are reduced; POSCAR rounding must
    not re-index the grid into the other one."""
    f = grid_fracs((12, 12, 1), centred=True)
    two_pi = 2 * np.pi
    band = -(
        np.cos(two_pi * f[:, 0]) + np.cos(two_pi * f[:, 1]) + np.cos(two_pi * (f[:, 0] - f[:, 1]))
    )

    fs = FermiSurface.from_ebs(mesh(f, band, BISB, (12, 12, 1), 0.0))

    np.testing.assert_array_equal(fs.ebs.reciprocal_lattice, BISB)
