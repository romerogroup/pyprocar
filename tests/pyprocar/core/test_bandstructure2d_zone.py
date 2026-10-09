"""BandStructure2D draws the whole first zone on any basis and any full grid (#302, P1 and P2).

The bands here are linear inside the cubic zone (a sawtooth, periodic over the lattice), so any
linear interpolation of the k-grid samples gives the band exactly at points whose cells stay
inside the zone. A value read off the drawn surface at a Cartesian point then does not depend on
where a build puts its (u, v) grid, and two builds of one lattice in two bases can be compared.
"""

import numpy as np
import pytest
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from pyprocar.core.bandstructure2D import (
    BandStructure2D,
    compute_plane_info,
    generate_band_2d_surfaces,
)
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from tests.pyprocar.core.test_bandstructure2d_grid import (
    HEXAGONAL,
    saw_band,
    tight_binding_graphene,
)
from tests.utils.user_warning import user_warning

CUBIC = np.eye(3)
SHEARED = np.array([[1.0, 0, 0], [3.0, 1, 0], [0, 0, 1]])
"""The cubic reciprocal lattice in the basis b1, b2 + 3 b1, b3."""


def grid_fracs(kgrid, centred: bool = False) -> np.ndarray:
    axes = []
    for n in kgrid:
        axis = np.arange(n) / n
        axes.append(np.where(axis > 0.5 + 1e-12, axis - 1.0, axis) if centred else axis)
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def mesh(frac, energies, lattice, kgrid) -> ElectronicBandStructureMesh:
    """Two bands, ``energies`` and ``energies + 5``, one spin."""
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=tuple(kgrid), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=frac,
        bands=np.stack([energies, energies + 5], axis=1)[:, :, None],
        projected=np.ones((len(frac), 2, 1, 1, 1)),
        fermi=0.0,
        reciprocal_lattice=lattice,
        orbital_names=["s"],
    )


def saw_mesh(lattice, kgrid=(16, 16, 16)) -> ElectronicBandStructureMesh:
    frac = grid_fracs(kgrid)
    return mesh(frac, saw_band(frac @ lattice), lattice, kgrid)


def values_at(bs: BandStructure2D, cart: np.ndarray, band: int) -> np.ndarray:
    """The drawn sheet's energy at Cartesian points of its plane, read off the sheet by linear
    interpolation in its (2 pi k.u, 2 pi k.v) coordinates."""
    sheet = np.asarray(bs.band_surfaces[(band, 0)].points)
    sheet = sheet[np.isfinite(sheet).all(axis=1)]
    uv = 2 * np.pi * np.column_stack([cart @ bs.u, cart @ bs.v])
    return LinearNDInterpolator(sheet[:, :2], sheet[:, 2])(uv)


IN_ZONE = np.array([[0.37, 0.36, 0.0], [-0.37, 0.36, 0.0], [0.1, -0.2, 0.0]])
"""Cartesian points of the plane z = 0, inside the cubic zone. On b2 = (3,1,0) the second sits at
fractional b1 = -1.45, beyond today's pad of 15 points on 16."""


def test_bs2d_on_a_sheared_basis_has_the_reduced_bands_at_every_zone_point():
    """P2: the plane z = 0 given as fractions of the sheared basis."""

    def build(lattice: np.ndarray) -> BandStructure2D:
        return BandStructure2D.from_ebs(
            saw_mesh(lattice),
            normal=(0, 0, 1),
            origin=(0, 0, 0),
            as_cartesian=False,
            grid_interpolation=(120, 120),
        )

    sheared, reduced = build(SHEARED), build(CUBIC)

    for band, offset in [(0, 0.0), (1, 5.0)]:
        got = values_at(sheared, IN_ZONE, band)
        np.testing.assert_allclose(got, values_at(reduced, IN_ZONE, band), rtol=0, atol=1e-9)
        np.testing.assert_allclose(got, np.array([1.09, 0.35, -0.3]) + offset, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(sheared.normal, [0, 0, 1])
    np.testing.assert_array_equal(sheared.origin, [0, 0, 0])
    assert sheared.as_cartesian is False


@pytest.mark.guards_existing_behaviour(
    reason="a fractional plane that today's pad covers keeps its bands once the grid is redrawn"
)
def test_fractional_plane_whose_normal_the_drawn_basis_changes_cuts_the_given_plane():
    """The user's plane f1 = 0.1 of the sheared basis is x - 3y = 0.1, whose fractional normal in
    the cubic basis is (1, -3, 0). A sheared cell spans 0.25 in x, so the points keep |x| <= 0.25
    for today's cells to stay inside the zone, where the band is linear."""
    bs = BandStructure2D.from_ebs(
        saw_mesh(SHEARED),
        normal=(1, 0, 0),
        origin=(0.1, 0, 0),
        as_cartesian=False,
        grid_interpolation=(120, 120),
    )
    on_plane = np.array([[0.25, 0.05, 0.2], [-0.14, -0.08, -0.3], [0.1, 0.0, 0.3]])

    np.testing.assert_allclose(values_at(bs, on_plane, 0), saw_band(on_plane), rtol=0, atol=1e-9)
    np.testing.assert_array_equal(bs.normal, [1, 0, 0])
    np.testing.assert_array_equal(bs.origin, [0.1, 0, 0])


@pytest.mark.guards_existing_behaviour(reason="the zone is the user's lattice's, as today")
def test_zone_of_a_grid_drawn_on_a_sublattice_is_the_crystals():
    """(16, 32, 16) on b2 = (3,1,0) is drawn on a sublattice tile twice the grid; the zone the
    plotter draws and clips to is still the cubic one, |kx|, |ky| <= pi with the 2 pi scale."""
    bs = BandStructure2D.from_ebs(saw_mesh(SHEARED, (16, 32, 16)), normal=(0, 0, 1))

    zone = bs.get_2d_brillouin_zone(e_min=-1.0, e_max=1.0)

    np.testing.assert_allclose(zone.bounds, [-np.pi, np.pi, -np.pi, np.pi, -1.0, 1.0], atol=1e-12)


HEXAGON_ZONE_AREA = 0.405231703 * 0.233960597 * 2 * (2 * np.pi) ** 2
"""Graphene's zone, |b1 x b2| of HEXAGONAL, with the plot's 2 pi scale."""


def projected_area(surface: pv.PolyData) -> float:
    points = np.array(surface.points)
    points[:, 2] = 0.0
    flat = surface.copy()
    flat.points = points
    return float(flat.area)


def sheet_holds(bs: BandStructure2D, points: np.ndarray) -> np.ndarray:
    """Whether the drawn sheet covers each point's (x, y)."""
    surface = np.asarray(bs.points)
    extent = Delaunay(surface[np.isfinite(surface).all(axis=1), :2])
    return extent.find_simplex(points[:, :2]) >= 0


def test_bs2d_on_a_zero_to_one_hexagonal_grid_draws_the_whole_zone():
    """P1: graphene on a [0, 1) 30 x 30 grid. The zone corner at fractional -2/3 lies 20 points
    below the grid, beyond the default pad of 15."""
    frac = grid_fracs((30, 30, 1))
    energies = tight_binding_graphene(frac)
    bs = BandStructure2D.from_ebs(
        mesh(frac, energies, HEXAGONAL, (30, 30, 1)),
        normal=(0, 0, 1),
        grid_interpolation=(60, 60),
    )
    plotter = BS2DPlotter(bs, off_screen=True)

    drawn = plotter.plot(show_brillouin_zone=False, clip_brillouin_zone=True)

    plotter.close()
    zone = bs.get_2d_brillouin_zone(e_min=-1.0, e_max=10.0)
    assert sheet_holds(bs, np.asarray(zone.points)).all()
    assert sorted(drawn) == [(0, 0), (1, 0)]
    for sheet in drawn.values():
        assert projected_area(sheet) == pytest.approx(HEXAGON_ZONE_AREA, rel=1e-9)


@pytest.mark.parametrize("padding", [3, 15])
def test_unclipped_bs2d_on_a_zero_to_one_grid_holds_the_zone_and_todays_pad(padding):
    """A pad of p points on a [0, 1) 30 x 30 grid spans fractional -p/30 to (29 + p)/30. Both
    pads miss the zone corner at -2/3, and both reach past the zone's far corner at 2/3 on the
    far side, so the drawn sheet must hold the zone and the whole pad."""
    frac = grid_fracs((30, 30, 1))
    bs = BandStructure2D.from_ebs(
        mesh(frac, tight_binding_graphene(frac), HEXAGONAL, (30, 30, 1)),
        normal=(0, 0, 1),
        padding=padding,
    )

    lo, hi = -padding / 30, (29 + padding) / 30
    pad = np.array([[lo, lo, 0.0], [hi, lo, 0.0], [lo, hi, 0.0], [hi, hi, 0.0]])
    # A hair inside the box, so a corner on the sheet's edge counts as held.
    pad = pad + 1e-6 * (pad.mean(axis=0) - pad)
    zone = bs.get_2d_brillouin_zone(e_min=-1.0, e_max=1.0)
    assert sheet_holds(bs, np.asarray(zone.points)).all()
    assert sheet_holds(bs, 2 * np.pi * pad @ HEXAGONAL).all()


def test_unclipped_bs2d_on_a_sheared_basis_draws_the_pad_of_its_reduced_basis():
    """The [0, 1) 16^3 grid on b2 = (3,1,0) is the cubic [0, 1) grid. Given in the cubic basis,
    a pad of 9 reaches the zone's box, one point past the face at -8, so that pad is the mesh
    both bases draw."""
    sheared = BandStructure2D.from_ebs(saw_mesh(SHEARED), normal=(0, 0, 1), padding=9)
    cubic = saw_mesh(CUBIC).pad(padding=9, inplace=False)

    def cartesian(ebs: ElectronicBandStructureMesh) -> np.ndarray:
        points = np.asarray(ebs.kpoints) @ np.asarray(ebs.reciprocal_lattice)
        return np.round(points[np.lexsort(points.T)], 9)

    # expand_single_dimension leaves a three-dimensional grid as it is.
    assert sheared.ebs.kgrid == cubic.kgrid == (34, 34, 34)
    np.testing.assert_array_equal(cartesian(sheared.ebs), cartesian(cubic))


def todays_pad_build(ebs, padding: int, normal, origin, as_cartesian: bool) -> BandStructure2D:
    """dev's from_ebs: pad the given grid, then cut the plane."""
    padded = ebs.pad(padding=padding, inplace=False).expand_single_dimension(inplace=False)
    info = compute_plane_info(padded, normal, origin, (40, 40), as_cartesian)
    surface, sheets, point_set = generate_band_2d_surfaces(
        ebs=padded, plane_info=info, original_ebs=ebs
    )
    return BandStructure2D(
        points=surface.points,
        faces=surface.faces,
        band_surfaces=sheets,
        point_set=point_set,
        original_ebs=ebs,
        ebs=padded,
        plane_info=info,
        point_data=dict(surface.point_data),
    )


CENTRED_HEX = grid_fracs((24, 24, 1), centred=True)
CENTRED_HEX_MESH = mesh(CENTRED_HEX, tight_binding_graphene(CENTRED_HEX), HEXAGONAL, (24, 24, 1))
REDUCED_CASES = {
    "hexagonal-centred-cartesian": (
        CENTRED_HEX_MESH,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        True,
    ),
    "hexagonal-centred-fractional": (
        CENTRED_HEX_MESH,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        False,
    ),
    "cubic-fractional-110": (saw_mesh(CUBIC), (1.0, 1.0, 0.0), (0.1, 0.0, 0.2), False),
}


@pytest.mark.guards_existing_behaviour(
    reason="a reduced grid whose pad covers the zone: dev's build"
)
@pytest.mark.parametrize("case", REDUCED_CASES)
def test_bs2d_on_a_reduced_grid_whose_pad_covers_the_zone_is_todays_build(case):
    ebs, normal, origin, as_cartesian = REDUCED_CASES[case]

    bs = BandStructure2D.from_ebs(
        ebs,
        normal=normal,
        origin=origin,
        grid_interpolation=(40, 40),
        as_cartesian=as_cartesian,
        padding=15,
    )

    today = todays_pad_build(ebs, 15, normal, origin, as_cartesian)
    np.testing.assert_array_equal(bs.points, today.points)
    np.testing.assert_array_equal(bs.faces, today.faces)
    assert bs.point_data.keys() == today.point_data.keys()
    for name in today.point_data:
        np.testing.assert_array_equal(bs.point_data[name], today.point_data[name])
    np.testing.assert_array_equal(bs.get_property("bands").value, today.get_property("bands").value)
    np.testing.assert_array_equal(bs.ebs.kpoints, today.ebs.kpoints)
    np.testing.assert_array_equal(bs.normal, normal)
    np.testing.assert_array_equal(bs.origin, origin)


def test_stretched_grid_on_a_sheared_basis_warns_that_it_draws_the_given_basis():
    """Points that are not one uniform grid keep today's pad; on a sheared basis that can miss
    part of the zone, so the user is told."""
    axis = (np.arange(16) / 16) ** 1.05
    frac = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    ebs = mesh(frac, saw_band(frac @ SHEARED), SHEARED, (16, 16, 16))

    with user_warning(__file__, match="not one uniform grid"):
        BandStructure2D.from_ebs(ebs, normal=(0, 0, 1))
