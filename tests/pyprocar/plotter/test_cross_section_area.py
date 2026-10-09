import itertools
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import pyvista as pv
from matplotlib.quiver import Quiver
from vtkmodules.vtkCommonTransforms import vtkTransform

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from pyprocar.plotter._periodic_cut import _BandCut, periodic_bands
from pyprocar.plotter._surface_plot import cross_section_areas, snap_normal
from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from pyprocar.plotter.fs_plot import (
    FS_AREA_SCALE_FACTOR,
    FermiPlotter,
    dHvA_frequency,
)
from tests.utils import DATA_DIR

RADIUS = 0.3


def _sphere(x: float) -> pv.PolyData:
    return pv.Sphere(radius=RADIUS, center=(x, 0, 0), theta_resolution=180, phi_resolution=180)


def _area_text(plotter) -> str:
    return cast(pv.CornerAnnotation, plotter.actors["area_text"]).GetText(2)


def _number(text: str) -> float:
    return float(text.split(":")[1].split()[0])


def test_cross_section_area_sums_the_closed_loops():
    two_spheres = _sphere(-0.5).merge(_sphere(0.5))
    plotter = FermiPlotter(off_screen=True)

    plotter.add_box_slicer(two_spheres, normal=(0, 0, 1), show_cross_section_area=True)

    text = _area_text(plotter)
    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(2 * np.pi * RADIUS**2 * FS_AREA_SCALE_FACTOR, rel=1e-3)
    plotter.close()


def test_van_alphen_frequency_uses_the_largest_closed_orbit():
    small = pv.Sphere(radius=0.1, center=(-0.5, 0, 0), theta_resolution=180, phi_resolution=180)
    plotter = FermiPlotter(off_screen=True)

    plotter.add_box_slicer(
        small.merge(_sphere(0.5)), normal=(0, 0, 1), show_van_alphen_frequency=True
    )

    assert _number(_area_text(plotter)) == pytest.approx(onsager_gauss(RADIUS), rel=1e-3)
    plotter.close()


def onsager_gauss(radius: float) -> float:
    """F = hbar A / (2 pi e) in SI with CODATA 2018 constants, converted from tesla to gauss.

    ``radius`` is in 1/Angstrom without 2 pi, so the physical orbit area is
    pi (2 pi radius)^2 in 1/Angstrom^2, times 1e20 for 1/m^2.
    """
    hbar, e = 1.054571817e-34, 1.602176634e-19
    area = np.pi * (2 * np.pi * radius) ** 2 * 1e20
    return hbar * area / (2 * np.pi * e) * 1e4


def test_dhva_frequency_of_one_inverse_square_angstrom():
    assert dHvA_frequency(1.0) == pytest.approx(1.04758e8, rel=1e-5)


def _bowl() -> pv.PolyData:
    return cast(pv.PolyData, _sphere(-0.5).clip(normal=(1, 0, 0), origin=(-0.6, 0, 0)))


def _slice_text(surface: pv.PolyData, origin=(0, 0, 0), normal=(0, 0, 1), **flags: bool) -> str:
    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(surface, normal=normal, origin=origin, **flags)
    text = _area_text(plotter)
    plotter.close()
    return text


def test_open_curves_are_reported_and_not_counted():
    text = _slice_text(_bowl().merge(_sphere(0.5)), show_cross_section_area=True)

    assert text.endswith(" Ang^-2 (1 open curve not counted)")
    assert _number(text) == pytest.approx(np.pi * RADIUS**2 * FS_AREA_SCALE_FACTOR, rel=1e-3)


def test_van_alphen_frequency_without_a_closed_orbit_says_so():
    text = _slice_text(_bowl(), origin=(-0.7, 0, 0), show_van_alphen_frequency=True)

    assert (
        text == "Van Alphen Frequency : no closed orbit through this cut (1 open curve not counted)"
    )


def test_van_alphen_frequency_reports_the_open_curves_it_skipped():
    text = _slice_text(_bowl().merge(_sphere(0.5)), show_van_alphen_frequency=True)

    assert text.endswith(" Gauss (1 open curve not counted)")
    assert _number(text) == pytest.approx(onsager_gauss(RADIUS), rel=1e-3)


def _band_surface_with_nan_energies() -> pv.PolyData:
    """E = kx^2 + ky^2 with NaN energies beyond |k| = 0.4, as BandStructure2D leaves them."""
    k = np.linspace(-0.5, 0.5, 41)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    energy = kx**2 + ky**2
    energy[np.hypot(kx, ky) > 0.4] = np.nan
    grid = pv.StructuredGrid(kx[..., None], ky[..., None], energy[..., None])
    surface = cast(pv.PolyData, grid.extract_surface().triangulate())
    surface.point_data["energy"] = surface.points[:, 2]
    surface.set_active_scalars("energy")
    return surface


@pytest.mark.parametrize(
    ("normal", "origin", "area", "suffix"),
    [
        ((0, 0, 1), (0, 0, 0.04), np.pi * 0.04, " Ang^-2"),
        ((1, 0, 0), (0, 0, 0), 0.0, " Ang^-2 (1 open curve not counted)"),
    ],
)
def test_band_surface_area_skips_nan_points(normal, origin, area, suffix):
    plotter = BS2DPlotter(SimpleNamespace(), off_screen=True)

    plotter.add_box_slicer(
        _band_surface_with_nan_energies(), normal=normal, origin=origin, cross_section_area=True
    )

    text = _area_text(plotter)
    assert text.endswith(suffix)
    assert _number(text) == pytest.approx(area, rel=0.01, abs=1e-12)
    plotter.close()


def test_saved_slice_draws_the_active_vectors_as_arrows(tmp_path):
    sphere = _sphere(0.0)
    sphere.point_data["spin"] = np.tile([0.0, 1.0, 0.0], (sphere.n_points, 1))
    sphere.point_data["spin-norm"] = np.ones(sphere.n_points)
    sphere.set_active_vectors("spin")
    sphere.set_active_scalars("spin-norm")
    plotter = FermiPlotter(off_screen=True)

    slice_plotter = plotter.save_slice_2d(
        sphere, (0, 0, 1), (0, 0, 0), tmp_path / "slice.png", cmap="viridis"
    )

    (arrows,) = [c for c in slice_plotter.ax.collections if isinstance(c, Quiver)]
    assert arrows.get_cmap().name == "viridis"
    np.testing.assert_allclose(arrows.get_array(), 1.0)
    plotter.close()


def _same(k: np.ndarray) -> np.ndarray:
    return k


def _periodic_surface(
    energy: Callable[[np.ndarray], np.ndarray],
    reciprocal_lattice: np.ndarray | None = None,
    stored_kpoints: Callable[[np.ndarray], np.ndarray] = _same,
    kgrid: tuple[int, int, int] = (16, 16, 16),
    kshift: tuple[float, float, float] = (0, 0, 0),
) -> FermiSurface:
    """The E_F = 0.1 surface of ``energy(k)`` on a 16^3 mesh, B = I unless given.

    ``stored_kpoints`` maps the grid k-points to the ones the band structure records.
    """
    axes = [(np.arange(n) + shift) / n for n, shift in zip(kgrid, kshift, strict=True)]
    kpoints = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    energies = np.stack([energy(kpoints), np.full(len(kpoints), 5.0)], axis=1)
    return FermiSurface.from_ebs(
        ElectronicBandStructureMesh(
            kgrid_info=KGridInfo(kgrid=kgrid, kgrid_mode=KGRID_MODE.GAMMA, kshift=kshift),
            kpoints=stored_kpoints(kpoints),
            bands=energies[..., np.newaxis],
            projected=np.ones((len(kpoints), 2, 1, 1, 1)),
            fermi=0.1,
            reciprocal_lattice=np.eye(3) if reciprocal_lattice is None else reciprocal_lattice,
            orbital_names=["s"],
        )
    )


def test_orbit_inside_the_zone_keeps_its_area():
    def sphere_around_gamma(k):
        to_gamma = (k + 0.5) % 1.0 - 0.5
        return np.sum(to_gamma**2, axis=1)

    text = _slice_text(_periodic_surface(sphere_around_gamma), show_cross_section_area=True)

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(np.pi * 0.1 * (2 * np.pi) ** 2, rel=0.02)


def test_orbit_around_the_zone_corner_closes_across_the_zone_boundary():
    def cylinder_around_m(k):
        to_m = k[:, :2] % 1.0 - 0.5
        return np.sum(to_m**2, axis=1)

    text = _slice_text(_periodic_surface(cylinder_around_m), show_cross_section_area=True)

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(np.pi * 0.1 * (2 * np.pi) ** 2, rel=0.02)


@pytest.mark.parametrize("height", [0.5, 0.5 - 1e-4, -0.5 + 1e-4])
def test_zone_face_cut_counts_the_orbit_around_m_once(height):
    """The orbit around M meets the zone at all four corners; its lattice translates are
    one orbit, so the cut has one area pi 0.1, not four."""

    def cylinder_around_m(k):
        to_m = k[:, :2] % 1.0 - 0.5
        return np.sum(to_m**2, axis=1)

    areas, n_open = cross_section_areas(
        _periodic_surface(cylinder_around_m), (0, 0, 1), (0, 0, height), np.eye(3)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([np.pi * 0.1], rel=0.02)


def test_generic_normal_cuts_the_cylinder_around_m_in_one_ellipse():
    """A plane with normal n cuts the cylinder |k_xy - M|^2 = 0.1 in an ellipse of area
    pi 0.1 / |n_z|. Two of the plane's lattice translates that cross the surface are only
    9.3e-4 apart along this normal, and each carries a piece of the ellipse.
    """

    def cylinder_around_m(k: np.ndarray) -> np.ndarray:
        return np.sum((k[:, :2] % 1.0 - 0.5) ** 2, axis=1)

    normal = np.array([0.630914259462603, -0.6191560458478862, 0.4675392904310453])

    areas, n_open = cross_section_areas(
        _periodic_surface(cylinder_around_m), normal, 0.5790289704673109 * normal, np.eye(3)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([np.pi * 0.1 / normal[2]], rel=0.02)


ROUNDED_111 = (0.5774, 0.5773, 0.5774)


def _cylinder_around_m(k: np.ndarray) -> np.ndarray:
    return np.sum((k[:, :2] % 1.0 - 0.5) ** 2, axis=1)


def test_rounded_111_normal_cuts_the_exact_111_plane():
    """Typed to 4 digits, the normal is 1e-4 rad off [1 1 1]. The (1 1 1) plane through
    Gamma cuts the cylinder around M in one ellipse of area pi 0.1 sqrt(3) per cell.
    """
    areas, n_open = cross_section_areas(
        _periodic_surface(_cylinder_around_m), ROUNDED_111, (0, 0, 0), np.eye(3)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([np.pi * 0.1 * np.sqrt(3)], rel=0.02)


A3_PLUS_6_A1 = np.array([[1, 0, 0], [0, 1, 0], [6, 0, 1]])
"""Rows a1, a2, a3 + 6 a1 of the simple cubic lattice in units of a: the same lattice, where
[1 1 1] is -5 a1' + a2' + a3'."""


def test_rounded_111_normal_snaps_in_a_sheared_basis():
    """The cubic lattice given with a3' = a3 + 6 a1: the (1 1 1) plane through Gamma still
    cuts the cylinder around M in one ellipse of area pi 0.1 sqrt(3) per cell."""
    reciprocal = np.linalg.inv(A3_PLUS_6_A1.astype(float)).T

    areas, n_open = cross_section_areas(
        _periodic_surface(lambda k: _cylinder_around_m(k @ reciprocal), reciprocal),
        ROUNDED_111,
        (0, 0, 0),
        reciprocal,
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([np.pi * 0.1 * np.sqrt(3)], rel=0.02)


def test_widget_says_when_it_snapped_the_normal():
    surface = _periodic_surface(_cylinder_around_m)

    snapped = _slice_text(surface, normal=ROUNDED_111, show_cross_section_area=True)
    exact = _slice_text(surface, normal=(1, 1, 1), show_cross_section_area=True)

    assert snapped.endswith(" Ang^-2 (normal snapped to [1 1 1])")
    assert exact.endswith(" Ang^-2")
    assert _number(snapped) == pytest.approx(_number(exact), rel=1e-6)


def _saddle_band(k: np.ndarray) -> np.ndarray:
    """Pockets around M that nearly touch at the zone-face saddle X: E(X) = E_F + 1e-6."""
    c = np.cos(2 * np.pi * k)
    return 0.1 + c[:, 0] + c[:, 1] + 0.3 * c[:, 0] * c[:, 1] + 0.3 + 1e-6


@pytest.mark.parametrize("height", [0.0, 0.123])
def test_pockets_that_nearly_touch_at_a_saddle_stay_separate_orbits(height):
    """The pocket around M has area 0.4376 (a contour of the analytic band on a 2001^2
    grid); the 16^3 mesh gives it within 2%. Joining the near-touching arcs of
    neighbouring pockets instead encloses the complement, 1 - 0.4332."""
    areas, n_open = cross_section_areas(
        _periodic_surface(_saddle_band), (0, 0, 1), (0, 0, height), np.eye(3)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([0.4376], rel=0.02)


@pytest.mark.parametrize(
    "normal", [(1, 2, 2), (2, 2, 1), (1, 1, 3), (3, 2, 4), (1, 1, 1), (0, 0, 1)]
)
def test_exact_lattice_normals_are_not_reported_as_snapped(normal):
    assert snap_normal(normal, np.eye(3))[1] is None


def test_exact_hexagonal_lattice_direction_is_not_reported_as_snapped():
    real = np.array([[2.46, 0, 0], [-1.23, 2.46 * np.sqrt(3) / 2, 0], [0, 0, 6.7]])
    reciprocal = np.linalg.inv(real).T

    for uvw in [(1, 1, 3), (2, -1, 3), (1, 0, 0)]:
        assert snap_normal(np.asarray(uvw, float) @ real, reciprocal)[1] is None


def test_drawn_slice_uses_the_snapped_normal():
    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(
        _periodic_surface(_cylinder_around_m),
        normal=ROUNDED_111,
        origin=(0, 0, 0),
        show_cross_section_area=True,
    )
    points = cast(pv.Actor, plotter.actors["slice"]).mapper.dataset.points
    plotter.close()

    np.testing.assert_allclose(np.asarray(points) @ (np.ones(3) / np.sqrt(3)), 0.0, atol=1e-6)


def _sum_of_cosines(k: np.ndarray) -> np.ndarray:
    return 0.1 - 0.3 * np.cos(2 * np.pi * k).sum(axis=1)


def test_plane_tangent_to_the_surface_at_a_translate_closes_its_orbit():
    """E = 0.1 - 0.3 sum cos(2 pi k), normal (1, 1, 1), through a mesh vertex one zone over,
    where a translate of the plane touches the surface at its highest point. Marching the
    same grid over 7^3 periods and slicing it with VTK gives one closed 0.7253 orbit; the
    analytic band gives 0.7261."""
    origin = (-0.5, 0.1875, 0.14229804277420044)

    areas, n_open = cross_section_areas(
        _periodic_surface(_sum_of_cosines), np.ones(3) / np.sqrt(3), origin, np.eye(3)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([0.7253], rel=1e-3)


def test_fcc_plane_through_grid_lines_counts_its_orbit_once():
    """The plane through Gamma normal to a3 of an fcc cell holds grid lines of the k mesh,
    so its cut passes through mesh vertices. Lattice translates of the orbit then have
    different node sets but the same area-weighted centroid modulo a lattice vector; the
    plane 1e-10 above cuts no vertex and gives the same single orbit."""
    fcc = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 4.08
    surface = _periodic_surface(lambda k: _sum_of_cosines(k) + 0.05, fcc)
    normal = np.linalg.inv(fcc).T[2]
    normal /= np.linalg.norm(normal)

    areas, n_open = cross_section_areas(surface, normal, (0, 0, 0), fcc)
    above, _ = cross_section_areas(surface, normal, 1e-10 * normal, fcc)

    assert n_open == 0
    assert len(areas) == len(above) == 1
    assert areas[0] == pytest.approx(above[0], rel=1e-8)


SHEARED_CUBIC = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize(
    ("normal", "height"),
    [
        ((0.4728, 0.1555, 0.8673), -0.1247),
        ((0.0788, -0.2114, 0.9742), 0.1563),
        ((-0.8043, 0.0128, 0.594), -0.0483),
    ],
)
def test_sheared_basis_counts_only_the_orbit_in_the_first_zone(normal, height):
    """b1 = (1, 0, 0), b2 = (3, 1, 0), b3 = (0, 0, 1) span the cubic lattice, whose first
    zone is the unit cube around Gamma. The sphere |k|^2 = 0.1 around Gamma lies inside it,
    so a plane at distance d from Gamma cuts one orbit of area pi (0.1 - d^2); the spheres
    around the other lattice points lie outside the zone."""
    normal = np.asarray(normal) / np.linalg.norm(normal)
    surface = _periodic_surface(
        lambda k: np.sum(((k @ SHEARED_CUBIC + 0.5) % 1.0 - 0.5) ** 2, axis=1),
        SHEARED_CUBIC,
        kgrid=(16, 48, 16),
    )

    areas, n_open = cross_section_areas(surface, normal, height * normal, SHEARED_CUBIC)

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([np.pi * (0.1 - height**2)], rel=0.04)


def test_sheared_basis_finds_the_orbit_at_a_zone_corner_six_cells_out():
    """b2 = (3, 1, 0), b3 = (-2, 2, 1) span the cubic lattice. The zone corner R = (0.5,
    0.5, -0.5) is the point (-5, 1.5, -0.5) of this basis. A plane 0.05 from R cuts the
    pocket |k - R|^2 = 0.02 in an orbit of area pi (0.02 - 0.05^2), and the sphere
    |k|^2 = 0.1 around Gamma at distance D in an orbit of area pi (0.1 - D^2)."""
    basis = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [-2.0, 2.0, 1.0]])
    normal = np.array([0.3971, 0.5523, 0.7330]) / np.linalg.norm([0.3971, 0.5523, 0.7330])
    origin = np.array([0.5, 0.5, -0.5]) + 0.05 * normal

    def sphere_and_pocket(k: np.ndarray) -> np.ndarray:
        cart = k @ basis
        to_gamma = np.sum(((cart + 0.5) % 1.0 - 0.5) ** 2, axis=1)
        return np.minimum(to_gamma, 5 * np.sum((cart % 1.0 - 0.5) ** 2, axis=1))

    surface = _periodic_surface(sphere_and_pocket, basis, kgrid=(16, 64, 48))

    areas, n_open = cross_section_areas(surface, normal, origin, basis)

    assert n_open == 0
    assert np.sort(areas) == pytest.approx(
        [np.pi * (0.02 - 0.05**2), np.pi * (0.1 - (origin @ normal) ** 2)], rel=0.08
    )


def test_shifted_grid_finds_the_line_through_a_zone_corner_below_the_start_cells():
    """The triclinic zone's corner V = (0.245642, 0.17609, -0.640012) lies at -1.939 b1 of the
    basis below. On a grid shifted half a spacing along b1, each translate of the period
    starts 0.125 b1 above an integer, so the plane 0.005 inside V meets the zone only in the
    translate three cells down. The level set f2 = f2(V) is a lattice plane through V that
    cuts the plane in one open line."""
    cell = np.array([[1.0, 0.1, 0.05], [0.3, 1.2, 0.0], [0.2, 0.35, 0.8]])
    basis = np.array([[-1, 0, 0], [0, -1, 0], [2, 0, 1]]) @ cell
    corner = np.array([0.245642, 0.17609, -0.640012])
    f2 = (corner @ np.linalg.inv(basis))[1]
    normal = corner / np.linalg.norm(corner) + np.array([0.05, -0.03, 0.02])
    normal /= np.linalg.norm(normal)
    surface = _periodic_surface(
        lambda k: 0.1 + np.sin(2 * np.pi * k[:, 1]) - np.sin(2 * np.pi * f2),
        basis,
        kgrid=(4, 24, 24),
        kshift=(0.5, 0, 0),
    )

    areas, n_open = cross_section_areas(surface, normal, corner - 0.005 * normal, basis)

    assert (areas, n_open) == ([], 1)


def test_orbit_reaching_five_cells_from_the_zone_closes():
    """|n_z| = 0.06 stretches the ellipse around each M cylinder to 10.5 cells. The
    verifier's reference (marching cubes translated 16 cells, sliced by VTK) finds two
    closed orbits of 5.1506; analytic pi 0.1 / |n_z| = 5.212."""
    normal = np.array([0.6027597240921518, -0.7956428358016405, 0.06027597240921519])

    areas, n_open = cross_section_areas(
        _periodic_surface(_cylinder_around_m), normal, (0.5, 0.5, 0), np.eye(3)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([5.1506, 5.1506], rel=1e-4)


def _sphere_and_cylinder(k: np.ndarray) -> np.ndarray:
    """A sphere of radius 0.2 around Gamma and a cylinder of radius sqrt(0.1) around M."""
    to_gamma = (k + 0.5) % 1.0 - 0.5
    to_m = k[:, :2] % 1.0 - 0.5
    return 0.1 * np.minimum((to_gamma**2).sum(axis=1) / 0.04, (to_m**2).sum(axis=1) / 0.1)


def _text_in_box(
    half_width: float,
    kz: float = 0.0,
    scale: tuple[float, float, float] | None = None,
    angle: float = 0.0,
    centre: tuple[float, float, float] = (0, 0, 0),
    **flags: bool,
) -> tuple[str, int]:
    """Widget text and drawn slice size at kz for the box [-1, 1]^3 scaled (by default to
    |kx|, |ky| <= half_width, |kz| <= 0.5), turned by ``angle`` degrees about z and moved to
    ``centre``."""
    surface = _periodic_surface(_sphere_and_cylinder)
    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(surface, normal=(0, 0, 1), origin=(0, 0, kz), **flags)
    box = plotter.box_widgets[0]
    box.SetPlaceFactor(1.0)
    box.PlaceWidget([-1, 1, -1, 1, -1, 1])
    transform = vtkTransform()
    transform.PostMultiply()
    transform.Scale(*(scale or (half_width, half_width, 0.5)))
    transform.RotateZ(angle)
    transform.Translate(*centre)
    box.SetTransform(transform)
    box.InvokeEvent("EndInteractionEvent")
    text = _area_text(plotter)
    slc = plotter.actors.get("slice")
    n_points = cast(pv.Actor, slc).mapper.dataset.n_points if slc is not None else 0
    plotter.close()
    return text, n_points


def test_box_around_nothing_counts_no_orbit():
    text, n_points = _text_in_box(0.1, show_cross_section_area=True)

    assert n_points == 0
    assert text == "Cross sectional area : 0.0000 Ang^-2"


def test_box_around_the_smaller_orbit_gives_its_frequency():
    """The full box holds both orbits and the text gives the cylinder's frequency; a box
    at |kx|, |ky| <= 0.25 holds the sphere's circle (radius 0.2) and none of the cylinder's
    (0.39 from Gamma at the closest), so it gives the sphere's."""
    full = _slice_text(_periodic_surface(_sphere_and_cylinder), show_van_alphen_frequency=True)
    boxed, n_points = _text_in_box(0.25, show_van_alphen_frequency=True)

    assert n_points > 0
    assert _number(full) == pytest.approx(
        dHvA_frequency(np.pi * 0.1 * FS_AREA_SCALE_FACTOR), rel=0.03
    )
    assert _number(boxed) == pytest.approx(
        dHvA_frequency(np.pi * 0.04 * FS_AREA_SCALE_FACTOR), rel=0.03
    )


def test_box_holding_a_short_arc_of_an_orbit_counts_it():
    """The turned box holds the cylinder's arc and, in one corner, a 0.027 arc of the
    sphere's circle with no mesh vertex cut inside the box; both orbits are drawn and count,
    as with the full box."""
    full = _slice_text(
        _periodic_surface(_sphere_and_cylinder),
        origin=(0, 0, -0.003),
        show_cross_section_area=True,
    )

    boxed, n_points = _text_in_box(
        0.0,
        kz=-0.003,
        scale=(0.146, 0.121, 0.6),
        angle=79.1,
        centre=(0.26, -0.247, 0),
        show_cross_section_area=True,
    )

    assert n_points > 0
    assert boxed == full


def test_thin_slab_through_an_orbit_counts_it():
    """The slab |kx - 0.09| <= 0.004 at kz = 0.03 holds two arcs of the sphere's circle,
    0.019 long in all, between two cut points; the text is that circle's area."""
    areas, _ = cross_section_areas(
        _periodic_surface(_sphere_and_cylinder), (0, 0, 1), (0, 0, 0.03), np.eye(3)
    )

    text, n_points = _text_in_box(
        0.0,
        kz=0.03,
        scale=(0.004, 0.6, 0.6),
        centre=(0.09, 0, 0),
        show_cross_section_area=True,
    )

    assert n_points > 0
    assert text == f"Cross sectional area : {min(areas) * FS_AREA_SCALE_FACTOR:.4f} Ang^-2"


@pytest.mark.parametrize("uvw", [(0, 0, 1), (1, 1, 1)])
def test_cut_through_mesh_vertices_keeps_every_straddling_triangle(uvw):
    """Planes along [u v w] of an fcc cell through mesh vertices: each triangle with vertices
    on both sides of the plane in a translate gives one segment, though the period's
    triangle heights, which pick the candidates, round differently from its own."""
    fcc = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 4.08
    bands = periodic_bands(_periodic_surface(lambda k: _sum_of_cosines(k) + 0.05, fcc))
    assert bands
    band = bands[0]
    normal = np.asarray(uvw, dtype=np.float64) @ np.linalg.inv(fcc).T
    normal /= np.linalg.norm(normal)
    steps = np.array(list(itertools.product(range(-2, 3), repeat=3)))

    for vertex in range(0, len(band.canon), len(band.canon) // 16):
        d = float(band.canon[vertex] @ fcc @ normal)
        cut = _BandCut(band, fcc, normal, d)
        s = cut.heights[band.tri_cls] + (band.tri_off[None] + steps[:, None, None]) @ cut.w - d
        straddling = int(((s > 0).any(axis=2) & ~(s > 0).all(axis=2)).sum())
        segments = cut.segments(steps)

        assert segments is not None
        assert len(segments[0]) == straddling


def test_kpoints_off_a_uniform_grid_say_orbits_are_not_joined():
    stretched = _periodic_surface(
        _cylinder_around_m,
        stored_kpoints=lambda k: np.column_stack([k[:, 0] ** 1.05, k[:, 1], k[:, 2]]),
    )

    text = _slice_text(stretched, show_cross_section_area=True)

    assert text.endswith(
        " (4 open curves not counted) (k-points are not a uniform 3D grid;"
        + " orbits crossing the zone boundary are not joined)"
    )


def test_kpoints_printed_with_3_decimals_still_join():
    rounded = _periodic_surface(_cylinder_around_m, stored_kpoints=lambda k: np.round(k, 3))

    text = _slice_text(rounded, show_cross_section_area=True)

    assert text == _slice_text(_periodic_surface(_cylinder_around_m), show_cross_section_area=True)
    assert _number(text) == pytest.approx(np.pi * 0.1 * FS_AREA_SCALE_FACTOR, rel=0.02)


def test_sheets_that_run_through_the_zone_stay_open():
    def planes_at_ky(k):
        to_gamma = (k[:, 1] + 0.5) % 1.0 - 0.5
        return to_gamma**2

    text = _slice_text(_periodic_surface(planes_at_ky), show_cross_section_area=True)

    assert text == "Cross sectional area : 0.0000 Ang^-2 (2 open curves not counted)"


def test_offset_oblique_cut_finds_the_orbit_in_the_neighbouring_zones():
    """The plane kx + ky = 3/4 cuts the sphere |k - M| = sqrt(0.1) mostly outside the first zone.

    Its distance to M is 1 / (4 sqrt 2), so the orbit has area pi (0.1 - 1/32).
    """

    def sphere_around_m(k):
        to_m = k % 1.0 - np.array([0.5, 0.5, 0.0])
        to_m[:, 2] = (k[:, 2] + 0.5) % 1.0 - 0.5
        return np.sum(to_m**2, axis=1)

    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(
        _periodic_surface(sphere_around_m),
        normal=(1, 1, 0),
        origin=(0.5, 0.25, 0),
        show_cross_section_area=True,
    )
    text = _area_text(plotter)
    plotter.close()

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(np.pi * (0.1 - 1 / 32) * (2 * np.pi) ** 2, rel=0.03)


@pytest.mark.data
def test_srvo3_band_16_offset_oblique_cut_finds_both_orbits():
    """SrVO3 band 16 on the plane kx + ky = 10/21 (fractional), normal (1, 1, 0).

    The plane holds the grid points (i, 10 - i, k). A matplotlib contour of the parsed
    energies on that periodic in-plane grid has two closed orbits per in-plane cell, each
    0.1632 of the cell |b1 - b2| |b3| = sqrt(2) / a^2.
    """
    a = 3.84652
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/non-spin-polarized"
    )
    band_16 = fs.select_bands([(16, 0)])
    assert isinstance(band_16, FermiSurface)

    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(
        band_16, normal=(1, 1, 0), origin=(10 / (21 * a), 0, 0), show_cross_section_area=True
    )
    text = _area_text(plotter)
    plotter.close()

    assert text.endswith(" Ang^-2")
    expected = 2 * 0.1632 * np.sqrt(2) / a**2 * (2 * np.pi) ** 2
    assert _number(text) == pytest.approx(expected, rel=0.03)


@pytest.mark.data
@pytest.mark.parametrize(
    ("normal", "origin", "area"),
    [
        ((0.02, 0.03, 1), (0.002559, 0.003839, 0.127955), 0.0104771),
        ((0.3, 0.7, 1), (0.016287, 0.038004, 0.054292), 0.12629),
    ],
)
def test_srvo3_band_16_orbits_cut_near_the_band_edge_close(normal, origin, area):
    """Each orbit passes through a lattice translate of the plane that cuts band 16 near its
    lowest or highest point along the normal.

    The expected areas come from the PR verifier's marching cubes on a periodic tile of the
    unfolded EIGENVAL energies, sliced by the plane with no zone clipping.
    """
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/non-spin-polarized"
    )
    band_16 = fs.select_bands([(16, 0)])
    assert isinstance(band_16, FermiSurface)

    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(band_16, normal=normal, origin=origin, show_cross_section_area=True)
    text = _area_text(plotter)
    plotter.close()

    assert _number(text) == pytest.approx(area * (2 * np.pi) ** 2, rel=0.01)


@pytest.mark.data
def test_srvo3_band_16_random_normal_cut_closes_its_orbit():
    """A random normal, so translate planes can lie closer together than 1e-3 |b|.

    The expected area is the PR verifier's independent reference: marching cubes on a
    periodic tile of the unfolded EIGENVAL energies, sliced by the plane with no zone
    clipping, gives one closed orbit of 0.2223563 1/A^2 (no 2 pi).
    """
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/non-spin-polarized"
    )
    band_16 = fs.select_bands([(16, 0)])
    assert isinstance(band_16, FermiSurface)
    normal = (-0.5452820360305133, -0.670617501299716, 0.5029310769210262)
    origin = tuple(0.054881091342609614 * np.asarray(normal))

    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(band_16, normal=normal, origin=origin, show_cross_section_area=True)
    text = _area_text(plotter)
    plotter.close()

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(0.2223563 * (2 * np.pi) ** 2, rel=0.01)


def _srvo3_band_16() -> FermiSurface:
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/non-spin-polarized"
    )
    band_16 = fs.select_bands([(16, 0)])
    assert isinstance(band_16, FermiSurface)
    return band_16


@pytest.mark.data
def test_srvo3_band_16_orbit_around_m_just_before_its_lifshitz_transition():
    """At kz = 0.2195874 b + 1e-7 b the four arcs around M are still separate chains,
    7.6e-6 apart at their closest ends. The orbit around M is 0.049788 1/A^2 (the
    verifier's value 1e-7 b below); the orbit past the transition is 0.017799."""
    b = 1 / 3.84652
    band_16 = _srvo3_band_16()

    areas, n_open = cross_section_areas(
        band_16, (0, 0, 1), (0, 0, (0.2195874 + 1e-7) * b), np.asarray(band_16.reciprocal_lattice)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([0.049788], rel=1e-3)


@pytest.mark.data
def test_srvo3_band_16_orbit_reaching_beyond_three_cells_closes():
    """The verifier's periodic reference (marching cubes translated 8-16 cells and sliced)
    finds one closed orbit of 0.234877 1/A^2; it reaches |k_frac| = 3.56."""
    normal = np.array([-0.273430119292263, 0.46320150330093784, -0.8430185865113354])
    band_16 = _srvo3_band_16()

    areas, n_open = cross_section_areas(
        band_16, normal, 0.04380787311312917 * normal, np.asarray(band_16.reciprocal_lattice)
    )

    assert n_open == 0
    assert np.asarray(areas) == pytest.approx([0.234877], rel=1e-3)


@pytest.mark.data
def test_srvo3_widget_does_not_say_it_snapped_an_exact_113_normal():
    text = _slice_text(_srvo3_band_16(), normal=(1, 1, 3), show_cross_section_area=True)

    assert text.endswith(" Ang^-2")


@pytest.mark.data
def test_gold_plane_through_grid_lines_counts_its_orbit_once():
    """(0.5, -0.2887, -0.8165) snaps to [1 0 -1]; the plane through Gamma then holds grid
    lines along b2. The orbit is 1.7729 Ang^-2, the same as 1e-9 off the plane."""
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/van-alphen", fermi=8.5642
    )
    band_5 = fs.select_bands([(5, 0)])
    assert isinstance(band_5, FermiSurface)

    text = _slice_text(band_5, normal=(0.5, -0.2887, -0.8165), show_cross_section_area=True)

    assert text == "Cross sectional area : 1.7729 Ang^-2 (normal snapped to [1 0 -1])"


@pytest.mark.data
def test_gold_open_orbits_near_110_stay_open():
    """3.5e-4 rad off [1 1 0], so not snapped. The verifier's periodic reference finds no
    closed orbit and 2 open orbits, still open 16 cells out."""
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/van-alphen", fermi=8.5642
    )
    band_5 = fs.select_bands([(5, 0)])
    assert isinstance(band_5, FermiSurface)
    normal = (0.866160402193743, 0.499766053396325, 0.00022258534303450556)
    origin = (-0.08530200352847554, -0.04924913473746253, 0.0)

    areas, n_open = cross_section_areas(
        band_5, normal, origin, np.asarray(band_5.reciprocal_lattice)
    )

    assert (areas, n_open) == ([], 2)


@pytest.mark.data
def test_srvo3_band_16_orbit_around_m_closes_across_the_zone_boundary():
    """SrVO3 band 16 at kz = 0 meets the zone boundary four times around M.

    The expected area is a matplotlib contour of the parsed band 16 energies on the
    exact i/21 mesh, tiled so the orbit around M = (1/2, 1/2, 0) lies inside one
    domain: 0.020664 1/A^2 without 2 pi.
    """
    fs = FermiSurface.from_code(
        code="vasp", dirpath=DATA_DIR / "examples/fermi3d/non-spin-polarized"
    )

    band_16 = fs.select_bands([(16, 0)])
    assert isinstance(band_16, FermiSurface)

    text = _slice_text(band_16, show_cross_section_area=True)

    assert text.endswith(" Ang^-2")
    assert _number(text) == pytest.approx(0.020664 * (2 * np.pi) ** 2, rel=0.01)


@pytest.mark.parametrize(
    ("flag", "text"),
    [
        ("show_cross_section_area", "Cross sectional area : 0.0000 Ang^-2"),
        ("show_van_alphen_frequency", "Van Alphen Frequency : no closed orbit through this cut"),
    ],
)
def test_moving_the_plane_to_an_empty_cut_clears_the_previous_result(flag, text):
    plotter = FermiPlotter(off_screen=True)
    plotter.add_box_slicer(_sphere(0.0), normal=(1, 0, 0), origin=(0, 0, 0), **{flag: True})
    widget = plotter.plane_widgets[0]

    widget.SetOrigin(0.32, 0.0, 0.0)
    widget.InvokeEvent("EndInteractionEvent")

    assert widget.GetOrigin()[0] > RADIUS
    assert _area_text(plotter) == text
    assert "slice" not in plotter.actors
    plotter.close()
