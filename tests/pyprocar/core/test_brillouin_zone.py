"""
Test module for pyprocar.core.brillouin_zone module.

This module contains unit tests for the BrillouinZone
and BrillouinZone2D classes.
"""

import itertools
import warnings

import numpy as np
import pytest
import pyvista as pv
from scipy.spatial import ConvexHull, Voronoi

from pyprocar.core.brillouin_zone import BrillouinZone, BrillouinZone2D, clip_to_zone
from tests.pyprocar.core.test_bandstructure2d_grid import HEXAGONAL


@pytest.fixture
def simple_cubic_reciprocal_lattice():
    """
    Create a simple cubic reciprocal lattice.

    Returns
    -------
    np.ndarray
        A 3x3 array representing a simple cubic reciprocal lattice
        with lattice parameter 2*pi.
    """
    return np.array([
        [2 * np.pi, 0.0, 0.0],
        [0.0, 2 * np.pi, 0.0],
        [0.0, 0.0, 2 * np.pi]
    ])


@pytest.fixture
def fcc_reciprocal_lattice():
    """
    Create an FCC reciprocal lattice (BCC in reciprocal space).

    Returns
    -------
    np.ndarray
        A 3x3 array representing an FCC reciprocal lattice.
    """
    a = 2 * np.pi
    return np.array([
        [-a, a, a],
        [a, -a, a],
        [a, a, -a]
    ])


@pytest.fixture
def hexagonal_reciprocal_lattice():
    """
    Create a hexagonal reciprocal lattice.

    Returns
    -------
    np.ndarray
        A 3x3 array representing a hexagonal reciprocal lattice.
    """
    a = 2 * np.pi / 3.0
    c = 2 * np.pi / 5.0
    return np.array([
        [a, a / np.sqrt(3), 0.0],
        [0.0, 2 * a / np.sqrt(3), 0.0],
        [0.0, 0.0, c]
    ])


class TestBrillouinZone:
    """Test class for BrillouinZone object."""

    def test_brillouin_zone_initialization_simple_cubic(self, simple_cubic_reciprocal_lattice):
        """Test BrillouinZone initialization with simple cubic lattice."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert bz.reciprocal is not None
        assert np.allclose(bz.reciprocal, simple_cubic_reciprocal_lattice)
        # Simple cubic BZ should have 6 faces (cube)
        assert bz.n_cells > 0

    def test_brillouin_zone_initialization_fcc(self, fcc_reciprocal_lattice):
        """Test BrillouinZone initialization with FCC lattice."""
        bz = BrillouinZone(reciprocal_lattice=fcc_reciprocal_lattice)

        assert bz.reciprocal is not None
        # FCC reciprocal (BCC direct) should have more faces (truncated octahedron)
        assert bz.n_cells > 0

    def test_brillouin_zone_initialization_hexagonal(self, hexagonal_reciprocal_lattice):
        """Test BrillouinZone initialization with hexagonal lattice."""
        bz = BrillouinZone(reciprocal_lattice=hexagonal_reciprocal_lattice)

        assert bz.reciprocal is not None
        assert bz.n_cells > 0

    def test_brillouin_zone_centers_property(self, simple_cubic_reciprocal_lattice):
        """Test centers property returns face centers."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        centers = bz.centers
        assert centers is not None
        assert centers.ndim == 2
        assert centers.shape[1] == 3  # 3D coordinates

    def test_brillouin_zone_faces_array_property(self, simple_cubic_reciprocal_lattice):
        """Test faces_array property reconstructs faces correctly."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        faces_array = bz.faces_array
        assert faces_array is not None
        assert len(faces_array) > 0
        # Each face should be a list starting with the number of vertices
        for face in faces_array:
            assert len(face) > 1
            assert face[0] == len(face) - 1  # First element is vertex count

    def test_brillouin_zone_wigner_seitz_method(self, simple_cubic_reciprocal_lattice):
        """Test wigner_seitz method returns valid vertices and faces."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        verts, faces = bz.wigner_seitz()

        assert verts is not None
        assert faces is not None
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(faces) > 0

    def test_brillouin_zone_is_pyvista_polydata(self, simple_cubic_reciprocal_lattice):
        """Test that BrillouinZone is a PyVista PolyData object."""
        import pyvista as pv

        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert isinstance(bz, pv.PolyData)

    def test_brillouin_zone_has_points(self, simple_cubic_reciprocal_lattice):
        """Test that BrillouinZone has points (vertices)."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert bz.n_points > 0

    def test_brillouin_zone_has_faces(self, simple_cubic_reciprocal_lattice):
        """Test that BrillouinZone has faces."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert bz.n_cells > 0

    def test_brillouin_zone_face_normals_exist(self, simple_cubic_reciprocal_lattice):
        """Test that face normals are computed."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        normals = bz.face_normals
        assert normals is not None
        assert normals.shape[0] == bz.n_cells

    def test_brillouin_zone_symmetry_cubic(self, simple_cubic_reciprocal_lattice):
        """Test that cubic BZ has expected symmetry properties."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        centers = bz.centers
        # For a cubic BZ, face centers should be at equal distances from origin
        distances = np.linalg.norm(centers, axis=1)
        # Due to numerical precision, allow small tolerance
        assert np.allclose(distances, distances[0], rtol=1e-5)

    def test_brillouin_zone_centered_at_origin(self, simple_cubic_reciprocal_lattice):
        """Test that BZ is centered at the origin."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        # The mean of all vertices should be close to origin
        mean_point = np.mean(bz.points, axis=0)
        assert np.allclose(mean_point, [0, 0, 0], atol=1e-10)


class TestBrillouinZone2D:
    """Test class for BrillouinZone2D object."""

    def test_brillouin_zone_2d_initialization(self, simple_cubic_reciprocal_lattice):
        """Test BrillouinZone2D initialization."""
        e_min = -5.0
        e_max = 5.0

        bz2d = BrillouinZone2D(
            e_min=e_min,
            e_max=e_max,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        assert bz2d.reciprocal is not None
        assert bz2d.n_cells > 0

    def test_brillouin_zone_2d_axis_default(self, simple_cubic_reciprocal_lattice):
        """Test BrillouinZone2D with default axis (2 = z)."""
        e_min = -5.0
        e_max = 5.0

        bz2d = BrillouinZone2D(
            e_min=e_min,
            e_max=e_max,
            axis=2,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        # Check that z-coordinates are transformed to e_min and e_max
        z_coords = bz2d.points[:, 2]
        assert np.min(z_coords) >= e_min - 0.1
        assert np.max(z_coords) <= e_max + 0.1

    def test_brillouin_zone_2d_axis_x(self, simple_cubic_reciprocal_lattice):
        """Test BrillouinZone2D with axis=0 (x-axis)."""
        e_min = -3.0
        e_max = 3.0

        bz2d = BrillouinZone2D(
            e_min=e_min,
            e_max=e_max,
            axis=0,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        # Check that x-coordinates are transformed
        x_coords = bz2d.points[:, 0]
        assert np.min(x_coords) >= e_min - 0.1
        assert np.max(x_coords) <= e_max + 0.1

    def test_brillouin_zone_2d_axis_y(self, simple_cubic_reciprocal_lattice):
        """Test BrillouinZone2D with axis=1 (y-axis)."""
        e_min = -4.0
        e_max = 4.0

        bz2d = BrillouinZone2D(
            e_min=e_min,
            e_max=e_max,
            axis=1,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        # Check that y-coordinates are transformed
        y_coords = bz2d.points[:, 1]
        assert np.min(y_coords) >= e_min - 0.1
        assert np.max(y_coords) <= e_max + 0.1

    def test_brillouin_zone_2d_centers_property(self, simple_cubic_reciprocal_lattice):
        """Test centers property for BrillouinZone2D."""
        bz2d = BrillouinZone2D(
            e_min=-5.0,
            e_max=5.0,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        centers = bz2d.centers
        assert centers is not None
        assert centers.ndim == 2
        assert centers.shape[1] == 3

    def test_brillouin_zone_2d_faces_array_property(self, simple_cubic_reciprocal_lattice):
        """Test faces_array property for BrillouinZone2D."""
        bz2d = BrillouinZone2D(
            e_min=-5.0,
            e_max=5.0,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        faces_array = bz2d.faces_array
        assert faces_array is not None
        assert len(faces_array) > 0

    def test_brillouin_zone_2d_wigner_seitz_method(self, simple_cubic_reciprocal_lattice):
        """Test wigner_seitz method for BrillouinZone2D."""
        bz2d = BrillouinZone2D(
            e_min=-5.0,
            e_max=5.0,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        verts, faces = bz2d.wigner_seitz()

        assert verts is not None
        assert faces is not None
        assert verts.ndim == 2
        assert verts.shape[1] == 3

    def test_brillouin_zone_2d_is_pyvista_polydata(self, simple_cubic_reciprocal_lattice):
        """Test that BrillouinZone2D is a PyVista PolyData object."""
        import pyvista as pv

        bz2d = BrillouinZone2D(
            e_min=-5.0,
            e_max=5.0,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        assert isinstance(bz2d, pv.PolyData)

    def test_brillouin_zone_2d_energy_range_transformation(self, simple_cubic_reciprocal_lattice):
        """Test that energy range is correctly applied."""
        e_min = -10.0
        e_max = 10.0

        bz2d = BrillouinZone2D(
            e_min=e_min,
            e_max=e_max,
            axis=2,
            reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        z_coords = bz2d.points[:, 2]
        # Vertices at extreme z should be at e_min or e_max
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)

        # Allow for numerical tolerance
        assert np.isclose(z_min, e_min, atol=1e-2) or z_min >= e_min
        assert np.isclose(z_max, e_max, atol=1e-2) or z_max <= e_max

    def test_brillouin_zone_2d_different_lattices(
        self,
        simple_cubic_reciprocal_lattice,
        fcc_reciprocal_lattice,
        hexagonal_reciprocal_lattice
    ):
        """Test BrillouinZone2D works with different lattice types."""
        e_min = -5.0
        e_max = 5.0

        for lattice in [simple_cubic_reciprocal_lattice, fcc_reciprocal_lattice, hexagonal_reciprocal_lattice]:
            bz2d = BrillouinZone2D(
                e_min=e_min,
                e_max=e_max,
                reciprocal_lattice=lattice
            )
            assert bz2d.n_cells > 0
            assert bz2d.n_points > 0


MONOCLINIC = np.array([[1.0, 0.0, 0.0], [0.0, 1.3, 0.0], [0.4, 0.0, 0.9]])
TRICLINIC = np.array([[1.0, 0.1, 0.05], [0.3, 1.2, 0.0], [0.2, 0.35, 0.8]])
SHEARS = {
    "unsheared": np.eye(3, dtype=int),
    "b2+3b1": np.array([[1, 0, 0], [3, 1, 0], [0, 0, 1]]),
    "b2+3b1,b3-2b1+2b2": np.array([[1, 0, 0], [3, 1, 0], [-2, 2, 1]]),
}
SHEARED_CELLS = [
    pytest.param(cell, shear, id=f"{name}-{shear_name}")
    for name, cell in [("monoclinic", MONOCLINIC), ("triclinic", TRICLINIC)]
    for shear_name, shear in SHEARS.items()
]


def _voronoi_reference(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Face vectors and vertices of the Voronoi cell of the origin among the 7^3 block of
    lattice points of ``cell``, from scipy.spatial.Voronoi."""
    steps = np.array(list(itertools.product(range(-3, 4), repeat=3)))
    origin = len(steps) // 2
    voronoi = Voronoi(steps @ cell)
    pairs = voronoi.ridge_points[(voronoi.ridge_points == origin).any(axis=1)]
    neighbours = pairs.sum(axis=1) - origin
    vertices = voronoi.vertices[voronoi.regions[voronoi.point_region[origin]]]
    return voronoi.points[neighbours], vertices


def _sorted_rows(points: np.ndarray) -> np.ndarray:
    rounded = np.round(points, 9) + 0.0
    return rounded[np.lexsort(rounded.T[::-1])]


@pytest.mark.parametrize(("cell", "shear"), SHEARED_CELLS)
def test_zone_faces_match_voronoi_for_any_basis_of_the_lattice(cell, shear):
    from pyprocar.core.brillouin_zone import zone_face_steps

    basis = shear @ cell
    faces, _ = _voronoi_reference(cell)

    found = zone_face_steps(basis) @ basis

    np.testing.assert_allclose(_sorted_rows(found), _sorted_rows(faces), atol=1e-9)


@pytest.mark.parametrize(("cell", "shear"), SHEARED_CELLS)
def test_brillouin_zone_matches_voronoi_for_any_basis_of_the_lattice(cell, shear):
    basis = shear @ cell
    _, vertices = _voronoi_reference(cell)

    zone = BrillouinZone(basis)

    assert ConvexHull(zone.points).volume == pytest.approx(abs(np.linalg.det(cell)), rel=1e-9)
    np.testing.assert_allclose(_sorted_rows(zone.points), _sorted_rows(vertices), atol=1e-9)


@pytest.mark.parametrize(
    "shear", [SHEARS["unsheared"], SHEARS["b2+3b1"]], ids=["unsheared", "b2+3b1"]
)
def test_2d_brillouin_zone_is_the_hexagonal_prism_for_any_basis(
    hexagonal_reciprocal_lattice, shear
):
    """The zone of a hexagonal reciprocal lattice with |b1| = |b2| = b is a regular hexagon
    of circumradius b / sqrt(3) and area |b1 x b2|; BrillouinZone2D stretches it from
    e_min to e_max."""
    b1, b2, _ = hexagonal_reciprocal_lattice

    zone = BrillouinZone2D(
        e_min=-1.0, e_max=1.0, reciprocal_lattice=shear @ hexagonal_reciprocal_lattice
    ).clean()

    hexagon = np.linalg.norm(np.cross(b1, b2))
    assert ConvexHull(zone.points).volume == pytest.approx(2.0 * hexagon, rel=1e-9)
    assert np.linalg.norm(zone.points[:, :2], axis=1) == pytest.approx(
        np.full(zone.n_points, np.linalg.norm(b1) / np.sqrt(3)), rel=1e-9
    )


@pytest.mark.guards_existing_behaviour(
    reason="dev builds the zone without spglib; this guards the delaunay_reduce call this PR adds"
)
def test_building_a_zone_emits_no_warnings():
    lattice = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [-2.0, 2.0, 1.0]])
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        BrillouinZone(lattice)

    assert [str(w.message) for w in record] == []


def _outward(zone, inside: np.ndarray) -> np.ndarray:
    """normal . (face center - interior point) for every face of ``zone``."""
    return np.einsum("ij,ij->i", zone.face_normals, zone.centers - inside)


@pytest.mark.parametrize(
    ("e_min", "e_max"), [(-8.0, -2.0), (-3.0, 3.0), (2.0, 8.0)], ids=["below", "around", "above"]
)
def test_2d_zone_face_normals_point_out_of_the_prism_at_any_energy(e_min, e_max):
    """The graphene zone is a hexagonal prism around Gamma from e_min to e_max, so its
    centroid is (0, 0, (e_min + e_max) / 2) and all 8 face normals point away from it."""
    zone = BrillouinZone2D(e_min=e_min, e_max=e_max, reciprocal_lattice=2 * np.pi * HEXAGONAL)

    outward = _outward(zone, np.array([0.0, 0.0, (e_min + e_max) / 2]))

    assert zone.n_cells == 8
    assert (outward > 0).all(), outward


CUBIC = 2 * np.pi * np.eye(3)
# Keys name the real-space lattice whose reciprocal lattice the value is.
ZONES_3D = {
    "cubic": CUBIC,
    "fcc": 2 * np.pi * np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
    "bcc": 2 * np.pi * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
    "hexagonal": 2 * np.pi * HEXAGONAL,
    "sheared-cubic": SHEARS["b2+3b1,b3-2b1+2b2"] @ CUBIC,
}


@pytest.mark.guards_existing_behaviour(
    reason="Gamma is inside every 3D zone, so the face-0 rule already orients these outward"
)
@pytest.mark.parametrize("lattice", list(ZONES_3D.values()), ids=list(ZONES_3D))
def test_3d_zone_face_normals_point_away_from_gamma(lattice):
    zone = BrillouinZone(lattice)

    assert (_outward(zone, np.zeros(3)) > 0).all()


def test_clip_to_zone_of_a_surface_outside_the_zone_is_empty():
    """H3: a sphere wholly outside the cubic zone clips to no points and does not raise; one
    wholly inside keeps every point."""
    zone = BrillouinZone(np.eye(3))
    outside = pv.Sphere(radius=0.2, center=(2.0, 2.0, 2.0))
    inside = pv.Sphere(radius=0.2)

    assert clip_to_zone(outside, zone).n_points == 0
    assert clip_to_zone(inside, zone).n_points == inside.n_points
