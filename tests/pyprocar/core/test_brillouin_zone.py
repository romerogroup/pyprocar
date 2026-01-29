"""
Test module for pyprocar.core.brillouin_zone module.

This module contains unit tests for the Lines, BrillouinZone,
and BrillouinZone2D classes.
"""

import numpy as np
import numpy.typing as npt
import pytest

from pyprocar.core.brillouin_zone import BrillouinZone, BrillouinZone2D, Lines


@pytest.fixture
def simple_cubic_reciprocal_lattice() -> npt.NDArray[np.float64]:
    """
    Create a simple cubic reciprocal lattice.

    Returns
    -------
    np.ndarray
        A 3x3 array representing a simple cubic reciprocal lattice
        with lattice parameter 2*pi.
    """
    return np.array([[2 * np.pi, 0.0, 0.0], [0.0, 2 * np.pi, 0.0], [0.0, 0.0, 2 * np.pi]])


@pytest.fixture
def fcc_reciprocal_lattice() -> npt.NDArray[np.float64]:
    """
    Create an FCC reciprocal lattice (BCC in reciprocal space).

    Returns
    -------
    np.ndarray
        A 3x3 array representing an FCC reciprocal lattice.
    """
    a = 2 * np.pi
    return np.array([[-a, a, a], [a, -a, a], [a, a, -a]])


@pytest.fixture
def hexagonal_reciprocal_lattice() -> npt.NDArray[np.float64]:
    """
    Create a hexagonal reciprocal lattice.

    Returns
    -------
    np.ndarray
        A 3x3 array representing a hexagonal reciprocal lattice.
    """
    a = 2 * np.pi / 3.0
    c = 2 * np.pi / 5.0
    return np.array([[a, a / np.sqrt(3), 0.0], [0.0, 2 * a / np.sqrt(3), 0.0], [0.0, 0.0, c]])


@pytest.fixture
def simple_verts() -> npt.NDArray[np.float64]:
    """Create simple vertices for Lines testing."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.fixture
def simple_faces() -> npt.NDArray[np.intp]:
    """Create simple faces for Lines testing (single square face)."""
    return np.array([[0, 1, 2, 3]], dtype=np.intp)


class TestLines:
    """Test class for Lines object."""

    def test_lines_initialization(
        self, simple_verts: npt.NDArray[np.float64], simple_faces: npt.NDArray[np.intp]
    ) -> None:
        """Test Lines initialization with vertices and faces."""
        lines = Lines(verts=simple_verts, faces=simple_faces)

        verts_val = lines.verts
        assert verts_val is not None
        assert np.allclose(verts_val, simple_verts)
        assert lines.faces is not None
        assert np.array_equal(lines.faces, simple_faces)

    def test_lines_nface_property(
        self, simple_verts: npt.NDArray[np.float64], simple_faces: npt.NDArray[np.intp]
    ) -> None:
        """Test nface property returns correct number of faces."""
        lines = Lines(verts=simple_verts, faces=simple_faces)

        assert lines.nface == 1

    def test_lines_multiple_faces(self, simple_verts: npt.NDArray[np.float64]) -> None:
        """Test Lines with multiple faces."""
        faces = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.intp)
        lines = Lines(verts=simple_verts, faces=faces)

        assert lines.nface == 3

    def test_lines_connectivity(
        self, simple_verts: npt.NDArray[np.float64], simple_faces: npt.NDArray[np.intp]
    ) -> None:
        """Test that connectivity is computed correctly."""
        lines = Lines(verts=simple_verts, faces=simple_faces)

        # Connectivity should include edges between consecutive points
        # plus closing edge from last to first
        assert len(lines.connectivity) > 0
        # Check that connectivity pairs are valid indices
        for conn in lines.connectivity:
            assert len(conn) == 2
            assert all(isinstance(idx, (int, np.integer)) for idx in conn)

    def test_lines_pyvista_line_initialized(
        self, simple_verts: npt.NDArray[np.float64], simple_faces: npt.NDArray[np.intp]
    ) -> None:
        """Test that PyVista line object is initialized."""
        lines = Lines(verts=simple_verts, faces=simple_faces)

        assert lines.pyvista_line is not None

    def test_lines_create_trimesh(
        self, simple_verts: npt.NDArray[np.float64], simple_faces: npt.NDArray[np.intp]
    ) -> None:
        """Test _create_trimesh method creates trimesh object."""
        lines = Lines(verts=simple_verts, faces=simple_faces)
        lines._create_trimesh()

        assert lines.trimesh_line is not None


class TestBrillouinZone:
    """Test class for BrillouinZone object."""

    def test_brillouin_zone_initialization_simple_cubic(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test BrillouinZone initialization with simple cubic lattice."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert bz.reciprocal is not None
        assert np.allclose(bz.reciprocal, simple_cubic_reciprocal_lattice)
        # Simple cubic BZ should have 6 faces (cube)
        assert bz.n_cells > 0

    def test_brillouin_zone_initialization_fcc(
        self, fcc_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test BrillouinZone initialization with FCC lattice."""
        bz = BrillouinZone(reciprocal_lattice=fcc_reciprocal_lattice)

        assert bz.reciprocal is not None
        # FCC reciprocal (BCC direct) should have more faces (truncated octahedron)
        assert bz.n_cells > 0

    def test_brillouin_zone_initialization_hexagonal(
        self, hexagonal_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test BrillouinZone initialization with hexagonal lattice."""
        bz = BrillouinZone(reciprocal_lattice=hexagonal_reciprocal_lattice)

        assert bz.reciprocal is not None
        assert bz.n_cells > 0

    def test_brillouin_zone_centers_property(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test centers property returns face centers."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        centers = bz.centers
        assert centers is not None
        assert centers.ndim == 2
        assert centers.shape[1] == 3  # 3D coordinates

    def test_brillouin_zone_faces_array_property(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test faces_array property reconstructs faces correctly."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        faces_array = bz.faces_array
        assert faces_array is not None
        assert len(faces_array) > 0
        # Each face should be a list starting with the number of vertices
        for face in faces_array:
            assert len(face) > 1
            assert face[0] == len(face) - 1  # First element is vertex count

    def test_brillouin_zone_wigner_seitz_method(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test wigner_seitz method returns valid vertices and faces."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        verts, faces = bz.wigner_seitz()

        assert verts is not None
        assert faces is not None
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert len(faces) > 0

    def test_brillouin_zone_is_pyvista_polydata(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that BrillouinZone is a PyVista PolyData object."""
        import pyvista as pv

        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert isinstance(bz, pv.PolyData)

    def test_brillouin_zone_has_points(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that BrillouinZone has points (vertices)."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert bz.n_points > 0

    def test_brillouin_zone_has_faces(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that BrillouinZone has faces."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        assert bz.n_cells > 0

    def test_brillouin_zone_face_normals_exist(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that face normals are computed."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        normals = bz.face_normals
        assert normals is not None
        assert normals.shape[0] == bz.n_cells

    def test_brillouin_zone_symmetry_cubic(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that cubic BZ has expected symmetry properties."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        centers = bz.centers
        # For a cubic BZ, face centers should be at equal distances from origin
        distances = np.linalg.norm(centers, axis=1)
        # Due to numerical precision, allow small tolerance
        assert np.allclose(distances, distances[0], rtol=1e-5)

    def test_brillouin_zone_centered_at_origin(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that BZ is centered at the origin."""
        bz = BrillouinZone(reciprocal_lattice=simple_cubic_reciprocal_lattice)

        # The mean of all vertices should be close to origin
        mean_point = np.mean(bz.points, axis=0)
        assert np.allclose(mean_point, [0, 0, 0], atol=1e-10)


class TestBrillouinZone2D:
    """Test class for BrillouinZone2D object."""

    def test_brillouin_zone_2d_initialization(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test BrillouinZone2D initialization."""
        e_min = -5.0
        e_max = 5.0

        bz2d = BrillouinZone2D(
            e_min=e_min, e_max=e_max, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        assert bz2d.reciprocal is not None
        assert bz2d.n_cells > 0

    def test_brillouin_zone_2d_axis_default(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test BrillouinZone2D with default axis (2 = z)."""
        e_min = -5.0
        e_max = 5.0

        bz2d = BrillouinZone2D(
            e_min=e_min, e_max=e_max, axis=2, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        # Check that z-coordinates are transformed to e_min and e_max
        z_coords = bz2d.points[:, 2]
        assert np.min(z_coords) >= e_min - 0.1
        assert np.max(z_coords) <= e_max + 0.1

    def test_brillouin_zone_2d_axis_x(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test BrillouinZone2D with axis=0 (x-axis)."""
        e_min = -3.0
        e_max = 3.0

        bz2d = BrillouinZone2D(
            e_min=e_min, e_max=e_max, axis=0, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        # Check that x-coordinates are transformed
        x_coords = bz2d.points[:, 0]
        assert np.min(x_coords) >= e_min - 0.1
        assert np.max(x_coords) <= e_max + 0.1

    def test_brillouin_zone_2d_axis_y(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test BrillouinZone2D with axis=1 (y-axis)."""
        e_min = -4.0
        e_max = 4.0

        bz2d = BrillouinZone2D(
            e_min=e_min, e_max=e_max, axis=1, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        # Check that y-coordinates are transformed
        y_coords = bz2d.points[:, 1]
        assert np.min(y_coords) >= e_min - 0.1
        assert np.max(y_coords) <= e_max + 0.1

    def test_brillouin_zone_2d_centers_property(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test centers property for BrillouinZone2D."""
        bz2d = BrillouinZone2D(
            e_min=-5.0, e_max=5.0, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        centers = bz2d.centers
        assert centers is not None
        assert centers.ndim == 2
        assert centers.shape[1] == 3

    def test_brillouin_zone_2d_faces_array_property(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test faces_array property for BrillouinZone2D."""
        bz2d = BrillouinZone2D(
            e_min=-5.0, e_max=5.0, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        faces_array = bz2d.faces_array
        assert faces_array is not None
        assert len(faces_array) > 0

    def test_brillouin_zone_2d_wigner_seitz_method(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test wigner_seitz method for BrillouinZone2D."""
        bz2d = BrillouinZone2D(
            e_min=-5.0, e_max=5.0, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        verts, faces = bz2d.wigner_seitz()

        assert verts is not None
        assert faces is not None
        assert verts.ndim == 2
        assert verts.shape[1] == 3

    def test_brillouin_zone_2d_is_pyvista_polydata(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that BrillouinZone2D is a PyVista PolyData object."""
        import pyvista as pv

        bz2d = BrillouinZone2D(
            e_min=-5.0, e_max=5.0, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        assert isinstance(bz2d, pv.PolyData)

    def test_brillouin_zone_2d_energy_range_transformation(
        self, simple_cubic_reciprocal_lattice: npt.NDArray[np.float64]
    ) -> None:
        """Test that energy range is correctly applied."""
        e_min = -10.0
        e_max = 10.0

        bz2d = BrillouinZone2D(
            e_min=e_min, e_max=e_max, axis=2, reciprocal_lattice=simple_cubic_reciprocal_lattice
        )

        z_coords = bz2d.points[:, 2]
        # Vertices at extreme z should be at e_min or e_max
        _z_min = np.min(z_coords)
        _z_max = np.max(z_coords)

        # Allow for numerical tolerance
        assert np.isclose(_z_min, e_min, atol=1e-2) or _z_min >= e_min
        assert np.isclose(_z_max, e_max, atol=1e-2) or _z_max <= e_max

    def test_brillouin_zone_2d_different_lattices(
        self,
        simple_cubic_reciprocal_lattice: npt.NDArray[np.float64],
        fcc_reciprocal_lattice: npt.NDArray[np.float64],
        hexagonal_reciprocal_lattice: npt.NDArray[np.float64],
    ) -> None:
        """Test BrillouinZone2D works with different lattice types."""
        e_min = -5.0
        e_max = 5.0

        lattices: list[npt.NDArray[np.float64]] = [
            simple_cubic_reciprocal_lattice,
            fcc_reciprocal_lattice,
            hexagonal_reciprocal_lattice,
        ]
        for lattice in lattices:
            bz2d = BrillouinZone2D(e_min=e_min, e_max=e_max, reciprocal_lattice=lattice)
            assert bz2d.n_cells > 0
            assert bz2d.n_points > 0
