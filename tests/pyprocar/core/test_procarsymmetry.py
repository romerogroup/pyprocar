"""Tests for ProcarSymmetry class."""

import numpy as np
import numpy.typing as npt
import pytest

from pyprocar.core.procarsymmetry import ProcarSymmetry


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_kpoints() -> npt.NDArray[np.float64]:
    """Simple k-points for testing."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ]
    )


@pytest.fixture
def simple_bands(simple_kpoints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Simple bands data matching k-points."""
    n_kpoints = len(simple_kpoints)
    n_bands = 4
    return np.arange(n_kpoints * n_bands).reshape(n_kpoints, n_bands).astype(float)


@pytest.fixture
def simple_character(
    simple_kpoints: npt.NDArray[np.float64], simple_bands: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Simple character data matching k-points and bands."""
    return np.ones((len(simple_kpoints), simple_bands.shape[1]))


@pytest.fixture
def spin_components(
    simple_kpoints: npt.NDArray[np.float64], simple_bands: npt.NDArray[np.float64]
) -> dict[str, npt.NDArray[np.float64]]:
    """Spin vector components for testing."""
    shape = (len(simple_kpoints), simple_bands.shape[1])
    return {
        "sx": np.ones(shape) * 0.5,
        "sy": np.ones(shape) * 0.3,
        "sz": np.ones(shape) * 0.4,
    }


class TestProcarSymmetryInit:
    """Test ProcarSymmetry initialization."""

    def test_init_minimal(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test initialization with only required parameters."""
        ps = ProcarSymmetry(kpoints=simple_kpoints, bands=simple_bands)

        assert np.allclose(ps.kpoints, simple_kpoints)
        assert np.allclose(ps.bands, simple_bands)
        assert ps.character.shape == (0,)
        assert ps.sx.shape == (0,)
        assert ps.sy.shape == (0,)
        assert ps.sz.shape == (0,)

    def test_init_with_character(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
        simple_character: npt.NDArray[np.float64],
    ) -> None:
        """Test initialization with character data."""
        ps = ProcarSymmetry(
            kpoints=simple_kpoints,
            bands=simple_bands,
            character=simple_character,
        )

        assert np.allclose(ps.character, simple_character)

    def test_init_with_spin_components(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
        spin_components: dict[str, npt.NDArray[np.float64]],
    ) -> None:
        """Test initialization with spin components."""
        ps = ProcarSymmetry(
            kpoints=simple_kpoints,
            bands=simple_bands,
            sx=spin_components["sx"],
            sy=spin_components["sy"],
            sz=spin_components["sz"],
        )

        assert np.allclose(ps.sx, spin_components["sx"])
        assert np.allclose(ps.sy, spin_components["sy"])
        assert np.allclose(ps.sz, spin_components["sz"])


class TestQuaternionMultiplication:
    """Test quaternion multiplication helper."""

    def test_q_mult_identity(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test multiplication with identity quaternion."""
        ps = ProcarSymmetry(kpoints=simple_kpoints, bands=simple_bands)

        # Identity quaternion: (1, 0, 0, 0)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.5, 0.1, 0.2, 0.3])

        result = ps._q_mult(identity, q)

        assert np.allclose(result, q)

    def test_q_mult_inverse(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test multiplication with inverse quaternion."""
        ps = ProcarSymmetry(kpoints=simple_kpoints, bands=simple_bands)

        # Unit quaternion
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_inv = np.array([0.5, -0.5, -0.5, -0.5])

        result = ps._q_mult(q, q_inv)

        # Should give identity (1, 0, 0, 0)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(result, expected, atol=1e-10)

    def test_q_mult_associative(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test quaternion multiplication is associative."""
        ps = ProcarSymmetry(kpoints=simple_kpoints, bands=simple_bands)

        q1 = np.array([0.5, 0.1, 0.2, 0.3])
        q2 = np.array([0.6, 0.2, 0.1, 0.4])
        q3 = np.array([0.4, 0.3, 0.5, 0.1])

        # (q1 * q2) * q3 should equal q1 * (q2 * q3)
        result1 = ps._q_mult(ps._q_mult(q1, q2), q3)
        result2 = ps._q_mult(q1, ps._q_mult(q2, q3))

        assert np.allclose(result1, result2)


class TestGeneralRotation:
    """Test general_rotation method."""

    def test_rotation_zero_angle(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test rotation by zero angle returns original kpoints."""
        ps = ProcarSymmetry(kpoints=simple_kpoints.copy(), bands=simple_bands)

        _kpoints, _sx, _sy, _sz = ps.general_rotation(angle=0, rotAxis="z", store=False)

        assert np.allclose(_kpoints, simple_kpoints, atol=1e-10)

    def test_rotation_360_degrees(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test rotation by 360 degrees returns original kpoints."""
        ps = ProcarSymmetry(kpoints=simple_kpoints.copy(), bands=simple_bands)

        _kpoints, _sx, _sy, _sz = ps.general_rotation(angle=360, rotAxis="z", store=False)

        assert np.allclose(_kpoints, simple_kpoints, atol=1e-10)

    def test_rotation_90_degrees_z_axis(
        self,
    ) -> None:
        """Test 90 degree rotation around z-axis."""
        # Point at (1, 0, 0) should rotate to (0, 1, 0)
        kpoints = np.array([[1.0, 0.0, 0.0]])
        bands = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands)
        new_kpoints, _sx, _sy, _sz = ps.general_rotation(angle=90, rotAxis="z", store=False)

        expected = np.array([[0.0, 1.0, 0.0]])
        assert np.allclose(new_kpoints, expected, atol=1e-10)

    def test_rotation_90_degrees_x_axis(
        self,
    ) -> None:
        """Test 90 degree rotation around x-axis."""
        # Point at (0, 1, 0) should rotate to (0, 0, 1)
        kpoints = np.array([[0.0, 1.0, 0.0]])
        bands = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands)
        new_kpoints, _sx, _sy, _sz = ps.general_rotation(angle=90, rotAxis="x", store=False)

        expected = np.array([[0.0, 0.0, 1.0]])
        assert np.allclose(new_kpoints, expected, atol=1e-10)

    def test_rotation_90_degrees_y_axis(
        self,
    ) -> None:
        """Test 90 degree rotation around y-axis."""
        # Point at (0, 0, 1) should rotate to (1, 0, 0)
        kpoints = np.array([[0.0, 0.0, 1.0]])
        bands = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands)
        new_kpoints, _sx, _sy, _sz = ps.general_rotation(angle=90, rotAxis="y", store=False)

        expected = np.array([[1.0, 0.0, 0.0]])
        assert np.allclose(new_kpoints, expected, atol=1e-10)

    def test_rotation_custom_axis(
        self,
    ) -> None:
        """Test rotation around custom axis."""
        kpoints = np.array([[1.0, 0.0, 0.0]])
        bands = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands)
        new_kpoints, _sx, _sy, _sz = ps.general_rotation(angle=180, rotAxis=[1, 1, 0], store=False)

        # 180 rotation around (1,1,0) axis: (1,0,0) -> (0,1,0)
        expected = np.array([[0.0, 1.0, 0.0]])
        assert np.allclose(new_kpoints, expected, atol=1e-10)

    def test_rotation_store_true(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
        spin_components: dict[str, npt.NDArray[np.float64]],
    ) -> None:
        """Test that store=True updates internal state."""
        ps = ProcarSymmetry(
            kpoints=simple_kpoints.copy(),
            bands=simple_bands,
            sx=spin_components["sx"],
            sy=spin_components["sy"],
            sz=spin_components["sz"],
        )
        original_kpoints = ps.kpoints.copy()

        ps.general_rotation(angle=90, rotAxis="z", store=True)

        assert not np.allclose(ps.kpoints, original_kpoints)

    def test_rotation_store_false(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test that store=False does not modify internal state."""
        ps = ProcarSymmetry(kpoints=simple_kpoints.copy(), bands=simple_bands)
        original_kpoints = ps.kpoints.copy()

        ps.general_rotation(angle=90, rotAxis="z", store=False)

        assert np.allclose(ps.kpoints, original_kpoints)

    def test_rotation_with_spin_vectors(
        self,
    ) -> None:
        """Test that spin vectors are also rotated."""
        # Spin pointing in x direction
        kpoints = np.array([[0.0, 0.0, 0.0]])
        bands = np.array([[1.0]])
        sx = np.array([[1.0]])
        sy = np.array([[0.0]])
        sz = np.array([[0.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, sx=sx, sy=sy, sz=sz)

        # 90 degree rotation around z: sx -> sy
        _new_kpoints, new_sx, new_sy, new_sz = ps.general_rotation(
            angle=90, rotAxis="z", store=False
        )

        assert np.allclose(new_sx, [[0.0]], atol=1e-10)
        assert np.allclose(new_sy, [[1.0]], atol=1e-10)
        assert np.allclose(new_sz, [[0.0]], atol=1e-10)


class TestRotSymmetryZ:
    """Test rot_symmetry_z method."""

    def test_rot_symmetry_z_order_2(
        self,
    ) -> None:
        """Test 2-fold rotational symmetry."""
        kpoints = np.array([[1.0, 0.0, 0.0]])
        bands = np.array([[1.0]])
        character = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character)
        ps.rot_symmetry_z(order=2)

        # Should have 2 k-points now
        assert ps.kpoints.shape[0] == 2
        assert ps.bands.shape[0] == 2
        assert ps.character.shape[0] == 2

        # Second point should be rotated 180 degrees
        expected_second = np.array([-1.0, 0.0, 0.0])
        assert np.allclose(ps.kpoints[1], expected_second, atol=1e-10)

    def test_rot_symmetry_z_order_4(
        self,
    ) -> None:
        """Test 4-fold rotational symmetry."""
        kpoints = np.array([[1.0, 0.0, 0.0]])
        bands = np.array([[1.0]])
        character = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character)
        ps.rot_symmetry_z(order=4)

        # Should have 4 k-points now
        assert ps.kpoints.shape[0] == 4

        # Check all rotated points
        expected = np.array(
            [
                [1.0, 0.0, 0.0],  # 0 degrees
                [0.0, 1.0, 0.0],  # 90 degrees
                [-1.0, 0.0, 0.0],  # 180 degrees
                [0.0, -1.0, 0.0],  # 270 degrees
            ]
        )
        assert np.allclose(ps.kpoints, expected, atol=1e-10)

    def test_rot_symmetry_z_order_6(
        self,
    ) -> None:
        """Test 6-fold rotational symmetry."""
        kpoints = np.array([[1.0, 0.0, 0.0]])
        bands = np.array([[1.0]])
        character = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character)
        ps.rot_symmetry_z(order=6)

        # Should have 6 k-points
        assert ps.kpoints.shape[0] == 6

    def test_rot_symmetry_z_with_spin(
        self,
    ) -> None:
        """Test rotational symmetry also rotates spin components."""
        kpoints = np.array([[1.0, 0.0, 0.0]])
        bands = np.array([[1.0]])
        character = np.array([[1.0]])
        sx = np.array([[1.0]])
        sy = np.array([[0.0]])
        sz = np.array([[0.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character, sx=sx, sy=sy, sz=sz)
        ps.rot_symmetry_z(order=4)

        assert ps.sx.shape[0] == 4
        assert ps.sy.shape[0] == 4
        assert ps.sz.shape[0] == 4


class TestMirrorX:
    """Test mirror_x method."""

    def test_mirror_x_kpoints(
        self,
    ) -> None:
        """Test mirror operation on k-points."""
        kpoints = np.array([[1.0, 2.0, 3.0]])
        bands = np.array([[1.0]])
        character = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character)
        ps.mirror_x()

        # Should have 2 k-points now
        assert ps.kpoints.shape[0] == 2

        # Second point should have y negated
        expected = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, -2.0, 3.0],
            ]
        )
        assert np.allclose(ps.kpoints, expected)

    def test_mirror_x_spin_components(self) -> None:
        """Test mirror operation on spin components."""
        kpoints = np.array([[0.0, 0.0, 0.0]])
        bands = np.array([[1.0]])
        character = np.array([[1.0]])
        sx = np.array([[1.0]])
        sy = np.array([[2.0]])
        sz = np.array([[3.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character, sx=sx, sy=sy, sz=sz)
        ps.mirror_x()

        # sx should be negated for mirrored point
        assert np.allclose(ps.sx[0], 1.0)
        assert np.allclose(ps.sx[1], -1.0)

        # sy and sz should remain same
        assert np.allclose(ps.sy[0], 2.0)
        assert np.allclose(ps.sy[1], 2.0)
        assert np.allclose(ps.sz[0], 3.0)
        assert np.allclose(ps.sz[1], 3.0)

    def test_mirror_x_bands_duplicated(
        self,
    ) -> None:
        """Test that bands and character are duplicated."""
        kpoints = np.array([[0.0, 0.0, 0.0]])
        bands = np.array([[1.0, 2.0]])
        character = np.array([[0.5, 0.5]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character)
        ps.mirror_x()

        assert ps.bands.shape[0] == 2
        assert np.allclose(ps.bands[0], ps.bands[1])
        assert ps.character.shape[0] == 2
        assert np.allclose(ps.character[0], ps.character[1])


class TestTranslate:
    """Test translate method."""

    def test_translate_by_coordinates(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test translation by coordinate vector."""
        ps = ProcarSymmetry(kpoints=simple_kpoints.copy(), bands=simple_bands)

        ps.translate(newOrigin=[0.5, 0.0, 0.0])

        # All k-points should be shifted
        expected = simple_kpoints - np.array([0.5, 0.0, 0.0])
        assert np.allclose(ps.kpoints, expected)

    def test_translate_by_index(
        self,
        simple_kpoints: npt.NDArray[np.float64],
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test translation using k-point index."""
        ps = ProcarSymmetry(kpoints=simple_kpoints.copy(), bands=simple_bands)

        # Translate to origin at k-point index 1
        ps.translate(newOrigin=[1])

        # K-point at index 1 should now be at origin
        assert np.allclose(ps.kpoints[1], [0.0, 0.0, 0.0])

    def test_translate_to_gamma(
        self,
        simple_bands: npt.NDArray[np.float64],
    ) -> None:
        """Test translation to Gamma point."""
        kpoints = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.5, 0.5, 0.5],
            ]
        )

        ps = ProcarSymmetry(kpoints=kpoints.copy(), bands=simple_bands[:2])

        # Translate so first point is at origin
        ps.translate(newOrigin=[0])

        assert np.allclose(ps.kpoints[0], [0.0, 0.0, 0.0])
        assert np.allclose(ps.kpoints[1], [0.4, 0.3, 0.2])


class TestProcarSymmetryIntegration:
    """Integration tests combining multiple operations."""

    def test_rotation_then_mirror(self) -> None:
        """Test combining rotation and mirror operations."""
        kpoints = np.array([[1.0, 0.0, 0.0]])
        bands = np.array([[1.0]])
        character = np.array([[1.0]])

        ps = ProcarSymmetry(kpoints=kpoints, bands=bands, character=character)

        # 4-fold rotation gives 4 points
        ps.rot_symmetry_z(order=4)
        assert ps.kpoints.shape[0] == 4

        # Mirror doubles to 8 points
        ps.mirror_x()
        assert ps.kpoints.shape[0] == 8

    def test_translate_then_rotate(self) -> None:
        """Test translation followed by rotation."""
        kpoints = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        bands = np.array([[1.0], [2.0]])
        character = np.array([[1.0], [1.0]])

        ps = ProcarSymmetry(kpoints=kpoints.copy(), bands=bands, character=character)

        # Translate to center at (1.5, 0, 0)
        ps.translate(newOrigin=[1.5, 0.0, 0.0])

        # Now points are at (-0.5, 0, 0) and (0.5, 0, 0)
        # 2-fold rotation should give symmetric points
        ps.rot_symmetry_z(order=2)

        assert ps.kpoints.shape[0] == 4
