"""Tests for ProcarSelect class."""

import numpy as np
import pytest


class MockProcarData:
    """Mock ProcarParser-like object for testing."""

    def __init__(
        self,
        n_kpoints: int = 10,
        n_bands: int = 8,
        n_spins: int = 2,
        n_atoms: int = 4,
        n_orbitals: int = 9,
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng(42)

        # spd shape: [kpoints, bands, spins, atoms+1, orbitals+2]
        # atoms+1 for total, orbitals+2 for atom number and total
        self.spd = rng.random((n_kpoints, n_bands, n_spins, n_atoms + 1, n_orbitals + 2))
        self.bands = rng.random((n_kpoints, n_bands))
        self.kpoints = rng.random((n_kpoints, 3))


@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def mock_procar_data(rng):
    """Create mock ProcarData with default dimensions."""
    return MockProcarData(rng=rng)


@pytest.fixture
def mock_procar_data_single_spin(rng):
    """Create mock ProcarData with single spin channel."""
    return MockProcarData(n_spins=1, rng=rng)


class TestProcarSelectInit:
    """Test ProcarSelect initialization."""

    def test_init_without_data(self):
        """Test initialization without ProcarData."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect()

        assert ps.spd is None
        assert ps.bands is None
        assert ps.kpoints is None

    def test_init_with_data_deep_copy(self, mock_procar_data):
        """Test initialization with ProcarData using deep copy."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data, deepCopy=True)

        assert ps.spd is not None
        assert ps.spd is not mock_procar_data.spd
        assert np.allclose(ps.spd, mock_procar_data.spd)

    def test_init_with_data_shallow_copy(self, mock_procar_data):
        """Test initialization with ProcarData using shallow copy."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data, deepCopy=False)

        assert ps.spd is mock_procar_data.spd

    def test_init_with_mode(self, mock_procar_data):
        """Test initialization with mode parameter."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data, mode="parametric")

        assert ps.mode == "parametric"


class TestProcarSelectSetData:
    """Test ProcarSelect.setData method."""

    def test_set_data_deep_copy(self, mock_procar_data):
        """Test setData with deep copy."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect()
        ps.setData(mock_procar_data, deepCopy=True)

        assert ps.spd is not mock_procar_data.spd
        assert np.allclose(ps.spd, mock_procar_data.spd)
        assert np.allclose(ps.bands, mock_procar_data.bands)
        assert np.allclose(ps.kpoints, mock_procar_data.kpoints)

    def test_set_data_shallow_copy(self, mock_procar_data):
        """Test setData with shallow copy."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect()
        ps.setData(mock_procar_data, deepCopy=False)

        assert ps.spd is mock_procar_data.spd

    def test_set_data_records_numspin(self, mock_procar_data):
        """Test that setData correctly records number of spins."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect()
        ps.setData(mock_procar_data)

        assert ps.numspin == mock_procar_data.spd.shape[2]


class TestProcarSelectIspin:
    """Test ProcarSelect.selectIspin method."""

    def test_select_ispin_density(self, mock_procar_data):
        """Test selecting spin density (value=[0])."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        original_shape = ps.spd.shape

        ps.selectIspin(value=[0], separate=False)

        # Shape should reduce from 5D to 4D
        assert len(ps.spd.shape) == 4
        assert ps.spd.shape == (original_shape[0], original_shape[1], original_shape[3], original_shape[4])

    def test_select_ispin_magnetization(self, mock_procar_data):
        """Test selecting spin magnetization (value=[1])."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)

        ps.selectIspin(value=[1], separate=False)

        assert len(ps.spd.shape) == 4

    def test_select_ispin_both_channels(self, mock_procar_data):
        """Test selecting both spin channels."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)

        ps.selectIspin(value=[0, 1], separate=False)

        assert len(ps.spd.shape) == 4

    def test_select_ispin_spin_up_separate(self, mock_procar_data):
        """Test selecting spin up separately."""
        from pyprocar.core.procarselect import ProcarSelect

        # Need even number of bands for separate spin selection
        mock_data = MockProcarData(n_bands=10)
        ps = ProcarSelect(ProcarData=mock_data)

        ps.selectIspin(value=[0], separate=True)

        # Should select first half of bands
        assert ps.spd.shape[1] == 5

    def test_select_ispin_spin_down_separate(self, mock_procar_data):
        """Test selecting spin down separately."""
        from pyprocar.core.procarselect import ProcarSelect

        mock_data = MockProcarData(n_bands=10)
        ps = ProcarSelect(ProcarData=mock_data)

        ps.selectIspin(value=[1], separate=True)

        # Should select second half of bands
        assert ps.spd.shape[1] == 5

    def test_select_ispin_wrong_dimensionality_raises_error(self, mock_procar_data):
        """Test that selectIspin raises error if array is not 5D."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        # First call reduces to 4D
        ps.selectIspin(value=[0])

        # Second call should raise error
        with pytest.raises(RuntimeError, match="Wrong dimensionality"):
            ps.selectIspin(value=[0])


class TestProcarSelectAtoms:
    """Test ProcarSelect.selectAtoms method."""

    def test_select_single_atom(self, mock_procar_data):
        """Test selecting a single atom."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        ps.selectIspin(value=[0])
        original_shape = ps.spd.shape

        ps.selectAtoms(value=[0])

        # Shape should reduce from 4D to 3D
        assert len(ps.spd.shape) == 3
        assert ps.spd.shape == (original_shape[0], original_shape[1], original_shape[3])

    def test_select_multiple_atoms(self, mock_procar_data):
        """Test selecting multiple atoms (summed)."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        ps.selectIspin(value=[0])

        ps.selectAtoms(value=[0, 1, 2])

        assert len(ps.spd.shape) == 3

    def test_select_atoms_fortran_indexing(self, mock_procar_data):
        """Test selecting atoms with 1-based indexing."""
        from pyprocar.core.procarselect import ProcarSelect

        ps1 = ProcarSelect(ProcarData=mock_procar_data)
        ps1.selectIspin(value=[0])
        ps1.selectAtoms(value=[0], fortran=False)

        ps2 = ProcarSelect(ProcarData=mock_procar_data)
        ps2.selectIspin(value=[0])
        ps2.selectAtoms(value=[1], fortran=True)

        # Both should select the same atom
        assert np.allclose(ps1.spd, ps2.spd)

    def test_select_atoms_wrong_dimensionality_raises_error(self, mock_procar_data):
        """Test that selectAtoms raises error if array is not 4D."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        # Skip selectIspin, so array is still 5D

        with pytest.raises(RuntimeError, match="Wrong dimensionality"):
            ps.selectAtoms(value=[0])


class TestProcarSelectOrbital:
    """Test ProcarSelect.selectOrbital method."""

    def test_select_single_orbital(self, mock_procar_data):
        """Test selecting a single orbital."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        ps.selectIspin(value=[0])
        ps.selectAtoms(value=[0])
        original_shape = ps.spd.shape

        ps.selectOrbital(value=[0])

        # Shape should reduce from 3D to 2D
        assert len(ps.spd.shape) == 2
        assert ps.spd.shape == (original_shape[0], original_shape[1])

    def test_select_multiple_orbitals(self, mock_procar_data):
        """Test selecting multiple orbitals (summed)."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        ps.selectIspin(value=[0])
        ps.selectAtoms(value=[0])

        ps.selectOrbital(value=[0, 1, 2])

        assert len(ps.spd.shape) == 2

    def test_select_total_orbital(self, mock_procar_data):
        """Test selecting total orbital using negative index."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        ps.selectIspin(value=[0])
        ps.selectAtoms(value=[0])

        ps.selectOrbital(value=[-1])

        assert len(ps.spd.shape) == 2

    def test_select_orbital_wrong_dimensionality_raises_error(self, mock_procar_data):
        """Test that selectOrbital raises error if array is not 3D."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)
        ps.selectIspin(value=[0])
        # Skip selectAtoms, so array is still 4D

        with pytest.raises(RuntimeError, match="Wrong dimensionality"):
            ps.selectOrbital(value=[0])


class TestProcarSelectFullPipeline:
    """Test complete selection pipeline."""

    def test_full_selection_pipeline(self, mock_procar_data):
        """Test complete pipeline: ispin -> atoms -> orbital."""
        from pyprocar.core.procarselect import ProcarSelect

        ps = ProcarSelect(ProcarData=mock_procar_data)

        # Verify initial 5D shape
        assert len(ps.spd.shape) == 5

        ps.selectIspin(value=[0])
        assert len(ps.spd.shape) == 4

        ps.selectAtoms(value=[0, 1])
        assert len(ps.spd.shape) == 3

        ps.selectOrbital(value=[0])
        assert len(ps.spd.shape) == 2

        # Final shape should be [kpoints, bands]
        assert ps.spd.shape == (mock_procar_data.kpoints.shape[0], mock_procar_data.bands.shape[1])
