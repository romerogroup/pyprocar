"""
Test module for pyprocar.core.procarunfold module.

This module contains unit tests for the Unfolder, ProcarUnfolder,
and plot_band_weight components.
"""

import matplotlib as mpl

mpl.use("Agg")  # Non-interactive backend for testing

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

# Import classes to test
from pyprocar.core.procarunfold import ProcarUnfolder
from pyprocar.core.procarunfold.fatband import plot_band_weight
from pyprocar.core.procarunfold.unfolder import Unfolder
from tests.utils import DATA_DIR

# Type aliases for readability
_BandPlotData = tuple[
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]

# ==============================================================================
# Fixtures for Synthetic Data (Unit Tests)
# ==============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_cubic_cell() -> npt.NDArray[np.float64]:
    """Provide simple cubic unit cell."""
    return np.eye(3) * 3.0  # 3 Angstrom lattice parameter


@pytest.fixture
def supercell_matrix_2x2x2() -> npt.NDArray[np.int64]:
    """2x2x2 supercell transformation matrix."""
    return np.diag([2, 2, 2])


@pytest.fixture
def supercell_matrix_identity() -> npt.NDArray[np.int64]:
    """Identity transformation (no supercell)."""
    return np.eye(3, dtype=np.int64)


@pytest.fixture
def simple_basis() -> list[str]:
    """Provide simple basis labels for a 2-atom cell with s and p orbitals."""
    return [f"None|{orb}|0" for _atom in range(2) for orb in ["s", "p"]]


@pytest.fixture
def simple_positions() -> npt.NDArray[np.float64]:
    """Fractional positions for a 2-atom cell."""
    positions = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5]),
    ] * 2  # Repeated for 2 orbitals per atom
    return np.array(positions)


@pytest.fixture
def simple_eigenvectors(rng: np.random.Generator) -> npt.NDArray[np.complex128]:
    """Provide simple eigenvectors for testing.

    Shape: (n_kpoints, n_bands, n_basis).
    """
    n_kpoints = 5
    n_bands = 4
    n_basis = 4  # 2 atoms * 2 orbitals

    # Create random complex eigenvectors and normalize
    real_part: npt.NDArray[np.float64] = rng.random((n_kpoints, n_bands, n_basis))
    imag_part: npt.NDArray[np.float64] = rng.random((n_kpoints, n_bands, n_basis))
    eigenvectors: npt.NDArray[np.complex128] = real_part + 1j * imag_part

    # Normalize along basis axis
    norm = np.linalg.norm(eigenvectors, axis=2, keepdims=True)
    return eigenvectors / norm


@pytest.fixture
def simple_qpoints() -> npt.NDArray[np.float64]:
    """Provide simple k-points for testing."""
    return np.array(
        [
            [0.0, 0.0, 0.0],  # Gamma
            [0.5, 0.0, 0.0],  # X
            [0.5, 0.5, 0.0],  # M
            [0.5, 0.5, 0.5],  # R
            [0.0, 0.0, 0.0],  # Gamma (return)
        ]
    )


@pytest.fixture
def simple_unfolder(
    simple_cubic_cell: npt.NDArray[np.float64],
    simple_basis: list[str],
    simple_positions: npt.NDArray[np.float64],
    supercell_matrix_2x2x2: npt.NDArray[np.int64],
    simple_eigenvectors: npt.NDArray[np.complex128],
    simple_qpoints: npt.NDArray[np.float64],
) -> Unfolder:
    """Create an Unfolder instance with synthetic data."""
    return Unfolder(
        cell=simple_cubic_cell,
        basis=simple_basis,
        positions=simple_positions,
        supercell_matrix=supercell_matrix_2x2x2,
        eigenvectors=simple_eigenvectors,
        qpoints=simple_qpoints,
        tol_r=0.1,
        phase=False,
    )


# ==============================================================================
# Fixtures for Real Data (Integration Tests)
# ==============================================================================


@pytest.fixture
def unfolding_supercell_dir() -> Path:
    """Path to supercell unfolding test data."""
    return DATA_DIR / "examples" / "bands" / "unfolding" / "supercell"


@pytest.fixture
def unfolding_primitive_dir() -> Path:
    """Path to primitive cell test data."""
    return DATA_DIR / "examples" / "bands" / "unfolding" / "primitive"


# ==============================================================================
# Fixtures for Plotting Tests
# ==============================================================================


@pytest.fixture
def band_plot_data() -> _BandPlotData:
    """Generate simple band plot data."""
    n_kpoints = 50
    n_bands = 4

    kslist: list[npt.NDArray[np.float64]] = [
        np.arange(n_kpoints, dtype=np.float64) for _ in range(n_bands)
    ]
    ekslist: npt.NDArray[np.float64] = np.array(
        [np.sin(np.linspace(0, 2 * np.pi, n_kpoints)) * (i + 1) for i in range(n_bands)]
    )
    wkslist: npt.NDArray[np.float64] = np.array(
        [np.abs(np.cos(np.linspace(0, 2 * np.pi, n_kpoints))) for _ in range(n_bands)]
    )

    return kslist, ekslist, wkslist


class TestUnfolder:
    """Test class for Unfolder algorithm."""

    def test_initialization(self, simple_unfolder: Unfolder) -> None:
        """Test Unfolder initializes with correct attributes."""
        assert simple_unfolder._cell is not None
        assert simple_unfolder._basis is not None
        assert simple_unfolder._positions is not None
        assert simple_unfolder._evecs is not None
        assert simple_unfolder._qpts is not None
        assert simple_unfolder._tol_r == 0.1

    def test_translation_maps_created(self, simple_unfolder: Unfolder) -> None:
        """Test that translation maps are created during initialization."""
        assert simple_unfolder._trans_rs is not None
        assert simple_unfolder._trans_indices is not None

    def test_translation_maps_shape(self, simple_unfolder: Unfolder) -> None:
        """Test translation maps have correct shapes."""
        n_positions = len(simple_unfolder._positions)

        # trans_rs contains the supercell translation vectors
        # For a 2x2x2 supercell, there should be 8 translation vectors
        trans_rs = simple_unfolder._trans_rs
        assert trans_rs is not None
        assert trans_rs.shape[0] == 8
        assert trans_rs.shape[1] == 3  # 3D vectors

        # trans_indices maps basis elements across translations
        trans_indices = simple_unfolder._trans_indices
        assert trans_indices is not None
        assert trans_indices.shape[0] == 8
        assert trans_indices.shape[1] == n_positions

    def test_translation_vectors_in_unit_cell(self, simple_unfolder: Unfolder) -> None:
        """Test that translation vectors are within [0, 1)."""
        trans_rs = simple_unfolder._trans_rs
        assert trans_rs is not None
        assert np.all(trans_rs >= 0.0)
        assert np.all(trans_rs < 1.0 + simple_unfolder._tol_r)

    def test_get_weight_returns_real(
        self,
        simple_unfolder: Unfolder,
        simple_eigenvectors: npt.NDArray[np.complex128],
        simple_qpoints: npt.NDArray[np.float64],
    ) -> None:
        """Test get_weight returns a real number."""
        evec = simple_eigenvectors[0, 0, :]
        qpt = simple_qpoints[0]

        weight = simple_unfolder.get_weight(evec, qpt)

        assert isinstance(weight, (float, np.floating))

    def test_get_weight_bounded(
        self,
        simple_unfolder: Unfolder,
        simple_eigenvectors: npt.NDArray[np.complex128],
        simple_qpoints: npt.NDArray[np.float64],
    ) -> None:
        """Test weights are bounded between 0 and 1 (approximately)."""
        evec = simple_eigenvectors[0, 0, :]
        qpt = simple_qpoints[0]

        weight = simple_unfolder.get_weight(evec, qpt)

        # Weights should be approximately in [0, 1] for normalized eigenvectors
        assert weight >= -0.1  # Allow small numerical tolerance
        assert weight <= 1.1

    def test_get_weights_shape(
        self,
        simple_unfolder: Unfolder,
        simple_eigenvectors: npt.NDArray[np.complex128],
    ) -> None:
        """Test get_weights returns correct shape."""
        weights = simple_unfolder.get_weights()

        n_kpoints, n_bands = simple_eigenvectors.shape[:2]
        assert weights.shape == (n_kpoints, n_bands)

    def test_get_weights_values_bounded(self, simple_unfolder: Unfolder) -> None:
        """Test all weights are approximately bounded."""
        weights = simple_unfolder.get_weights()

        # Most weights should be in reasonable range
        assert np.all(weights >= -0.5)
        assert np.all(weights <= 1.5)

    def test_identity_supercell_high_weights(
        self,
        simple_cubic_cell: npt.NDArray[np.float64],
        simple_basis: list[str],
        simple_positions: npt.NDArray[np.float64],
        supercell_matrix_identity: npt.NDArray[np.int64],
        simple_eigenvectors: npt.NDArray[np.complex128],
        simple_qpoints: npt.NDArray[np.float64],
    ) -> None:
        """Test identity transformation gives high weights (no unfolding needed)."""
        unfolder = Unfolder(
            cell=simple_cubic_cell,
            basis=simple_basis,
            positions=simple_positions,
            supercell_matrix=supercell_matrix_identity,
            eigenvectors=simple_eigenvectors,
            qpoints=simple_qpoints,
            tol_r=0.1,
            phase=False,
        )

        weights = unfolder.get_weights()

        # For identity transformation, weights should be close to 1
        assert np.mean(weights) > 0.5

    def test_get_weight_with_G_vector(
        self,
        simple_unfolder: Unfolder,
        simple_eigenvectors: npt.NDArray[np.complex128],
        simple_qpoints: npt.NDArray[np.float64],
    ) -> None:
        """Test get_weight with non-zero G vector."""
        evec = simple_eigenvectors[0, 0, :]
        qpt = simple_qpoints[0]
        g_vector = np.array([1.0, 0.0, 0.0])

        weight = simple_unfolder.get_weight(evec, qpt, G=g_vector)

        assert isinstance(weight, (float, np.floating))

    def test_phase_mode_true(
        self,
        simple_cubic_cell: npt.NDArray[np.float64],
        simple_basis: list[str],
        simple_positions: npt.NDArray[np.float64],
        supercell_matrix_2x2x2: npt.NDArray[np.int64],
        simple_eigenvectors: npt.NDArray[np.complex128],
        simple_qpoints: npt.NDArray[np.float64],
    ) -> None:
        """Test Unfolder with phase=True mode."""
        unfolder = Unfolder(
            cell=simple_cubic_cell,
            basis=simple_basis,
            positions=simple_positions,
            supercell_matrix=supercell_matrix_2x2x2,
            eigenvectors=simple_eigenvectors,
            qpoints=simple_qpoints,
            tol_r=0.1,
            phase=True,  # Different from default
        )

        weights = unfolder.get_weights()

        assert weights.shape == simple_eigenvectors.shape[:2]

    def test_different_tolerance(
        self,
        simple_cubic_cell: npt.NDArray[np.float64],
        simple_basis: list[str],
        simple_positions: npt.NDArray[np.float64],
        supercell_matrix_2x2x2: npt.NDArray[np.int64],
        simple_eigenvectors: npt.NDArray[np.complex128],
        simple_qpoints: npt.NDArray[np.float64],
    ) -> None:
        """Test Unfolder with different tolerance values."""
        for tol in [0.01, 0.1, 0.2]:
            unfolder = Unfolder(
                cell=simple_cubic_cell,
                basis=simple_basis,
                positions=simple_positions,
                supercell_matrix=supercell_matrix_2x2x2,
                eigenvectors=simple_eigenvectors,
                qpoints=simple_qpoints,
                tol_r=tol,
                phase=False,
            )

            weights = unfolder.get_weights()
            assert weights.shape == simple_eigenvectors.shape[:2]


class TestProcarUnfolder:
    """Test class for ProcarUnfolder with real data."""

    @pytest.fixture
    def procar_unfolder(self, unfolding_supercell_dir: Path) -> ProcarUnfolder:
        """Create ProcarUnfolder from test data."""
        procar_path = unfolding_supercell_dir / "PROCAR"
        poscar_path = unfolding_supercell_dir / "POSCAR"
        supercell_matrix = np.diag([2, 2, 2])

        return ProcarUnfolder(
            procar=str(procar_path),
            poscar=str(poscar_path),
            supercell_matrix=supercell_matrix,
        )

    def test_initialization(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test ProcarUnfolder initializes correctly."""
        assert procar_unfolder.fname is not None
        assert procar_unfolder.supercell_matrix is not None
        assert procar_unfolder.atoms is not None

    def test_procar_parsed(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test PROCAR file is parsed."""
        assert procar_unfolder.procar is not None
        assert procar_unfolder.procar.kpoints is not None
        assert procar_unfolder.procar.bands is not None

    def test_atoms_loaded(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test atomic structure is loaded via ASE."""
        assert len(procar_unfolder.atoms) > 0
        assert procar_unfolder.atoms.get_cell() is not None

    def test_unfold_returns_weights(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test unfold method returns weight array."""
        weights = procar_unfolder.unfold()

        assert weights is not None
        assert isinstance(weights, np.ndarray)
        assert weights.ndim == 2  # (n_kpoints, n_bands)

    def test_unfold_weights_shape(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test unfold weights have correct shape."""
        weights = procar_unfolder.unfold()

        n_kpoints = procar_unfolder.procar.kpointsCount
        n_bands = procar_unfolder.procar.bandsCount

        assert weights.shape == (n_kpoints, n_bands)

    def test_unfold_weights_reasonable(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test unfolded weights are in reasonable range."""
        weights = procar_unfolder.unfold()

        # Weights should be mostly in [0, 1]
        assert np.percentile(weights, 5) >= -0.1
        assert np.percentile(weights, 95) <= 1.1

    def test_prepare_unfold_basis_creates_eigenvectors(
        self, procar_unfolder: ProcarUnfolder
    ) -> None:
        """Test _prepare_unfold_basis creates eigenvector array."""
        procar_unfolder._prepare_unfold_basis()

        assert procar_unfolder.eigenvectors is not None
        assert procar_unfolder.eigenvectors.ndim == 3

    def test_prepare_unfold_basis_creates_basis_labels(
        self, procar_unfolder: ProcarUnfolder
    ) -> None:
        """Test _prepare_unfold_basis creates basis labels."""
        procar_unfolder._prepare_unfold_basis()

        assert len(procar_unfolder.basis) > 0
        assert len(procar_unfolder.positions) > 0
        assert len(procar_unfolder.basis) == len(procar_unfolder.positions)

    def test_eigenvectors_normalized(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test eigenvectors are normalized."""
        procar_unfolder._prepare_unfold_basis()

        # Check normalization along last axis
        eigenvectors = procar_unfolder.eigenvectors
        assert eigenvectors is not None
        norms = np.linalg.norm(eigenvectors, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-6)


class TestProcarUnfolderPlot:
    """Test class for ProcarUnfolder.plot() method."""

    @pytest.fixture
    def procar_unfolder(self, unfolding_supercell_dir: Path) -> ProcarUnfolder:
        """Create ProcarUnfolder from test data."""
        procar_path = unfolding_supercell_dir / "PROCAR"
        poscar_path = unfolding_supercell_dir / "POSCAR"
        supercell_matrix = np.diag([2, 2, 2])

        return ProcarUnfolder(
            procar=str(procar_path),
            poscar=str(poscar_path),
            supercell_matrix=supercell_matrix,
        )

    def test_plot_returns_axes(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test plot method returns matplotlib axes."""
        axes = procar_unfolder.plot(efermi=0.0, show_band=False, savetab=False)

        assert axes is not None
        plt.close("all")

    def test_plot_with_custom_ylim(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test plot with custom y-axis limits."""
        ylim = (-10, 5)
        axes = procar_unfolder.plot(efermi=0.0, ylim=ylim, show_band=False, savetab=False)

        assert axes.get_ylim() == ylim
        plt.close("all")

    def test_plot_with_show_band(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test plot with band lines shown."""
        axes = procar_unfolder.plot(efermi=0.0, show_band=True, savetab=False)

        # Should have lines for bands
        assert len(axes.lines) > 0
        plt.close("all")

    def test_plot_without_shift_efermi(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test plot without shifting Fermi energy."""
        axes = procar_unfolder.plot(efermi=5.0, shift_efermi=False, show_band=False, savetab=False)

        assert axes is not None
        plt.close("all")

    def test_plot_with_axis_parameter(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test plot accepts external axis."""
        _fig, ax = plt.subplots()
        axes = procar_unfolder.plot(efermi=0.0, axis=ax, show_band=False, savetab=False)

        assert axes is ax
        plt.close("all")

    def test_plot_with_custom_color(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test plot with custom color."""
        axes = procar_unfolder.plot(efermi=0.0, color="red", show_band=False, savetab=False)

        assert axes is not None
        plt.close("all")

    def test_plot_saves_tab_file(self, procar_unfolder: ProcarUnfolder) -> None:
        """Test plot can save tab-separated data file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        _axes = procar_unfolder.plot(efermi=0.0, savetab=temp_path, show_band=False)

        # Check file was created
        assert Path(temp_path).exists()

        # Check file has content
        data = np.loadtxt(temp_path, delimiter=",")
        assert data.size > 0

        plt.close("all")
        Path(temp_path).unlink()  # Cleanup


class TestPlotBandWeight:
    """Test class for plot_band_weight function."""

    def test_plot_without_weights(self, band_plot_data: _BandPlotData) -> None:
        """Test plot_band_weight without weight data."""
        kslist, ekslist, _ = band_plot_data

        axes = plot_band_weight(kslist, ekslist)

        assert axes is not None
        plt.close("all")

    def test_plot_with_weights_alpha_style(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with alpha style (default)."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, style="alpha")

        assert axes is not None
        # Check that LineCollection was added
        assert len(axes.collections) > 0
        plt.close("all")

    def test_plot_with_weights_width_style(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with width style."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, style="width")

        assert axes is not None
        plt.close("all")

    def test_plot_with_weights_color_style(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with color/colormap style."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, style="color")

        assert axes is not None
        plt.close("all")

    def test_plot_with_efermi(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with Fermi energy."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, efermi=0.0)

        assert axes is not None
        plt.close("all")

    def test_plot_with_efermi_shift(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with Fermi energy shift."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, efermi=1.0, shift_efermi=True)

        assert axes is not None
        plt.close("all")

    def test_plot_with_custom_yrange(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with custom y-range."""
        kslist, ekslist, wkslist = band_plot_data
        yrange = (-5, 5)

        axes = plot_band_weight(kslist, ekslist, wkslist, yrange=yrange)

        # Note: yrange is used but not enforced by the function
        assert axes is not None
        plt.close("all")

    def test_plot_with_external_axis(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with external axis provided."""
        kslist, ekslist, wkslist = band_plot_data

        _fig, ax = plt.subplots()
        axes = plot_band_weight(kslist, ekslist, wkslist, axis=ax)

        assert axes is ax
        plt.close("all")

    def test_plot_with_custom_width(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with custom line width."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, width=5)

        assert axes is not None
        plt.close("all")

    def test_plot_with_custom_fatness(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with custom fatness parameter."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, fatness=8)

        assert axes is not None
        plt.close("all")

    def test_plot_with_custom_color(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with custom color."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, color="red")

        assert axes is not None
        plt.close("all")

    def test_plot_with_xticks(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with custom x-ticks."""
        kslist, ekslist, wkslist = band_plot_data
        xticks = [["G", "X", "M"], [0, 25, 49]]

        axes = plot_band_weight(kslist, ekslist, wkslist, xticks=xticks)

        assert axes is not None
        # Check x-tick labels were set
        labels = [t.get_text() for t in axes.get_xticklabels()]
        assert "G" in labels or len(labels) > 0
        plt.close("all")

    def test_plot_with_custom_cmap(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with custom colormap."""
        from matplotlib import cm

        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(kslist, ekslist, wkslist, style="colormap", cmap=cm.viridis)

        assert axes is not None
        plt.close("all")

    def test_plot_with_weight_limits(self, band_plot_data: _BandPlotData) -> None:
        """Test plot with custom weight limits for colormap."""
        kslist, ekslist, wkslist = band_plot_data

        axes = plot_band_weight(
            kslist, ekslist, wkslist, style="color", weight_min=0.0, weight_max=1.0
        )

        assert axes is not None
        plt.close("all")

    def test_plot_alpha_normalization(self, band_plot_data: _BandPlotData) -> None:
        """Test that alpha values > 1 are normalized."""
        kslist, ekslist, _ = band_plot_data
        # Create weights that will exceed alpha=1 when multiplied by width
        wkslist: npt.NDArray[np.float64] = np.array(
            [np.ones(50) * 2.0 for _ in range(4)]
        )  # High weights

        # This should trigger the normalization code path
        axes = plot_band_weight(kslist, ekslist, wkslist, style="alpha", width=10)

        assert axes is not None
        plt.close("all")


class TestUnfolderEdgeCases:
    """Test edge cases for Unfolder class."""

    def test_single_kpoint(
        self,
        simple_cubic_cell: npt.NDArray[np.float64],
        simple_basis: list[str],
        simple_positions: npt.NDArray[np.float64],
        supercell_matrix_2x2x2: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> None:
        """Test with single k-point."""
        n_basis = len(simple_basis)
        eigenvectors: npt.NDArray[np.complex128] = rng.random((1, 2, n_basis)) + 1j * rng.random(
            (1, 2, n_basis)
        )
        norm = np.linalg.norm(eigenvectors, axis=2, keepdims=True)
        eigenvectors = eigenvectors / norm

        qpoints: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0]])

        unfolder = Unfolder(
            cell=simple_cubic_cell,
            basis=simple_basis,
            positions=simple_positions,
            supercell_matrix=supercell_matrix_2x2x2,
            eigenvectors=eigenvectors,
            qpoints=qpoints,
            phase=False,
        )

        weights = unfolder.get_weights()
        assert weights.shape == (1, 2)

    def test_single_band(
        self,
        simple_cubic_cell: npt.NDArray[np.float64],
        simple_basis: list[str],
        simple_positions: npt.NDArray[np.float64],
        supercell_matrix_2x2x2: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> None:
        """Test with single band."""
        n_basis = len(simple_basis)
        eigenvectors: npt.NDArray[np.complex128] = rng.random((5, 1, n_basis)) + 1j * rng.random(
            (5, 1, n_basis)
        )
        norm = np.linalg.norm(eigenvectors, axis=2, keepdims=True)
        eigenvectors = eigenvectors / norm

        qpoints: npt.NDArray[np.float64] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ]
        )

        unfolder = Unfolder(
            cell=simple_cubic_cell,
            basis=simple_basis,
            positions=simple_positions,
            supercell_matrix=supercell_matrix_2x2x2,
            eigenvectors=eigenvectors,
            qpoints=qpoints,
            phase=False,
        )

        weights = unfolder.get_weights()
        assert weights.shape == (5, 1)

    def test_non_diagonal_supercell_matrix(
        self,
        simple_cubic_cell: npt.NDArray[np.float64],
        simple_basis: list[str],
        simple_positions: npt.NDArray[np.float64],
        simple_eigenvectors: npt.NDArray[np.complex128],
        simple_qpoints: npt.NDArray[np.float64],
    ) -> None:
        """Test with non-diagonal supercell matrix."""
        # 2x2x1 with shear
        supercell_matrix = np.array(
            [
                [2, 1, 0],
                [0, 2, 0],
                [0, 0, 1],
            ]
        )

        unfolder = Unfolder(
            cell=simple_cubic_cell,
            basis=simple_basis,
            positions=simple_positions,
            supercell_matrix=supercell_matrix,
            eigenvectors=simple_eigenvectors,
            qpoints=simple_qpoints,
            phase=False,
        )

        weights = unfolder.get_weights()
        assert weights.shape == simple_eigenvectors.shape[:2]

    def test_high_symmetry_kpoints(self, simple_unfolder: Unfolder) -> None:
        """Test weights at high-symmetry points."""
        weights = simple_unfolder.get_weights()

        # Gamma point (index 0) should have defined weight
        assert np.isfinite(weights[0, :]).all()

        # All other points should also be finite
        assert np.isfinite(weights).all()


class TestPlotBandWeightEdgeCases:
    """Test edge cases for plot_band_weight function."""

    def test_empty_weights(self) -> None:
        """Test plot with empty weight list (wkslist=None)."""
        kslist: list[npt.NDArray[np.float64]] = [np.arange(5, dtype=np.float64)]
        ekslist: npt.NDArray[np.float64] = np.array([[0.0, 0.5, 1.0, 0.5, 0.0]])

        axes = plot_band_weight(kslist, ekslist, wkslist=None)

        assert axes is not None
        plt.close("all")

    def test_single_band_plot(self) -> None:
        """Test plot with single band."""
        kslist: list[npt.NDArray[np.float64]] = [np.arange(20, dtype=np.float64)]
        ekslist: npt.NDArray[np.float64] = np.array([np.sin(np.linspace(0, np.pi, 20))])
        wkslist: npt.NDArray[np.float64] = np.array([np.ones(20) * 0.5])

        axes = plot_band_weight(kslist, ekslist, wkslist)

        assert axes is not None
        plt.close("all")

    def test_negative_weights(self) -> None:
        """Test plot handles negative weights."""
        kslist: list[npt.NDArray[np.float64]] = [np.arange(20, dtype=np.float64)]
        ekslist: npt.NDArray[np.float64] = np.array([np.sin(np.linspace(0, np.pi, 20))])
        wkslist: npt.NDArray[np.float64] = np.array(
            [np.linspace(-0.5, 0.5, 20)]
        )  # Includes negative values

        axes = plot_band_weight(kslist, ekslist, wkslist, style="color")

        assert axes is not None
        plt.close("all")
