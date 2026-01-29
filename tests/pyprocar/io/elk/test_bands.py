"""Tests for ElkBands extractor."""

import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.elk.bands import ElkBands
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


# Example BANDLINES.OUT content (high-symmetry point positions)
# Format: x-position, energy_min, newline, x-position, energy_max, blank line
BANDLINES_OUT = """   0.000000000      -4.832432739
   0.000000000       2.461158505

  0.4327918353      -4.832432739
  0.4327918353       2.461158505

  0.8655836707      -4.832432739
  0.8655836707       2.461158505

"""

# Minimal BANDS.OUT content for testing (non-spin-polarized)
# Format: x-position, energy (Hartree), projections...
# This represents 5 k-points and 2 bands
BANDS_OUT_NON_SPIN = """   0.000000000      -2.401220419        0.000002    0.000000
  0.1081979588      -2.401219876        0.000002    0.000000
  0.2163959177      -2.401222821        0.000001    0.000000
  0.3245938765      -2.401223004        0.000000    0.000000
  0.4327918353      -2.401221850        0.000000    0.000001

   0.000000000      -1.452042593        0.000000    0.000009
  0.1081979588      -1.452224071        0.000050    0.000008
  0.2163959177      -1.452686930        0.000168    0.000005
  0.3245938765      -1.453108368        0.000283    0.000001
  0.4327918353      -1.453290430        0.000330    0.000000

"""

# Minimal BANDS.OUT content for spin-polarized case
# With nspin=2, raw_nbands is doubled, then split into spin channels
BANDS_OUT_SPIN = """   0.000000000      -2.401220419        0.000002    0.000000
  0.1081979588      -2.401219876        0.000002    0.000000
  0.2163959177      -2.401222821        0.000001    0.000000
  0.3245938765      -2.401223004        0.000000    0.000000
  0.4327918353      -2.401221850        0.000000    0.000001

   0.000000000      -1.452042593        0.000000    0.000009
  0.1081979588      -1.452224071        0.000050    0.000008
  0.2163959177      -1.452686930        0.000168    0.000005
  0.3245938765      -1.453108368        0.000283    0.000001
  0.4327918353      -1.453290430        0.000330    0.000000

   0.000000000      -2.301220419        0.000002    0.000000
  0.1081979588      -2.301219876        0.000002    0.000000
  0.2163959177      -2.301222821        0.000001    0.000000
  0.3245938765      -2.301223004        0.000000    0.000000
  0.4327918353      -2.301221850        0.000000    0.000001

   0.000000000      -1.352042593        0.000000    0.000009
  0.1081979588      -1.352224071        0.000050    0.000008
  0.2163959177      -1.352686930        0.000168    0.000005
  0.3245938765      -1.353108368        0.000283    0.000001
  0.4327918353      -1.353290430        0.000330    0.000000

"""

# High-symmetry points for k-path interpolation
HIGH_SYM_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # Gamma
        [0.5, 0.0, 0.0],  # X
        [0.5, 0.5, 0.0],  # M
    ]
)


@pytest.fixture
def bands_files_non_spin(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary band files for non-spin-polarized case."""
    bands_file = tmp_path / "BANDS.OUT"
    bands_file.write_text(BANDS_OUT_NON_SPIN)
    bandlines_file = tmp_path / "BANDLINES.OUT"
    bandlines_file.write_text(BANDLINES_OUT)
    return bands_file, bandlines_file


@pytest.fixture
def bands_files_spin(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary band files for spin-polarized case."""
    bands_file = tmp_path / "BANDS.OUT"
    bands_file.write_text(BANDS_OUT_SPIN)
    bandlines_file = tmp_path / "BANDLINES.OUT"
    bandlines_file.write_text(BANDLINES_OUT)
    return bands_file, bandlines_file


class TestElkBandsInit(BaseTest):
    def test_bands_from_filepath(self, bands_files_non_spin: tuple[Path, Path]) -> None:
        """Test loading ElkBands from file paths."""
        bands_file, bandlines_file = bands_files_non_spin
        bands = ElkBands(
            bands_filepath=bands_file,
            bandlines_filepath=bandlines_file,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert bands is not None

    def test_bands_from_str(self) -> None:
        """Test loading ElkBands from string content."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert bands is not None


class TestElkBandsDimensions(BaseTest):
    def test_nkpoints(self) -> None:
        """Test number of k-points."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert bands.nkpoints == 5

    def test_nspin_non_polarized(self) -> None:
        """Test nspin for non-spin-polarized."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert bands.nspin == 1

    def test_nspin_polarized(self) -> None:
        """Test nspin for spin-polarized."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=2,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert bands.nspin == 2

    def test_nbands_non_spin(self) -> None:
        """Test number of bands for non-spin-polarized."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert bands.nbands == 2

    def test_nbands_spin_polarized(self) -> None:
        """Test number of bands for spin-polarized (half of raw)."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=2,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert bands.nbands == 2  # 4 raw bands / 2 spins


class TestElkBandsKpoints(BaseTest):
    def test_kpoints_shape_no_high_sym(self) -> None:
        """Test k-points array shape when no high-symmetry points provided."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=None,  # No high-sym points
        )
        # Should return zeros array of correct shape
        assert bands.kpoints.shape == (5, 3)
        assert np.allclose(bands.kpoints, 0)

    def test_kticks(self) -> None:
        """Test high-symmetry k-point indices."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        # kticks should be indices where high-sym points occur
        assert isinstance(bands.kticks, list)
        assert all(isinstance(t, int) for t in bands.kticks)

    def test_ngrids(self) -> None:
        """Test number of points per segment."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        assert isinstance(bands.ngrids, np.ndarray)


class TestElkBandsEnergies(BaseTest):
    def test_bands_hartree_shape(self) -> None:
        """Test bands_hartree array shape."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        # Shape is (nkpoints, raw_nbands)
        assert bands.bands_hartree.shape == (5, 2)

    def test_bands_shape_non_spin(self) -> None:
        """Test bands array shape for non-spin-polarized."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        # Shape is (nkpoints, nbands, nspin)
        assert bands.bands.shape == (5, 2, 1)

    def test_bands_shape_spin_polarized(self) -> None:
        """Test bands array shape for spin-polarized."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=2,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        # Shape is (nkpoints, nbands, nspin)
        assert bands.bands.shape == (5, 2, 2)

    def test_bands_in_ev(self) -> None:
        """Test bands are converted to eV."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        # First band, first k-point energy should be ~-65 eV
        # -2.401220419 Hartree * 27.211386 = -65.34 eV
        assert bands.bands[0, 0, 0] == pytest.approx(-65.34, rel=0.01)

    def test_bands_hartree_values(self) -> None:
        """Test raw Hartree values are correctly parsed."""
        bands = ElkBands.from_str(
            bands_content=BANDS_OUT_NON_SPIN,
            bandlines_content=BANDLINES_OUT,
            nkpoints=5,
            nspin=1,
            high_symmetry_points=HIGH_SYM_POINTS,
        )
        # First band, first k-point
        assert bands.bands_hartree[0, 0] == pytest.approx(-2.401220419)
        # Second band, first k-point
        assert bands.bands_hartree[0, 1] == pytest.approx(-1.452042593)
