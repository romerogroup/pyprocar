"""Tests for ElkProjections extractor."""

import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.elk.projections import ElkProjections
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


# Example BAND_S01_A0001.OUT content (projections for atom 1)
# Format: x-position, energy (Hartree), 16 orbital projections
# Minimal example: 3 k-points, 2 bands
BAND_S01_A0001 = """   0.000000000      -2.401220419        0.000002    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.1081979588      -2.401219876        0.000002    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2163959177      -2.401222821        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000

   0.000000000      -1.452042593        0.000000    0.000009    0.000000    0.000009    0.000000    0.000000    0.000000    0.000000    0.000000    0.000002    0.000000    0.000001    0.000000    0.000001    0.000000    0.000002
  0.1081979588      -1.452224071        0.000050    0.000008    0.000000    0.000008    0.000000    0.000000    0.000000    0.000000    0.000000    0.000002    0.000000    0.000001    0.000000    0.000001    0.000000    0.000002
  0.2163959177      -1.452686930        0.000168    0.000005    0.000000    0.000005    0.000001    0.000000    0.000001    0.000000    0.000001    0.000001    0.000000    0.000001    0.000000    0.000001    0.000000    0.000001

"""

# Example BAND_S02_A0001.OUT content (projections for atom 2)
BAND_S02_A0001 = """   0.000000000      -2.401220419        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.1081979588      -2.401219876        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2163959177      -2.401222821        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000

   0.000000000      -1.452042593        0.000000    0.000005    0.000000    0.000005    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.1081979588      -1.452224071        0.000025    0.000004    0.000000    0.000004    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.2163959177      -1.452686930        0.000084    0.000002    0.000000    0.000002    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000

"""


@pytest.fixture
def projection_files(tmp_path: Path) -> list[Path]:
    """Create temporary projection files."""
    file1 = tmp_path / "BAND_S01_A0001.OUT"
    file1.write_text(BAND_S01_A0001)
    file2 = tmp_path / "BAND_S02_A0001.OUT"
    file2.write_text(BAND_S02_A0001)
    return [file1, file2]


class TestElkProjectionsInit(BaseTest):
    def test_projections_from_filepaths(self, projection_files: list[Path]) -> None:
        """Test loading ElkProjections from file paths."""
        proj = ElkProjections(
            filepaths=projection_files,
            nkpoints=3,
            nbands=2,
            nspin=1,
            natoms=2,
        )
        assert proj is not None

    def test_projections_from_str(self) -> None:
        """Test loading ElkProjections from string content."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        assert proj is not None


class TestElkProjectionsDimensions(BaseTest):
    def test_nkpoints(self) -> None:
        """Test nkpoints property."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        assert proj.nkpoints == 3

    def test_nbands(self) -> None:
        """Test nbands property."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        assert proj.nbands == 2

    def test_nspin(self) -> None:
        """Test nspin property."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        assert proj.nspin == 1

    def test_natoms(self) -> None:
        """Test natoms property."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        assert proj.natoms == 2


class TestElkProjectionsSPD(BaseTest):
    def test_spd_shape(self) -> None:
        """Test SPD array shape."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        # Shape: (nkpoints, nbands, nspin, natoms+1, norbitals+2)
        assert proj.spd.shape == (3, 2, 1, 3, 18)

    def test_spd_not_empty(self) -> None:
        """Test SPD array has data."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        assert proj.spd.size > 0
        # Should have some non-zero values
        assert np.any(proj.spd != 0)


class TestElkProjectionsProjected(BaseTest):
    def test_projected_shape(self) -> None:
        """Test projected array shape in canonical format."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        # Shape: (nkpoints, nbands, natoms, nprincipals, norbitals, nspin)
        assert proj.projected is not None
        assert proj.projected.shape == (3, 2, 2, 1, 16, 1)

    def test_projected_not_none(self) -> None:
        """Test projected is not None when data exists."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        assert proj.projected is not None

    def test_projected_values(self) -> None:
        """Test specific projected values are parsed correctly."""
        proj = ElkProjections.from_str(
            file_contents=[BAND_S01_A0001, BAND_S02_A0001],
            nkpoints=3,
            nbands=2,
            nspin=1,
        )
        # First k-point, first band, first atom, first orbital should be ~0.000002
        projected = proj.projected
        assert projected is not None
        assert projected[0, 0, 0, 0, 0, 0] == pytest.approx(0.000002, abs=1e-7)

    def test_projected_empty_when_no_files(self) -> None:
        """Test projected is None when no file contents."""
        proj = ElkProjections(
            nkpoints=3,
            nbands=2,
            nspin=1,
            natoms=0,
        )
        assert proj.projected is None
