"""Tests for ElkIn extractor."""

import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.elk.elkin import ElkIn
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


# Example elk.in content from non-spin-polarized bands calculation
ELKIN_NON_SPIN_BANDS = """! plotting the wannier functions in SrVO3

tasks
  0
  22

ngridk
  8 8 8

scale
  7.2589

avec
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

reducebf
0.5

sppath
  '/home/uthpala/elk/elk-6.3.2/species/'

atoms
  3                                    : nspecies
  'Sr.in'                              : spfname
  1                                    : natoms; atpos, bfcmt below
  0.00000000000  0.00000000000  0.00000000000  0. 0. 0.
  'V.in'
  1
   0.5 0.5 0.5 0.0 0.0 1.0
  'O.in'                               : spfname
  3                                    : natoms; atpos, bfcmt below
  0.5 0.5 0.0 0. 0. 0.
  0.5 0.0 0.5 0. 0. 0.
  0.0 0.5 0.5 0. 0. 0.

! These are the vertices to be joined for the band structure plot
plot1d
  6 50
   0.0      0.0      0.0 : \\Gamma
   0.5      0.0      0.0 : X
   0.5      0.5      0.0 : M
   0.0      0.0      0.0 : \\Gamma
   0.5      0.5      0.5 : R
   0.5      0.0      0.0 : X
"""

# Example elk.in content from spin-polarized bands calculation
ELKIN_SPIN_POLARIZED_BANDS = """! plotting the wannier functions in SrVO3

tasks
  0
  22

ngridk
  8 8 8

scale
  7.2589

avec
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

highq
.true.

spinpol
.true.

wrtvars
.true.

sppath
  '/home/uthpala/elk/elk-6.3.2/species/'

atoms
  3                                    : nspecies
  'Sr.in'                              : spfname
  1                                    : natoms; atpos, bfcmt below
  .00000000000  .00000000000  .00000000000  0. 0. 0.
  'V.in'
  1
   0.5 0.5 0.5 0.0 0.0 1.0
  'O.in'                               : spfname
  3                                    : natoms; atpos, bfcmt below
  0.5 0.5 0.0 0. 0. 0.
  0.5 0.0 0.5 0. 0. 0.
  0.0 0.5 0.5 0. 0. 0.

! These are the vertices to be joined for the band structure plot
plot1d
  6 40
   0.0      0.0      0.0 : \\Gamma
   0.5      0.0      0.0 : X
   0.5      0.5      0.0 : M
   0.0      0.0      0.0 : \\Gamma
   0.5      0.5      0.5 : R
   0.5      0.0      0.0 : X
"""

# Example elk.in for DOS calculation (no plot1d block)
ELKIN_DOS = """! DOS calculation for SrVO3

tasks
  0
  10

ngridk
  8 8 8

scale
  7.2589

avec
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

sppath
  '/home/uthpala/elk/elk-6.3.2/species/'

atoms
  3                                    : nspecies
  'Sr.in'                              : spfname
  1                                    : natoms; atpos, bfcmt below
  0.00000000000  0.00000000000  0.00000000000  0. 0. 0.
  'V.in'
  1
   0.5 0.5 0.5 0.0 0.0 1.0
  'O.in'                               : spfname
  3                                    : natoms; atpos, bfcmt below
  0.5 0.5 0.0 0. 0. 0.
  0.5 0.0 0.5 0. 0. 0.
  0.0 0.5 0.5 0. 0. 0.
"""


@pytest.fixture
def elkin_non_spin_bands(tmp_path: Path) -> Path:
    """Create a temporary elk.in file for non-spin-polarized bands."""
    elkin_file = tmp_path / "elk.in"
    elkin_file.write_text(ELKIN_NON_SPIN_BANDS)
    return elkin_file


@pytest.fixture
def elkin_spin_polarized_bands(tmp_path: Path) -> Path:
    """Create a temporary elk.in file for spin-polarized bands."""
    elkin_file = tmp_path / "elk.in"
    elkin_file.write_text(ELKIN_SPIN_POLARIZED_BANDS)
    return elkin_file


class TestElkInInit(BaseTest):
    def test_elkin_from_filepath(self, elkin_non_spin_bands: Path) -> None:
        """Test loading ElkIn from elk.in file."""
        elkin = ElkIn(elkin_non_spin_bands)
        assert elkin is not None
        assert elkin.filepath == elkin_non_spin_bands

    def test_elkin_from_str(self) -> None:
        """Test loading ElkIn from string content."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin is not None
        assert elkin.filepath is None


class TestElkInTasks(BaseTest):
    def test_tasks_parsing_bands(self) -> None:
        """Test that task numbers are extracted correctly."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.tasks == [0, 22]

    def test_tasks_parsing_dos(self) -> None:
        """Test that task numbers are extracted correctly for DOS."""
        elkin = ElkIn.from_str(ELKIN_DOS)
        assert elkin.tasks == [0, 10]

    def test_is_bands_calculation_true(self) -> None:
        """Test is_bands_calculation returns True for task 22."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.is_bands_calculation is True

    def test_is_bands_calculation_false(self) -> None:
        """Test is_bands_calculation returns False for DOS task."""
        elkin = ElkIn.from_str(ELKIN_DOS)
        assert elkin.is_bands_calculation is False


class TestElkInSpin(BaseTest):
    def test_spinpol_false(self) -> None:
        """Test spinpol detection when not spin-polarized."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.spinpol is False

    def test_spinpol_true(self) -> None:
        """Test spinpol detection when spin-polarized."""
        elkin = ElkIn.from_str(ELKIN_SPIN_POLARIZED_BANDS)
        assert elkin.spinpol is True

    def test_nspin_non_polarized(self) -> None:
        """Test nspin is 1 for non-spin-polarized."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.nspin == 1

    def test_nspin_polarized(self) -> None:
        """Test nspin is 2 for spin-polarized."""
        elkin = ElkIn.from_str(ELKIN_SPIN_POLARIZED_BANDS)
        assert elkin.nspin == 2


class TestElkInStructure(BaseTest):
    def test_nspecies(self) -> None:
        """Test number of species."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.nspecies == 3

    def test_composition(self) -> None:
        """Test species composition dictionary."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.composition == {"Sr": 1, "V": 1, "O": 3}

    def test_atoms(self) -> None:
        """Test atoms list."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.atoms == ["Sr", "V", "O", "O", "O"]

    def test_natoms(self) -> None:
        """Test number of atoms."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.natoms == 5

    def test_lattice_shape(self) -> None:
        """Test lattice vectors shape."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.lattice.shape == (3, 3)

    def test_lattice_values(self) -> None:
        """Test lattice vectors with scale factor applied."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        expected = np.array(
            [
                [7.2589, 0.0, 0.0],
                [0.0, 7.2589, 0.0],
                [0.0, 0.0, 7.2589],
            ]
        )
        np.testing.assert_allclose(elkin.lattice, expected)

    def test_fractional_coordinates_shape(self) -> None:
        """Test fractional coordinates shape."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.fractional_coordinates.shape == (5, 3)

    def test_fractional_coordinates_values(self) -> None:
        """Test fractional coordinates values."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],  # Sr
                [0.5, 0.5, 0.5],  # V
                [0.5, 0.5, 0.0],  # O
                [0.5, 0.0, 0.5],  # O
                [0.0, 0.5, 0.5],  # O
            ]
        )
        np.testing.assert_allclose(elkin.fractional_coordinates, expected)


class TestElkInKpath(BaseTest):
    def test_has_kpath_true(self) -> None:
        """Test has_kpath is True for bands calculation."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.has_kpath is True

    def test_has_kpath_false(self) -> None:
        """Test has_kpath is False for DOS calculation."""
        elkin = ElkIn.from_str(ELKIN_DOS)
        assert elkin.has_kpath is False

    def test_n_high_sym(self) -> None:
        """Test number of high-symmetry points."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.n_high_sym == 6

    def test_nkpoints_non_spin(self) -> None:
        """Test total number of k-points."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.nkpoints == 50

    def test_nkpoints_spin_polarized(self) -> None:
        """Test total number of k-points for spin-polarized."""
        elkin = ElkIn.from_str(ELKIN_SPIN_POLARIZED_BANDS)
        assert elkin.nkpoints == 40

    def test_n_segments(self) -> None:
        """Test number of path segments."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.n_segments == 5

    def test_high_symmetry_points_shape(self) -> None:
        """Test high-symmetry points array shape."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert elkin.high_symmetry_points.shape == (6, 3)

    def test_high_symmetry_points_values(self) -> None:
        """Test high-symmetry points values."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        expected = np.array(
            [
                [0.0, 0.0, 0.0],  # Gamma
                [0.5, 0.0, 0.0],  # X
                [0.5, 0.5, 0.0],  # M
                [0.0, 0.0, 0.0],  # Gamma
                [0.5, 0.5, 0.5],  # R
                [0.5, 0.0, 0.0],  # X
            ]
        )
        np.testing.assert_allclose(elkin.high_symmetry_points, expected)

    def test_knames(self) -> None:
        """Test k-point labels."""
        elkin = ElkIn.from_str(ELKIN_NON_SPIN_BANDS)
        assert len(elkin.knames) == 5  # 5 segments
        # Each segment has [start, end] labels
        for segment in elkin.knames:
            assert len(segment) == 2
