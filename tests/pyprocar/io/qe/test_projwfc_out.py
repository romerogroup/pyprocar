"""Tests for ProjwfcOut parser."""

from pathlib import Path

import pytest

from pyprocar.io.qe.projwfc import ProjwfcOut

# =============================================================================
# Inline String Fixtures
# =============================================================================

NON_SPIN_POLARIZED_PROJWFC_OUT = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     This program is part of the open-source Quantum ESPRESSO suite

     Parallel version (MPI), running on     1 processors

     Reading xml data from directory:
     ./tmp/test.save/

     IMPORTANT: XC functional enforced from input :
     Exchange-correlation= PBE
                           (   1   4   3   4   0   0   0)
     Any further DFT definition will be discarded
     Please, verify this is what you really want


     G-vector sticks info
     --------------------
     sticks:   dense  smooth     PW     G-vecs:    dense   smooth      PW
     Sum        1000     500    150                20000    10000    2000

     Check: negative core charge=   -0.000001

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Sr ), wfc  2 (l=1 m= 1)
     state #   3: atom   1 (Sr ), wfc  2 (l=1 m= 2)
     state #   4: atom   1 (Sr ), wfc  2 (l=1 m= 3)
     state #   5: atom   2 (V  ), wfc  1 (l=0 m= 1)
     state #   6: atom   2 (V  ), wfc  2 (l=1 m= 1)
     state #   7: atom   2 (V  ), wfc  2 (l=1 m= 2)
     state #   8: atom   2 (V  ), wfc  2 (l=1 m= 3)
     state #   9: atom   2 (V  ), wfc  3 (l=2 m= 1)
     state #  10: atom   2 (V  ), wfc  3 (l=2 m= 2)
     state #  11: atom   2 (V  ), wfc  3 (l=2 m= 3)
     state #  12: atom   2 (V  ), wfc  3 (l=2 m= 4)
     state #  13: atom   2 (V  ), wfc  3 (l=2 m= 5)
     state #  14: atom   3 (O  ), wfc  1 (l=0 m= 1)
     state #  15: atom   3 (O  ), wfc  2 (l=1 m= 1)
     state #  16: atom   3 (O  ), wfc  2 (l=1 m= 2)
     state #  17: atom   3 (O  ), wfc  2 (l=1 m= 3)
     state #  18: atom   4 (O  ), wfc  1 (l=0 m= 1)
     state #  19: atom   4 (O  ), wfc  2 (l=1 m= 1)
     state #  20: atom   4 (O  ), wfc  2 (l=1 m= 2)
     state #  21: atom   4 (O  ), wfc  2 (l=1 m= 3)
     state #  22: atom   5 (O  ), wfc  1 (l=0 m= 1)
     state #  23: atom   5 (O  ), wfc  2 (l=1 m= 1)
     state #  24: atom   5 (O  ), wfc  2 (l=1 m= 2)
     state #  25: atom   5 (O  ), wfc  2 (l=1 m= 3)

 Parallelization info
 --------------------
 sticks:   dense  smooth     PW     G-vecs:    dense   smooth      PW
 Min        1000     500    150                20000    10000    2000
 Max        1000     500    150                20000    10000    2000
 Sum        1000     500    150                20000    10000    2000


     Calling projwave .... 

     natomwfc =   25
     nbnd     =   24
     nkstot   =   29
     nspin    =    1

     k =   0.0000   0.0000   0.0000
     k =   0.1250   0.0000   0.0000
     k =   0.2500   0.0000   0.0000
     k =   0.3750   0.0000   0.0000
     k =   0.5000   0.0000   0.0000
     k =   0.1250   0.1250   0.0000
     k =   0.2500   0.1250   0.0000
     k =   0.3750   0.1250   0.0000
     k =   0.5000   0.1250   0.0000
     k =   0.2500   0.2500   0.0000
     k =   0.3750   0.2500   0.0000
     k =   0.5000   0.2500   0.0000
     k =   0.3750   0.3750   0.0000
     k =   0.5000   0.3750   0.0000
     k =   0.5000   0.5000   0.0000
     k =   0.1250   0.1250   0.1250
     k =   0.2500   0.1250   0.1250
     k =   0.3750   0.1250   0.1250
     k =   0.5000   0.1250   0.1250
     k =   0.2500   0.2500   0.1250
     k =   0.3750   0.2500   0.1250
     k =   0.5000   0.2500   0.1250
     k =   0.3750   0.3750   0.1250
     k =   0.5000   0.3750   0.1250
     k =   0.2500   0.2500   0.2500
     k =   0.3750   0.2500   0.2500
     k =   0.5000   0.2500   0.2500
     k =   0.3750   0.3750   0.2500
     k =   0.5000   0.5000   0.2500

     PROJWFC      :      1.00s CPU      1.10s WALL


   This run was terminated on:  12: 0: 1   1Jan2026            

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""

SPIN_POLARIZED_PROJWFC_OUT = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Sr ), wfc  2 (l=1 m= 1)
     state #   3: atom   2 (V  ), wfc  1 (l=0 m= 1)
     state #   4: atom   2 (V  ), wfc  2 (l=2 m= 1)
     state #   5: atom   3 (O  ), wfc  1 (l=0 m= 1)

     Calling projwave .... 

     natomwfc =    5
     nbnd     =   24
     nkstot   =   58
     nspin    =    2

     k =   0.0000   0.0000   0.0000
     k =   0.1250   0.0000   0.0000
     k =   0.2500   0.0000   0.0000
     k =   0.3750   0.0000   0.0000
     k =   0.5000   0.0000   0.0000
     k =   0.1250   0.1250   0.0000
     k =   0.2500   0.1250   0.0000
     k =   0.3750   0.1250   0.0000
     k =   0.5000   0.1250   0.0000
     k =   0.2500   0.2500   0.0000
     k =   0.3750   0.2500   0.0000
     k =   0.5000   0.2500   0.0000
     k =   0.3750   0.3750   0.0000
     k =   0.5000   0.3750   0.0000
     k =   0.5000   0.5000   0.0000
     k =   0.1250   0.1250   0.1250
     k =   0.2500   0.1250   0.1250
     k =   0.3750   0.1250   0.1250
     k =   0.5000   0.1250   0.1250
     k =   0.2500   0.2500   0.1250
     k =   0.3750   0.2500   0.1250
     k =   0.5000   0.2500   0.1250
     k =   0.3750   0.3750   0.1250
     k =   0.5000   0.3750   0.1250
     k =   0.2500   0.2500   0.2500
     k =   0.3750   0.2500   0.2500
     k =   0.5000   0.2500   0.2500
     k =   0.3750   0.3750   0.2500
     k =   0.5000   0.5000   0.2500
     k =   0.0000   0.0000   0.0000
     k =   0.1250   0.0000   0.0000
     k =   0.2500   0.0000   0.0000
     k =   0.3750   0.0000   0.0000
     k =   0.5000   0.0000   0.0000
     k =   0.1250   0.1250   0.0000
     k =   0.2500   0.1250   0.0000
     k =   0.3750   0.1250   0.0000
     k =   0.5000   0.1250   0.0000
     k =   0.2500   0.2500   0.0000
     k =   0.3750   0.2500   0.0000
     k =   0.5000   0.2500   0.0000
     k =   0.3750   0.3750   0.0000
     k =   0.5000   0.3750   0.0000
     k =   0.5000   0.5000   0.0000
     k =   0.1250   0.1250   0.1250
     k =   0.2500   0.1250   0.1250
     k =   0.3750   0.1250   0.1250
     k =   0.5000   0.1250   0.1250
     k =   0.2500   0.2500   0.1250
     k =   0.3750   0.2500   0.1250
     k =   0.5000   0.2500   0.1250
     k =   0.3750   0.3750   0.1250
     k =   0.5000   0.3750   0.1250
     k =   0.2500   0.2500   0.2500
     k =   0.3750   0.2500   0.2500
     k =   0.5000   0.2500   0.2500
     k =   0.3750   0.3750   0.2500
     k =   0.5000   0.5000   0.2500

     PROJWFC      :      2.00s CPU      2.20s WALL

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""

PW_INPUT_FILE = """
&CONTROL
  calculation = 'scf'
  prefix = 'test'
/
&SYSTEM
  ibrav = 0
  nat = 5
  ntyp = 3
  ecutwfc = 60.0
/
&ELECTRONS
/
ATOMIC_SPECIES
Sr  87.620  Sr.upf
V   50.942  V.upf
O   15.999  O.upf
ATOMIC_POSITIONS crystal
Sr  0.500  0.500  0.500
V   0.000  0.000  0.000
O   0.500  0.000  0.000
O   0.000  0.500  0.000
O   0.000  0.000  0.500
K_POINTS automatic
8 8 8 0 0 0
CELL_PARAMETERS angstrom
3.842  0.000  0.000
0.000  3.842  0.000
0.000  0.000  3.842
"""

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized test."""
    filepath = tmp_path / "projwfc.out"
    filepath.write_text(NON_SPIN_POLARIZED_PROJWFC_OUT)
    return filepath


@pytest.fixture
def non_spin_parser(non_spin_filepath: Path) -> ProjwfcOut:
    """Create parser instance for non-spin-polarized test."""
    return ProjwfcOut(filepath=non_spin_filepath)


@pytest.fixture
def spin_polarized_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized test."""
    filepath = tmp_path / "projwfc.out"
    filepath.write_text(SPIN_POLARIZED_PROJWFC_OUT)
    return filepath


@pytest.fixture
def spin_polarized_parser(spin_polarized_filepath: Path) -> ProjwfcOut:
    """Create parser instance for spin-polarized test."""
    return ProjwfcOut(filepath=spin_polarized_filepath)


@pytest.fixture
def pw_input_filepath(tmp_path: Path) -> Path:
    """Create temporary file for PW input test."""
    filepath = tmp_path / "scf.in"
    filepath.write_text(PW_INPUT_FILE)
    return filepath


# =============================================================================
# Tests: File Type Identification
# =============================================================================


def test_is_file_of_type_returns_true_for_valid_projwfc_output(
    non_spin_filepath: Path,
) -> None:
    """Test that is_file_of_type correctly identifies valid projwfc.out files."""
    assert ProjwfcOut.is_file_of_type(non_spin_filepath) is True


def test_is_file_of_type_returns_false_for_pw_output(pw_input_filepath: Path) -> None:
    """Test that is_file_of_type returns False for PW input files."""
    assert ProjwfcOut.is_file_of_type(pw_input_filepath) is False


# =============================================================================
# Tests: Problem Size Parsing
# =============================================================================


def test_n_atomic_wfc_returns_correct_count(non_spin_parser: ProjwfcOut) -> None:
    """Test that natomwfc parses correctly."""
    assert non_spin_parser.natomwfc == 25


def test_n_bands_returns_correct_count(non_spin_parser: ProjwfcOut) -> None:
    """Test that nbnd parses correctly."""
    assert non_spin_parser.nbnd == 24


def test_n_kpoints_non_spin_polarized(non_spin_parser: ProjwfcOut) -> None:
    """Test that nkstot parses correctly for non-spin-polarized."""
    assert non_spin_parser.nkstot == 29


def test_n_kpoints_spin_polarized(spin_polarized_parser: ProjwfcOut) -> None:
    """Test that nkstot parses correctly for spin-polarized (doubled k-points)."""
    assert spin_polarized_parser.nkstot == 58


# =============================================================================
# Tests: Spin Channel Detection
# =============================================================================


def test_n_spin_channels_non_spin_polarized(non_spin_parser: ProjwfcOut) -> None:
    """Test that non-spin-polarized has 1 spin channel."""
    assert non_spin_parser.n_spin_channels == 1


def test_n_spin_channels_spin_polarized(spin_polarized_parser: ProjwfcOut) -> None:
    """Test that spin-polarized has 2 spin channels."""
    assert spin_polarized_parser.n_spin_channels == 2


# =============================================================================
# Tests: Atomic Wavefunction Info Parsing
# =============================================================================


def test_atomic_wfc_info_length_matches_n_atomic_wfc(
    non_spin_parser: ProjwfcOut,
) -> None:
    """Test that atm_wfcs list length matches natomwfc."""
    assert len(non_spin_parser.atm_wfcs) == non_spin_parser.natomwfc


def test_atomic_wfc_info_contains_atom_index(non_spin_parser: ProjwfcOut) -> None:
    """Test that atm_wfcs contains atom_num field."""
    first_wfc = non_spin_parser.atm_wfcs[0]
    assert first_wfc["atom_num"] == 1


def test_atomic_wfc_info_contains_element_name(non_spin_parser: ProjwfcOut) -> None:
    """Test that atm_wfcs contains element field."""
    first_wfc = non_spin_parser.atm_wfcs[0]
    assert first_wfc["element"] == "Sr"


def test_atomic_wfc_info_contains_l_quantum_number(non_spin_parser: ProjwfcOut) -> None:
    """Test that atm_wfcs contains l quantum number."""
    first_wfc = non_spin_parser.atm_wfcs[0]
    assert first_wfc["l"] == 0


def test_atomic_wfc_info_contains_m_quantum_number(non_spin_parser: ProjwfcOut) -> None:
    """Test that atm_wfcs contains m quantum number."""
    first_wfc = non_spin_parser.atm_wfcs[0]
    assert first_wfc["m"] == 1
