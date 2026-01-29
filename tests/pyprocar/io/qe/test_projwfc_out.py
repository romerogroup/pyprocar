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

NON_COLINEAR_PROJWFC_OUT = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 j=0.5 m_j=-0.5)
     state #   2: atom   1 (Sr ), wfc  1 (l=0 j=0.5 m_j= 0.5)
     state #   3: atom   1 (Sr ), wfc  2 (l=0 j=0.5 m_j=-0.5)
     state #   4: atom   1 (Sr ), wfc  2 (l=0 j=0.5 m_j= 0.5)
     state #   5: atom   1 (Sr ), wfc  3 (l=1 j=0.5 m_j=-0.5)
     state #   6: atom   1 (Sr ), wfc  3 (l=1 j=0.5 m_j= 0.5)
     state #   7: atom   1 (Sr ), wfc  4 (l=1 j=1.5 m_j=-1.5)
     state #   8: atom   1 (Sr ), wfc  4 (l=1 j=1.5 m_j=-0.5)
     state #   9: atom   1 (Sr ), wfc  4 (l=1 j=1.5 m_j= 0.5)
     state #  10: atom   1 (Sr ), wfc  4 (l=1 j=1.5 m_j= 1.5)
     state #  11: atom   2 (V  ), wfc  1 (l=2 j=1.5 m_j=-1.5)
     state #  12: atom   2 (V  ), wfc  1 (l=2 j=1.5 m_j=-0.5)
     state #  13: atom   2 (V  ), wfc  1 (l=2 j=1.5 m_j= 0.5)
     state #  14: atom   2 (V  ), wfc  1 (l=2 j=1.5 m_j= 1.5)
     state #  15: atom   2 (V  ), wfc  2 (l=2 j=2.5 m_j=-2.5)
     state #  16: atom   2 (V  ), wfc  2 (l=2 j=2.5 m_j=-1.5)
     state #  17: atom   2 (V  ), wfc  2 (l=2 j=2.5 m_j=-0.5)
     state #  18: atom   2 (V  ), wfc  2 (l=2 j=2.5 m_j= 0.5)
     state #  19: atom   2 (V  ), wfc  2 (l=2 j=2.5 m_j= 1.5)
     state #  20: atom   2 (V  ), wfc  2 (l=2 j=2.5 m_j= 2.5)

     Calling projwave .... 

     natomwfc =   20
     nbnd     =   50
     nkstot   =  151

     k =   0.0000   0.0000   0.0000
     k =   0.0500   0.0000   0.0000

     PROJWFC      :      1.00s CPU      1.10s WALL

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
def non_colinear_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-colinear test."""
    filepath = tmp_path / "projwfc.out"
    filepath.write_text(NON_COLINEAR_PROJWFC_OUT)
    return filepath


@pytest.fixture
def non_colinear_parser(non_colinear_filepath: Path) -> ProjwfcOut:
    """Create parser instance for non-colinear test."""
    return ProjwfcOut(filepath=non_colinear_filepath)


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


# =============================================================================
# Tests: Non-colinear Detection
# =============================================================================


def test_is_non_colinear_true_for_non_colinear(
    non_colinear_parser: ProjwfcOut,
) -> None:
    """Test that is_non_colinear is True for non-colinear calculation."""
    assert non_colinear_parser.is_non_colinear is True


def test_is_non_colinear_false_for_colinear(non_spin_parser: ProjwfcOut) -> None:
    """Test that is_non_colinear is False for colinear calculation."""
    assert non_spin_parser.is_non_colinear is False


# =============================================================================
# Tests: Non-colinear Quantum Numbers
# =============================================================================


def test_atomic_wfc_info_contains_j_quantum_number(
    non_colinear_parser: ProjwfcOut,
) -> None:
    """Test that atm_wfcs contains j quantum number for non-colinear."""
    first_wfc = non_colinear_parser.atm_wfcs[0]
    assert first_wfc["j"] == pytest.approx(0.5)


def test_atomic_wfc_info_contains_m_j_quantum_number(
    non_colinear_parser: ProjwfcOut,
) -> None:
    """Test that atm_wfcs contains m_j quantum number for non-colinear."""
    first_wfc = non_colinear_parser.atm_wfcs[0]
    assert first_wfc["m_j"] == pytest.approx(-0.5)


def test_atomic_wfc_info_j_is_none_for_colinear(non_spin_parser: ProjwfcOut) -> None:
    """Test that atm_wfcs j is None for colinear calculation."""
    first_wfc = non_spin_parser.atm_wfcs[0]
    assert first_wfc["j"] is None


def test_atomic_wfc_info_m_j_is_none_for_colinear(non_spin_parser: ProjwfcOut) -> None:
    """Test that atm_wfcs m_j is None for colinear calculation."""
    first_wfc = non_spin_parser.atm_wfcs[0]
    assert first_wfc["m_j"] is None


def test_atomic_wfc_info_m_is_none_for_non_colinear(
    non_colinear_parser: ProjwfcOut,
) -> None:
    """Test that atm_wfcs m is None for non-colinear calculation."""
    first_wfc = non_colinear_parser.atm_wfcs[0]
    assert first_wfc["m"] is None


def test_n_atomic_wfc_non_colinear(non_colinear_parser: ProjwfcOut) -> None:
    """Test that natomwfc parses correctly for non-colinear."""
    assert non_colinear_parser.natomwfc == 20


def test_atomic_wfc_info_length_matches_n_atomic_wfc_non_colinear(
    non_colinear_parser: ProjwfcOut,
) -> None:
    """Test that atm_wfcs list length matches natomwfc for non-colinear."""
    assert len(non_colinear_parser.atm_wfcs) == non_colinear_parser.natomwfc


def test_atomic_wfc_info_d_orbital_j_1_5(non_colinear_parser: ProjwfcOut) -> None:
    """Test that d orbital with j=1.5 is parsed correctly."""
    # State 11 is atom 2 (V), wfc 1, l=2, j=1.5, m_j=-1.5
    wfc_11 = non_colinear_parser.atm_wfcs[10]  # 0-indexed
    assert wfc_11["l"] == 2
    assert wfc_11["j"] == pytest.approx(1.5)
    assert wfc_11["m_j"] == pytest.approx(-1.5)


def test_atomic_wfc_info_d_orbital_j_2_5(non_colinear_parser: ProjwfcOut) -> None:
    """Test that d orbital with j=2.5 is parsed correctly."""
    # State 15 is atom 2 (V), wfc 2, l=2, j=2.5, m_j=-2.5
    wfc_15 = non_colinear_parser.atm_wfcs[14]  # 0-indexed
    assert wfc_15["l"] == 2
    assert wfc_15["j"] == pytest.approx(2.5)
    assert wfc_15["m_j"] == pytest.approx(-2.5)


# =============================================================================
# Inline String Fixtures for k-point Resolved Band Projections (kpdos.out)
# =============================================================================

# Non-spin-polarized kpdos.out with 2 kpoints and 2 bands
KPDOS_NON_SPIN_POLARIZED = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Sr ), wfc  2 (l=0 m= 1)
     state #   3: atom   1 (Sr ), wfc  3 (l=1 m= 1)
     state #   4: atom   1 (Sr ), wfc  3 (l=1 m= 2)
     state #   5: atom   1 (Sr ), wfc  3 (l=1 m= 3)
     state #   6: atom   2 (V  ), wfc  1 (l=0 m= 1)
     state #   7: atom   2 (V  ), wfc  2 (l=1 m= 1)
     state #   8: atom   2 (V  ), wfc  2 (l=1 m= 2)
     state #   9: atom   2 (V  ), wfc  2 (l=1 m= 3)
     state #  10: atom   2 (V  ), wfc  3 (l=2 m= 1)

     Calling projwave .... 

     natomwfc =   10
     nbnd     =    2
     nkstot   =    2
     nspin    =    1

 k =   0.0000000000  0.0000000000  0.0000000000
==== e(   1) =   -53.32013 eV ==== 
     psi = 0.942*[#   9]+0.054*[#   2]+0.002*[#  10]
    |psi|^2 = 0.998
==== e(   2) =   -27.14377 eV ==== 
     psi = 0.327*[#   3]+0.327*[#   4]+0.327*[#   5]+0.006*[#   6]
    |psi|^2 = 0.987

 k =   0.0166666667  0.0000000000  0.0000000000
==== e(   1) =   -53.32014 eV ==== 
     psi = 0.942*[#   9]+0.054*[#   2]+0.002*[#  10]
    |psi|^2 = 0.998
==== e(   2) =   -27.14388 eV ==== 
     psi = 0.327*[#   3]+0.327*[#   4]+0.327*[#   5]+0.006*[#   6]
    |psi|^2 = 0.987

     PROJWFC      :      1.00s CPU      1.10s WALL

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""

# Spin-polarized kpdos.out with 4 kpoints (2 per spin) and 2 bands
KPDOS_SPIN_POLARIZED = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Sr ), wfc  2 (l=0 m= 1)
     state #   3: atom   1 (Sr ), wfc  3 (l=1 m= 1)
     state #   4: atom   1 (Sr ), wfc  3 (l=1 m= 2)
     state #   5: atom   1 (Sr ), wfc  3 (l=1 m= 3)
     state #   6: atom   2 (V  ), wfc  1 (l=0 m= 1)
     state #   7: atom   2 (V  ), wfc  2 (l=1 m= 1)
     state #   8: atom   2 (V  ), wfc  2 (l=1 m= 2)
     state #   9: atom   2 (V  ), wfc  2 (l=1 m= 3)
     state #  10: atom   2 (V  ), wfc  3 (l=2 m= 1)

     Calling projwave .... 

     natomwfc =   10
     nbnd     =    2
     nkstot   =    4
     nspin    =    2

 k =   0.0000000000  0.0000000000  0.0000000000
==== e(   1) =   -53.30767 eV ==== 
     psi = 0.942*[#   9]+0.054*[#   2]+0.002*[#  10]
    |psi|^2 = 0.998
==== e(   2) =   -27.13208 eV ==== 
     psi = 0.327*[#   3]+0.327*[#   4]+0.327*[#   5]+0.006*[#   6]
    |psi|^2 = 0.987

 k =   0.0166666667  0.0000000000  0.0000000000
==== e(   1) =   -53.30767 eV ==== 
     psi = 0.942*[#   9]+0.054*[#   2]+0.002*[#  10]
    |psi|^2 = 0.998
==== e(   2) =   -27.13218 eV ==== 
     psi = 0.327*[#   3]+0.327*[#   4]+0.327*[#   5]+0.006*[#   6]
    |psi|^2 = 0.987

 k =   0.0000000000  0.0000000000  0.0000000000
==== e(   1) =   -53.30589 eV ==== 
     psi = 0.940*[#   9]+0.055*[#   2]+0.003*[#  10]
    |psi|^2 = 0.998
==== e(   2) =   -27.13033 eV ==== 
     psi = 0.325*[#   3]+0.325*[#   4]+0.325*[#   5]+0.008*[#   6]
    |psi|^2 = 0.983

 k =   0.0166666667  0.0000000000  0.0000000000
==== e(   1) =   -53.30590 eV ==== 
     psi = 0.940*[#   9]+0.055*[#   2]+0.003*[#  10]
    |psi|^2 = 0.998
==== e(   2) =   -27.13044 eV ==== 
     psi = 0.325*[#   3]+0.325*[#   4]+0.325*[#   5]+0.008*[#   6]
    |psi|^2 = 0.983

     PROJWFC      :      2.00s CPU      2.20s WALL

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""


# =============================================================================
# Pytest Fixtures for k-point Resolved Band Projections
# =============================================================================


@pytest.fixture
def kpdos_non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized kpdos test."""
    filepath = tmp_path / "kpdos.out"
    filepath.write_text(KPDOS_NON_SPIN_POLARIZED)
    return filepath


@pytest.fixture
def kpdos_non_spin_parser(kpdos_non_spin_filepath: Path) -> ProjwfcOut:
    """Create parser instance for non-spin-polarized kpdos test."""
    return ProjwfcOut(filepath=kpdos_non_spin_filepath)


@pytest.fixture
def kpdos_spin_polarized_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized kpdos test."""
    filepath = tmp_path / "kpdos.out"
    filepath.write_text(KPDOS_SPIN_POLARIZED)
    return filepath


@pytest.fixture
def kpdos_spin_polarized_parser(kpdos_spin_polarized_filepath: Path) -> ProjwfcOut:
    """Create parser instance for spin-polarized kpdos test."""
    return ProjwfcOut(filepath=kpdos_spin_polarized_filepath)


# =============================================================================
# Tests: kpdos.out Band Parsing - Non-spin-polarized
# =============================================================================


def test_kpdos_non_spin_bands_shape(kpdos_non_spin_parser: ProjwfcOut) -> None:
    """Test that bands array has correct shape for non-spin-polarized kpdos."""
    bands = kpdos_non_spin_parser.bands
    assert bands is not None
    # Shape: (nkstot, nbnd, n_spin_channels) = (2, 2, 1)
    assert bands.shape == (2, 2, 1)


def test_kpdos_non_spin_bands_first_kpoint_first_band(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test band energy for first k-point, first band in non-spin-polarized."""
    bands = kpdos_non_spin_parser.bands
    assert bands is not None
    assert bands[0, 0, 0] == pytest.approx(-53.32013, rel=1e-5)


def test_kpdos_non_spin_bands_first_kpoint_second_band(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test band energy for first k-point, second band in non-spin-polarized."""
    bands = kpdos_non_spin_parser.bands
    assert bands is not None
    assert bands[0, 1, 0] == pytest.approx(-27.14377, rel=1e-5)


def test_kpdos_non_spin_bands_second_kpoint_first_band(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test band energy for second k-point, first band in non-spin-polarized."""
    bands = kpdos_non_spin_parser.bands
    assert bands is not None
    assert bands[1, 0, 0] == pytest.approx(-53.32014, rel=1e-5)


def test_kpdos_non_spin_bands_second_kpoint_second_band(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test band energy for second k-point, second band in non-spin-polarized."""
    bands = kpdos_non_spin_parser.bands
    assert bands is not None
    assert bands[1, 1, 0] == pytest.approx(-27.14388, rel=1e-5)


# =============================================================================
# Tests: kpdos.out psi2 Parsing - Non-spin-polarized
# =============================================================================


def test_kpdos_non_spin_psi2_shape(kpdos_non_spin_parser: ProjwfcOut) -> None:
    """Test that psi2 array has correct shape for non-spin-polarized kpdos."""
    psi2 = kpdos_non_spin_parser.psi2
    assert psi2 is not None
    # Shape: (nkstot, nbnd, n_spin_channels) = (2, 2, 1)
    assert psi2.shape == (2, 2, 1)


def test_kpdos_non_spin_psi2_first_kpoint_first_band(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test psi2 value for first k-point, first band in non-spin-polarized."""
    psi2 = kpdos_non_spin_parser.psi2
    assert psi2 is not None
    assert psi2[0, 0, 0] == pytest.approx(0.998, rel=1e-3)


def test_kpdos_non_spin_psi2_first_kpoint_second_band(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test psi2 value for first k-point, second band in non-spin-polarized."""
    psi2 = kpdos_non_spin_parser.psi2
    assert psi2 is not None
    assert psi2[0, 1, 0] == pytest.approx(0.987, rel=1e-3)


# =============================================================================
# Tests: kpdos.out psi_coeffs Parsing - Non-spin-polarized
# =============================================================================


def test_kpdos_non_spin_psi_coeffs_shape(kpdos_non_spin_parser: ProjwfcOut) -> None:
    """Test that psi_coeffs array has correct shape for non-spin-polarized kpdos."""
    psi_coeffs = kpdos_non_spin_parser.psi_coeffs
    assert psi_coeffs is not None
    # Shape: (nkstot, nbnd, n_spin_channels, natomwfc) = (2, 2, 1, 10)
    assert psi_coeffs.shape == (2, 2, 1, 10)


def test_kpdos_non_spin_psi_coeffs_first_kpoint_first_band_state_9(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test psi coefficient for state #9 at first k-point, first band."""
    psi_coeffs = kpdos_non_spin_parser.psi_coeffs
    assert psi_coeffs is not None
    # State #9 is index 8 (0-based), coefficient should be 0.942
    assert psi_coeffs[0, 0, 0, 8] == pytest.approx(0.942, rel=1e-3)


def test_kpdos_non_spin_psi_coeffs_first_kpoint_first_band_state_2(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test psi coefficient for state #2 at first k-point, first band."""
    psi_coeffs = kpdos_non_spin_parser.psi_coeffs
    assert psi_coeffs is not None
    # State #2 is index 1 (0-based), coefficient should be 0.054
    assert psi_coeffs[0, 0, 0, 1] == pytest.approx(0.054, rel=1e-3)


def test_kpdos_non_spin_psi_coeffs_first_kpoint_second_band_state_3(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test psi coefficient for state #3 at first k-point, second band."""
    psi_coeffs = kpdos_non_spin_parser.psi_coeffs
    assert psi_coeffs is not None
    # State #3 is index 2 (0-based), coefficient should be 0.327
    assert psi_coeffs[0, 1, 0, 2] == pytest.approx(0.327, rel=1e-3)


def test_kpdos_non_spin_psi_coeffs_zero_for_missing_state(
    kpdos_non_spin_parser: ProjwfcOut,
) -> None:
    """Test that missing psi coefficients are zero."""
    psi_coeffs = kpdos_non_spin_parser.psi_coeffs
    assert psi_coeffs is not None
    # State #1 is index 0, not present in first band psi, should be 0.0
    assert psi_coeffs[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Tests: kpdos.out Band Parsing - Spin-polarized
# NOTE: For spin-polarized, reshape results in indexing [spin, kpoint, band]
# =============================================================================


def test_kpdos_spin_bands_shape(kpdos_spin_polarized_parser: ProjwfcOut) -> None:
    """Test that bands array has correct shape for spin-polarized kpdos."""
    bands = kpdos_spin_polarized_parser.bands
    assert bands is not None
    # Shape: (2, 2, 2) - actual layout is [spin, kpoint, band]
    assert bands.shape == (2, 2, 2)


def test_kpdos_spin_bands_first_kpoint_first_band_spin_up(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test band energy for first k-point, first band, spin up."""
    bands = kpdos_spin_polarized_parser.bands
    assert bands is not None
    # Indexing: [spin, kpoint, band]
    assert bands[0, 0, 0] == pytest.approx(-53.30767, rel=1e-5)


def test_kpdos_spin_bands_first_kpoint_second_band_spin_up(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test band energy for first k-point, second band, spin up."""
    bands = kpdos_spin_polarized_parser.bands
    assert bands is not None
    # Indexing: [spin, kpoint, band]
    assert bands[0, 0, 1] == pytest.approx(-27.13208, rel=1e-5)


def test_kpdos_spin_bands_first_kpoint_first_band_spin_down(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test band energy for first k-point, first band, spin down."""
    bands = kpdos_spin_polarized_parser.bands
    assert bands is not None
    # Indexing: [spin, kpoint, band]
    assert bands[1, 0, 0] == pytest.approx(-53.30589, rel=1e-5)


def test_kpdos_spin_bands_first_kpoint_second_band_spin_down(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test band energy for first k-point, second band, spin down."""
    bands = kpdos_spin_polarized_parser.bands
    assert bands is not None
    # Indexing: [spin, kpoint, band]
    assert bands[1, 0, 1] == pytest.approx(-27.13033, rel=1e-5)


def test_kpdos_spin_bands_second_kpoint_second_band_spin_down(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test band energy for second k-point, second band, spin down."""
    bands = kpdos_spin_polarized_parser.bands
    assert bands is not None
    # Indexing: [spin, kpoint, band]
    assert bands[1, 1, 1] == pytest.approx(-27.13044, rel=1e-5)


# =============================================================================
# Tests: kpdos.out psi2 Parsing - Spin-polarized
# NOTE: For spin-polarized, reshape results in indexing [spin, kpoint, band]
# =============================================================================


def test_kpdos_spin_psi2_shape(kpdos_spin_polarized_parser: ProjwfcOut) -> None:
    """Test that psi2 array has correct shape for spin-polarized kpdos."""
    psi2 = kpdos_spin_polarized_parser.psi2
    assert psi2 is not None
    # Shape: (2, 2, 2) - actual layout is [spin, kpoint, band]
    assert psi2.shape == (2, 2, 2)


def test_kpdos_spin_psi2_first_kpoint_first_band_spin_up(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test psi2 value for first k-point, first band, spin up."""
    psi2 = kpdos_spin_polarized_parser.psi2
    assert psi2 is not None
    # Indexing: [spin, kpoint, band]
    assert psi2[0, 0, 0] == pytest.approx(0.998, rel=1e-3)


def test_kpdos_spin_psi2_first_kpoint_second_band_spin_down(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test psi2 value for first k-point, second band, spin down."""
    psi2 = kpdos_spin_polarized_parser.psi2
    assert psi2 is not None
    # Indexing: [spin, kpoint, band]
    assert psi2[1, 0, 1] == pytest.approx(0.983, rel=1e-3)


# =============================================================================
# Tests: kpdos.out psi_coeffs Parsing - Spin-polarized
# NOTE: For spin-polarized, reshape results in indexing [spin, kpoint, band, atomwfc]
# =============================================================================


def test_kpdos_spin_psi_coeffs_shape(kpdos_spin_polarized_parser: ProjwfcOut) -> None:
    """Test that psi_coeffs array has correct shape for spin-polarized kpdos."""
    psi_coeffs = kpdos_spin_polarized_parser.psi_coeffs
    assert psi_coeffs is not None
    # Shape: (2, 2, 2, 10) - actual layout is [spin, kpoint, band, atomwfc]
    assert psi_coeffs.shape == (2, 2, 2, 10)


def test_kpdos_spin_psi_coeffs_first_kpoint_first_band_spin_up_state_9(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test psi coefficient for state #9 at first k-point, first band, spin up."""
    psi_coeffs = kpdos_spin_polarized_parser.psi_coeffs
    assert psi_coeffs is not None
    # Indexing: [spin, kpoint, band, atomwfc]
    # State #9 is index 8 (0-based), coefficient should be 0.942 for spin up
    assert psi_coeffs[0, 0, 0, 8] == pytest.approx(0.942, rel=1e-3)


def test_kpdos_spin_psi_coeffs_first_kpoint_first_band_spin_down_state_9(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test psi coefficient for state #9 at first k-point, first band, spin down."""
    psi_coeffs = kpdos_spin_polarized_parser.psi_coeffs
    assert psi_coeffs is not None
    # Indexing: [spin, kpoint, band, atomwfc]
    # State #9 is index 8 (0-based), coefficient should be 0.940 for spin down
    assert psi_coeffs[1, 0, 0, 8] == pytest.approx(0.940, rel=1e-3)


def test_kpdos_spin_psi_coeffs_first_kpoint_second_band_spin_up_state_3(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test psi coefficient for state #3 at first k-point, second band, spin up."""
    psi_coeffs = kpdos_spin_polarized_parser.psi_coeffs
    assert psi_coeffs is not None
    # Indexing: [spin, kpoint, band, atomwfc]
    # State #3 is index 2 (0-based), coefficient should be 0.327 for spin up
    assert psi_coeffs[0, 0, 1, 2] == pytest.approx(0.327, rel=1e-3)


def test_kpdos_spin_psi_coeffs_first_kpoint_second_band_spin_down_state_3(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test psi coefficient for state #3 at first k-point, second band, spin down."""
    psi_coeffs = kpdos_spin_polarized_parser.psi_coeffs
    assert psi_coeffs is not None
    # Indexing: [spin, kpoint, band, atomwfc]
    # State #3 is index 2 (0-based), coefficient should be 0.325 for spin down
    assert psi_coeffs[1, 0, 1, 2] == pytest.approx(0.325, rel=1e-3)


def test_kpdos_spin_psi_coeffs_different_between_spin_channels(
    kpdos_spin_polarized_parser: ProjwfcOut,
) -> None:
    """Test that psi coefficients differ between spin channels."""
    psi_coeffs = kpdos_spin_polarized_parser.psi_coeffs
    assert psi_coeffs is not None
    # Indexing: [spin, kpoint, band, atomwfc]
    # State #6 (index 5) has different values for spin up (0.006) vs spin down (0.008)
    assert psi_coeffs[0, 0, 1, 5] == pytest.approx(0.006, rel=1e-2)
    assert psi_coeffs[1, 0, 1, 5] == pytest.approx(0.008, rel=1e-2)


# =============================================================================
# Tests: kpdos.out kpoints Parsing
# =============================================================================


def test_kpdos_non_spin_kpoints_shape(kpdos_non_spin_parser: ProjwfcOut) -> None:
    """Test that kpoints array has correct shape for non-spin-polarized kpdos."""
    kpoints = kpdos_non_spin_parser.kpoints
    assert kpoints is not None
    # Shape: (nkstot, 3) = (2, 3)
    assert kpoints.shape == (2, 3)


def test_kpdos_non_spin_kpoints_first(kpdos_non_spin_parser: ProjwfcOut) -> None:
    """Test first k-point coordinates for non-spin-polarized."""
    kpoints = kpdos_non_spin_parser.kpoints
    assert kpoints is not None
    assert kpoints[0, 0] == pytest.approx(0.0, abs=1e-9)
    assert kpoints[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert kpoints[0, 2] == pytest.approx(0.0, abs=1e-9)


def test_kpdos_non_spin_kpoints_second(kpdos_non_spin_parser: ProjwfcOut) -> None:
    """Test second k-point coordinates for non-spin-polarized."""
    kpoints = kpdos_non_spin_parser.kpoints
    assert kpoints is not None
    assert kpoints[1, 0] == pytest.approx(0.0166666667, rel=1e-6)
    assert kpoints[1, 1] == pytest.approx(0.0, abs=1e-9)
    assert kpoints[1, 2] == pytest.approx(0.0, abs=1e-9)


def test_kpdos_spin_kpoints_shape(kpdos_spin_polarized_parser: ProjwfcOut) -> None:
    """Test that kpoints array has correct shape for spin-polarized kpdos."""
    kpoints = kpdos_spin_polarized_parser.kpoints
    assert kpoints is not None
    # Shape: (nkstot//2, 3) = (2, 3) because duplicate k-points are removed
    assert kpoints.shape == (2, 3)


# =============================================================================
# Inline String Fixtures for Lowdin Charges Parsing
# =============================================================================

# Non-spin-polarized projwfc.out with Lowdin charges
LOWDIN_NON_SPIN_POLARIZED = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Sr ), wfc  2 (l=1 m= 1)
     state #   3: atom   1 (Sr ), wfc  2 (l=1 m= 2)
     state #   4: atom   1 (Sr ), wfc  2 (l=1 m= 3)
     state #   5: atom   1 (Sr ), wfc  3 (l=2 m= 1)
     state #   6: atom   1 (Sr ), wfc  3 (l=2 m= 2)
     state #   7: atom   1 (Sr ), wfc  3 (l=2 m= 3)
     state #   8: atom   1 (Sr ), wfc  3 (l=2 m= 4)
     state #   9: atom   1 (Sr ), wfc  3 (l=2 m= 5)
     state #  10: atom   2 (V  ), wfc  1 (l=0 m= 1)
     state #  11: atom   2 (V  ), wfc  2 (l=1 m= 1)
     state #  12: atom   2 (V  ), wfc  2 (l=1 m= 2)
     state #  13: atom   2 (V  ), wfc  2 (l=1 m= 3)
     state #  14: atom   2 (V  ), wfc  3 (l=2 m= 1)
     state #  15: atom   2 (V  ), wfc  3 (l=2 m= 2)
     state #  16: atom   2 (V  ), wfc  3 (l=2 m= 3)
     state #  17: atom   2 (V  ), wfc  3 (l=2 m= 4)
     state #  18: atom   2 (V  ), wfc  3 (l=2 m= 5)
     state #  19: atom   3 (O  ), wfc  1 (l=0 m= 1)
     state #  20: atom   3 (O  ), wfc  2 (l=1 m= 1)
     state #  21: atom   3 (O  ), wfc  2 (l=1 m= 2)
     state #  22: atom   3 (O  ), wfc  2 (l=1 m= 3)
     state #  23: atom   4 (O  ), wfc  1 (l=0 m= 1)
     state #  24: atom   4 (O  ), wfc  2 (l=1 m= 1)
     state #  25: atom   4 (O  ), wfc  2 (l=1 m= 2)
     state #  26: atom   4 (O  ), wfc  2 (l=1 m= 3)
     state #  27: atom   5 (O  ), wfc  1 (l=0 m= 1)
     state #  28: atom   5 (O  ), wfc  2 (l=1 m= 1)
     state #  29: atom   5 (O  ), wfc  2 (l=1 m= 2)
     state #  30: atom   5 (O  ), wfc  2 (l=1 m= 3)

     Calling projwave .... 

     natomwfc =   30
     nbnd     =   24
     nkstot   =   29
     nspin    =    1

Lowdin Charges: 

     Atom #   1: total charge =   8.7713, s =  2.2034, 
     Atom #   1: total charge =   8.7713, p =  6.5679, pz=  2.1893, px=  2.1893, py=  2.1893, 
     Atom #   1: total charge =   8.7713, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
     Atom #   2: total charge =  12.3518, s =  2.2697, 
     Atom #   2: total charge =  12.3518, p =  5.9915, pz=  1.9972, px=  1.9972, py=  1.9972, 
     Atom #   2: total charge =  12.3518, d =  4.0906, dz2=  0.5649, dxz=  0.9869, dyz=  0.9869, dx2-y2=  0.5649, dxy=  0.9869, 
     Atom #   3: total charge =   6.8633, s =  1.7126, 
     Atom #   3: total charge =   6.8633, p =  5.1507, pz=  1.7498, px=  1.7498, py=  1.6510, 
     Atom #   3: total charge =   6.8633, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
     Atom #   4: total charge =   6.8633, s =  1.7126, 
     Atom #   4: total charge =   6.8633, p =  5.1507, pz=  1.7498, px=  1.6510, py=  1.7498, 
     Atom #   4: total charge =   6.8633, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
     Atom #   5: total charge =   6.8633, s =  1.7126, 
     Atom #   5: total charge =   6.8633, p =  5.1507, pz=  1.6510, px=  1.7498, py=  1.7498, 
     Atom #   5: total charge =   6.8633, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
     Spilling Parameter:  -0.0174
 
     PROJWFC      :     32.27s CPU   1m26.28s WALL


   This run was terminated on:  11:30: 1  19Jul2024         

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""

# Spin-polarized projwfc.out with Lowdin charges
LOWDIN_SPIN_POLARIZED = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Sr ), wfc  2 (l=1 m= 1)
     state #   3: atom   1 (Sr ), wfc  2 (l=1 m= 2)
     state #   4: atom   1 (Sr ), wfc  2 (l=1 m= 3)
     state #   5: atom   1 (Sr ), wfc  3 (l=2 m= 1)
     state #   6: atom   1 (Sr ), wfc  3 (l=2 m= 2)
     state #   7: atom   1 (Sr ), wfc  3 (l=2 m= 3)
     state #   8: atom   1 (Sr ), wfc  3 (l=2 m= 4)
     state #   9: atom   1 (Sr ), wfc  3 (l=2 m= 5)
     state #  10: atom   2 (V  ), wfc  1 (l=0 m= 1)
     state #  11: atom   2 (V  ), wfc  2 (l=1 m= 1)
     state #  12: atom   2 (V  ), wfc  2 (l=1 m= 2)
     state #  13: atom   2 (V  ), wfc  2 (l=1 m= 3)
     state #  14: atom   2 (V  ), wfc  3 (l=2 m= 1)
     state #  15: atom   2 (V  ), wfc  3 (l=2 m= 2)
     state #  16: atom   2 (V  ), wfc  3 (l=2 m= 3)
     state #  17: atom   2 (V  ), wfc  3 (l=2 m= 4)
     state #  18: atom   2 (V  ), wfc  3 (l=2 m= 5)
     state #  19: atom   3 (O  ), wfc  1 (l=0 m= 1)
     state #  20: atom   3 (O  ), wfc  2 (l=1 m= 1)
     state #  21: atom   3 (O  ), wfc  2 (l=1 m= 2)
     state #  22: atom   3 (O  ), wfc  2 (l=1 m= 3)
     state #  23: atom   4 (O  ), wfc  1 (l=0 m= 1)
     state #  24: atom   4 (O  ), wfc  2 (l=1 m= 1)
     state #  25: atom   4 (O  ), wfc  2 (l=1 m= 2)
     state #  26: atom   4 (O  ), wfc  2 (l=1 m= 3)
     state #  27: atom   5 (O  ), wfc  1 (l=0 m= 1)
     state #  28: atom   5 (O  ), wfc  2 (l=1 m= 1)
     state #  29: atom   5 (O  ), wfc  2 (l=1 m= 2)
     state #  30: atom   5 (O  ), wfc  2 (l=1 m= 3)

     Calling projwave .... 

     natomwfc =   30
     nbnd     =   24
     nkstot   =   58
     nspin    =    2

 k =   0.0000000000  0.0000000000  0.0000000000
 k =   0.0000000000  0.0000000000  0.0000000000

Lowdin Charges: 

     Atom #   1: total charge =   8.7712, s =  2.2033, p =  6.5679, d =  0.0000, 
                 spin up      =   4.3856, s =  1.1017, 
                 spin up      =   4.3856, p =  3.2839, pz=  1.0946, px=  1.0946, py=  1.0946, 
                 spin up      =   4.3856, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 spin down    =   4.3856, s =  1.1017, 
                 spin down    =   4.3856, p =  3.2839, pz=  1.0946, px=  1.0946, py=  1.0946, 
                 spin down    =   4.3856, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 polarization =  -0.0000, s = -0.0000, p = -0.0000, d =  0.0000, 
     Atom #   2: total charge =  12.3367, s =  2.2696, p =  5.9915, d =  4.0755, 
                 spin up      =   6.1689, s =  1.1348, 
                 spin up      =   6.1689, p =  2.9958, pz=  0.9986, px=  0.9986, py=  0.9986, 
                 spin up      =   6.1689, d =  2.0383, dz2=  0.2821, dxz=  0.4914, dyz=  0.4914, dx2-y2=  0.2821, dxy=  0.4914, 
                 spin down    =   6.1677, s =  1.1348, 
                 spin down    =   6.1677, p =  2.9958, pz=  0.9986, px=  0.9986, py=  0.9986, 
                 spin down    =   6.1677, d =  2.0372, dz2=  0.2821, dxz=  0.4910, dyz=  0.4910, dx2-y2=  0.2821, dxy=  0.4910, 
                 polarization =   0.0012, s =  0.0000, p = -0.0000, d =  0.0012, 
     Atom #   3: total charge =   6.8639, s =  1.7127, p =  5.1513, d =  0.0000, 
                 spin up      =   3.4320, s =  0.8563, 
                 spin up      =   3.4320, p =  2.5756, pz=  0.8750, px=  0.8750, py=  0.8257, 
                 spin up      =   3.4320, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 spin down    =   3.4320, s =  0.8563, 
                 spin down    =   3.4320, p =  2.5756, pz=  0.8749, px=  0.8749, py=  0.8257, 
                 spin down    =   3.4320, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 polarization =  -0.0000, s =  0.0000, p = -0.0000, d =  0.0000, 
     Atom #   4: total charge =   6.8639, s =  1.7127, p =  5.1513, d =  0.0000, 
                 spin up      =   3.4320, s =  0.8563, 
                 spin up      =   3.4320, p =  2.5756, pz=  0.8750, px=  0.8257, py=  0.8750, 
                 spin up      =   3.4320, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 spin down    =   3.4320, s =  0.8563, 
                 spin down    =   3.4320, p =  2.5756, pz=  0.8749, px=  0.8257, py=  0.8749, 
                 spin down    =   3.4320, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 polarization =  -0.0000, s =  0.0000, p = -0.0000, d =  0.0000, 
     Atom #   5: total charge =   6.8639, s =  1.7127, p =  5.1513, d =  0.0000, 
                 spin up      =   3.4320, s =  0.8563, 
                 spin up      =   3.4320, p =  2.5756, pz=  0.8257, px=  0.8750, py=  0.8750, 
                 spin up      =   3.4320, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 spin down    =   3.4320, s =  0.8563, 
                 spin down    =   3.4320, p =  2.5756, pz=  0.8257, px=  0.8749, py=  0.8749, 
                 spin down    =   3.4320, d =  0.0000, dz2=  0.0000, dxz=  0.0000, dyz=  0.0000, dx2-y2=  0.0000, dxy=  0.0000, 
                 polarization =  -0.0000, s =  0.0000, p = -0.0000, d =  0.0000, 
     Spilling Parameter:  -0.0171
 
     PROJWFC      :     41.67s CPU   1m38.01s WALL

 
   This run was terminated on:  11:30:30  19Jul2024            

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""

# Non-colinear projwfc.out with Lowdin charges
LOWDIN_NON_COLINEAR = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     Calling projwave .... 

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Sr ), wfc  1 (l=0 j=0.5 m_j=-0.5)
     state #   2: atom   1 (Sr ), wfc  1 (l=0 j=0.5 m_j= 0.5)
     state #   3: atom   1 (Sr ), wfc  2 (l=1 j=0.5 m_j=-0.5)
     state #   4: atom   1 (Sr ), wfc  2 (l=1 j=0.5 m_j= 0.5)
     state #   5: atom   1 (Sr ), wfc  3 (l=1 j=1.5 m_j=-1.5)
     state #   6: atom   1 (Sr ), wfc  3 (l=1 j=1.5 m_j=-0.5)
     state #   7: atom   1 (Sr ), wfc  3 (l=1 j=1.5 m_j= 0.5)
     state #   8: atom   1 (Sr ), wfc  3 (l=1 j=1.5 m_j= 1.5)
     state #   9: atom   2 (V  ), wfc  1 (l=0 j=0.5 m_j=-0.5)
     state #  10: atom   2 (V  ), wfc  1 (l=0 j=0.5 m_j= 0.5)
     state #  11: atom   2 (V  ), wfc  2 (l=1 j=0.5 m_j=-0.5)
     state #  12: atom   2 (V  ), wfc  2 (l=1 j=0.5 m_j= 0.5)
     state #  13: atom   2 (V  ), wfc  3 (l=1 j=1.5 m_j=-1.5)
     state #  14: atom   2 (V  ), wfc  3 (l=1 j=1.5 m_j=-0.5)
     state #  15: atom   2 (V  ), wfc  3 (l=1 j=1.5 m_j= 0.5)
     state #  16: atom   2 (V  ), wfc  3 (l=1 j=1.5 m_j= 1.5)
     state #  17: atom   2 (V  ), wfc  4 (l=2 j=1.5 m_j=-1.5)
     state #  18: atom   2 (V  ), wfc  4 (l=2 j=1.5 m_j=-0.5)
     state #  19: atom   2 (V  ), wfc  4 (l=2 j=1.5 m_j= 0.5)
     state #  20: atom   2 (V  ), wfc  4 (l=2 j=1.5 m_j= 1.5)
     state #  21: atom   3 (O  ), wfc  1 (l=0 j=0.5 m_j=-0.5)
     state #  22: atom   3 (O  ), wfc  1 (l=0 j=0.5 m_j= 0.5)
     state #  23: atom   3 (O  ), wfc  2 (l=1 j=0.5 m_j=-0.5)
     state #  24: atom   3 (O  ), wfc  2 (l=1 j=0.5 m_j= 0.5)
     state #  25: atom   4 (O  ), wfc  1 (l=0 j=0.5 m_j=-0.5)
     state #  26: atom   4 (O  ), wfc  1 (l=0 j=0.5 m_j= 0.5)
     state #  27: atom   4 (O  ), wfc  2 (l=1 j=0.5 m_j=-0.5)
     state #  28: atom   4 (O  ), wfc  2 (l=1 j=0.5 m_j= 0.5)
     state #  29: atom   5 (O  ), wfc  1 (l=0 j=0.5 m_j=-0.5)
     state #  30: atom   5 (O  ), wfc  1 (l=0 j=0.5 m_j= 0.5)

     Calling projwave .... 

     natomwfc =   30
     nbnd     =   50
     nkstot   =  151

Lowdin Charges: 

     Atom #   1: total charge =   8.7714, s =  2.2034, 
     Atom #   1: total charge =   8.7714, p =  6.5680, 
     Atom #   1: total charge =   8.7714, d =  0.0000, 
     Atom #   2: total charge =  12.3437, s =  2.2697, 
     Atom #   2: total charge =  12.3437, p =  5.9915, 
     Atom #   2: total charge =  12.3437, d =  4.0824, 
     Atom #   3: total charge =   6.7339, s =  1.7441, 
     Atom #   3: total charge =   6.7339, p =  4.9897, 
     Atom #   3: total charge =   6.7339, d =  0.0000, 
     Atom #   4: total charge =   6.7339, s =  1.7441, 
     Atom #   4: total charge =   6.7339, p =  4.9897, 
     Atom #   4: total charge =   6.7339, d =  0.0000, 
     Atom #   5: total charge =   7.1212, s =  1.6496, 
     Atom #   5: total charge =   7.1212, p =  5.4716, 
     Atom #   5: total charge =   7.1212, d =  0.0000, 
     Spilling Parameter:  -0.0172
 
     PROJWFC      :     52.32s CPU   2m16.28s WALL

 
   This run was terminated on:  11:32:24  19Jul2024            

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""


# =============================================================================
# Pytest Fixtures for Lowdin Charges
# =============================================================================


@pytest.fixture
def lowdin_non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized Lowdin charges test."""
    filepath = tmp_path / "projwfc.out"
    filepath.write_text(LOWDIN_NON_SPIN_POLARIZED)
    return filepath


@pytest.fixture
def lowdin_non_spin_parser(lowdin_non_spin_filepath: Path) -> ProjwfcOut:
    """Create parser instance for non-spin-polarized Lowdin charges test."""
    return ProjwfcOut(filepath=lowdin_non_spin_filepath)


@pytest.fixture
def lowdin_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized Lowdin charges test."""
    filepath = tmp_path / "projwfc.out"
    filepath.write_text(LOWDIN_SPIN_POLARIZED)
    return filepath


@pytest.fixture
def lowdin_spin_parser(lowdin_spin_filepath: Path) -> ProjwfcOut:
    """Create parser instance for spin-polarized Lowdin charges test."""
    return ProjwfcOut(filepath=lowdin_spin_filepath)


@pytest.fixture
def lowdin_non_colinear_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-colinear Lowdin charges test."""
    filepath = tmp_path / "projwfc.out"
    filepath.write_text(LOWDIN_NON_COLINEAR)
    return filepath


@pytest.fixture
def lowdin_non_colinear_parser(lowdin_non_colinear_filepath: Path) -> ProjwfcOut:
    """Create parser instance for non-colinear Lowdin charges test."""
    return ProjwfcOut(filepath=lowdin_non_colinear_filepath)


# =============================================================================
# Tests: Lowdin Charges - Non-spin-polarized
# =============================================================================


def test_lowdin_non_spin_returns_not_none(lowdin_non_spin_parser: ProjwfcOut) -> None:
    """Test that lowdin_charges returns a dict for non-spin-polarized."""
    assert lowdin_non_spin_parser.lowdin_charges is not None


def test_lowdin_non_spin_spilling_parameter(lowdin_non_spin_parser: ProjwfcOut) -> None:
    """Test spilling parameter for non-spin-polarized."""
    assert lowdin_non_spin_parser.spilling_parameter == pytest.approx(-0.0174, rel=1e-3)


def test_lowdin_non_spin_total_charges_shape(lowdin_non_spin_parser: ProjwfcOut) -> None:
    """Test total charges array shape for non-spin-polarized."""
    total_charges = lowdin_non_spin_parser.total_lowdin_charges
    assert total_charges is not None
    # Shape: (n_atoms, n_l_orbitals) = (5, 4)
    assert total_charges.shape == (5, 4)


def test_lowdin_non_spin_atom1_s_orbital_charge(
    lowdin_non_spin_parser: ProjwfcOut,
) -> None:
    """Test atom 1 (Sr) s-orbital charge for non-spin-polarized."""
    charges_per_l = lowdin_non_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, s-orbital (l=0), spin channel 0
    assert charges_per_l[0, 0, 0] == pytest.approx(2.2034, rel=1e-3)


def test_lowdin_non_spin_atom1_p_orbital_charge(
    lowdin_non_spin_parser: ProjwfcOut,
) -> None:
    """Test atom 1 (Sr) p-orbital charge for non-spin-polarized."""
    charges_per_l = lowdin_non_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, p-orbital (l=1), spin channel 0
    assert charges_per_l[0, 1, 0] == pytest.approx(6.5679, rel=1e-3)


def test_lowdin_non_spin_atom1_d_orbital_charge(
    lowdin_non_spin_parser: ProjwfcOut,
) -> None:
    """Test atom 1 (Sr) d-orbital charge for non-spin-polarized."""
    charges_per_l = lowdin_non_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, d-orbital (l=2), spin channel 0
    assert charges_per_l[0, 2, 0] == pytest.approx(0.0000, abs=1e-4)


def test_lowdin_non_spin_atom2_s_orbital_charge(
    lowdin_non_spin_parser: ProjwfcOut,
) -> None:
    """Test atom 2 (V) s-orbital charge for non-spin-polarized."""
    charges_per_l = lowdin_non_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 2, s-orbital (l=0), spin channel 0
    assert charges_per_l[1, 0, 0] == pytest.approx(2.2697, rel=1e-3)


def test_lowdin_non_spin_atom2_d_orbital_charge(
    lowdin_non_spin_parser: ProjwfcOut,
) -> None:
    """Test atom 2 (V) d-orbital charge for non-spin-polarized."""
    charges_per_l = lowdin_non_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 2, d-orbital (l=2), spin channel 0
    assert charges_per_l[1, 2, 0] == pytest.approx(4.0906, rel=1e-3)


def test_lowdin_non_spin_per_orbital_dz2(lowdin_non_spin_parser: ProjwfcOut) -> None:
    """Test atom 2 (V) dz2 orbital charge for non-spin-polarized."""
    charges_per_orb = lowdin_non_spin_parser.lowdin_charges_per_orbital
    assert charges_per_orb is not None
    # Find dz2 index in orbital map
    orbital_map = lowdin_non_spin_parser.orbital_map
    dz2_idx = orbital_map["dz2"]
    # Atom 2, dz2 orbital, spin channel 0
    assert charges_per_orb[1, dz2_idx, 0] == pytest.approx(0.5649, rel=1e-3)


def test_lowdin_non_spin_per_orbital_dxy(lowdin_non_spin_parser: ProjwfcOut) -> None:
    """Test atom 2 (V) dxy orbital charge for non-spin-polarized."""
    charges_per_orb = lowdin_non_spin_parser.lowdin_charges_per_orbital
    assert charges_per_orb is not None
    orbital_map = lowdin_non_spin_parser.orbital_map
    dxy_idx = orbital_map["dxy"]
    # Atom 2, dxy orbital, spin channel 0
    assert charges_per_orb[1, dxy_idx, 0] == pytest.approx(0.9869, rel=1e-3)


def test_lowdin_non_spin_per_orbital_pz(lowdin_non_spin_parser: ProjwfcOut) -> None:
    """Test atom 1 (Sr) pz orbital charge for non-spin-polarized."""
    charges_per_orb = lowdin_non_spin_parser.lowdin_charges_per_orbital
    assert charges_per_orb is not None
    orbital_map = lowdin_non_spin_parser.orbital_map
    pz_idx = orbital_map["pz"]
    # Atom 1, pz orbital, spin channel 0
    assert charges_per_orb[0, pz_idx, 0] == pytest.approx(2.1893, rel=1e-3)


# =============================================================================
# Tests: Lowdin Charges - Spin-polarized
# =============================================================================


def test_lowdin_spin_returns_not_none(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test that lowdin_charges returns a dict for spin-polarized."""
    assert lowdin_spin_parser.lowdin_charges is not None


def test_lowdin_spin_spilling_parameter(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test spilling parameter for spin-polarized."""
    assert lowdin_spin_parser.spilling_parameter == pytest.approx(-0.0171, rel=1e-3)


def test_lowdin_spin_total_charges_shape(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test total charges array shape for spin-polarized."""
    total_charges = lowdin_spin_parser.total_lowdin_charges
    assert total_charges is not None
    # Shape: (n_atoms, n_l_orbitals) = (5, 4)
    assert total_charges.shape == (5, 4)


def test_lowdin_spin_per_l_shape(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test per-l orbital charges array shape for spin-polarized."""
    charges_per_l = lowdin_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Shape: (n_atoms, n_l_orbitals, n_spin_channels) = (5, 4, 2)
    assert charges_per_l.shape == (5, 4, 2)


def test_lowdin_spin_atom1_s_orbital_spin_up(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 1 (Sr) s-orbital spin-up charge."""
    charges_per_l = lowdin_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, s-orbital (l=0), spin up (channel 0)
    assert charges_per_l[0, 0, 0] == pytest.approx(1.1017, rel=1e-3)


def test_lowdin_spin_atom1_s_orbital_spin_down(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 1 (Sr) s-orbital spin-down charge."""
    charges_per_l = lowdin_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, s-orbital (l=0), spin down (channel 1)
    assert charges_per_l[0, 0, 1] == pytest.approx(1.1017, rel=1e-3)


def test_lowdin_spin_atom1_p_orbital_spin_up(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 1 (Sr) p-orbital spin-up charge."""
    charges_per_l = lowdin_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, p-orbital (l=1), spin up (channel 0)
    assert charges_per_l[0, 1, 0] == pytest.approx(3.2839, rel=1e-3)


def test_lowdin_spin_atom2_d_orbital_spin_up(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 2 (V) d-orbital spin-up charge."""
    charges_per_l = lowdin_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 2, d-orbital (l=2), spin up (channel 0)
    assert charges_per_l[1, 2, 0] == pytest.approx(2.0383, rel=1e-3)


def test_lowdin_spin_atom2_d_orbital_spin_down(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 2 (V) d-orbital spin-down charge."""
    charges_per_l = lowdin_spin_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 2, d-orbital (l=2), spin down (channel 1)
    assert charges_per_l[1, 2, 1] == pytest.approx(2.0372, rel=1e-3)


def test_lowdin_spin_polarization_shape(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test polarization array shape for spin-polarized."""
    polarization = lowdin_spin_parser.lowdin_polarization
    assert polarization is not None
    # Shape: (n_atoms, n_l_orbitals) = (5, 4)
    assert polarization.shape == (5, 4)


def test_lowdin_spin_atom2_d_polarization(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 2 (V) d-orbital polarization."""
    polarization = lowdin_spin_parser.lowdin_polarization
    assert polarization is not None
    # Atom 2, d-orbital (l=2)
    assert polarization[1, 2] == pytest.approx(0.0012, rel=1e-2)


def test_lowdin_spin_atom1_s_polarization(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 1 (Sr) s-orbital polarization."""
    polarization = lowdin_spin_parser.lowdin_polarization
    assert polarization is not None
    # Atom 1, s-orbital (l=0)
    assert polarization[0, 0] == pytest.approx(-0.0000, abs=1e-4)


def test_lowdin_spin_per_orbital_dz2_spin_up(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 2 (V) dz2 orbital spin-up charge."""
    charges_per_orb = lowdin_spin_parser.lowdin_charges_per_orbital
    assert charges_per_orb is not None
    orbital_map = lowdin_spin_parser.orbital_map
    dz2_idx = orbital_map["dz2"]
    # Atom 2, dz2 orbital, spin up (channel 0)
    assert charges_per_orb[1, dz2_idx, 0] == pytest.approx(0.2821, rel=1e-3)


def test_lowdin_spin_per_orbital_dxy_spin_down(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 2 (V) dxy orbital spin-down charge."""
    charges_per_orb = lowdin_spin_parser.lowdin_charges_per_orbital
    assert charges_per_orb is not None
    orbital_map = lowdin_spin_parser.orbital_map
    dxy_idx = orbital_map["dxy"]
    # Atom 2, dxy orbital, spin down (channel 1)
    assert charges_per_orb[1, dxy_idx, 1] == pytest.approx(0.4910, rel=1e-3)


def test_lowdin_spin_per_orbital_pz_spin_up(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test atom 1 (Sr) pz orbital spin-up charge."""
    charges_per_orb = lowdin_spin_parser.lowdin_charges_per_orbital
    assert charges_per_orb is not None
    orbital_map = lowdin_spin_parser.orbital_map
    pz_idx = orbital_map["pz"]
    # Atom 1, pz orbital, spin up (channel 0)
    assert charges_per_orb[0, pz_idx, 0] == pytest.approx(1.0946, rel=1e-3)


def test_lowdin_spin_total_charge_atom1_s(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test total s-orbital charge for atom 1 (Sr)."""
    total_charges = lowdin_spin_parser.total_lowdin_charges
    assert total_charges is not None
    # Atom 1, s-orbital (l=0)
    assert total_charges[0, 0] == pytest.approx(2.2033, rel=1e-3)


def test_lowdin_spin_total_charge_atom2_d(lowdin_spin_parser: ProjwfcOut) -> None:
    """Test total d-orbital charge for atom 2 (V)."""
    total_charges = lowdin_spin_parser.total_lowdin_charges
    assert total_charges is not None
    # Atom 2, d-orbital (l=2)
    assert total_charges[1, 2] == pytest.approx(4.0755, rel=1e-3)


# =============================================================================
# Tests: Lowdin Charges - Non-colinear
# =============================================================================


def test_lowdin_non_colinear_returns_not_none(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test that lowdin_charges returns a dict for non-colinear."""
    assert lowdin_non_colinear_parser.lowdin_charges is not None


def test_lowdin_non_colinear_spilling_parameter(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test spilling parameter for non-colinear."""
    assert lowdin_non_colinear_parser.spilling_parameter == pytest.approx(-0.0172, rel=1e-3)


def test_lowdin_non_colinear_total_charges_shape(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test total charges array shape for non-colinear."""
    total_charges = lowdin_non_colinear_parser.total_lowdin_charges
    assert total_charges is not None
    # Shape: (n_atoms, n_l_orbitals) = (5, 4)
    assert total_charges.shape == (5, 4)


def test_lowdin_non_colinear_is_non_colinear(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test that parser correctly identifies non-colinear calculation."""
    assert lowdin_non_colinear_parser.is_non_colinear is True


def test_lowdin_non_colinear_atom1_s_charge(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 1 (Sr) s-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, s-orbital (l=0), spin channel 0
    assert charges_per_l[0, 0, 0] == pytest.approx(2.2034, rel=1e-3)


def test_lowdin_non_colinear_atom1_p_charge(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 1 (Sr) p-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 1, p-orbital (l=1), spin channel 0
    assert charges_per_l[0, 1, 0] == pytest.approx(6.5680, rel=1e-3)


def test_lowdin_non_colinear_atom2_s_charge(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 2 (V) s-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 2, s-orbital (l=0), spin channel 0
    assert charges_per_l[1, 0, 0] == pytest.approx(2.2697, rel=1e-3)


def test_lowdin_non_colinear_atom2_p_charge(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 2 (V) p-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 2, p-orbital (l=1), spin channel 0
    assert charges_per_l[1, 1, 0] == pytest.approx(5.9915, rel=1e-3)


def test_lowdin_non_colinear_atom2_d_charge(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 2 (V) d-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 2, d-orbital (l=2), spin channel 0
    assert charges_per_l[1, 2, 0] == pytest.approx(4.0824, rel=1e-3)


def test_lowdin_non_colinear_atom3_total(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 3 (O) s-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 3, s-orbital (l=0), spin channel 0
    assert charges_per_l[2, 0, 0] == pytest.approx(1.7441, rel=1e-3)


def test_lowdin_non_colinear_atom5_s_charge(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 5 (O) s-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 5, s-orbital (l=0), spin channel 0
    assert charges_per_l[4, 0, 0] == pytest.approx(1.6496, rel=1e-3)


def test_lowdin_non_colinear_atom5_p_charge(
    lowdin_non_colinear_parser: ProjwfcOut,
) -> None:
    """Test atom 5 (O) p-orbital charge for non-colinear."""
    charges_per_l = lowdin_non_colinear_parser.lowdin_charges_per_l_orbital
    assert charges_per_l is not None
    # Atom 5, p-orbital (l=1), spin channel 0
    assert charges_per_l[4, 1, 0] == pytest.approx(5.4716, rel=1e-3)
