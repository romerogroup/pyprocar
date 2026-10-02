"""Tests for ProjwfcDOS and ProjwfcPDOSFile parsers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyprocar.io.qe.projwfc import ProjwfcDOS
from pyprocar.io.qe.projwfc.projwfc_dos import ProjwfcPDOSFile

# =============================================================================
# Inline String Fixtures
# =============================================================================

# PDOS file for atom 1 (Sr), s orbital
PDOS_ATM1_S = """# E (eV)  ldos(E)  pdos(E)
 -10.00   0.000   0.000
   5.00   0.005   0.005
"""

# PDOS file for atom 2 (V), s orbital
PDOS_ATM2_S = """# E (eV)  ldos(E)  pdos(E)
 -10.00   0.000   0.000
   5.00   0.010   0.010
"""

# Spin-polarized PDOS file for atom 1
PDOS_ATM1_S_SPIN = """# E (eV)  ldosup(E)  ldosdw(E)  pdosup(E)  pdosdw(E)
 -10.00   0.000   0.000   0.000   0.000
  -5.00   0.005   0.004   0.005   0.004
   0.00   0.030   0.025   0.030   0.025
   5.00   0.005   0.004   0.005   0.004
"""

# Spin-polarized PDOS file for atom 2
PDOS_ATM2_S_SPIN = """# E (eV)  ldosup(E)  ldosdw(E)  pdosup(E)  pdosdw(E)
 -10.00   0.000   0.000   0.000   0.000
  -5.00   0.010   0.008   0.010   0.008
   0.00   0.060   0.050   0.060   0.050
   5.00   0.010   0.008   0.010   0.008
"""

# Non-colinear k-resolved PDOS file content (s orbital with j=0.5, 2 m_j components)
# Format matches real QE output: SrVO3.k.pdos_atm#1(Sr)_wfc#1(s_j0.5)
PDOS_ATM1_S_J05_NON_COLINEAR = """# ik    E (eV)   ldos(E)   pdos(E)_1   pdos(E)_2   
    1  -54.824  0.313E-08  0.138E-08  0.175E-08
    1  -54.814  0.353E-08  0.156E-08  0.197E-08
    2  -54.824  0.280E-08  0.120E-08  0.160E-08
    2  -54.814  0.320E-08  0.140E-08  0.180E-08
"""

# Non-colinear k-resolved PDOS for p orbital with j=1.5 (4 m_j components)
# Format matches real QE output: SrVO3.k.pdos_atm#2(V)_wfc#4(p_j1.5)
PDOS_ATM2_P_J15_NON_COLINEAR = """# ik    E (eV)   ldos(E)   pdos(E)_1   pdos(E)_2   pdos(E)_3   pdos(E)_4   
    1  -54.824  0.154E-20  0.905E-22  0.965E-21  0.355E-21  0.126E-21
    1  -54.814  0.173E-20  0.102E-21  0.109E-20  0.401E-21  0.142E-21
    2  -54.824  0.148E-20  0.880E-22  0.940E-21  0.340E-21  0.120E-21
    2  -54.814  0.165E-20  0.980E-22  0.105E-20  0.385E-21  0.135E-21
"""

# Non-colinear k-resolved PDOS for d orbital with j=1.5 (4 m_j components)
# Format matches real QE output: SrVO3.k.pdos_atm#2(V)_wfc#5(d_j1.5)
PDOS_ATM2_D_J15_NON_COLINEAR = """# ik    E (eV)   ldos(E)   pdos(E)_1   pdos(E)_2   pdos(E)_3   pdos(E)_4   
    1  -54.824  0.886E-16  0.160E-19  0.449E-16  0.437E-16  0.474E-20
    1  -54.814  0.999E-16  0.181E-19  0.506E-16  0.493E-16  0.536E-20
    2  -54.824  0.850E-16  0.155E-19  0.430E-16  0.420E-16  0.460E-20
    2  -54.814  0.960E-16  0.175E-19  0.490E-16  0.475E-16  0.520E-20
"""


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def non_spin_dos_dir(tmp_path: Path) -> Path:
    """Create temporary directory with non-spin-polarized PDOS files."""
    dos_dir = tmp_path / "dos"
    dos_dir.mkdir()

    # Create PDOS files with proper naming convention
    pdos_file1 = dos_dir / "test.pdos_atm#1(Sr)_wfc#1(s)"
    pdos_file1.write_text(PDOS_ATM1_S)

    pdos_file2 = dos_dir / "test.pdos_atm#2(V)_wfc#1(s)"
    pdos_file2.write_text(PDOS_ATM2_S)

    return dos_dir


@pytest.fixture
def non_spin_parser(non_spin_dos_dir: Path) -> ProjwfcDOS:
    """Create parser instance for non-spin-polarized DOS."""
    return ProjwfcDOS(paths=non_spin_dos_dir)


@pytest.fixture
def spin_polarized_dos_dir(tmp_path: Path) -> Path:
    """Create temporary directory with spin-polarized PDOS files."""
    dos_dir = tmp_path / "dos"
    dos_dir.mkdir()

    # Create PDOS files with proper naming convention
    pdos_file1 = dos_dir / "test.pdos_atm#1(Sr)_wfc#1(s)"
    pdos_file1.write_text(PDOS_ATM1_S_SPIN)

    pdos_file2 = dos_dir / "test.pdos_atm#2(V)_wfc#1(s)"
    pdos_file2.write_text(PDOS_ATM2_S_SPIN)

    return dos_dir


@pytest.fixture
def spin_polarized_parser(spin_polarized_dos_dir: Path) -> ProjwfcDOS:
    """Create parser instance for spin-polarized DOS."""
    return ProjwfcDOS(paths=spin_polarized_dos_dir)


@pytest.fixture
def non_colinear_dos_dir(tmp_path: Path) -> Path:
    """Create temporary directory with non-colinear PDOS files for different orbitals."""
    dos_dir = tmp_path / "dos"
    dos_dir.mkdir()

    # s orbital with j=0.5 (2 m_j components)
    pdos_s = dos_dir / "SrVO3.k.pdos_atm#1(Sr)_wfc#1(s_j0.5)"
    pdos_s.write_text(PDOS_ATM1_S_J05_NON_COLINEAR)

    # p orbital with j=1.5 (4 m_j components)
    pdos_p = dos_dir / "SrVO3.k.pdos_atm#2(V)_wfc#4(p_j1.5)"
    pdos_p.write_text(PDOS_ATM2_P_J15_NON_COLINEAR)

    # d orbital with j=1.5 (4 m_j components)
    pdos_d = dos_dir / "SrVO3.k.pdos_atm#2(V)_wfc#5(d_j1.5)"
    pdos_d.write_text(PDOS_ATM2_D_J15_NON_COLINEAR)

    return dos_dir


@pytest.fixture
def non_colinear_parser(non_colinear_dos_dir: Path) -> ProjwfcDOS:
    """Create parser instance for non-colinear DOS."""
    return ProjwfcDOS(paths=non_colinear_dos_dir)


@pytest.fixture
def pdos_file_non_colinear_s_j05(tmp_path: Path) -> Path:
    """Create a non-colinear s orbital PDOS file (j=0.5, 2 m_j components)."""
    filepath = tmp_path / "SrVO3.k.pdos_atm#1(Sr)_wfc#1(s_j0.5)"
    filepath.write_text(PDOS_ATM1_S_J05_NON_COLINEAR)
    return filepath


@pytest.fixture
def pdos_file_non_colinear_p_j15(tmp_path: Path) -> Path:
    """Create a non-colinear p orbital PDOS file (j=1.5, 4 m_j components)."""
    filepath = tmp_path / "SrVO3.k.pdos_atm#2(V)_wfc#4(p_j1.5)"
    filepath.write_text(PDOS_ATM2_P_J15_NON_COLINEAR)
    return filepath


@pytest.fixture
def pdos_file_non_colinear_d_j15(tmp_path: Path) -> Path:
    """Create a non-colinear d orbital PDOS file (j=1.5, 4 m_j components)."""
    filepath = tmp_path / "SrVO3.k.pdos_atm#2(V)_wfc#5(d_j1.5)"
    filepath.write_text(PDOS_ATM2_D_J15_NON_COLINEAR)
    return filepath


# =============================================================================
# Tests: Atom Count
# =============================================================================


def test_n_atoms_returns_correct_count(non_spin_parser: ProjwfcDOS) -> None:
    """Test that n_atoms parses correctly from file names."""
    assert non_spin_parser.n_atoms == 2


# =============================================================================
# Tests: Energy Grid
# =============================================================================


def test_energies_returns_array(non_spin_parser: ProjwfcDOS) -> None:
    """Test that energies returns a numpy array."""
    energies = non_spin_parser.energies
    assert isinstance(energies, np.ndarray)


def test_n_energies_returns_correct_count(non_spin_parser: ProjwfcDOS) -> None:
    """Test that n_energies returns the correct count of energy points."""
    assert non_spin_parser.n_energies == 2


def test_energy_values_match_fixture(non_spin_parser: ProjwfcDOS) -> None:
    """Test that energy values match the fixture values."""
    energies = non_spin_parser.energies
    # First energy point should be -10.0
    assert pytest.approx(energies[0, 0]) == -10.0
    # Last energy point should be 5.0
    assert pytest.approx(energies[0, -1]) == 5.0


# =============================================================================
# Tests: Spin Polarization Detection
# =============================================================================


def test_is_spin_polarized_false_for_non_spin(non_spin_parser: ProjwfcDOS) -> None:
    """Test that is_spin_polarized is False for non-spin-polarized data."""
    assert non_spin_parser.is_spin_polarized is False


def test_is_spin_polarized_true_for_spin(spin_polarized_parser: ProjwfcDOS) -> None:
    """Test that is_spin_polarized is True for spin-polarized data."""
    assert spin_polarized_parser.is_spin_polarized is True


def test_n_spin_channels_non_spin_polarized(non_spin_parser: ProjwfcDOS) -> None:
    """Test that n_spin_channels is 1 for non-spin-polarized."""
    assert non_spin_parser.n_spin_channels == 1


def test_n_spin_channels_spin_polarized(spin_polarized_parser: ProjwfcDOS) -> None:
    """Test that n_spin_channels is 2 for spin-polarized."""
    assert spin_polarized_parser.n_spin_channels == 2


# =============================================================================
# Tests: Data Loading
# =============================================================================


def test_data_returns_list(non_spin_parser: ProjwfcDOS) -> None:
    """Test that data returns a list of DataFrames."""
    data = non_spin_parser.data
    assert isinstance(data, list)
    assert len(data) == 2  # Two PDOS files


def test_data_contains_dataframes(non_spin_parser: ProjwfcDOS) -> None:
    """Test that data contains pandas DataFrames."""
    data = non_spin_parser.data
    assert isinstance(data[0], pd.DataFrame)


def test_files_metadata_contains_atom_info(non_spin_parser: ProjwfcDOS) -> None:
    """Test that files_metadata contains atom information."""
    metadata = non_spin_parser.files_metadata
    assert len(metadata) == 2
    # First file should be atom 1 (Sr)
    assert metadata[0]["atom_index"] == 1
    assert metadata[0]["atom_symbol"] == "Sr"
    assert metadata[0]["orbital"] == "s"


# =============================================================================
# ProjwfcPDOSFile Test Fixtures and Data
# =============================================================================

# Non-spin-polarized PDOS file content
PDOS_FILE_CONTENT_NON_SPIN = """# E (eV)  ldos(E)  pdos(E)
 -10.00   0.000   0.000
  -5.00   0.005   0.005
   0.00   0.030   0.030
   5.00   0.005   0.005
"""

# Spin-polarized PDOS file content
PDOS_FILE_CONTENT_SPIN = """# E (eV)  ldosup(E)  ldosdw(E)  pdosup(E)  pdosdw(E)
 -10.00   0.000   0.000   0.000   0.000
  -5.00   0.005   0.004   0.005   0.004
   0.00   0.030   0.025   0.030   0.025
   5.00   0.005   0.004   0.005   0.004
"""

# K-resolved PDOS file content
PDOS_FILE_CONTENT_KRESOLVED = """# ik  E (eV)  ldos(E)  pdos(E)
1 -10.00   0.000   0.000
1  -5.00   0.005   0.005
2 -10.00   0.001   0.001
2  -5.00   0.006   0.006
"""

# PDOS file with multiple m-components (p orbital)
PDOS_FILE_CONTENT_P_ORBITAL = """# E (eV)  ldos(E)  pdos(pz)  pdos(px)  pdos(py)
 -10.00   0.000   0.000   0.000   0.000
   0.00   0.100   0.033   0.033   0.034
"""


@pytest.fixture
def pdos_file_colinear(tmp_path: Path) -> Path:
    """Create a colinear PDOS file (no j value in filename)."""
    filepath = tmp_path / "test.pdos_atm#1(Fe)_wfc#2(d)"
    filepath.write_text(PDOS_FILE_CONTENT_NON_SPIN)
    return filepath


@pytest.fixture
def pdos_file_non_colinear(tmp_path: Path) -> Path:
    """Create a non-colinear PDOS file (with j value in filename)."""
    filepath = tmp_path / "test.pdos_atm#3(O)_wfc#1(p_j1.5)"
    filepath.write_text(PDOS_FILE_CONTENT_NON_SPIN)
    return filepath


@pytest.fixture
def pdos_file_spin_polarized(tmp_path: Path) -> Path:
    """Create a spin-polarized PDOS file."""
    filepath = tmp_path / "test.pdos_atm#1(Sr)_wfc#1(s)"
    filepath.write_text(PDOS_FILE_CONTENT_SPIN)
    return filepath


@pytest.fixture
def pdos_file_kresolved(tmp_path: Path) -> Path:
    """Create a k-resolved PDOS file."""
    filepath = tmp_path / "test.pdos_atm#1(Sr)_wfc#1(s)"
    filepath.write_text(PDOS_FILE_CONTENT_KRESOLVED)
    return filepath


@pytest.fixture
def pdos_file_p_orbital(tmp_path: Path) -> Path:
    """Create a PDOS file for p orbital with multiple m-components."""
    filepath = tmp_path / "test.pdos_atm#2(V)_wfc#1(p)"
    filepath.write_text(PDOS_FILE_CONTENT_P_ORBITAL)
    return filepath


@pytest.fixture
def pdos_file_non_matching_name(tmp_path: Path) -> Path:
    """Create a PDOS file with non-matching filename."""
    filepath = tmp_path / "some_random_file.txt"
    filepath.write_text(PDOS_FILE_CONTENT_NON_SPIN)
    return filepath


# =============================================================================
# Tests: ProjwfcPDOSFile Initialization
# =============================================================================


def test_pdos_file_init_sets_filepath(pdos_file_colinear: Path) -> None:
    """Test that initialization sets the filepath correctly."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert pdos_file.filepath == pdos_file_colinear


def test_pdos_file_init_accepts_string_path(pdos_file_colinear: Path) -> None:
    """Test that initialization accepts string path."""
    pdos_file = ProjwfcPDOSFile(str(pdos_file_colinear))
    assert pdos_file.filepath == pdos_file_colinear


def test_pdos_file_text_returns_file_content(pdos_file_colinear: Path) -> None:
    """Test that text property returns the file content."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert pdos_file.text == PDOS_FILE_CONTENT_NON_SPIN


# =============================================================================
# Tests: ProjwfcPDOSFile filename_info Parsing
# =============================================================================


def test_pdos_file_filename_info_parses_atom_index(pdos_file_colinear: Path) -> None:
    """Test that filename_info correctly parses atom index."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert pdos_file.filename_info["atom_index"] == 1


def test_pdos_file_filename_info_parses_atom_symbol(pdos_file_colinear: Path) -> None:
    """Test that filename_info correctly parses atom symbol."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert pdos_file.filename_info["atom_symbol"] == "Fe"


def test_pdos_file_filename_info_parses_wfc_index(pdos_file_colinear: Path) -> None:
    """Test that filename_info correctly parses wfc index."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert pdos_file.filename_info["wfc_index"] == 2


def test_pdos_file_filename_info_parses_orbital(pdos_file_colinear: Path) -> None:
    """Test that filename_info correctly parses orbital."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert pdos_file.filename_info["orbital"] == "d"


def test_pdos_file_filename_info_j_value_is_none_for_colinear(
    pdos_file_colinear: Path,
) -> None:
    """Test that filename_info j_value is None for colinear files."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert pdos_file.filename_info["j_value"] is None


def test_pdos_file_filename_info_parses_j_value_for_non_colinear(
    pdos_file_non_colinear: Path,
) -> None:
    """Test that filename_info correctly parses j value for non-colinear files."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear)
    assert pdos_file.filename_info["j_value"] == pytest.approx(1.5)


def test_pdos_file_filename_info_parses_atom_index_non_colinear(
    pdos_file_non_colinear: Path,
) -> None:
    """Test that filename_info parses atom index for non-colinear files."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear)
    assert pdos_file.filename_info["atom_index"] == 3


def test_pdos_file_filename_info_parses_atom_symbol_non_colinear(
    pdos_file_non_colinear: Path,
) -> None:
    """Test that filename_info parses atom symbol for non-colinear files."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear)
    assert pdos_file.filename_info["atom_symbol"] == "O"


def test_pdos_file_filename_info_parses_orbital_non_colinear(
    pdos_file_non_colinear: Path,
) -> None:
    """Test that filename_info parses orbital for non-colinear files."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear)
    assert pdos_file.filename_info["orbital"] == "p"


def test_pdos_file_filename_info_returns_empty_dict_for_non_matching(
    pdos_file_non_matching_name: Path,
) -> None:
    """Test that filename_info returns empty dict for non-matching filename."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_matching_name)
    assert pdos_file.filename_info == {}


# =============================================================================
# Tests: ProjwfcPDOSFile lines Property
# =============================================================================


def test_pdos_file_lines_returns_list(pdos_file_colinear: Path) -> None:
    """Test that lines property returns a list."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert isinstance(pdos_file.lines, list)


def test_pdos_file_lines_contains_strings(pdos_file_colinear: Path) -> None:
    """Test that lines property contains strings."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert all(isinstance(line, str) for line in pdos_file.lines)


def test_pdos_file_lines_count_matches_file(pdos_file_colinear: Path) -> None:
    """Test that lines count matches file content."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    expected_lines = PDOS_FILE_CONTENT_NON_SPIN.splitlines()
    assert len(pdos_file.lines) == len(expected_lines)


# =============================================================================
# Tests: ProjwfcPDOSFile columns Property
# =============================================================================


def test_pdos_file_columns_returns_list(pdos_file_colinear: Path) -> None:
    """Test that columns property returns a list."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert isinstance(pdos_file.columns, list)


def test_pdos_file_columns_extracts_column_names(pdos_file_colinear: Path) -> None:
    """Test that columns property extracts column names."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    expected_columns = ["E", "ldos(E)", "pdos(E)"]
    assert pdos_file.columns == expected_columns


def test_pdos_file_columns_removes_eV_unit(pdos_file_colinear: Path) -> None:
    """Test that columns property removes (eV) from header."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert "(eV)" not in pdos_file.columns


def test_pdos_file_columns_spin_polarized(pdos_file_spin_polarized: Path) -> None:
    """Test that columns property handles spin-polarized files."""
    pdos_file = ProjwfcPDOSFile(pdos_file_spin_polarized)
    expected_columns = ["E", "ldosup(E)", "ldosdw(E)", "pdosup(E)", "pdosdw(E)"]
    assert pdos_file.columns == expected_columns


def test_pdos_file_columns_kresolved(pdos_file_kresolved: Path) -> None:
    """Test that columns property handles k-resolved files.

    Note: The current implementation only removes (eV) when it's at index 1,
    so for k-resolved files where (eV) is at index 2, it remains in the columns.
    """
    pdos_file = ProjwfcPDOSFile(pdos_file_kresolved)
    # Current implementation doesn't remove (eV) at index 2
    expected_columns = ["ik", "E", "(eV)", "ldos(E)", "pdos(E)"]
    assert pdos_file.columns == expected_columns


def test_pdos_file_columns_p_orbital(pdos_file_p_orbital: Path) -> None:
    """Test that columns property handles multiple orbital components."""
    pdos_file = ProjwfcPDOSFile(pdos_file_p_orbital)
    expected_columns = ["E", "ldos(E)", "pdos(pz)", "pdos(px)", "pdos(py)"]
    assert pdos_file.columns == expected_columns


# =============================================================================
# Tests: ProjwfcPDOSFile data Property
# =============================================================================


def test_pdos_file_data_returns_dataframe(pdos_file_colinear: Path) -> None:
    """Test that data property returns a DataFrame."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    assert isinstance(pdos_file.data, pd.DataFrame)


def test_pdos_file_data_has_correct_shape(pdos_file_colinear: Path) -> None:
    """Test that data DataFrame has correct shape."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    # 4 data rows (excluding header), 3 columns
    assert pdos_file.data.shape == (4, 3)


def test_pdos_file_data_has_correct_columns(pdos_file_colinear: Path) -> None:
    """Test that data DataFrame has correct column names."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    expected_columns = ["E", "ldos(E)", "pdos(E)"]
    assert list(pdos_file.data.columns) == expected_columns


def test_pdos_file_data_energy_values(pdos_file_colinear: Path) -> None:
    """Test that data DataFrame contains correct energy values."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    expected_energies = [-10.0, -5.0, 0.0, 5.0]
    actual_energies = pdos_file.data["E"].tolist()
    assert actual_energies == pytest.approx(expected_energies)


def test_pdos_file_data_ldos_values(pdos_file_colinear: Path) -> None:
    """Test that data DataFrame contains correct ldos values."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    expected_ldos = [0.0, 0.005, 0.030, 0.005]
    actual_ldos = pdos_file.data["ldos(E)"].tolist()
    assert actual_ldos == pytest.approx(expected_ldos)


def test_pdos_file_data_pdos_values(pdos_file_colinear: Path) -> None:
    """Test that data DataFrame contains correct pdos values."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    expected_pdos = [0.0, 0.005, 0.030, 0.005]
    actual_pdos = pdos_file.data["pdos(E)"].tolist()
    assert actual_pdos == pytest.approx(expected_pdos)


def test_pdos_file_data_spin_polarized_shape(pdos_file_spin_polarized: Path) -> None:
    """Test that spin-polarized data has correct shape."""
    pdos_file = ProjwfcPDOSFile(pdos_file_spin_polarized)
    # 4 data rows, 5 columns (E, ldosup, ldosdw, pdosup, pdosdw)
    assert pdos_file.data.shape == (4, 5)


def test_pdos_file_data_spin_polarized_up_values(
    pdos_file_spin_polarized: Path,
) -> None:
    """Test that spin-up pdos values are correctly parsed."""
    pdos_file = ProjwfcPDOSFile(pdos_file_spin_polarized)
    expected_pdosup = [0.0, 0.005, 0.030, 0.005]
    actual_pdosup = pdos_file.data["pdosup(E)"].tolist()
    assert actual_pdosup == pytest.approx(expected_pdosup)


def test_pdos_file_data_spin_polarized_down_values(
    pdos_file_spin_polarized: Path,
) -> None:
    """Test that spin-down pdos values are correctly parsed."""
    pdos_file = ProjwfcPDOSFile(pdos_file_spin_polarized)
    expected_pdosdw = [0.0, 0.004, 0.025, 0.004]
    actual_pdosdw = pdos_file.data["pdosdw(E)"].tolist()
    assert actual_pdosdw == pytest.approx(expected_pdosdw)


def test_pdos_file_data_kresolved_shape(pdos_file_kresolved: Path) -> None:
    """Test that k-resolved data has correct shape."""
    pdos_file = ProjwfcPDOSFile(pdos_file_kresolved)
    # 4 data rows (2 k-points * 2 energies), 4 columns
    assert pdos_file.data.shape == (4, 4)


def test_pdos_file_data_kresolved_ik_values(pdos_file_kresolved: Path) -> None:
    """Test that k-point indices are correctly parsed as integers."""
    pdos_file = ProjwfcPDOSFile(pdos_file_kresolved)
    expected_ik = [1, 1, 2, 2]
    actual_ik = pdos_file.data["ik"].tolist()
    assert actual_ik == expected_ik


def test_pdos_file_data_p_orbital_shape(pdos_file_p_orbital: Path) -> None:
    """Test that p orbital data has correct shape."""
    pdos_file = ProjwfcPDOSFile(pdos_file_p_orbital)
    # 2 data rows, 5 columns (E, ldos, pz, px, py)
    assert pdos_file.data.shape == (2, 5)


# =============================================================================
# Tests: ProjwfcPDOSFile _first_data_line Method
# =============================================================================


def test_pdos_file_first_data_line_returns_first_data(
    pdos_file_colinear: Path,
) -> None:
    """Test that _first_data_line returns first non-comment line."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    first_line = pdos_file._first_data_line()
    assert first_line is not None
    assert "-10.00" in first_line


def test_pdos_file_first_data_line_skips_header(pdos_file_colinear: Path) -> None:
    """Test that _first_data_line skips comment header."""
    pdos_file = ProjwfcPDOSFile(pdos_file_colinear)
    first_line = pdos_file._first_data_line()
    assert first_line is not None
    assert not first_line.strip().startswith("#")


def test_pdos_file_first_data_line_returns_none_for_empty_file(
    tmp_path: Path,
) -> None:
    """Test that _first_data_line returns None for file with only comments."""
    filepath = tmp_path / "test.pdos_atm#1(Fe)_wfc#1(s)"
    filepath.write_text("# Comment only\n# Another comment\n")
    pdos_file = ProjwfcPDOSFile(filepath)
    assert pdos_file._first_data_line() is None


# =============================================================================
# Tests: ProjwfcDOS Non-colinear Detection
# =============================================================================


def test_is_non_colinear_true_for_non_colinear(
    non_colinear_parser: ProjwfcDOS,
) -> None:
    """Test that is_non_colinear is True for non-colinear DOS files."""
    assert non_colinear_parser.is_non_colinear is True


def test_is_non_colinear_false_for_colinear(non_spin_parser: ProjwfcDOS) -> None:
    """Test that is_non_colinear is False for colinear DOS files."""
    assert non_spin_parser.is_non_colinear is False


# =============================================================================
# Tests: ProjwfcDOS Non-colinear File Discovery
# =============================================================================


def test_non_colinear_n_atoms(non_colinear_parser: ProjwfcDOS) -> None:
    """Test that n_atoms is correct for non-colinear files (Sr and V atoms)."""
    assert non_colinear_parser.n_atoms == 2


def test_non_colinear_finds_all_orbital_files(non_colinear_parser: ProjwfcDOS) -> None:
    """Test that s_j0.5, p_j1.5, and d_j1.5 files are found."""
    assert len(non_colinear_parser.filepaths) == 3


def test_non_colinear_files_metadata_has_j_values(
    non_colinear_parser: ProjwfcDOS,
) -> None:
    """Test that files_metadata contains j_value for non-colinear files."""
    for metadata in non_colinear_parser.files_metadata:
        assert metadata["j_value"] is not None


def test_non_colinear_files_metadata_covers_different_orbitals(
    non_colinear_parser: ProjwfcDOS,
) -> None:
    """Test that files_metadata covers s, p, and d orbitals."""
    orbitals = {metadata["orbital"] for metadata in non_colinear_parser.files_metadata}
    assert orbitals == {"s", "p", "d"}


# =============================================================================
# Tests: ProjwfcDOS Non-colinear K-resolved Data
# =============================================================================


def test_non_colinear_is_kresolved_true(non_colinear_parser: ProjwfcDOS) -> None:
    """Test that k-resolved non-colinear data is detected."""
    assert non_colinear_parser.is_kresolved is True


# =============================================================================
# Tests: ProjwfcPDOSFile Non-colinear K-resolved Columns by Orbital Type
# =============================================================================


def test_pdos_file_s_j05_has_2_pdos_columns(pdos_file_non_colinear_s_j05: Path) -> None:
    """Test that s orbital (j=0.5) has 2 m_j pdos columns in header."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_s_j05)
    # Header: ik, E, (eV), ldos(E), pdos(E)_1, pdos(E)_2
    expected_columns = ["ik", "E", "(eV)", "ldos(E)", "pdos(E)_1", "pdos(E)_2"]
    assert pdos_file.columns == expected_columns


def test_pdos_file_p_j15_has_4_pdos_columns(pdos_file_non_colinear_p_j15: Path) -> None:
    """Test that p orbital (j=1.5) has 4 m_j pdos columns in header."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_p_j15)
    # Header: ik, E, (eV), ldos(E), pdos(E)_1, pdos(E)_2, pdos(E)_3, pdos(E)_4
    expected_columns = [
        "ik",
        "E",
        "(eV)",
        "ldos(E)",
        "pdos(E)_1",
        "pdos(E)_2",
        "pdos(E)_3",
        "pdos(E)_4",
    ]
    assert pdos_file.columns == expected_columns


def test_pdos_file_d_j15_has_4_pdos_columns(pdos_file_non_colinear_d_j15: Path) -> None:
    """Test that d orbital (j=1.5) has 4 m_j pdos columns in header."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_d_j15)
    # Header: ik, E, (eV), ldos(E), pdos(E)_1, pdos(E)_2, pdos(E)_3, pdos(E)_4
    expected_columns = [
        "ik",
        "E",
        "(eV)",
        "ldos(E)",
        "pdos(E)_1",
        "pdos(E)_2",
        "pdos(E)_3",
        "pdos(E)_4",
    ]
    assert pdos_file.columns == expected_columns


def test_pdos_file_s_j05_data_shape(pdos_file_non_colinear_s_j05: Path) -> None:
    """Test that s orbital data has correct shape (4 rows, 5 data columns)."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_s_j05)
    # 4 data rows (2 k-points * 2 energies), 5 data columns
    assert pdos_file.data.shape == (4, 5)


def test_pdos_file_p_j15_data_shape(pdos_file_non_colinear_p_j15: Path) -> None:
    """Test that p orbital data has correct shape (4 rows, 7 data columns)."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_p_j15)
    # 4 data rows (2 k-points * 2 energies), 7 data columns
    assert pdos_file.data.shape == (4, 7)


def test_pdos_file_d_j15_data_shape(pdos_file_non_colinear_d_j15: Path) -> None:
    """Test that d orbital data has correct shape (4 rows, 7 data columns)."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_d_j15)
    # 4 data rows (2 k-points * 2 energies), 7 data columns
    assert pdos_file.data.shape == (4, 7)


def test_pdos_file_non_colinear_ik_values(pdos_file_non_colinear_s_j05: Path) -> None:
    """Test that k-point indices are correctly parsed."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_s_j05)
    ik_values = pdos_file.data["ik"].tolist()
    expected_ik = [1, 1, 2, 2]
    assert ik_values == expected_ik


def test_pdos_file_s_j05_filename_info_j_value(
    pdos_file_non_colinear_s_j05: Path,
) -> None:
    """Test that j_value is correctly parsed from s_j0.5 filename."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_s_j05)
    assert pdos_file.filename_info["j_value"] == pytest.approx(0.5)


def test_pdos_file_p_j15_filename_info_j_value(
    pdos_file_non_colinear_p_j15: Path,
) -> None:
    """Test that j_value is correctly parsed from p_j1.5 filename."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_p_j15)
    assert pdos_file.filename_info["j_value"] == pytest.approx(1.5)


def test_pdos_file_d_j15_filename_info_orbital(
    pdos_file_non_colinear_d_j15: Path,
) -> None:
    """Test that orbital is correctly parsed from d_j1.5 filename."""
    pdos_file = ProjwfcPDOSFile(pdos_file_non_colinear_d_j15)
    assert pdos_file.filename_info["orbital"] == "d"
