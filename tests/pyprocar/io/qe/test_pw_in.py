"""Tests for PwIn parser."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.qe.pw import PwIn

# =============================================================================
# Inline String Fixtures
# =============================================================================

NON_SPIN_POLARIZED_PW_IN = """&CONTROL
  calculation = 'scf'
  prefix = 'test'
  outdir = './tmp'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 0
  nat = 5
  ntyp = 3
  ecutwfc = 60.0
  ecutrho = 600.0
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0d-8
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

SPIN_POLARIZED_PW_IN = """&CONTROL
  calculation = 'scf'
  prefix = 'test'
  outdir = './tmp'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 0
  nat = 5
  ntyp = 3
  ecutwfc = 60.0
  ecutrho = 600.0
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
  nspin = 2
  starting_magnetization(1) = 0.0
  starting_magnetization(2) = 0.5
  starting_magnetization(3) = 0.0
/
&ELECTRONS
  conv_thr = 1.0d-8
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

K_POINTS_GAMMA_PW_IN = """&CONTROL
  calculation = 'scf'
  prefix = 'test'
/
&SYSTEM
  ibrav = 0
  nat = 1
  ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS
/
ATOMIC_SPECIES
H  1.008  H.upf
ATOMIC_POSITIONS crystal
H  0.0  0.0  0.0
K_POINTS gamma
CELL_PARAMETERS angstrom
2.0  0.0  0.0
0.0  2.0  0.0
0.0  0.0  2.0
"""

K_POINTS_CRYSTAL_B_PW_IN = """&CONTROL
  calculation = 'bands'
  prefix = 'test'
/
&SYSTEM
  ibrav = 0
  nat = 1
  ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS
/
ATOMIC_SPECIES
H  1.008  H.upf
ATOMIC_POSITIONS crystal
H  0.0  0.0  0.0
K_POINTS crystal_b
4
0.0  0.0  0.0  20  ! G
0.5  0.0  0.0  20  ! X
0.5  0.5  0.0  20  ! M
0.0  0.0  0.0  1   ! G
CELL_PARAMETERS angstrom
2.0  0.0  0.0
0.0  2.0  0.0
0.0  0.0  2.0
"""

INVALID_FILE = """This is not a valid QE input file.
It does not contain the required namelists.
"""

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized test."""
    filepath = tmp_path / "scf.in"
    filepath.write_text(NON_SPIN_POLARIZED_PW_IN)
    return filepath


@pytest.fixture
def spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized test."""
    filepath = tmp_path / "scf_spin.in"
    filepath.write_text(SPIN_POLARIZED_PW_IN)
    return filepath


@pytest.fixture
def gamma_filepath(tmp_path: Path) -> Path:
    """Create temporary file for gamma-point test."""
    filepath = tmp_path / "scf_gamma.in"
    filepath.write_text(K_POINTS_GAMMA_PW_IN)
    return filepath


@pytest.fixture
def crystal_b_filepath(tmp_path: Path) -> Path:
    """Create temporary file for crystal_b bands test."""
    filepath = tmp_path / "bands.in"
    filepath.write_text(K_POINTS_CRYSTAL_B_PW_IN)
    return filepath


@pytest.fixture
def invalid_filepath(tmp_path: Path) -> Path:
    """Create temporary file with invalid content."""
    filepath = tmp_path / "invalid.in"
    filepath.write_text(INVALID_FILE)
    return filepath


@pytest.fixture
def non_spin_parser(non_spin_filepath: Path) -> PwIn:
    """Create parser instance for non-spin-polarized test."""
    return PwIn(filepath=non_spin_filepath)


@pytest.fixture
def spin_parser(spin_filepath: Path) -> PwIn:
    """Create parser instance for spin-polarized test."""
    return PwIn(filepath=spin_filepath)


@pytest.fixture
def gamma_parser(gamma_filepath: Path) -> PwIn:
    """Create parser instance for gamma-point test."""
    return PwIn(filepath=gamma_filepath)


@pytest.fixture
def crystal_b_parser(crystal_b_filepath: Path) -> PwIn:
    """Create parser instance for crystal_b test."""
    return PwIn(filepath=crystal_b_filepath)


# =============================================================================
# Tests: File Type Identification
# =============================================================================


def test_is_file_of_type_returns_true_for_valid_scf_input(non_spin_filepath: Path) -> None:
    """Test that is_file_of_type correctly identifies valid files."""
    assert PwIn.is_file_of_type(non_spin_filepath) is True


def test_is_file_of_type_returns_false_for_invalid_file(invalid_filepath: Path) -> None:
    """Test that is_file_of_type rejects invalid files."""
    assert PwIn.is_file_of_type(invalid_filepath) is False


# =============================================================================
# Tests: Namelist Parsing
# =============================================================================


def test_control_namelist_parses_calculation_type(non_spin_parser: PwIn) -> None:
    """Test that control_card parses calculation type correctly."""
    control_data = non_spin_parser.data["CONTROL"].data
    assert control_data["calculation"] == "scf"


def test_control_namelist_parses_prefix(non_spin_parser: PwIn) -> None:
    """Test that control_card parses prefix correctly."""
    control_data = non_spin_parser.data["CONTROL"].data
    assert control_data["prefix"] == "test"


def test_system_namelist_parses_nat(non_spin_parser: PwIn) -> None:
    """Test that system_card parses nat correctly."""
    system_data = non_spin_parser.data["SYSTEM"].data
    assert int(system_data["nat"]) == 5


def test_system_namelist_parses_ntyp(non_spin_parser: PwIn) -> None:
    """Test that system_card parses ntyp correctly."""
    system_data = non_spin_parser.data["SYSTEM"].data
    assert int(system_data["ntyp"]) == 3


def test_system_namelist_parses_ecutwfc(non_spin_parser: PwIn) -> None:
    """Test that system_card parses ecutwfc correctly."""
    system_data = non_spin_parser.data["SYSTEM"].data
    assert float(system_data["ecutwfc"]) == 60.0


def test_system_namelist_parses_nspin_for_spin_polarized(spin_parser: PwIn) -> None:
    """Test that system_card parses nspin for spin-polarized calculations."""
    system_data = spin_parser.data["SYSTEM"].data
    assert int(system_data["nspin"]) == 2


# =============================================================================
# Tests: Atomic Species/Positions
# =============================================================================


def test_atomic_species_returns_correct_count(non_spin_parser: PwIn) -> None:
    """Test that atomic_species_card returns correct number of species."""
    species_card = non_spin_parser.data["ATOMIC_SPECIES"]
    assert len(species_card.labels) == 3


def test_atomic_species_contains_element_names(non_spin_parser: PwIn) -> None:
    """Test that atomic_species_card contains expected element names."""
    species_card = non_spin_parser.data["ATOMIC_SPECIES"]
    assert "Sr" in species_card.labels
    assert "V" in species_card.labels
    assert "O" in species_card.labels


def test_atomic_positions_returns_correct_shape(non_spin_parser: PwIn) -> None:
    """Test that atomic_positions_card returns correct shape."""
    positions_card = non_spin_parser.data["ATOMIC_POSITIONS"]
    assert positions_card.positions.shape == (5, 3)


def test_atomic_positions_returns_correct_coordinates(non_spin_parser: PwIn) -> None:
    """Test that atomic_positions_card returns correct coordinates."""
    positions_card = non_spin_parser.data["ATOMIC_POSITIONS"]
    # Check first position (Sr at 0.5, 0.5, 0.5)
    np.testing.assert_array_almost_equal(positions_card.positions[0], [0.5, 0.5, 0.5])


# =============================================================================
# Tests: K_POINTS Modes
# =============================================================================


def test_k_points_automatic_returns_grid(non_spin_parser: PwIn) -> None:
    """Test that kpoints_card with automatic mode returns grid."""
    kpoints_card = non_spin_parser.data["K_POINTS"]
    assert kpoints_card.options.lower() == "automatic"
    assert kpoints_card.nk1 == 8
    assert kpoints_card.nk2 == 8
    assert kpoints_card.nk3 == 8


def test_k_points_gamma_returns_gamma_point(gamma_parser: PwIn) -> None:
    """Test that kpoints_card with gamma mode is identified correctly."""
    kpoints_card = gamma_parser.data["K_POINTS"]
    assert kpoints_card.options.lower() == "gamma"


def test_k_points_crystal_b_returns_path_with_labels(crystal_b_parser: PwIn) -> None:
    """Test that kpoints_card with crystal_b mode returns path with labels."""
    kpoints_card = crystal_b_parser.data["K_POINTS"]
    assert kpoints_card.options.lower() == "crystal_b"
    labels = crystal_b_parser.bands_kpoint_names
    assert labels is not None
    assert "G" in labels


# =============================================================================
# Tests: Cell Parameters
# =============================================================================


def test_cell_parameters_returns_3x3_matrix(non_spin_parser: PwIn) -> None:
    """Test that cell_card block can be parsed into 3x3 matrix."""
    cell_card = non_spin_parser.data["CELL_PARAMETERS"]
    # Parse the block manually
    lines = [l.strip() for l in cell_card.block.splitlines() if l.strip()]
    cell_matrix = np.array([[float(v) for v in line.split()] for line in lines])
    assert cell_matrix.shape == (3, 3)


def test_cell_parameters_returns_correct_values(non_spin_parser: PwIn) -> None:
    """Test that cell_card block contains correct lattice values."""
    cell_card = non_spin_parser.data["CELL_PARAMETERS"]
    # Parse the block manually
    lines = [l.strip() for l in cell_card.block.splitlines() if l.strip()]
    cell_matrix = np.array([[float(v) for v in line.split()] for line in lines])
    # Check diagonal values are 3.842
    np.testing.assert_array_almost_equal(np.diag(cell_matrix), [3.842, 3.842, 3.842])
