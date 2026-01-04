"""Tests for PwIn parser."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.qe.pw import PwIn
from pyprocar.io.qe.pw.pwin import (
    AtomicPositionsCard,
    AtomicSpeciesCard,
    ControlCard,
    ElectronsCard,
    KPointsCard,
    SystemCard,
)

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
    control_card = non_spin_parser.data["CONTROL"]
    assert isinstance(control_card, ControlCard)
    assert control_card.data["calculation"] == "scf"


def test_control_namelist_parses_prefix(non_spin_parser: PwIn) -> None:
    """Test that control_card parses prefix correctly."""
    control_card = non_spin_parser.data["CONTROL"]
    assert isinstance(control_card, ControlCard)
    assert control_card.data["prefix"] == "test"


def test_system_namelist_parses_nat(non_spin_parser: PwIn) -> None:
    """Test that system_card parses nat correctly."""
    system_card = non_spin_parser.data["SYSTEM"]
    assert isinstance(system_card, SystemCard)
    nat = system_card.data["nat"]
    assert isinstance(nat, (int, float, str))
    assert int(nat) == 5


def test_system_namelist_parses_ntyp(non_spin_parser: PwIn) -> None:
    """Test that system_card parses ntyp correctly."""
    system_card = non_spin_parser.data["SYSTEM"]
    assert isinstance(system_card, SystemCard)
    ntyp = system_card.data["ntyp"]
    assert isinstance(ntyp, (int, float, str))
    assert int(ntyp) == 3


def test_system_namelist_parses_ecutwfc(non_spin_parser: PwIn) -> None:
    """Test that system_card parses ecutwfc correctly."""
    system_card = non_spin_parser.data["SYSTEM"]
    assert isinstance(system_card, SystemCard)
    ecutwfc = system_card.data["ecutwfc"]
    assert isinstance(ecutwfc, (int, float, str))
    assert float(ecutwfc) == 60.0


def test_system_namelist_parses_nspin_for_spin_polarized(spin_parser: PwIn) -> None:
    """Test that system_card parses nspin for spin-polarized calculations."""
    system_card = spin_parser.data["SYSTEM"]
    assert isinstance(system_card, SystemCard)
    nspin = system_card.data["nspin"]
    assert isinstance(nspin, (int, float, str))
    assert int(nspin) == 2


# =============================================================================
# Tests: Atomic Species/Positions
# =============================================================================


def test_atomic_species_returns_correct_count(non_spin_parser: PwIn) -> None:
    """Test that atomic_species_card returns correct number of species."""
    species_card = non_spin_parser.data["ATOMIC_SPECIES"]
    assert isinstance(species_card, AtomicSpeciesCard)
    assert len(species_card.labels) == 3


def test_atomic_species_contains_element_names(non_spin_parser: PwIn) -> None:
    """Test that atomic_species_card contains expected element names."""
    species_card = non_spin_parser.data["ATOMIC_SPECIES"]
    assert isinstance(species_card, AtomicSpeciesCard)
    assert "Sr" in species_card.labels
    assert "V" in species_card.labels
    assert "O" in species_card.labels


def test_atomic_positions_returns_correct_shape(non_spin_parser: PwIn) -> None:
    """Test that atomic_positions_card returns correct shape."""
    positions_card = non_spin_parser.data["ATOMIC_POSITIONS"]
    assert isinstance(positions_card, AtomicPositionsCard)
    assert positions_card.positions is not None
    assert positions_card.positions.shape == (5, 3)


def test_atomic_positions_returns_correct_coordinates(non_spin_parser: PwIn) -> None:
    """Test that atomic_positions_card returns correct coordinates."""
    positions_card = non_spin_parser.data["ATOMIC_POSITIONS"]
    assert isinstance(positions_card, AtomicPositionsCard)
    assert positions_card.positions is not None
    # Check first position (Sr at 0.5, 0.5, 0.5)
    np.testing.assert_array_almost_equal(positions_card.positions[0], [0.5, 0.5, 0.5])


# =============================================================================
# Tests: K_POINTS Modes
# =============================================================================


def test_k_points_automatic_returns_grid(non_spin_parser: PwIn) -> None:
    """Test that kpoints_card with automatic mode returns grid."""
    kpoints_card = non_spin_parser.data["K_POINTS"]
    assert isinstance(kpoints_card, KPointsCard)
    assert kpoints_card.options.lower() == "automatic"
    assert kpoints_card.nk1 == 8
    assert kpoints_card.nk2 == 8
    assert kpoints_card.nk3 == 8


def test_k_points_gamma_returns_gamma_point(gamma_parser: PwIn) -> None:
    """Test that kpoints_card with gamma mode is identified correctly."""
    kpoints_card = gamma_parser.data["K_POINTS"]
    assert isinstance(kpoints_card, KPointsCard)
    assert kpoints_card.options.lower() == "gamma"


def test_k_points_crystal_b_returns_path_with_labels(crystal_b_parser: PwIn) -> None:
    """Test that kpoints_card with crystal_b mode returns path with labels."""
    kpoints_card = crystal_b_parser.data["K_POINTS"]
    assert isinstance(kpoints_card, KPointsCard)
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
    lines = [raw_line.strip() for raw_line in cell_card.block.splitlines() if raw_line.strip()]
    cell_matrix = np.array([[float(v) for v in line.split()] for line in lines])
    assert cell_matrix.shape == (3, 3)


def test_cell_parameters_returns_correct_values(non_spin_parser: PwIn) -> None:
    """Test that cell_card block contains correct lattice values."""
    cell_card = non_spin_parser.data["CELL_PARAMETERS"]
    # Parse the block manually
    lines = [raw_line.strip() for raw_line in cell_card.block.splitlines() if raw_line.strip()]
    cell_matrix = np.array([[float(v) for v in line.split()] for line in lines])
    # Check diagonal values are 3.842
    np.testing.assert_array_almost_equal(np.diag(cell_matrix), [3.842, 3.842, 3.842])


# =============================================================================
# Tests: ControlCard Direct Parsing
# =============================================================================


CONTROL_BLOCK_SCF = """calculation = 'scf',
   outdir = './out',
   pseudo_dir = '.',
   prefix = 'SrVO3',"""

CONTROL_BLOCK_BANDS = """calculation = 'bands',
      outdir = './out',
      pseudo_dir = '.',
      prefix = 'SrVO3',"""


def test_control_card_parses_calculation_scf() -> None:
    """Test ControlCard parses calculation type 'scf'."""
    card = ControlCard(name="CONTROL", options="", block=CONTROL_BLOCK_SCF)
    assert card.data["calculation"] == "scf"


def test_control_card_parses_calculation_bands() -> None:
    """Test ControlCard parses calculation type 'bands'."""
    card = ControlCard(name="CONTROL", options="", block=CONTROL_BLOCK_BANDS)
    assert card.data["calculation"] == "bands"


def test_control_card_parses_outdir() -> None:
    """Test ControlCard parses outdir correctly."""
    card = ControlCard(name="CONTROL", options="", block=CONTROL_BLOCK_SCF)
    assert card.data["outdir"] == "./out"


def test_control_card_parses_pseudo_dir() -> None:
    """Test ControlCard parses pseudo_dir correctly."""
    card = ControlCard(name="CONTROL", options="", block=CONTROL_BLOCK_SCF)
    assert card.data["pseudo_dir"] == "."


def test_control_card_parses_prefix() -> None:
    """Test ControlCard parses prefix correctly."""
    card = ControlCard(name="CONTROL", options="", block=CONTROL_BLOCK_SCF)
    assert card.data["prefix"] == "SrVO3"


# =============================================================================
# Tests: SystemCard Direct Parsing
# =============================================================================


SYSTEM_BLOCK_NON_SPIN = """ibrav = 1
    celldm(1) = 7.268850437
    ntyp = 3
    nat = 5
    ecutwfc = 50.0
    ecutrho = 600.0
    occupations = "smearing"
    degauss     = 0.014"""

SYSTEM_BLOCK_SPIN_POLARIZED = """ibrav = 1
    celldm(1) = 7.268850437
    ntyp = 3
    nat = 5
    ecutwfc = 50.0
    ecutrho = 600.0
    nspin = 2,
    starting_magnetization(1)= 0.7
    occupations = "smearing"
    degauss     = 0.014"""

SYSTEM_BLOCK_NON_COLINEAR = """ibrav = 1
    celldm(1) = 7.268850437
    ntyp = 3
    nat = 5
    ecutwfc = 50.0
    ecutrho = 600.0
    noncolin = .true.,
    lspinorb = .true.,
    starting_magnetization(1)= 0.7
    occupations = "smearing"
    degauss     = 0.014"""


def test_system_card_parses_ibrav() -> None:
    """Test SystemCard parses ibrav correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    ibrav = card.data["ibrav"]
    assert isinstance(ibrav, (int, float, str))
    assert int(ibrav) == 1


def test_system_card_parses_celldm() -> None:
    """Test SystemCard parses celldm(1) correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    # Array notation celldm(1) is stored as dict with tuple index
    celldm = card.data["celldm"]
    assert isinstance(celldm, dict)
    celldm_1 = celldm[(1,)]
    assert isinstance(celldm_1, (int, float, str))
    assert float(celldm_1) == pytest.approx(7.268850437)


def test_system_card_parses_ntyp() -> None:
    """Test SystemCard parses ntyp correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    ntyp = card.data["ntyp"]
    assert isinstance(ntyp, (int, float, str))
    assert int(ntyp) == 3


def test_system_card_parses_nat() -> None:
    """Test SystemCard parses nat correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    nat = card.data["nat"]
    assert isinstance(nat, (int, float, str))
    assert int(nat) == 5


def test_system_card_parses_ecutwfc() -> None:
    """Test SystemCard parses ecutwfc correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    ecutwfc = card.data["ecutwfc"]
    assert isinstance(ecutwfc, (int, float, str))
    assert float(ecutwfc) == pytest.approx(50.0)


def test_system_card_parses_ecutrho() -> None:
    """Test SystemCard parses ecutrho correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    ecutrho = card.data["ecutrho"]
    assert isinstance(ecutrho, (int, float, str))
    assert float(ecutrho) == pytest.approx(600.0)


def test_system_card_parses_occupations() -> None:
    """Test SystemCard parses occupations correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    assert card.data["occupations"] == "smearing"


def test_system_card_parses_degauss() -> None:
    """Test SystemCard parses degauss correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    degauss = card.data["degauss"]
    assert isinstance(degauss, (int, float, str))
    assert float(degauss) == pytest.approx(0.014)


def test_system_card_parses_nspin_for_spin_polarized() -> None:
    """Test SystemCard parses nspin=2 for spin-polarized calculations."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_SPIN_POLARIZED)
    nspin = card.data["nspin"]
    assert isinstance(nspin, (int, float, str))
    assert int(nspin) == 2


def test_system_card_parses_starting_magnetization() -> None:
    """Test SystemCard parses starting_magnetization correctly."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_SPIN_POLARIZED)
    # Array notation starting_magnetization(1) is stored as dict with tuple index
    start_mag = card.data["starting_magnetization"]
    assert isinstance(start_mag, dict)
    start_mag_1 = start_mag[(1,)]
    assert isinstance(start_mag_1, (int, float, str))
    assert float(start_mag_1) == pytest.approx(0.7)


def test_system_card_parses_noncolin_true() -> None:
    """Test SystemCard parses noncolin=.true. for non-colinear calculations."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_COLINEAR)
    assert card.data["noncolin"] is True


def test_system_card_parses_lspinorb_true() -> None:
    """Test SystemCard parses lspinorb=.true. for SOC calculations."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_COLINEAR)
    assert card.data["lspinorb"] is True


def test_system_card_no_nspin_for_non_spin() -> None:
    """Test SystemCard has no nspin key for non-spin-polarized calculations."""
    card = SystemCard(name="SYSTEM", options="", block=SYSTEM_BLOCK_NON_SPIN)
    assert "nspin" not in card.data


# =============================================================================
# Tests: ElectronsCard Direct Parsing
# =============================================================================


ELECTRONS_BLOCK_EMPTY = ""

ELECTRONS_BLOCK_WITH_PARAMS = """conv_thr = 1.0d-8
    mixing_beta = 0.7"""


def test_electrons_card_empty_block() -> None:
    """Test ElectronsCard handles empty block."""
    card = ElectronsCard(name="ELECTRONS", options="", block=ELECTRONS_BLOCK_EMPTY)
    assert card.data == {}


def test_electrons_card_parses_conv_thr() -> None:
    """Test ElectronsCard parses conv_thr correctly."""
    card = ElectronsCard(name="ELECTRONS", options="", block=ELECTRONS_BLOCK_WITH_PARAMS)
    conv_thr = card.data["conv_thr"]
    assert isinstance(conv_thr, (int, float, str))
    assert float(conv_thr) == pytest.approx(1.0e-8)


def test_electrons_card_parses_mixing_beta() -> None:
    """Test ElectronsCard parses mixing_beta correctly."""
    card = ElectronsCard(name="ELECTRONS", options="", block=ELECTRONS_BLOCK_WITH_PARAMS)
    mixing_beta = card.data["mixing_beta"]
    assert isinstance(mixing_beta, (int, float, str))
    assert float(mixing_beta) == pytest.approx(0.7)


# =============================================================================
# Tests: AtomicSpeciesCard Direct Parsing
# =============================================================================


ATOMIC_SPECIES_BLOCK = """Sr   87.6200000000  Sr.pbe-spn-kjpaw_psl.1.0.0.UPF
  V   50.9415000000  V.pbe-spn-kjpaw_psl.1.0.0.UPF
  O   15.9994000000  O.pbe-n-kjpaw_psl.0.1.upf"""


def test_atomic_species_card_parses_labels() -> None:
    """Test AtomicSpeciesCard parses element labels correctly."""
    card = AtomicSpeciesCard(name="ATOMIC_SPECIES", options="", block=ATOMIC_SPECIES_BLOCK)
    assert card.labels == ["Sr", "V", "O"]


def test_atomic_species_card_parses_masses() -> None:
    """Test AtomicSpeciesCard parses atomic masses correctly."""
    card = AtomicSpeciesCard(name="ATOMIC_SPECIES", options="", block=ATOMIC_SPECIES_BLOCK)
    assert card.masses is not None
    np.testing.assert_array_almost_equal(card.masses, [87.62, 50.9415, 15.9994], decimal=4)


def test_atomic_species_card_parses_pseudopotentials() -> None:
    """Test AtomicSpeciesCard parses pseudopotential filenames correctly."""
    card = AtomicSpeciesCard(name="ATOMIC_SPECIES", options="", block=ATOMIC_SPECIES_BLOCK)
    assert card.pseudopotentials[0] == "Sr.pbe-spn-kjpaw_psl.1.0.0.UPF"
    assert card.pseudopotentials[1] == "V.pbe-spn-kjpaw_psl.1.0.0.UPF"
    assert card.pseudopotentials[2] == "O.pbe-n-kjpaw_psl.0.1.upf"


def test_atomic_species_card_infers_upf_format() -> None:
    """Test AtomicSpeciesCard infers UPF pseudo format correctly."""
    card = AtomicSpeciesCard(name="ATOMIC_SPECIES", options="", block=ATOMIC_SPECIES_BLOCK)
    assert card.pseudo_formats == ["UPF", "UPF", "UPF"]


def test_atomic_species_card_species_dict() -> None:
    """Test AtomicSpeciesCard species property returns correct mapping."""
    card = AtomicSpeciesCard(name="ATOMIC_SPECIES", options="", block=ATOMIC_SPECIES_BLOCK)
    assert card.species["Sr"] == pytest.approx(87.62)
    assert card.species["V"] == pytest.approx(50.9415)
    assert card.species["O"] == pytest.approx(15.9994)


def test_atomic_species_card_with_comments() -> None:
    """Test AtomicSpeciesCard handles lines with comments."""
    block = """Sr   87.62  Sr.upf  ! Strontium
    V   50.94  V.upf   # Vanadium"""
    card = AtomicSpeciesCard(name="ATOMIC_SPECIES", options="", block=block)
    assert card.labels == ["Sr", "V"]


# =============================================================================
# Tests: AtomicPositionsCard Direct Parsing
# =============================================================================


ATOMIC_POSITIONS_BLOCK_CRYSTAL = """Sr   0.00000000000000   0.00000000000000   0.00000000000000
  V   0.50000000000000   0.50000000000000   0.50000000000000
  O   0.50000000000000   0.00000000000000   0.50000000000000
  O   0.00000000000000   0.50000000000000   0.50000000000000
  O   0.50000000000000   0.50000000000000   0.00000000000000"""


def test_atomic_positions_card_parses_mode_crystal() -> None:
    """Test AtomicPositionsCard parses mode 'crystal' correctly."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="crystal", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    assert card.mode == "crystal"


def test_atomic_positions_card_parses_mode_angstrom() -> None:
    """Test AtomicPositionsCard parses mode 'angstrom' correctly."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="angstrom", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    assert card.mode == "angstrom"


def test_atomic_positions_card_parses_mode_bohr() -> None:
    """Test AtomicPositionsCard parses mode 'bohr' correctly."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="bohr", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    assert card.mode == "bohr"


def test_atomic_positions_card_parses_mode_alat_default() -> None:
    """Test AtomicPositionsCard defaults to 'alat' mode when unspecified."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    assert card.mode == "alat"


def test_atomic_positions_card_parses_labels() -> None:
    """Test AtomicPositionsCard parses atom labels correctly."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="crystal", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    assert card.labels == ["Sr", "V", "O", "O", "O"]


def test_atomic_positions_card_parses_positions_shape() -> None:
    """Test AtomicPositionsCard parses positions with correct shape."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="crystal", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    assert card.positions is not None
    assert card.positions.shape == (5, 3)


def test_atomic_positions_card_parses_positions_values() -> None:
    """Test AtomicPositionsCard parses position values correctly."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="crystal", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    assert card.positions is not None
    # Sr at origin
    np.testing.assert_array_almost_equal(card.positions[0], [0.0, 0.0, 0.0])
    # V at body center
    np.testing.assert_array_almost_equal(card.positions[1], [0.5, 0.5, 0.5])


def test_atomic_positions_card_parses_constraints_default() -> None:
    """Test AtomicPositionsCard defaults constraints to (1,1,1) when not specified."""
    card = AtomicPositionsCard(
        name="ATOMIC_POSITIONS", options="crystal", block=ATOMIC_POSITIONS_BLOCK_CRYSTAL
    )
    # All atoms should have default constraints (1, 1, 1) = mobile
    assert card.constraints is not None
    assert card.constraints.shape == (5, 3)
    np.testing.assert_array_equal(card.constraints[0], [1, 1, 1])


def test_atomic_positions_card_parses_constraints_with_braces() -> None:
    """Test AtomicPositionsCard parses constraints in braces."""
    block = """Sr   0.0   0.0   0.0 { 0 0 0 }
    V   0.5   0.5   0.5 { 1 1 1 }"""
    card = AtomicPositionsCard(name="ATOMIC_POSITIONS", options="crystal", block=block)
    assert card.constraints is not None
    # Sr fixed
    np.testing.assert_array_equal(card.constraints[0], [0, 0, 0])
    # V mobile
    np.testing.assert_array_equal(card.constraints[1], [1, 1, 1])


def test_atomic_positions_card_with_arithmetic_expressions() -> None:
    """Test AtomicPositionsCard handles arithmetic expressions in coordinates."""
    block = """H   1/4   0.5   1.0-0.75"""
    card = AtomicPositionsCard(name="ATOMIC_POSITIONS", options="crystal", block=block)
    assert card.positions is not None
    np.testing.assert_array_almost_equal(card.positions[0], [0.25, 0.5, 0.25])


def test_atomic_positions_card_with_comments() -> None:
    """Test AtomicPositionsCard handles lines with comments."""
    block = """Sr   0.0   0.0   0.0 ! Strontium
    V   0.5   0.5   0.5 # Vanadium"""
    card = AtomicPositionsCard(name="ATOMIC_POSITIONS", options="crystal", block=block)
    assert card.labels == ["Sr", "V"]


# =============================================================================
# Tests: KPointsCard Direct Parsing - Automatic Mode
# =============================================================================


KPOINTS_BLOCK_AUTOMATIC = """15 15 15 0 0 0"""


def test_kpoints_card_automatic_parses_mode() -> None:
    """Test KPointsCard with automatic mode parses mode correctly."""
    card = KPointsCard(name="K_POINTS", options="automatic", block=KPOINTS_BLOCK_AUTOMATIC)
    assert card.mode == "automatic"


def test_kpoints_card_automatic_parses_nk1() -> None:
    """Test KPointsCard with automatic mode parses nk1 correctly."""
    card = KPointsCard(name="K_POINTS", options="automatic", block=KPOINTS_BLOCK_AUTOMATIC)
    assert card.nk1 == 15


def test_kpoints_card_automatic_parses_nk2() -> None:
    """Test KPointsCard with automatic mode parses nk2 correctly."""
    card = KPointsCard(name="K_POINTS", options="automatic", block=KPOINTS_BLOCK_AUTOMATIC)
    assert card.nk2 == 15


def test_kpoints_card_automatic_parses_nk3() -> None:
    """Test KPointsCard with automatic mode parses nk3 correctly."""
    card = KPointsCard(name="K_POINTS", options="automatic", block=KPOINTS_BLOCK_AUTOMATIC)
    assert card.nk3 == 15


def test_kpoints_card_automatic_parses_shift() -> None:
    """Test KPointsCard with automatic mode parses shift correctly."""
    card = KPointsCard(name="K_POINTS", options="automatic", block=KPOINTS_BLOCK_AUTOMATIC)
    assert card.sk1 == 0
    assert card.sk2 == 0
    assert card.sk3 == 0


def test_kpoints_card_automatic_with_shift() -> None:
    """Test KPointsCard with automatic mode with non-zero shift."""
    block = "8 8 8 1 1 1"
    card = KPointsCard(name="K_POINTS", options="automatic", block=block)
    assert card.nk1 == 8
    assert card.sk1 == 1
    assert card.sk2 == 1
    assert card.sk3 == 1


# =============================================================================
# Tests: KPointsCard Direct Parsing - Gamma Mode
# =============================================================================


def test_kpoints_card_gamma_parses_mode() -> None:
    """Test KPointsCard with gamma mode parses mode correctly."""
    card = KPointsCard(name="K_POINTS", options="gamma", block="")
    assert card.mode == "gamma"


def test_kpoints_card_gamma_is_gamma_flag() -> None:
    """Test KPointsCard with gamma mode sets is_gamma flag."""
    card = KPointsCard(name="K_POINTS", options="gamma", block="")
    assert card.is_gamma is True


def test_kpoints_card_gamma_kpoints_at_origin() -> None:
    """Test KPointsCard with gamma mode sets kpoint at origin."""
    card = KPointsCard(name="K_POINTS", options="gamma", block="")
    assert card.kpoints is not None
    np.testing.assert_array_almost_equal(card.kpoints, [[0.0, 0.0, 0.0]])


def test_kpoints_card_gamma_weights_one() -> None:
    """Test KPointsCard with gamma mode sets weight to 1.0."""
    card = KPointsCard(name="K_POINTS", options="gamma", block="")
    assert card.weights is not None
    np.testing.assert_array_almost_equal(card.weights, [1.0])


# =============================================================================
# Tests: KPointsCard Direct Parsing - Crystal_b Mode (Band Path)
# =============================================================================


KPOINTS_BLOCK_CRYSTAL_B = """6
0.0      0.0      0.0 30 !Gamma
0.5      0.0      0.0 30 !X
0.5      0.5      0.0 30 !M
0.0      0.0      0.0 30 !G
0.5      0.5      0.5 30 !R
0.5      0.0      0.0 30 !X"""


def test_kpoints_card_crystal_b_parses_mode() -> None:
    """Test KPointsCard with crystal_b mode parses mode correctly."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    assert card.mode == "crystal_b"


def test_kpoints_card_crystal_b_parses_nhigh_sym() -> None:
    """Test KPointsCard with crystal_b mode parses number of high-symmetry points."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    assert card.nhigh_sym == 6


def test_kpoints_card_crystal_b_parses_high_symmetry_points_shape() -> None:
    """Test KPointsCard with crystal_b mode parses high-symmetry points with correct shape."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    assert card.high_symmetry_points is not None
    assert card.high_symmetry_points.shape == (6, 3)


def test_kpoints_card_crystal_b_parses_gamma_point() -> None:
    """Test KPointsCard with crystal_b mode parses Gamma point correctly."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    assert card.high_symmetry_points is not None
    np.testing.assert_array_almost_equal(card.high_symmetry_points[0], [0.0, 0.0, 0.0])


def test_kpoints_card_crystal_b_parses_x_point() -> None:
    """Test KPointsCard with crystal_b mode parses X point correctly."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    assert card.high_symmetry_points is not None
    np.testing.assert_array_almost_equal(card.high_symmetry_points[1], [0.5, 0.0, 0.0])


def test_kpoints_card_crystal_b_parses_line_points() -> None:
    """Test KPointsCard with crystal_b mode parses line point counts."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    assert card.line_points is not None
    np.testing.assert_array_equal(card.line_points, [30, 30, 30, 30, 30, 30])


def test_kpoints_card_crystal_b_parses_knames() -> None:
    """Test KPointsCard with crystal_b mode parses k-point names."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    assert card.knames == ["Gamma", "X", "M", "G", "R", "X"]


def test_kpoints_card_crystal_b_parses_kticks() -> None:
    """Test KPointsCard with crystal_b mode parses k-tick positions."""
    card = KPointsCard(name="K_POINTS", options="crystal_b", block=KPOINTS_BLOCK_CRYSTAL_B)
    # kticks should be cumulative positions
    assert len(card.kticks) == 6


# =============================================================================
# Tests: KPointsCard Direct Parsing - Explicit (tpiba) Mode
# =============================================================================


KPOINTS_BLOCK_EXPLICIT = """3
0.0  0.0  0.0  1.0
0.5  0.0  0.0  2.0
0.5  0.5  0.0  1.0"""


def test_kpoints_card_explicit_parses_mode() -> None:
    """Test KPointsCard with tpiba mode parses mode correctly."""
    card = KPointsCard(name="K_POINTS", options="tpiba", block=KPOINTS_BLOCK_EXPLICIT)
    assert card.mode == "tpiba"


def test_kpoints_card_explicit_parses_kpoints_shape() -> None:
    """Test KPointsCard with explicit mode parses kpoints with correct shape."""
    card = KPointsCard(name="K_POINTS", options="tpiba", block=KPOINTS_BLOCK_EXPLICIT)
    assert card.kpoints is not None
    assert card.kpoints.shape == (3, 3)


def test_kpoints_card_explicit_parses_kpoints_values() -> None:
    """Test KPointsCard with explicit mode parses kpoint values correctly."""
    card = KPointsCard(name="K_POINTS", options="tpiba", block=KPOINTS_BLOCK_EXPLICIT)
    assert card.kpoints is not None
    np.testing.assert_array_almost_equal(card.kpoints[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(card.kpoints[1], [0.5, 0.0, 0.0])


def test_kpoints_card_explicit_parses_weights() -> None:
    """Test KPointsCard with explicit mode parses weights correctly."""
    card = KPointsCard(name="K_POINTS", options="tpiba", block=KPOINTS_BLOCK_EXPLICIT)
    assert card.weights is not None
    np.testing.assert_array_almost_equal(card.weights, [1.0, 2.0, 1.0])


def test_kpoints_card_explicit_with_labels() -> None:
    """Test KPointsCard with explicit mode parses labels."""
    block = """2
0.0  0.0  0.0  1.0 !Gamma
0.5  0.0  0.0  1.0 !X"""
    card = KPointsCard(name="K_POINTS", options="tpiba", block=block)
    assert card.line_comments == ["Gamma", "X"]


# =============================================================================
# Tests: KPointsCard Direct Parsing - Crystal Mode
# =============================================================================


KPOINTS_BLOCK_CRYSTAL = """4
0.0  0.0  0.0  1.0 !Gamma
0.5  0.0  0.0  1.0 !X
0.5  0.5  0.0  1.0 !M
0.0  0.0  0.0  1.0 !Gamma"""


def test_kpoints_card_crystal_parses_mode() -> None:
    """Test KPointsCard with crystal mode parses mode correctly."""
    card = KPointsCard(name="K_POINTS", options="crystal", block=KPOINTS_BLOCK_CRYSTAL)
    assert card.mode == "crystal"


def test_kpoints_card_crystal_parses_knames() -> None:
    """Test KPointsCard with crystal mode parses k-point names."""
    card = KPointsCard(name="K_POINTS", options="crystal", block=KPOINTS_BLOCK_CRYSTAL)
    assert card.knames == ["Gamma", "X", "M", "Gamma"]


def test_kpoints_card_crystal_parses_nhigh_sym() -> None:
    """Test KPointsCard with crystal mode parses number of high-symmetry points."""
    card = KPointsCard(name="K_POINTS", options="crystal", block=KPOINTS_BLOCK_CRYSTAL)
    assert card.nhigh_sym == 4
