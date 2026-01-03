"""Tests for ProjwfcIn parser."""

from pathlib import Path

import pytest

from pyprocar.io.qe.projwfc import ProjwfcIn

# =============================================================================
# Inline String Fixtures
# =============================================================================

BASIC_PROJWFC_IN = """
&projwfc
  outdir = './tmp'
  prefix = 'test'
  filpdos = 'test.pdos'
  filproj = 'test.proj'
  DeltaE = 0.01
  ngauss = 0
  degauss = 0.01
/
"""

MINIMAL_PROJWFC_IN = """
&projwfc
  prefix = 'test'
/
"""

PROJWFC_IN_WITH_KRESOLVEDDOS = """
&projwfc
  outdir = './tmp'
  prefix = 'test'
  filpdos = 'test.pdos'
  kresolveddos = .true.
/
"""

INVALID_FILE = """This is not a valid input file for the projection program.
"""

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def basic_filepath(tmp_path: Path) -> Path:
    """Create temporary file for basic test."""
    filepath = tmp_path / "projwfc.in"
    filepath.write_text(BASIC_PROJWFC_IN)
    return filepath


@pytest.fixture
def minimal_filepath(tmp_path: Path) -> Path:
    """Create temporary file for minimal test."""
    filepath = tmp_path / "projwfc_min.in"
    filepath.write_text(MINIMAL_PROJWFC_IN)
    return filepath


@pytest.fixture
def kresolved_filepath(tmp_path: Path) -> Path:
    """Create temporary file for kresolveddos test."""
    filepath = tmp_path / "projwfc_kres.in"
    filepath.write_text(PROJWFC_IN_WITH_KRESOLVEDDOS)
    return filepath


@pytest.fixture
def invalid_filepath(tmp_path: Path) -> Path:
    """Create temporary file with invalid content."""
    filepath = tmp_path / "invalid.in"
    filepath.write_text(INVALID_FILE)
    return filepath


@pytest.fixture
def basic_parser(basic_filepath: Path) -> ProjwfcIn:
    """Create parser instance for basic test."""
    return ProjwfcIn(filepath=basic_filepath)


@pytest.fixture
def minimal_parser(minimal_filepath: Path) -> ProjwfcIn:
    """Create parser instance for minimal test."""
    return ProjwfcIn(filepath=minimal_filepath)


@pytest.fixture
def kresolved_parser(kresolved_filepath: Path) -> ProjwfcIn:
    """Create parser instance for kresolveddos test."""
    return ProjwfcIn(filepath=kresolved_filepath)


# =============================================================================
# Tests: File Type Identification
# =============================================================================


def test_is_file_of_type_returns_true_for_valid_projwfc_input(basic_filepath: Path) -> None:
    """Test that is_file_of_type correctly identifies valid files."""
    assert ProjwfcIn.is_file_of_type(basic_filepath) is True


def test_is_file_of_type_returns_false_for_pw_input(invalid_filepath: Path) -> None:
    """Test that is_file_of_type rejects invalid files."""
    assert ProjwfcIn.is_file_of_type(invalid_filepath) is False


# =============================================================================
# Tests: Namelist Parsing
# =============================================================================


def test_projwfc_namelist_parses_prefix(basic_parser: ProjwfcIn) -> None:
    """Test that data parses prefix correctly."""
    assert basic_parser.data["prefix"] == "test"


def test_projwfc_namelist_parses_outdir(basic_parser: ProjwfcIn) -> None:
    """Test that data parses outdir correctly."""
    assert basic_parser.data["outdir"] == "./tmp"


def test_filpdos_returns_correct_value(basic_parser: ProjwfcIn) -> None:
    """Test that filpdos parses correctly."""
    assert basic_parser.data["filpdos"] == "test.pdos"


def test_filproj_returns_correct_value(basic_parser: ProjwfcIn) -> None:
    """Test that filproj parses correctly."""
    assert basic_parser.data["filproj"] == "test.proj"


def test_deltae_returns_correct_value(basic_parser: ProjwfcIn) -> None:
    """Test that DeltaE parses correctly."""
    assert float(basic_parser.data["deltae"]) == 0.01


def test_ngauss_returns_correct_value(basic_parser: ProjwfcIn) -> None:
    """Test that ngauss parses correctly."""
    assert int(basic_parser.data["ngauss"]) == 0


def test_degauss_returns_correct_value(basic_parser: ProjwfcIn) -> None:
    """Test that degauss parses correctly."""
    assert float(basic_parser.data["degauss"]) == 0.01


def test_minimal_input_has_default_values(minimal_parser: ProjwfcIn) -> None:
    """Test that minimal input has at least prefix."""
    assert minimal_parser.data["prefix"] == "test"
    # Other values should not be present
    assert "filpdos" not in minimal_parser.data or minimal_parser.data["filpdos"] is None
