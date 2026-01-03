"""Tests for ProjwfcDOS parser."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyprocar.io.qe.projwfc import ProjwfcDOS

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
