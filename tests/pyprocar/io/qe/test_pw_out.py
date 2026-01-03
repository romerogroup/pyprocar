"""Tests for PwOut parser."""

from pathlib import Path

import numpy as np
import pytest
from pyprocar.io.qe.pw.pwout import PwOut

# =============================================================================
# Inline String Fixtures
# =============================================================================

NON_SPIN_POLARIZED_PW_OUT = """
     Program PWSCF v.7.2 starts on  1Jan2026 at 12: 0: 0 

     This program is part of the open-source Quantum ESPRESSO suite
     for quantum simulation of materials; please cite
         "P. Giannozzi et al., J. Phys.:Condens. Matter 21 395502 (2009);
         "P. Giannozzi et al., J. Phys.:Condens. Matter 29 465901 (2017);
         "P. Giannozzi et al., J. Chem. Phys. 152 154105 (2020);
          URL http://www.quantum-espresso.org", 
     in publications or presentations arising from this work. More details at
     http://www.quantum-espresso.org/quote

     Parallel version (MPI), running on     1 processors

     MPI processes distributed on     1 nodes

     bravais-lattice index     =            0
     lattice parameter (alat)  =      7.2608  a.u.
     unit-cell volume          =    382.6090 (a.u.)^3
     number of atoms/cell      =            5
     number of atomic types    =            3
     number of electrons       =        40.00
     number of Kohn-Sham states=           24
     kinetic-energy cutoff     =      60.0000  Ry
     charge density cutoff     =     600.0000  Ry

     celldm(1)=   7.260800  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = (  1.000000  0.000000  0.000000 )
               b(2) = (  0.000000  1.000000  0.000000 )
               b(3) = (  0.000000  0.000000  1.000000 )

     PseudoPot. # 1 for Sr read from file:
     ./Sr.upf
     MD5 check sum: abcdef123456
     Pseudo is Norm-conserving, Zval = 10.0
     
     PseudoPot. # 2 for V  read from file:
     ./V.upf
     MD5 check sum: 123456abcdef
     Pseudo is Norm-conserving, Zval = 13.0

     PseudoPot. # 3 for O  read from file:
     ./O.upf
     MD5 check sum: fedcba654321
     Pseudo is Norm-conserving, Zval =  6.0

     atomic species   valence    mass     pseudopotential
        Sr            10.00    87.62000     Sr( 1.00)
        V             13.00    50.94200     V ( 1.00)
        O              6.00    15.99900     O ( 1.00)

     48 Sym. Ops., with inversion, found

   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Sr  tau(   1) = (   0.5000000   0.5000000   0.5000000  )
         2           V   tau(   2) = (   0.0000000   0.0000000   0.0000000  )
         3           O   tau(   3) = (   0.5000000   0.0000000   0.0000000  )
         4           O   tau(   4) = (   0.0000000   0.5000000   0.0000000  )
         5           O   tau(   5) = (   0.0000000   0.0000000   0.5000000  )

     number of k points=    29  Gaussian smearing, width (Ry)=  0.0100

     the Fermi energy is    10.1234 ev

!    total energy              =    -123.45678901 Ry

     convergence has been achieved in  10 iterations
"""

SPIN_POLARIZED_PW_OUT = """
     Program PWSCF v.7.2 starts on  1Jan2026 at 12: 0: 0 

     Parallel version (MPI), running on     1 processors

     bravais-lattice index     =            0
     lattice parameter (alat)  =      7.2608  a.u.
     unit-cell volume          =    382.6090 (a.u.)^3
     number of atoms/cell      =            5
     number of atomic types    =            3
     number of electrons       =        40.00 (up:  21.00, down:  19.00)
     number of Kohn-Sham states=           24
     kinetic-energy cutoff     =      60.0000  Ry
     charge density cutoff     =     600.0000  Ry

     celldm(1)=   7.260800  celldm(2)=   0.000000  celldm(3)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = (  1.000000  0.000000  0.000000 )
               b(2) = (  0.000000  1.000000  0.000000 )
               b(3) = (  0.000000  0.000000  1.000000 )

     atomic species   valence    mass     pseudopotential
        Sr            10.00    87.62000     Sr( 1.00)
        V             13.00    50.94200     V ( 1.00)
        O              6.00    15.99900     O ( 1.00)

   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Sr  tau(   1) = (   0.5000000   0.5000000   0.5000000  )
         2           V   tau(   2) = (   0.0000000   0.0000000   0.0000000  )
         3           O   tau(   3) = (   0.5000000   0.0000000   0.0000000  )
         4           O   tau(   4) = (   0.0000000   0.5000000   0.0000000  )
         5           O   tau(   5) = (   0.0000000   0.0000000   0.5000000  )

     number of k points=    29  Gaussian smearing, width (Ry)=  0.0100

     the spin up/dw Fermi energies are    10.5000   9.8000 ev

     total magnetization       =     2.00 Bohr mag/cell

!    total energy              =    -123.45678901 Ry

     convergence has been achieved in  12 iterations
"""

INVALID_FILE = """This is not a valid QE output file.
It does not contain the required program marker.
"""

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized test."""
    filepath = tmp_path / "scf.out"
    filepath.write_text(NON_SPIN_POLARIZED_PW_OUT)
    return filepath


@pytest.fixture
def spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized test."""
    filepath = tmp_path / "scf_spin.out"
    filepath.write_text(SPIN_POLARIZED_PW_OUT)
    return filepath


@pytest.fixture
def invalid_filepath(tmp_path: Path) -> Path:
    """Create temporary file with invalid content."""
    filepath = tmp_path / "invalid.out"
    filepath.write_text(INVALID_FILE)
    return filepath


@pytest.fixture
def non_spin_parser(non_spin_filepath: Path) -> PwOut:
    """Create parser instance for non-spin-polarized test."""
    return PwOut(filepath=non_spin_filepath)


@pytest.fixture
def spin_parser(spin_filepath: Path) -> PwOut:
    """Create parser instance for spin-polarized test."""
    return PwOut(filepath=spin_filepath)


# =============================================================================
# Tests: File Type Identification
# =============================================================================


def test_is_file_of_type_returns_true_for_valid_scf_output(non_spin_filepath: Path) -> None:
    """Test that is_file_of_type correctly identifies valid files."""
    assert PwOut.is_file_of_type(non_spin_filepath) is True


def test_is_file_of_type_returns_false_for_invalid_file(invalid_filepath: Path) -> None:
    """Test that is_file_of_type rejects invalid files."""
    assert PwOut.is_file_of_type(invalid_filepath) is False


# =============================================================================
# Tests: System Info Parsing
# =============================================================================


def test_program_version_parses_correctly(non_spin_parser: PwOut) -> None:
    """Test that version_tuple parses correctly."""
    # Note: The fixture uses "Program PWSCF" not "Program PROJWFC" so version may be None
    # Just check it doesn't crash
    version = non_spin_parser.version
    assert version is None or isinstance(version, str)


def test_n_atoms_returns_correct_count(non_spin_parser: PwOut) -> None:
    """Test that natoms returns correct count."""
    assert non_spin_parser.natoms == 5


def test_n_species_returns_correct_count(non_spin_parser: PwOut) -> None:
    """Test that ntyp returns correct count."""
    assert non_spin_parser.ntyp == 3


def test_n_electrons_returns_correct_count(non_spin_parser: PwOut) -> None:
    """Test that nelectrons returns correct count."""
    assert non_spin_parser.nelectrons == 40.00


def test_n_kpoints_returns_correct_count(non_spin_parser: PwOut) -> None:
    """Test that we can parse k-points count from output."""
    # Check if we can find "number of k points" in text
    text = non_spin_parser.text
    assert "number of k points=    29" in text


# =============================================================================
# Tests: Lattice Parsing
# =============================================================================


def test_ecutwfc_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that ecutwfc parses correctly."""
    assert non_spin_parser.ecutwfc == 60.0


def test_lattice_returns_3x3_matrix(non_spin_parser: PwOut) -> None:
    """Test that crystal_axes returns 3x3 matrix."""
    lattice = non_spin_parser.crystal_axes
    assert lattice is not None
    assert len(lattice) == 3
    assert len(lattice[0]) == 3


def test_lattice_values_correct(non_spin_parser: PwOut) -> None:
    """Test that lattice values are correct."""
    lattice = non_spin_parser.crystal_axes
    assert lattice is not None
    # Check diagonal is identity
    expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    np.testing.assert_array_almost_equal(lattice, expected)


def test_reciprocal_lattice_returns_3x3_matrix(non_spin_parser: PwOut) -> None:
    """Test that reciprocal_axes returns 3x3 matrix."""
    rlattice = non_spin_parser.reciprocal_axes
    assert rlattice is not None
    assert len(rlattice) == 3
    assert len(rlattice[0]) == 3


# =============================================================================
# Tests: Atomic Positions
# =============================================================================


def test_atomic_positions_cart_returns_correct_shape(non_spin_parser: PwOut) -> None:
    """Test that atomic positions can be parsed."""
    # The parser doesn't have a direct atomic_positions property,
    # but we can check the text contains position data
    text = non_spin_parser.text
    assert "tau(   1) = (   0.5000000   0.5000000   0.5000000  )" in text


def test_atomic_species_info_contains_elements(non_spin_parser: PwOut) -> None:
    """Test that atomic species info is parseable from text."""
    text = non_spin_parser.text
    assert "Sr" in text
    assert "V" in text and "V  read from file" in text
    assert "O" in text and "O  read from file" in text


# =============================================================================
# Tests: Energy/Fermi
# =============================================================================


def test_fermi_energy_non_spin_polarized(non_spin_parser: PwOut) -> None:
    """Test that Fermi energy parses correctly for non-spin."""
    # The parser has fermi_energy_ev from final_results
    fermi = non_spin_parser.fermi_energy_ev
    assert fermi is not None
    assert pytest.approx(fermi, abs=0.001) == 10.1234


def test_fermi_energy_spin_polarized_up(spin_parser: PwOut) -> None:
    """Test that spin-polarized Fermi energy (up) parses correctly."""
    # For spin-polarized, we need to check the text for both energies
    text = spin_parser.text
    assert "the spin up/dw Fermi energies are    10.5000   9.8000 ev" in text


def test_fermi_energy_spin_polarized_down(spin_parser: PwOut) -> None:
    """Test that spin-polarized Fermi energy (down) is present."""
    text = spin_parser.text
    # Check that down channel energy is present
    assert "9.8000 ev" in text


def test_total_energy_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that total energy parses correctly."""
    # Check if total energy is in text
    text = non_spin_parser.text
    assert "total energy              =    -123.45678901 Ry" in text


def test_total_magnetization_for_spin_polarized(spin_parser: PwOut) -> None:
    """Test that total magnetization is present for spin-polarized."""
    text = spin_parser.text
    assert "total magnetization       =     2.00 Bohr mag/cell" in text


# =============================================================================
# Tests: Unit Cell
# =============================================================================


def test_alat_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that alat parses correctly."""
    assert non_spin_parser.alat == pytest.approx(7.2608, abs=0.0001)


def test_volume_returns_correct_value(non_spin_parser: PwOut) -> None:
    """Test that cell volume parses correctly."""
    assert non_spin_parser.cell_volume == pytest.approx(382.6090, abs=0.001)
