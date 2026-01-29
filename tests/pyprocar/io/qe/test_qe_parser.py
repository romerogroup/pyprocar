"""Tests for QEParser (auto-detection and integration)."""

from pathlib import Path

import pytest

from pyprocar.io.qe.parser import QEParser

# =============================================================================
# Inline String Fixtures
# =============================================================================

# Minimal SCF input file
SCF_IN = """
&CONTROL
  calculation = 'scf'
  prefix = 'test'
  outdir = './tmp'
  pseudo_dir = './'
/
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
  ecutwfc = 30.0
/
&ELECTRONS
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
Si  28.086  Si.upf
ATOMIC_POSITIONS crystal
Si  0.000  0.000  0.000
Si  0.250  0.250  0.250
K_POINTS automatic
4 4 4 0 0 0
CELL_PARAMETERS angstrom
5.43  0.000  0.000
0.000  5.43  0.000
0.000  0.000  5.43
"""

# Minimal SCF output file
SCF_OUT = """

     Program PWSCF v.7.2 starts on  1Jan2026 at 12: 0: 0

     bravais-lattice index     =            0
     lattice parameter (alat)  =     10.2608  a.u.
     unit-cell volume          =   1080.5090 (a.u.)^3
     number of atoms/cell      =            2
     number of atomic types    =            1
     number of electrons       =         8.00
     number of Kohn-Sham states=           12
     kinetic-energy cutoff     =      30.0000  Ry

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = (  1.000000  0.000000  0.000000 )
               b(2) = (  0.000000  1.000000  0.000000 )
               b(3) = (  0.000000  0.000000  1.000000 )

     atomic species   valence    mass     pseudopotential
        Si             4.00    28.08600     Si( 1.00)

   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Si  tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2           Si  tau(   2) = (   0.2500000   0.2500000   0.2500000  )

     number of k points=    10

     the Fermi energy is     6.5000 ev

!    total energy              =     -15.85472100 Ry

     convergence has been achieved in  8 iterations
"""

# Minimal Projwfc input file
PROJWFC_IN = """
&projwfc
  outdir = './tmp'
  prefix = 'test'
  filpdos = 'test.pdos'
/
"""

# Minimal Projwfc output file
PROJWFC_OUT = """

     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0

     Parallel version (MPI), running on     1 processors

     Calling projwave ....

     Atomic states used for projection
     (read from pseudopotential files):

     state #   1: atom   1 (Si ), wfc  1 (l=0 m= 1)
     state #   2: atom   1 (Si ), wfc  2 (l=1 m= 1)
     state #   3: atom   1 (Si ), wfc  2 (l=1 m= 2)
     state #   4: atom   1 (Si ), wfc  2 (l=1 m= 3)
     state #   5: atom   2 (Si ), wfc  1 (l=0 m= 1)
     state #   6: atom   2 (Si ), wfc  2 (l=1 m= 1)
     state #   7: atom   2 (Si ), wfc  2 (l=1 m= 2)
     state #   8: atom   2 (Si ), wfc  2 (l=1 m= 3)

     natomwfc =    8
     nbnd     =   12
     nkstot   =   10
     nspin    =    1

     k =   0.0000   0.0000   0.0000
     k =   0.2500   0.0000   0.0000
     k =   0.5000   0.0000   0.0000
     k =   0.2500   0.2500   0.0000
     k =   0.5000   0.2500   0.0000
     k =   0.5000   0.5000   0.0000
     k =   0.2500   0.2500   0.2500
     k =   0.5000   0.2500   0.2500
     k =   0.5000   0.5000   0.2500
     k =   0.5000   0.5000   0.5000

     PROJWFC      :      1.00s CPU      1.10s WALL

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""

# Minimal data-file-schema.xml
DATA_FILE_SCHEMA_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <general_info>
    <xml_format NAME="QEXSD" VERSION="21.11.01">QEXSD_21.11.01</xml_format>
  </general_info>
  <input>
    <control_variables>
      <calculation>scf</calculation>
    </control_variables>
    <spin>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </spin>
    <bands>
      <nbnd>12</nbnd>
    </bands>
  </input>
  <output>
    <atomic_structure nat="2" alat="10.2608">
      <atomic_positions>
        <atom name="Si" index="1">0.000 0.000 0.000</atom>
        <atom name="Si" index="2">2.565 2.565 2.565</atom>
      </atomic_positions>
      <cell>
        <a1>5.43 0.0 0.0</a1>
        <a2>0.0 5.43 0.0</a2>
        <a3>0.0 0.0 5.43</a3>
      </cell>
    </atomic_structure>
    <band_structure>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
      <nbnd>12</nbnd>
      <nelec>8.0</nelec>
      <fermi_energy>0.239</fermi_energy>
      <starting_k_points>
        <nk>10</nk>
      </starting_k_points>
      <ks_energies>
        <k_point weight="0.001">0.0 0.0 0.0</k_point>
        <npw>1000</npw>
        <eigenvalues size="12">
          -0.2 -0.1 0.0 0.05 0.1 0.15 0.2 0.21 0.22 0.23 0.3 0.4
        </eigenvalues>
        <occupations size="12">
          1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        </occupations>
      </ks_energies>
    </band_structure>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def scf_calculation_dir(tmp_path: Path) -> Path:
    """Create temporary directory with SCF calculation files."""
    calc_dir = tmp_path / "scf_calc"
    calc_dir.mkdir()

    # Create SCF input file
    scf_in_file = calc_dir / "scf.in"
    scf_in_file.write_text(SCF_IN)

    # Create SCF output file
    scf_out_file = calc_dir / "scf.out"
    scf_out_file.write_text(SCF_OUT)

    return calc_dir


@pytest.fixture
def scf_parser(scf_calculation_dir: Path) -> QEParser:
    """Create parser instance for SCF calculation."""
    return QEParser(dirpath=scf_calculation_dir)


@pytest.fixture
def dos_calculation_dir(tmp_path: Path) -> Path:
    """Create temporary directory with DOS calculation files."""
    calc_dir = tmp_path / "dos_calc"
    calc_dir.mkdir()

    # Create SCF input file
    scf_in_file = calc_dir / "scf.in"
    scf_in_file.write_text(SCF_IN)

    # Create SCF output file
    scf_out_file = calc_dir / "scf.out"
    scf_out_file.write_text(SCF_OUT)

    # Create projwfc input file
    projwfc_in_file = calc_dir / "projwfc.in"
    projwfc_in_file.write_text(PROJWFC_IN)

    # Create projwfc output file
    projwfc_out_file = calc_dir / "projwfc.out"
    projwfc_out_file.write_text(PROJWFC_OUT)

    return calc_dir


@pytest.fixture
def dos_parser(dos_calculation_dir: Path) -> QEParser:
    """Create parser instance for DOS calculation."""
    return QEParser(dirpath=dos_calculation_dir)


@pytest.fixture
def xml_calculation_dir(tmp_path: Path) -> Path:
    """Create temporary directory with XML-based calculation files."""
    calc_dir = tmp_path / "xml_calc"
    calc_dir.mkdir()

    # Create .save directory structure (standard QE output)
    save_dir = calc_dir / "test.save"
    save_dir.mkdir()

    # Create data-file-schema.xml
    data_file = save_dir / "data-file-schema.xml"
    data_file.write_text(DATA_FILE_SCHEMA_XML)

    # Create SCF input file
    scf_in_file = calc_dir / "scf.in"
    scf_in_file.write_text(SCF_IN)

    # Create SCF output file
    scf_out_file = calc_dir / "scf.out"
    scf_out_file.write_text(SCF_OUT)

    return calc_dir


@pytest.fixture
def xml_parser(xml_calculation_dir: Path) -> QEParser:
    """Create parser instance for XML-based calculation."""
    return QEParser(dirpath=xml_calculation_dir)


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Create empty temporary directory."""
    calc_dir = tmp_path / "empty"
    calc_dir.mkdir()
    return calc_dir


# =============================================================================
# Tests: File Detection
# =============================================================================


def test_parser_detects_scf_input(scf_parser: QEParser) -> None:
    """Test that parser detects SCF input file."""
    assert scf_parser.scf_in is not None


def test_parser_detects_scf_output(scf_parser: QEParser) -> None:
    """Test that parser detects SCF output file."""
    assert scf_parser.scf_out is not None


def test_parser_detects_projwfc_input(dos_parser: QEParser) -> None:
    """Test that parser detects projwfc input file."""
    assert dos_parser.projwfc_in is not None


def test_parser_detects_projwfc_output(dos_parser: QEParser) -> None:
    """Test that parser detects projwfc output file."""
    assert dos_parser.projwfc_out is not None


def test_parser_detects_data_file_schema_xml(xml_parser: QEParser) -> None:
    """Test that parser detects data-file-schema.xml."""
    assert xml_parser.data_file_schema_xml is not None


def test_summary_returns_dict(scf_parser: QEParser) -> None:
    """Test that summary returns a dictionary."""
    summary = scf_parser.summary()
    assert isinstance(summary, dict)


def test_summary_contains_dirpath(scf_parser: QEParser) -> None:
    """Test that summary contains dirpath."""
    summary = scf_parser.summary()
    assert "dirpath" in summary


def test_summary_contains_parsers_status(scf_parser: QEParser) -> None:
    """Test that summary contains parsers status."""
    summary = scf_parser.summary()
    assert "parsers" in summary
    assert summary["parsers"]["scf_in"] is True
    assert summary["parsers"]["scf_out"] is True


# =============================================================================
# Tests: Empty Directory Handling
# =============================================================================


def test_parser_handles_empty_directory(empty_dir: Path) -> None:
    """Test that parser handles empty directory gracefully."""
    parser = QEParser(dirpath=empty_dir)
    assert parser.scf_in is None
    assert parser.scf_out is None


def test_parser_handles_nonexistent_directory(tmp_path: Path) -> None:
    """Test that parser handles nonexistent directory gracefully."""
    nonexistent = tmp_path / "nonexistent"
    parser = QEParser(dirpath=nonexistent)
    # Should not raise, just have empty detections
    summary = parser.summary()
    assert summary["parsers"]["scf_in"] is False
