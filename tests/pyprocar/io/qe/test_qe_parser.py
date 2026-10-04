"""Tests for QEParser (auto-detection and integration)."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.ebs import (
    ElectronicBandStructure,
    ElectronicBandStructureMesh,
    ElectronicBandStructurePath,
)
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo, get_kpoints_from_kgrid
from pyprocar.io import get_parser
from pyprocar.io.qe.parser import QEParser
from tests.utils import DATA_DIR

QE_CODES_DIR = DATA_DIR / "codes" / "qe" / "7.2" / "SrVO3"

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


# =============================================================================
# Tests: Summary
# =============================================================================


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


def test_structure_reads_the_bohr_cartesian_positions_as_fractions_of_the_cell(
    xml_parser: QEParser,
) -> None:
    structure = xml_parser.structure

    assert structure is not None and structure.fractional_coordinates is not None
    np.testing.assert_allclose(structure.fractional_coordinates[1], [2.565 / 5.43] * 3)


def test_structure_is_none_when_lattice_is_missing(tmp_path: Path) -> None:
    parser = QEParser(tmp_path)
    parser.__dict__["species"] = ["Sr", "V", "O", "O", "O"]

    assert parser.structure is None


@pytest.mark.parametrize(
    ("mag", "pdos_columns"),
    [
        ("non-spin-polarized", [2]),
        ("spin-polarized-colinear", [3, 4]),
        ("non-colinear", [2]),
    ],
)
@pytest.mark.data
def test_projected_dos_sums_to_the_pdos_tot_column(
    mag: str, pdos_columns: list[int]
) -> None:
    calc_dir = QE_CODES_DIR / mag / "dos"
    pdos_tot = np.loadtxt(calc_dir / "SrVO3.k.pdos_tot")

    dos = get_parser("qe", calc_dir).dos

    assert dos is not None and dos.projected is not None
    summed = dos.projected.to_array().sum(axis=(2, 3))
    np.testing.assert_allclose(summed, pdos_tot[:, pdos_columns], rtol=1e-2, atol=1e-2)


def test_reciprocal_lattice_has_no_two_pi_and_kpoints_stay_fractional(dos_parser: QEParser) -> None:
    reciprocal_lattice, alat = dos_parser.reciprocal_lattice, dos_parser.alat
    kpoints = dos_parser.kpoints
    assert reciprocal_lattice is not None and alat is not None and kpoints is not None
    assert np.allclose(reciprocal_lattice * alat, np.eye(3))
    assert np.allclose(
        kpoints,
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.25, 0.0],
            [0.5, 0.25, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.25, 0.25],
            [0.5, 0.5, 0.25],
            [0.5, 0.5, 0.5],
        ],
    )


def test_shifted_automatic_grid_is_gamma_centred_with_half_step_shift(tmp_path: Path) -> None:
    (tmp_path / "nscf.in").write_text(SCF_IN.replace("4 4 4 0 0 0", "4 4 4 1 1 1"))
    kgrid_info = QEParser(dirpath=tmp_path).kgrid_info

    assert kgrid_info is not None
    assert kgrid_info == KGridInfo(
        kgrid=(4, 4, 4), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.5, 0.5, 0.5)
    )
    kpoints = get_kpoints_from_kgrid(
        kgrid_info.kgrid, kshift=kgrid_info.kshift, mode=kgrid_info.kgrid_mode
    )
    assert sorted(set(np.round(kpoints[:, 0], 6))) == [-0.375, -0.125, 0.125, 0.375]


TETRAGONAL_FRACTIONAL_KPOINTS = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5],
    [0.25, 0.25, 0.5],
]
TETRAGONAL_CARTESIAN_KPOINTS = [
    "0.0 0.0 0.0",
    "0.5 0.0 0.0",
    "0.0 0.0 0.25",
    "0.25 0.25 0.25",
]

TETRAGONAL_PW_XML = (
    """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <atomic_structure nat="1" alat="10.2608">
      <atomic_positions>
        <atom name="Si" index="1">0.0 0.0 0.0</atom>
      </atomic_positions>
      <cell>
        <a1>10.2608 0.0 0.0</a1>
        <a2>0.0 10.2608 0.0</a2>
        <a3>0.0 0.0 20.5216</a3>
      </cell>
    </atomic_structure>
    <basis_set>
      <reciprocal_lattice>
        <b1>1.0 0.0 0.0</b1>
        <b2>0.0 1.0 0.0</b2>
        <b3>0.0 0.0 0.5</b3>
      </reciprocal_lattice>
    </basis_set>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
    </magnetization>
    <band_structure>
      <nbnd>1</nbnd>
      <nks>4</nks>
"""
    + "".join(
        f"""      <ks_energies>
        <k_point weight="0.25">{k}</k_point>
        <npw>100</npw>
        <eigenvalues size="1">0.1</eigenvalues>
        <occupations size="1">1.0</occupations>
      </ks_energies>
"""
        for k in TETRAGONAL_CARTESIAN_KPOINTS
    )
    + """    </band_structure>
  </output>
</qes:espresso>
"""
)


TETRAGONAL_PROJWFC_FILES = {
    "scf.out": SCF_OUT.replace(
        "b(3) = (  0.000000  0.000000  1.000000 )",
        "b(3) = (  0.000000  0.000000  0.500000 )",
    ),
    "projwfc.out": PROJWFC_OUT.replace("nkstot   =   10", "nkstot   =    4").split(
        "     k ="
    )[0]
    + "".join(f"     k =   {k}\n" for k in TETRAGONAL_CARTESIAN_KPOINTS),
}


@pytest.mark.parametrize(
    "files",
    [
        pytest.param(TETRAGONAL_PROJWFC_FILES, id="projwfc.out"),
        pytest.param({"test.xml": TETRAGONAL_PW_XML}, id="pw.xml"),
        pytest.param(
            {"test.save/data-file-schema.xml": TETRAGONAL_PW_XML},
            id="data-file-schema.xml",
        ),
    ],
)
def test_every_kpoint_source_yields_the_same_fractional_kpoints(
    tmp_path: Path, files: dict[str, str]
) -> None:
    for relative_path, content in files.items():
        (tmp_path / relative_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative_path).write_text(content)
    parser = QEParser(dirpath=tmp_path)

    assert parser.reciprocal_lattice is not None and parser.alat is not None
    assert np.allclose(
        parser.reciprocal_lattice * parser.alat, np.diag([1.0, 1.0, 0.5])
    )
    assert parser.kpoints is not None
    assert np.allclose(parser.kpoints, TETRAGONAL_FRACTIONAL_KPOINTS)


@pytest.mark.data
def test_pw_xml_fallback_matches_atomic_proj_kpoints_on_srvo3(tmp_path: Path) -> None:
    calc_dir = QE_CODES_DIR / "non-spin-polarized" / "dos"
    for relative_path in ["out/SrVO3.xml", "scf.in", "scf.out"]:
        (tmp_path / relative_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative_path).write_bytes((calc_dir / relative_path).read_bytes())

    primary, fallback = QEParser(dirpath=calc_dir), QEParser(dirpath=tmp_path)

    assert primary.atomic_proj_xml is not None and fallback.atomic_proj_xml is None
    assert fallback.pw_xml is not None
    assert primary.kpoints is not None and fallback.kpoints is not None
    assert np.allclose(fallback.kpoints[:2], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0625]])
    assert np.allclose(fallback.kpoints, primary.kpoints)
    fallback_lattice, primary_lattice = fallback.reciprocal_lattice, primary.reciprocal_lattice
    assert fallback_lattice is not None and primary_lattice is not None
    assert np.allclose(fallback_lattice, primary_lattice)


def _tetragonal_bands_dir(
    tmp_path: Path, mode: str, card: str, cartesian_kpoints: list[str]
) -> Path:
    head, _, tail = TETRAGONAL_PW_XML.partition("      <ks_energies>")
    tail = tail.rpartition("      </ks_energies>\n")[2]
    (tmp_path / "test.xml").write_text(
        head.replace("<nks>4</nks>", f"<nks>{len(cartesian_kpoints)}</nks>")
        + "".join(
            f"""      <ks_energies>
        <k_point weight="1.0">{k}</k_point>
        <npw>100</npw>
        <eigenvalues size="1">0.1</eigenvalues>
        <occupations size="1">1.0</occupations>
      </ks_energies>
"""
            for k in cartesian_kpoints
        )
        + tail
    )
    (tmp_path / "bands.in").write_text(
        SCF_IN.replace("'scf'", "'bands'").replace(
            "K_POINTS automatic\n4 4 4 0 0 0\n", f"K_POINTS {mode}\n{card}"
        )
    )
    return tmp_path


TETRAGONAL_GXZ_PATH = [
    *(f"{0.05 * i:.2f} 0.0 0.0" for i in range(10)),
    *(f"{0.5 - 0.05 * i:.2f} 0.0 {0.025 * i:.3f}" for i in range(11)),
]


@pytest.mark.parametrize(
    ("mode", "card"),
    [
        ("crystal_b", "3\n0 0 0 10 !G\n0.5 0 0 10 !X\n0 0 0.5 1 !Z\n"),
        ("tpiba_b", "3\n0 0 0 10 !G\n0.5 0 0 10 !X\n0 0 0.25 1 !Z\n"),
    ],
    ids=["crystal_b", "tpiba_b"],
)
def test_band_path_vertices_in_crystal_or_tpiba_units_give_the_same_kpath(
    tmp_path: Path, mode: str, card: str
) -> None:
    kpath = QEParser(_tetragonal_bands_dir(tmp_path, mode, card, TETRAGONAL_GXZ_PATH)).kpath

    assert kpath is not None
    assert kpath.segment_names == [("Γ", "X"), ("X", "Z")]
    kpoints = np.asarray(kpath.kpoints)
    assert len(kpoints) == 22
    assert np.allclose(
        kpoints[[0, 10, 11, 21]], [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0, 0, 0.5]]
    )


@pytest.mark.parametrize(
    ("mode", "card"),
    [
        ("crystal_c", "3\n0 0 0 1\n0.5 0 0 2\n0 0 0.5 2\n"),
        ("tpiba_c", "3\n0 0 0 1\n0.5 0 0 2\n0 0 0.25 2\n"),
    ],
    ids=["crystal_c", "tpiba_c"],
)
def test_contour_mode_has_no_kpath_and_keeps_the_mesh_kpoints(
    tmp_path: Path, mode: str, card: str
) -> None:
    mesh = ["0 0 0", "0.25 0 0", "0 0 0.125", "0.25 0 0.125"]
    parser = QEParser(_tetragonal_bands_dir(tmp_path, mode, card, mesh))

    assert parser.kpath is None
    assert parser.kpoints is not None
    assert np.allclose(parser.kpoints, [[0, 0, 0], [0.25, 0, 0], [0, 0, 0.25], [0.25, 0, 0.25]])


@pytest.mark.parametrize(
    ("mode", "card"),
    [
        ("crystal", "4\n0 0 0 1\n0.5 0 0 1\n0 0 0.5 1\n0.25 0.25 0.5 1\n"),
        ("tpiba", "4\n0 0 0 1\n0.5 0 0 1\n0 0 0.25 1\n0.25 0.25 0.25 1\n"),
        ("automatic", "4 4 2 0 0 0\n"),
    ],
    ids=["crystal", "tpiba", "automatic"],
)
def test_bands_run_without_a_band_path_is_a_plain_ebs_of_the_computed_kpoints(
    tmp_path: Path, mode: str, card: str
) -> None:
    calc_dir = _tetragonal_bands_dir(tmp_path, mode, card, TETRAGONAL_CARTESIAN_KPOINTS)
    xml = (calc_dir / "test.xml").read_text()
    (calc_dir / "test.xml").write_text(
        xml.replace(
            "<nbnd>1</nbnd>",
            "<nbnd>1</nbnd>\n      <fermi_energy>0.2</fermi_energy>\n"
            + '      <starting_k_points><monkhorst_pack nk1="4" nk2="4" nk3="2"'
            + ' k1="0" k2="0" k3="0"/></starting_k_points>',
        )
    )
    parser = QEParser(calc_dir)
    ebs = parser.ebs

    assert parser.kpath is None and parser.kgrid_info is None
    assert type(ebs) is ElectronicBandStructure
    assert np.allclose(ebs.kpoints, TETRAGONAL_FRACTIONAL_KPOINTS)


def test_mixed_directory_follows_the_run_that_wrote_the_xml(tmp_path: Path) -> None:
    card = "3\n0 0 0 10 !G\n0.5 0 0 10 !X\n0 0 0.5 1 !Z\n"
    calc_dir = _tetragonal_bands_dir(tmp_path, "crystal_b", card, TETRAGONAL_CARTESIAN_KPOINTS)
    xml = (calc_dir / "test.xml").read_text()
    (calc_dir / "test.xml").write_text(
        xml.replace(
            "  <output>",
            "  <input><control_variables><calculation>nscf</calculation>"
            + "</control_variables></input>\n  <output>",
        ).replace(
            "<nbnd>1</nbnd>",
            '<nbnd>1</nbnd>\n      <starting_k_points><monkhorst_pack nk1="4" nk2="4" nk3="2"'
            + ' k1="0" k2="0" k3="0"/></starting_k_points>',
        )
    )
    parser = QEParser(calc_dir)

    assert parser.kpath is None
    assert parser.kgrid_info == KGridInfo(
        kgrid=(4, 4, 2), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
    )


@pytest.mark.data
def test_dos_directory_with_a_bands_input_still_gives_the_full_mesh(tmp_path: Path) -> None:
    dos_dir = QE_CODES_DIR / "non-spin-polarized" / "dos"
    for path in dos_dir.rglob("*"):
        if path.is_file() and "pdos_" not in path.name and path.suffix != ".pkl":
            target = tmp_path / path.relative_to(dos_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    bands_in = QE_CODES_DIR / "non-spin-polarized" / "bands" / "bands.in"
    (tmp_path / "bands.in").write_bytes(bands_in.read_bytes())
    ebs = QEParser(tmp_path).ebs

    assert type(ebs) is ElectronicBandStructureMesh
    assert ebs.kpoints.shape == (4096, 3)


def test_projwfc_kpoints_from_a_bands_run_win_over_a_later_nscf_xml(tmp_path: Path) -> None:
    for relative_path, content in TETRAGONAL_PROJWFC_FILES.items():
        (tmp_path / relative_path).write_text(content)
    head, _, tail = TETRAGONAL_PW_XML.partition("      <ks_energies>")
    (tmp_path / "test.xml").write_text(
        head.replace(
            "  <output>",
            "  <input><control_variables><calculation>nscf</calculation>"
            + "</control_variables></input>\n  <output>",
        ).replace("<nks>4</nks>", "<nks>1</nks>")
        + "      <ks_energies>"
        + tail.split("      <ks_energies>")[0]
        + tail.rpartition("      </ks_energies>\n")[2]
    )
    (tmp_path / "bands.in").write_text(
        SCF_IN.replace("'scf'", "'bands'").replace(
            "K_POINTS automatic\n4 4 4 0 0 0\n",
            "K_POINTS crystal_b\n3\n0 0 0 1 !G\n0.5 0 0 1 !X\n0 0 0.5 1 !Z\n",
        )
    )
    parser = QEParser(tmp_path)

    assert parser.pw_xml is not None and parser.pw_xml.kpoints is not None
    assert len(parser.pw_xml.kpoints) == 1
    assert parser.is_bands_run
    assert parser.kgrid_info is None


@pytest.mark.data
def test_bands_projwfc_then_nscf_in_one_directory_still_plots_the_bands(tmp_path: Path) -> None:
    bands_dir = QE_CODES_DIR / "non-spin-polarized" / "bands"
    dos_dir = QE_CODES_DIR / "non-spin-polarized" / "dos"
    for path in bands_dir.rglob("*"):
        if path.is_file() and "pdos_" not in path.name and path.suffix != ".pkl":
            target = tmp_path / path.relative_to(bands_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    for name in ("out/SrVO3.xml", "nscf.in", "nscf.out"):
        (tmp_path / name).write_bytes((dos_dir / name).read_bytes())
    ebs = QEParser(tmp_path).ebs

    assert type(ebs) is ElectronicBandStructurePath
    assert ebs.kpoints.shape == (155, 3)


def test_an_nscf_xml_with_as_many_but_different_kpoints_does_not_win(tmp_path: Path) -> None:
    for relative_path, content in TETRAGONAL_PROJWFC_FILES.items():
        (tmp_path / relative_path).write_text(content)
    shifted = [
        " ".join(f"{float(x) + 0.125:.3f}" for x in k.split()) for k in TETRAGONAL_CARTESIAN_KPOINTS
    ]
    xml = TETRAGONAL_PW_XML.replace(
        "  <output>",
        "  <input><control_variables><calculation>nscf</calculation>"
        + "</control_variables></input>\n  <output>",
    )
    for original, moved in zip(TETRAGONAL_CARTESIAN_KPOINTS, shifted, strict=True):
        xml = xml.replace(f">{original}</k_point>", f">{moved}</k_point>")
    (tmp_path / "test.xml").write_text(xml)
    (tmp_path / "bands.in").write_text(
        SCF_IN.replace("'scf'", "'bands'").replace(
            "K_POINTS automatic\n4 4 4 0 0 0\n",
            "K_POINTS crystal_b\n3\n0 0 0 1 !G\n0.5 0 0 1 !X\n0 0 0.5 1 !Z\n",
        )
    )
    parser = QEParser(tmp_path)

    assert parser.pw_xml is not None and parser.pw_xml.kpoints is not None
    assert len(parser.pw_xml.kpoints) == 4
    assert parser.is_bands_run
    assert parser.kgrid_info is None


@pytest.mark.data
@pytest.mark.parametrize(
    ("mag", "gamma_band_0"),
    [
        ("non-spin-polarized", [-53.32013]),
        ("spin-polarized-colinear", [-53.3077, -53.3059]),
        ("non-colinear", None),
    ],
)
def test_projwfc_out_bands_match_atomic_proj_bands_in_ev(
    tmp_path: Path, mag: str, gamma_band_0: list[float] | None
) -> None:
    bands_dir = QE_CODES_DIR / mag / "bands"
    for path in bands_dir.rglob("*"):
        skip = "pdos_" in path.name or path.suffix == ".pkl" or path.name == "atomic_proj.xml"
        if path.is_file() and not skip:
            target = tmp_path / path.relative_to(bands_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    primary, fallback = QEParser(bands_dir), QEParser(tmp_path)

    assert fallback.atomic_proj_xml is None and fallback.projwfc_out is not None
    assert primary.bands is not None and fallback.bands is not None
    if gamma_band_0 is not None:
        assert fallback.bands[0, 0, :] == pytest.approx(gamma_band_0, abs=1e-4)
    np.testing.assert_allclose(fallback.bands, primary.bands, atol=1e-4)
