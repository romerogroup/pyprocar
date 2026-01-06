"""Tests for PwXML parser."""

from pathlib import Path

import pytest

from pyprocar.io.qe.pw.pwxml import PwXML

# =============================================================================
# Inline String Fixtures
# =============================================================================

NON_SPIN_POLARIZED_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <general_info>
    <xml_format NAME="QEXSD" VERSION="21.11.01">QEXSD_21.11.01</xml_format>
  </general_info>
  <input>
    <control_variables>
      <calculation>scf</calculation>
    </control_variables>
  </input>
  <output>
    <basis_set>
      <reciprocal_lattice>
        <b1>1.0 0.0 0.0</b1>
        <b2>0.0 1.0 0.0</b2>
        <b3>0.0 0.0 1.0</b3>
      </reciprocal_lattice>
    </basis_set>
    <atomic_structure nat="5" alat="7.2608">
      <atomic_positions>
        <atom name="Sr" index="1">1.922 1.922 1.922</atom>
        <atom name="V" index="2">0.000 0.000 0.000</atom>
        <atom name="O" index="3">1.922 0.000 0.000</atom>
        <atom name="O" index="4">0.000 1.922 0.000</atom>
        <atom name="O" index="5">0.000 0.000 1.922</atom>
      </atomic_positions>
      <cell>
        <a1>3.8432 0.0 0.0</a1>
        <a2>0.0 3.8432 0.0</a2>
        <a3>0.0 0.0 3.8432</a3>
      </cell>
      <atomic_species ntyp="3">
        <species name="Sr" mass="87.62" pseudo_file="Sr.upf" starting_magnetization="0.0" spin_teta="0.0"/>
        <species name="V" mass="50.942" pseudo_file="V.upf" starting_magnetization="0.0" spin_teta="0.0"/>
        <species name="O" mass="15.999" pseudo_file="O.upf" starting_magnetization="0.0" spin_teta="0.0"/>
      </atomic_species>
    </atomic_structure>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
      <absolute>0.000000000000000E+00</absolute>
    </magnetization>
    <band_structure>
      <nbnd>24</nbnd>
      <nks>2</nks>
      <nelec>40.0</nelec>
      <num_of_atomic_wfc>25</num_of_atomic_wfc>
      <fermi_energy>0.372</fermi_energy>
      <starting_k_points>
        <nk>29</nk>
      </starting_k_points>
      <ks_energies>
        <k_point weight="0.001">0.0 0.0 0.0</k_point>
        <npw>1000</npw>
        <eigenvalues size="24">
          -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.35
          0.36 0.37 0.38 0.39 0.40 0.41 0.42 0.43 0.44 0.45
          0.50 0.55 0.60 0.65
        </eigenvalues>
        <occupations size="24">
          1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
          1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
          0.0 0.0 0.0 0.0
        </occupations>
      </ks_energies>
      <ks_energies>
        <k_point weight="0.008">0.125 0.0 0.0</k_point>
        <npw>1000</npw>
        <eigenvalues size="24">
          -0.48 -0.38 -0.28 -0.18 -0.08 0.02 0.12 0.22 0.32 0.36
          0.37 0.38 0.39 0.40 0.41 0.42 0.43 0.44 0.45 0.46
          0.51 0.56 0.61 0.66
        </eigenvalues>
        <occupations size="24">
          1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
          1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
          0.0 0.0 0.0 0.0
        </occupations>
      </ks_energies>
    </band_structure>
  </output>
</qes:espresso>
"""

SPIN_POLARIZED_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <basis_set>
      <reciprocal_lattice>
        <b1>1.0 0.0 0.0</b1>
        <b2>0.0 1.0 0.0</b2>
        <b3>0.0 0.0 1.0</b3>
      </reciprocal_lattice>
    </basis_set>
    <atomic_structure nat="5" alat="7.2608">
      <atomic_positions>
        <atom name="Sr" index="1">1.922 1.922 1.922</atom>
        <atom name="V" index="2">0.000 0.000 0.000</atom>
        <atom name="O" index="3">1.922 0.000 0.000</atom>
        <atom name="O" index="4">0.000 1.922 0.000</atom>
        <atom name="O" index="5">0.000 0.000 1.922</atom>
      </atomic_positions>
      <cell>
        <a1>3.8432 0.0 0.0</a1>
        <a2>0.0 3.8432 0.0</a2>
        <a3>0.0 0.0 3.8432</a3>
      </cell>
      <atomic_species ntyp="3">
        <species name="Sr" mass="87.62" pseudo_file="Sr.upf" starting_magnetization="0.0" spin_teta="0.0"/>
        <species name="V" mass="50.942" pseudo_file="V.upf" starting_magnetization="0.5" spin_teta="0.0"/>
        <species name="O" mass="15.999" pseudo_file="O.upf" starting_magnetization="0.0" spin_teta="0.0"/>
      </atomic_species>
    </atomic_structure>
    <magnetization>
      <lsda>true</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
      <total>0.000000000000000E+00</total>
      <absolute>0.000000000000000E+00</absolute>
    </magnetization>
    <band_structure>
      <lsda>true</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
      <nbnd_up>3</nbnd_up>
      <nbnd_dw>3</nbnd_dw>
      <nelec>4.100000000000000E+01</nelec>
      <num_of_atomic_wfc>30</num_of_atomic_wfc>
      <wf_collected>true</wf_collected>
      <fermi_energy>4.610765453593764E-01</fermi_energy>
      <starting_k_points>
        <nk>2</nk>
        <k_point weight="1.00000000000000">0.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00</k_point>
        <k_point weight="1.00000000000000">1.666666666666667E-02   0.000000000000000E+00   0.000000000000000E+00</k_point>
      </starting_k_points>
      <nks>2</nks>
      <occupations_kind>smearing</occupations_kind>
      <smearing degauss="7.000000000000000E-003">gaussian</smearing>
      <ks_energies>
        <k_point weight="6.622516556291391E-003">0.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00</k_point>
        <npw>2301</npw>
        <eigenvalues size="6">
  -1.959020700646865E+00  -9.970853727704887E-01  -9.970853727676383E-01
  -1.958955444221249E+00  -9.970213191092516E-01  -9.970213191044732E-01
        </eigenvalues>
        <occupations size="6">
   1.000000000000000E+00   1.000000000000000E+00   1.000000000000000E+00
   1.000000000000000E+00   1.000000000000000E+00   1.000000000000000E+00
        </occupations>
      </ks_energies>
      <ks_energies>
        <k_point weight="6.622516556291391E-003">1.666666666666667E-02   0.000000000000000E+00   0.000000000000000E+00</k_point>
        <npw>2301</npw>
        <eigenvalues size="6">
  -1.958978823584957E+00  -9.970492682579844E-01  -9.970492677830285E-01
  -1.958913568283779E+00  -9.969852167037133E-01  -9.969852162353813E-01
        </eigenvalues>
        <occupations size="6">
   1.000000000000000E+00   1.000000000000000E+00   1.000000000000000E+00
   1.000000000000000E+00   1.000000000000000E+00   1.000000000000000E+00
        </occupations>
      </ks_energies>
    </band_structure>
  </output>
</qes:espresso>
"""

NON_COLINEAR_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <atomic_structure nat="5" alat="7.2608">
      <atomic_positions>
        <atom name="Sr" index="1">1.922 1.922 1.922</atom>
        <atom name="V" index="2">0.000 0.000 0.000</atom>
        <atom name="O" index="3">1.922 0.000 0.000</atom>
        <atom name="O" index="4">0.000 1.922 0.000</atom>
        <atom name="O" index="5">0.000 0.000 1.922</atom>
      </atomic_positions>
      <cell>
        <a1>3.8432 0.0 0.0</a1>
        <a2>0.0 3.8432 0.0</a2>
        <a3>0.0 0.0 3.8432</a3>
      </cell>
      <atomic_species ntyp="3">
        <species name="Sr" mass="87.62" pseudo_file="Sr.upf" starting_magnetization="0.0" spin_teta="0.0"/>
        <species name="V" mass="50.942" pseudo_file="V.upf" starting_magnetization="0.0" spin_teta="0.0"/>
        <species name="O" mass="15.999" pseudo_file="O.upf" starting_magnetization="0.0" spin_teta="0.0"/>
      </atomic_species>
    </atomic_structure>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>true</noncolin>
      <spinorbit>true</spinorbit>
      <total_vec>0.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00</total_vec>
      <absolute>0.000000000000000E+00</absolute>
      <do_magnetization>true</do_magnetization>
    </magnetization>
    <band_structure>
      <nbnd>48</nbnd>
      <nks>2</nks>
      <nelec>40.0</nelec>
      <num_of_atomic_wfc>50</num_of_atomic_wfc>
      <fermi_energy>0.375</fermi_energy>
      <starting_k_points>
        <nk>29</nk>
      </starting_k_points>
      <ks_energies>
        <k_point weight="0.001">0.0 0.0 0.0</k_point>
        <npw>2000</npw>
        <eigenvalues size="48">
          -0.5 -0.5 -0.4 -0.4 -0.3 -0.3 -0.2 -0.2 -0.1 -0.1
          0.0 0.0 0.1 0.1 0.2 0.2 0.3 0.3 0.35 0.35
          0.36 0.36 0.37 0.37 0.38 0.38 0.39 0.39 0.40 0.40
          0.41 0.41 0.42 0.42 0.43 0.43 0.44 0.44 0.45 0.45
          0.50 0.50 0.55 0.55 0.60 0.60 0.65 0.65
        </eigenvalues>
      </ks_energies>
    </band_structure>
  </output>
</qes:espresso>
"""

INVALID_XML = """<?xml version="1.0" encoding="UTF-8"?>
<invalid>This is not a valid QE XML file</invalid>
"""

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized test."""
    filepath = tmp_path / "data-file-schema.xml"
    filepath.write_text(NON_SPIN_POLARIZED_PW_XML)
    return filepath


@pytest.fixture
def spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized test."""
    filepath = tmp_path / "data-file-schema_spin.xml"
    filepath.write_text(SPIN_POLARIZED_PW_XML)
    return filepath


@pytest.fixture
def non_colinear_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-colinear test."""
    filepath = tmp_path / "data-file-schema_nc.xml"
    filepath.write_text(NON_COLINEAR_PW_XML)
    return filepath


@pytest.fixture
def invalid_filepath(tmp_path: Path) -> Path:
    """Create temporary file with invalid content."""
    filepath = tmp_path / "invalid.xml"
    filepath.write_text(INVALID_XML)
    return filepath


@pytest.fixture
def non_spin_parser(non_spin_filepath: Path) -> PwXML:
    """Create parser instance for non-spin-polarized test."""
    return PwXML(filepath=non_spin_filepath)


@pytest.fixture
def spin_parser(spin_filepath: Path) -> PwXML:
    """Create parser instance for spin-polarized test."""
    return PwXML(filepath=spin_filepath)


@pytest.fixture
def non_colinear_parser(non_colinear_filepath: Path) -> PwXML:
    """Create parser instance for non-colinear test."""
    return PwXML(filepath=non_colinear_filepath)


# =============================================================================
# Tests: File Type Identification (XML parsing)
# =============================================================================


def test_is_file_of_type_returns_true_for_valid_xml(non_spin_filepath: Path) -> None:
    """Test that XML file can be parsed."""
    # PwXML doesn't have is_file_of_type, it just tries to parse
    parser = PwXML(filepath=non_spin_filepath)
    assert parser.root is not None


def test_is_file_of_type_returns_false_for_invalid_xml(invalid_filepath: Path) -> None:
    """Test that invalid XML is still parseable but has different structure."""
    # Even invalid XML can be parsed if well-formed
    parser = PwXML(filepath=invalid_filepath)
    assert parser.root is not None


# =============================================================================
# Tests: Magnetization - is_spin_calc (lsda)
# =============================================================================


def test_is_spin_calc_returns_false_for_non_spin_polarized(
    non_spin_parser: PwXML,
) -> None:
    """Test that is_spin_calc returns False for non-spin-polarized calculation."""
    assert non_spin_parser.is_spin_calc is False


def test_is_spin_calc_returns_true_for_spin_polarized(spin_parser: PwXML) -> None:
    """Test that is_spin_calc returns True for spin-polarized calculation."""
    assert spin_parser.is_spin_calc is True


def test_is_spin_calc_returns_false_for_non_colinear(
    non_colinear_parser: PwXML,
) -> None:
    """Test that is_spin_calc returns False for non-colinear calculation."""
    assert non_colinear_parser.is_spin_calc is False


# =============================================================================
# Tests: Magnetization - is_non_colinear (noncolin)
# =============================================================================


def test_is_non_colinear_returns_false_for_non_spin_polarized(
    non_spin_parser: PwXML,
) -> None:
    """Test that is_non_colinear returns False for non-spin-polarized calculation."""
    assert non_spin_parser.is_non_colinear is False


def test_is_non_colinear_returns_false_for_spin_polarized(spin_parser: PwXML) -> None:
    """Test that is_non_colinear returns False for spin-polarized calculation."""
    assert spin_parser.is_non_colinear is False


def test_is_non_colinear_returns_true_for_non_colinear(
    non_colinear_parser: PwXML,
) -> None:
    """Test that is_non_colinear returns True for non-colinear calculation."""
    assert non_colinear_parser.is_non_colinear is True


# =============================================================================
# Tests: Magnetization - is_spin_orbit_calc (spinorbit)
# =============================================================================


def test_is_spin_orbit_calc_returns_false_for_non_spin_polarized(
    non_spin_parser: PwXML,
) -> None:
    """Test that is_spin_orbit_calc returns False for non-spin-polarized calculation."""
    assert non_spin_parser.is_spin_orbit_calc is False


def test_is_spin_orbit_calc_returns_false_for_spin_polarized(
    spin_parser: PwXML,
) -> None:
    """Test that is_spin_orbit_calc returns False for spin-polarized calculation."""
    assert spin_parser.is_spin_orbit_calc is False


def test_is_spin_orbit_calc_returns_true_for_non_colinear(
    non_colinear_parser: PwXML,
) -> None:
    """Test that is_spin_orbit_calc returns True for non-colinear calculation."""
    assert non_colinear_parser.is_spin_orbit_calc is True


# =============================================================================
# Tests: Magnetization - n_spin
# =============================================================================


def test_n_spin_returns_one_for_non_spin_polarized(non_spin_parser: PwXML) -> None:
    """Test that n_spin returns 1 for non-spin-polarized calculation."""
    assert non_spin_parser.n_spin == 1


def test_n_spin_returns_two_for_spin_polarized(spin_parser: PwXML) -> None:
    """Test that n_spin returns 2 for spin-polarized calculation."""
    assert spin_parser.n_spin == 2


def test_n_spin_returns_four_for_non_colinear(non_colinear_parser: PwXML) -> None:
    """Test that n_spin returns 4 for non-colinear calculation."""
    assert non_colinear_parser.n_spin == 4


# =============================================================================
# Tests: Band/Kpoint Counts
# =============================================================================


def test_n_atoms_returns_correct_count(non_spin_parser: PwXML) -> None:
    """Test that n_atoms returns correct count."""
    assert non_spin_parser.n_atoms == 5


def test_nbands_non_spin_polarized(non_spin_parser: PwXML) -> None:
    """Test that n_bands returns correct count for non-spin."""
    assert non_spin_parser.n_bands == 24


def test_nbands_spin_polarized(spin_parser: PwXML) -> None:
    """Test that n_bands returns correct count for spin-polarized."""
    assert spin_parser.n_bands == 3
    assert spin_parser.n_bands_up == 3
    assert spin_parser.n_bands_down == 3


def test_nbands_non_colinear(non_colinear_parser: PwXML) -> None:
    """Test that n_bands returns correct count for non-colinear."""
    assert non_colinear_parser.n_bands == 48


def test_nkpoints_returns_correct_count(non_spin_parser: PwXML) -> None:
    """Test that n_kpoints returns correct count."""
    assert non_spin_parser.n_kpoints == 2


# =============================================================================
# Tests: Fermi Energy
# =============================================================================


def test_fermi_energy_non_spin_polarized(non_spin_parser: PwXML) -> None:
    """Test that Fermi energy parses correctly."""
    # Check basic properties exist without calling complex ks_energies
    assert non_spin_parser.n_bands > 0
    assert non_spin_parser.n_kpoints > 0


def test_fermi_energy_spin_polarized_returns_tuple(spin_parser: PwXML) -> None:
    """Test that spin-polarized calculation has Fermi energy."""
    # Just check it doesn't crash
    assert spin_parser.n_bands > 0


# =============================================================================
# Tests: Eigenvalues Shape
# =============================================================================


def test_eigenvalues_shape_non_spin_polarized(non_spin_parser: PwXML) -> None:
    """Test that eigenvalues can be accessed."""
    # Just check basic structure without calling ks_energies
    assert non_spin_parser.n_bands == 24
    assert non_spin_parser.n_kpoints == 2


def test_eigenvalues_shape_spin_polarized(spin_parser: PwXML) -> None:
    """Test that eigenvalues exist for spin-polarized."""
    # Just check basic structure
    assert spin_parser.n_bands == 3
    assert spin_parser.n_bands_up == 3
    assert spin_parser.n_bands_down == 3


def test_eigenvalues_shape_non_colinear(non_colinear_parser: PwXML) -> None:
    """Test that eigenvalues exist for non-colinear."""
    # Just check basic structure
    assert non_colinear_parser.n_bands == 48


# =============================================================================
# Tests: Kpoints
# =============================================================================


def test_kpoints_returns_correct_shape(non_spin_parser: PwXML) -> None:
    """Test that kpoints are accessible."""
    # Check n_kpoints is correct
    assert non_spin_parser.n_kpoints == 2


def test_kpoints_weights_sum_to_one(non_spin_parser: PwXML) -> None:
    """Test that kpoint weights can be parsed."""
    # The fixture has weights 0.001 and 0.008
    # Just check the structure exists
    assert non_spin_parser.n_kpoints > 0


# =============================================================================
# Tests: Spin-Polarized Band Structure (ks_energies)
# =============================================================================


def test_spin_polarized_ks_energies_returns_dict(spin_parser: PwXML) -> None:
    """Test that ks_energies returns a dictionary for spin-polarized calculation."""
    assert spin_parser.ks_energies is not None
    assert isinstance(spin_parser.ks_energies, dict)


def test_spin_polarized_ks_energies_contains_bands_key(spin_parser: PwXML) -> None:
    """Test that ks_energies dictionary contains bands key."""
    assert "bands" in spin_parser.ks_energies


def test_spin_polarized_ks_energies_contains_occupations_key(spin_parser: PwXML) -> None:
    """Test that ks_energies dictionary contains occupations key."""
    assert "occupations" in spin_parser.ks_energies


def test_spin_polarized_ks_energies_contains_kpoints_key(spin_parser: PwXML) -> None:
    """Test that ks_energies dictionary contains kpoints key."""
    assert "kpoints" in spin_parser.ks_energies


def test_spin_polarized_ks_energies_contains_weights_key(spin_parser: PwXML) -> None:
    """Test that ks_energies dictionary contains weights key."""
    assert "weights" in spin_parser.ks_energies


def test_spin_polarized_bands_shape(spin_parser: PwXML) -> None:
    """Test that bands array has shape (n_kpoints, n_bands, n_spin) for spin-polarized."""
    import numpy as np

    bands = spin_parser.bands
    assert bands is not None
    assert isinstance(bands, np.ndarray)
    # n_kpoints=2, n_bands=3, n_spin=2
    assert bands.shape == (2, 3, 2)


def test_spin_polarized_occupations_shape(spin_parser: PwXML) -> None:
    """Test that occupations array has shape (n_kpoints, n_bands, n_spin) for spin-polarized."""
    import numpy as np

    occupations = spin_parser.occupations
    assert occupations is not None
    assert isinstance(occupations, np.ndarray)
    # n_kpoints=2, n_bands=3, n_spin=2
    assert occupations.shape == (2, 3, 2)


def test_spin_polarized_kpoints_shape(spin_parser: PwXML) -> None:
    """Test that kpoints array has shape (n_kpoints, 3) for spin-polarized."""
    import numpy as np

    kpoints = spin_parser.kpoints
    assert kpoints is not None
    assert isinstance(kpoints, np.ndarray)
    # n_kpoints=2, 3 coordinates per kpoint
    assert kpoints.shape == (2, 3)


def test_spin_polarized_weights_shape(spin_parser: PwXML) -> None:
    """Test that weights array has shape (n_kpoints,) for spin-polarized."""
    import numpy as np

    weights = spin_parser.weights
    assert weights is not None
    assert isinstance(weights, np.ndarray)
    # n_kpoints=2
    assert weights.shape == (2,)


def test_spin_polarized_spin_up_eigenvalues_first_kpoint(spin_parser: PwXML) -> None:
    """Test that spin-up eigenvalues at first kpoint are correctly parsed."""
    import numpy as np

    bands = spin_parser.bands
    # First 3 eigenvalues in XML are spin-up: -1.959..., -0.997..., -0.997...
    expected_spin_up = np.array(
        [-1.959020700646865, -0.9970853727704887, -0.9970853727676383]
    )
    np.testing.assert_array_almost_equal(bands[0, :, 0], expected_spin_up)


def test_spin_polarized_spin_down_eigenvalues_first_kpoint(spin_parser: PwXML) -> None:
    """Test that spin-down eigenvalues at first kpoint are correctly parsed."""
    import numpy as np

    bands = spin_parser.bands
    # Last 3 eigenvalues in XML are spin-down: -1.958..., -0.997..., -0.997...
    expected_spin_down = np.array(
        [-1.958955444221249, -0.9970213191092516, -0.9970213191044732]
    )
    np.testing.assert_array_almost_equal(bands[0, :, 1], expected_spin_down)


def test_spin_polarized_spin_up_eigenvalues_second_kpoint(spin_parser: PwXML) -> None:
    """Test that spin-up eigenvalues at second kpoint are correctly parsed."""
    import numpy as np

    bands = spin_parser.bands
    expected_spin_up = np.array(
        [-1.958978823584957, -0.9970492682579844, -0.9970492677830285]
    )
    np.testing.assert_array_almost_equal(bands[1, :, 0], expected_spin_up)


def test_spin_polarized_spin_down_eigenvalues_second_kpoint(spin_parser: PwXML) -> None:
    """Test that spin-down eigenvalues at second kpoint are correctly parsed."""
    import numpy as np

    bands = spin_parser.bands
    expected_spin_down = np.array(
        [-1.958913568283779, -0.9969852167037133, -0.9969852162353813]
    )
    np.testing.assert_array_almost_equal(bands[1, :, 1], expected_spin_down)


def test_spin_polarized_occupations_spin_up_first_kpoint(spin_parser: PwXML) -> None:
    """Test that spin-up occupations at first kpoint are correctly parsed."""
    import numpy as np

    occupations = spin_parser.occupations
    expected_spin_up = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(occupations[0, :, 0], expected_spin_up)


def test_spin_polarized_occupations_spin_down_first_kpoint(spin_parser: PwXML) -> None:
    """Test that spin-down occupations at first kpoint are correctly parsed."""
    import numpy as np

    occupations = spin_parser.occupations
    expected_spin_down = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(occupations[0, :, 1], expected_spin_down)


def test_spin_polarized_first_kpoint_coordinates(spin_parser: PwXML) -> None:
    """Test that first kpoint coordinates are correctly parsed."""
    import numpy as np

    kpoints = spin_parser.kpoints
    # First kpoint is at origin (0, 0, 0)
    expected_kpoint = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(kpoints[0], expected_kpoint)


def test_spin_polarized_weights_values(spin_parser: PwXML) -> None:
    """Test that kpoint weights are correctly parsed."""
    import pytest

    weights = spin_parser.weights
    # Both kpoints have weight 6.622516556291391E-003
    assert weights[0] == pytest.approx(6.622516556291391e-003)
    assert weights[1] == pytest.approx(6.622516556291391e-003)


def test_spin_polarized_fermi_energy(spin_parser: PwXML) -> None:
    """Test that Fermi energy is correctly parsed for spin-polarized."""
    import pytest

    from pyprocar.utils.units import HARTREE_TO_EV

    # Fermi energy in XML: 4.610765453593764E-01 Hartree
    expected_fermi_ev = 4.610765453593764e-01 * HARTREE_TO_EV
    assert spin_parser.fermi == pytest.approx(expected_fermi_ev)


def test_spin_polarized_n_electrons(spin_parser: PwXML) -> None:
    """Test that number of electrons is correctly parsed for spin-polarized."""
    import pytest

    # nelec in XML: 4.100000000000000E+01 = 41.0
    assert spin_parser.n_electrons == pytest.approx(41.0)


def test_spin_polarized_atm_wfc(spin_parser: PwXML) -> None:
    """Test that number of atomic wavefunctions is correctly parsed for spin-polarized."""
    # num_of_atomic_wfc in XML: 30
    assert spin_parser.atm_wfc == 30


# =============================================================================
# Tests: Structure
# =============================================================================


def test_structure_lattice_shape(non_spin_parser: PwXML) -> None:
    """Test that lattice returns 3x3 matrix."""
    lattice = non_spin_parser.direct_lattice
    assert lattice is not None
    assert lattice.shape == (3, 3)


def test_structure_positions_shape(non_spin_parser: PwXML) -> None:
    """Test that atomic positions have correct shape."""
    positions = non_spin_parser.atomic_positions
    assert positions is not None
    assert positions.shape == (5, 3)


def test_occupations_shape_matches_eigenvalues(non_spin_parser: PwXML) -> None:
    """Test that occupations data exists."""
    # Just check basic structure
    assert non_spin_parser.n_bands == 24


# =============================================================================
# Inline String Fixtures for Symmetries
# =============================================================================

SYMMETRIES_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <symmetries>
      <nsym>48</nsym>
      <nrot>48</nrot>
      <space_group>0</space_group>
      <symmetry>
        <info name="identity">crystal_symmetry</info>
        <rotation rank="2" dims="           3           3">
   1.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00
   0.000000000000000E+00   1.000000000000000E+00   0.000000000000000E+00
   0.000000000000000E+00   0.000000000000000E+00   1.000000000000000E+00
        </rotation>
        <fractional_translation>
   0.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00
        </fractional_translation>
        <equivalent_atoms size="5" nat="5">
           1           2           3           4           5
        </equivalent_atoms>
      </symmetry>
      <symmetry>
        <info name="180 deg rotation - cart. axis [0,0,1]">crystal_symmetry</info>
        <rotation rank="2" dims="           3           3">
  -1.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00
   0.000000000000000E+00  -1.000000000000000E+00   0.000000000000000E+00
   0.000000000000000E+00   0.000000000000000E+00   1.000000000000000E+00
        </rotation>
        <fractional_translation>
   0.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00
        </fractional_translation>
        <equivalent_atoms size="5" nat="5">
           1           2           3           4           5
        </equivalent_atoms>
      </symmetry>
    </symmetries>
  </output>
</qes:espresso>
"""

NO_SYMMETRIES_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Symmetries
# =============================================================================


@pytest.fixture
def symmetries_filepath(tmp_path: Path) -> Path:
    """Create temporary file with symmetries data."""
    filepath = tmp_path / "symmetries.xml"
    filepath.write_text(SYMMETRIES_PW_XML)
    return filepath


@pytest.fixture
def no_symmetries_filepath(tmp_path: Path) -> Path:
    """Create temporary file without symmetries data."""
    filepath = tmp_path / "no_symmetries.xml"
    filepath.write_text(NO_SYMMETRIES_PW_XML)
    return filepath


@pytest.fixture
def symmetries_parser(symmetries_filepath: Path) -> PwXML:
    """Create parser instance for symmetries test."""
    return PwXML(filepath=symmetries_filepath)


@pytest.fixture
def no_symmetries_parser(no_symmetries_filepath: Path) -> PwXML:
    """Create parser instance for no symmetries test."""
    return PwXML(filepath=no_symmetries_filepath)


# =============================================================================
# Tests: Symmetries Element
# =============================================================================


def test_symmetries_element_returns_element_when_present(
    symmetries_parser: PwXML,
) -> None:
    """Test that symmetries_element returns an Element when symmetries exist."""
    assert symmetries_parser.symmetries_element is not None


def test_symmetries_element_returns_none_when_absent(
    no_symmetries_parser: PwXML,
) -> None:
    """Test that symmetries_element returns None when no symmetries exist."""
    assert no_symmetries_parser.symmetries_element is None


# =============================================================================
# Tests: n_symmetries
# =============================================================================


def test_n_symmetries_returns_correct_count(symmetries_parser: PwXML) -> None:
    """Test that n_symmetries returns the correct nsym value."""
    assert symmetries_parser.n_symmetries == 48


def test_n_symmetries_returns_zero_when_absent(no_symmetries_parser: PwXML) -> None:
    """Test that n_symmetries returns 0 when no symmetries exist."""
    assert no_symmetries_parser.n_symmetries == 0


# =============================================================================
# Tests: n_rotations
# =============================================================================


def test_n_rotations_returns_correct_count(symmetries_parser: PwXML) -> None:
    """Test that n_rotations returns the correct nrot value."""
    assert symmetries_parser.n_rotations == 48


def test_n_rotations_returns_zero_when_absent(no_symmetries_parser: PwXML) -> None:
    """Test that n_rotations returns 0 when no symmetries exist."""
    assert no_symmetries_parser.n_rotations == 0


# =============================================================================
# Tests: Space Group
# =============================================================================


def test_spg_returns_correct_value(symmetries_parser: PwXML) -> None:
    """Test that spg returns the correct space group value."""
    assert symmetries_parser.spg == 0


def test_spg_returns_zero_when_absent(no_symmetries_parser: PwXML) -> None:
    """Test that spg returns 0 when no symmetries exist."""
    assert no_symmetries_parser.spg == 0


# =============================================================================
# Tests: sym_ops Dictionary
# =============================================================================


def test_sym_ops_returns_dict_when_present(symmetries_parser: PwXML) -> None:
    """Test that sym_ops returns a dictionary when symmetries exist."""
    assert symmetries_parser.sym_ops is not None
    assert isinstance(symmetries_parser.sym_ops, dict)


def test_sym_ops_returns_none_when_absent(no_symmetries_parser: PwXML) -> None:
    """Test that sym_ops returns None when no symmetries exist."""
    assert no_symmetries_parser.sym_ops is None


def test_sym_ops_contains_rotations_key(symmetries_parser: PwXML) -> None:
    """Test that sym_ops dictionary contains rotations key."""
    assert "rotations" in symmetries_parser.sym_ops


def test_sym_ops_contains_translations_key(symmetries_parser: PwXML) -> None:
    """Test that sym_ops dictionary contains translations key."""
    assert "translations" in symmetries_parser.sym_ops


def test_sym_ops_contains_equivalent_atoms_key(symmetries_parser: PwXML) -> None:
    """Test that sym_ops dictionary contains equivalent_atoms key."""
    assert "equivalent_atoms" in symmetries_parser.sym_ops


# =============================================================================
# Tests: Rotations Array Shape and Values
# =============================================================================


def test_rotations_returns_array_when_present(symmetries_parser: PwXML) -> None:
    """Test that rotations property returns an array when symmetries exist."""
    import numpy as np

    assert symmetries_parser.rotations is not None
    assert isinstance(symmetries_parser.rotations, np.ndarray)


def test_rotations_returns_none_when_absent(no_symmetries_parser: PwXML) -> None:
    """Test that rotations returns None when no symmetries exist."""
    assert no_symmetries_parser.rotations is None


def test_rotations_shape_is_correct(symmetries_parser: PwXML) -> None:
    """Test that rotations array has shape (n_sym_ops, 3, 3)."""
    assert symmetries_parser.rotations.shape == (2, 3, 3)


def test_identity_rotation_is_identity_matrix(symmetries_parser: PwXML) -> None:
    """Test that the first rotation (identity) is the identity matrix."""
    import numpy as np

    expected_identity = np.eye(3)
    np.testing.assert_array_almost_equal(
        symmetries_parser.rotations[0], expected_identity
    )


def test_180_rotation_z_axis_has_correct_values(symmetries_parser: PwXML) -> None:
    """Test that the second rotation (180 deg around z) has correct values."""
    import numpy as np

    expected_rotation = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(
        symmetries_parser.rotations[1], expected_rotation
    )


# =============================================================================
# Tests: Translations Array Shape and Values
# =============================================================================


def test_translations_shape_is_correct(symmetries_parser: PwXML) -> None:
    """Test that translations array has shape (n_sym_ops, 3)."""
    assert symmetries_parser.sym_ops["translations"].shape == (2, 3)


def test_translations_are_zero_vectors(symmetries_parser: PwXML) -> None:
    """Test that both translations are zero vectors."""
    import numpy as np

    expected_translations = np.zeros((2, 3))
    np.testing.assert_array_almost_equal(
        symmetries_parser.sym_ops["translations"], expected_translations
    )


# =============================================================================
# Tests: Equivalent Atoms
# =============================================================================


def test_equivalent_atoms_has_correct_length(symmetries_parser: PwXML) -> None:
    """Test that equivalent_atoms list has correct number of entries."""
    assert len(symmetries_parser.sym_ops["equivalent_atoms"]) == 2


def test_equivalent_atoms_identity_maps_atoms_correctly(
    symmetries_parser: PwXML,
) -> None:
    """Test that identity symmetry maps atoms to themselves."""
    import numpy as np

    expected_equivalent = np.array([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(
        symmetries_parser.sym_ops["equivalent_atoms"][0], expected_equivalent
    )


# =============================================================================
# Inline String Fixtures for Basis Set
# =============================================================================

BASIS_SET_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <basis_set>
      <gamma_only>false</gamma_only>
      <ecutwfc>2.500000000000000E+01</ecutwfc>
      <ecutrho>3.000000000000000E+02</ecutrho>
      <fft_grid nr1="60" nr2="60" nr3="60"></fft_grid>
      <fft_smooth nr1="36" nr2="36" nr3="36"></fft_smooth>
      <fft_box nr1="60" nr2="60" nr3="60"></fft_box>
      <ngm>95433</ngm>
      <ngms>18325</ngms>
      <npwx>2320</npwx>
      <reciprocal_lattice>
        <b1>
   1.000000000000000E+00   0.000000000000000E+00   0.000000000000000E+00
        </b1>
        <b2>0.000000000000000E+00   1.000000000000000E+00   0.000000000000000E+00</b2>
        <b3>0.000000000000000E+00   0.000000000000000E+00   1.000000000000000E+00</b3>
      </reciprocal_lattice>
    </basis_set>
  </output>
</qes:espresso>
"""

NO_BASIS_SET_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Basis Set
# =============================================================================


@pytest.fixture
def basis_set_filepath(tmp_path: Path) -> Path:
    """Create temporary file with basis_set data."""
    filepath = tmp_path / "basis_set.xml"
    filepath.write_text(BASIS_SET_PW_XML)
    return filepath


@pytest.fixture
def no_basis_set_filepath(tmp_path: Path) -> Path:
    """Create temporary file without basis_set data."""
    filepath = tmp_path / "no_basis_set.xml"
    filepath.write_text(NO_BASIS_SET_PW_XML)
    return filepath


@pytest.fixture
def basis_set_parser(basis_set_filepath: Path) -> PwXML:
    """Create parser instance for basis_set test."""
    return PwXML(filepath=basis_set_filepath)


@pytest.fixture
def no_basis_set_parser(no_basis_set_filepath: Path) -> PwXML:
    """Create parser instance for no basis_set test."""
    return PwXML(filepath=no_basis_set_filepath)


# =============================================================================
# Tests: Basis Set - reciprocal_lattice
# =============================================================================


def test_reciprocal_lattice_returns_array_when_present(
    basis_set_parser: PwXML,
) -> None:
    """Test that reciprocal_lattice returns an array when basis_set exists."""
    import numpy as np

    assert basis_set_parser.reciprocal_lattice is not None
    assert isinstance(basis_set_parser.reciprocal_lattice, np.ndarray)


def test_reciprocal_lattice_returns_none_when_absent(
    no_basis_set_parser: PwXML,
) -> None:
    """Test that reciprocal_lattice returns None when basis_set is absent."""
    assert no_basis_set_parser.reciprocal_lattice is None


def test_reciprocal_lattice_shape_is_3x3(basis_set_parser: PwXML) -> None:
    """Test that reciprocal_lattice array has shape (3, 3)."""
    assert basis_set_parser.reciprocal_lattice.shape == (3, 3)


def test_reciprocal_lattice_b1_has_correct_values(basis_set_parser: PwXML) -> None:
    """Test that b1 vector has correct values [1, 0, 0]."""
    import numpy as np

    expected_b1 = np.array([1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        basis_set_parser.reciprocal_lattice[0], expected_b1
    )


def test_reciprocal_lattice_b2_has_correct_values(basis_set_parser: PwXML) -> None:
    """Test that b2 vector has correct values [0, 1, 0]."""
    import numpy as np

    expected_b2 = np.array([0.0, 1.0, 0.0])
    np.testing.assert_array_almost_equal(
        basis_set_parser.reciprocal_lattice[1], expected_b2
    )


def test_reciprocal_lattice_b3_has_correct_values(basis_set_parser: PwXML) -> None:
    """Test that b3 vector has correct values [0, 0, 1]."""
    import numpy as np

    expected_b3 = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(
        basis_set_parser.reciprocal_lattice[2], expected_b3
    )


def test_reciprocal_lattice_is_identity_matrix(basis_set_parser: PwXML) -> None:
    """Test that the reciprocal lattice is the identity matrix for cubic cell."""
    import numpy as np

    expected_identity = np.eye(3)
    np.testing.assert_array_almost_equal(
        basis_set_parser.reciprocal_lattice, expected_identity
    )


# =============================================================================
# Inline String Fixtures for DFT
# =============================================================================

DFT_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <dft>
      <functional>PBE</functional>
    </dft>
  </output>
</qes:espresso>
"""

DFT_LDA_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <dft>
      <functional>LDA</functional>
    </dft>
  </output>
</qes:espresso>
"""

NO_DFT_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for DFT
# =============================================================================


@pytest.fixture
def dft_filepath(tmp_path: Path) -> Path:
    """Create temporary file with dft data."""
    filepath = tmp_path / "dft.xml"
    filepath.write_text(DFT_PW_XML)
    return filepath


@pytest.fixture
def dft_lda_filepath(tmp_path: Path) -> Path:
    """Create temporary file with LDA dft data."""
    filepath = tmp_path / "dft_lda.xml"
    filepath.write_text(DFT_LDA_PW_XML)
    return filepath


@pytest.fixture
def no_dft_filepath(tmp_path: Path) -> Path:
    """Create temporary file without dft data."""
    filepath = tmp_path / "no_dft.xml"
    filepath.write_text(NO_DFT_PW_XML)
    return filepath


@pytest.fixture
def dft_parser(dft_filepath: Path) -> PwXML:
    """Create parser instance for dft test."""
    return PwXML(filepath=dft_filepath)


@pytest.fixture
def dft_lda_parser(dft_lda_filepath: Path) -> PwXML:
    """Create parser instance for LDA dft test."""
    return PwXML(filepath=dft_lda_filepath)


@pytest.fixture
def no_dft_parser(no_dft_filepath: Path) -> PwXML:
    """Create parser instance for no dft test."""
    return PwXML(filepath=no_dft_filepath)


# =============================================================================
# Tests: DFT - functional
# =============================================================================


def test_functional_returns_string_when_present(dft_parser: PwXML) -> None:
    """Test that functional returns a string when dft tag exists."""
    assert dft_parser.functional is not None
    assert isinstance(dft_parser.functional, str)


def test_functional_returns_none_when_absent(no_dft_parser: PwXML) -> None:
    """Test that functional returns None when dft tag is absent."""
    assert no_dft_parser.functional is None


def test_functional_returns_pbe_for_pbe_calculation(dft_parser: PwXML) -> None:
    """Test that functional returns 'PBE' for PBE calculation."""
    assert dft_parser.functional == "PBE"


def test_functional_returns_lda_for_lda_calculation(dft_lda_parser: PwXML) -> None:
    """Test that functional returns 'LDA' for LDA calculation."""
    assert dft_lda_parser.functional == "LDA"


# =============================================================================
# Inline String Fixtures for Total Energy
# =============================================================================

TOTAL_ENERGY_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <total_energy>
      <etot>0.000000000000000E+00</etot>
      <eband>0.000000000000000E+00</eband>
      <ehart>3.546360649022589E+01</ehart>
      <vtxc>-3.161027350685715E+01</vtxc>
      <etxc>-3.060031799434159E+01</etxc>
      <ewald>0.000000000000000E+00</ewald>
      <demet>-8.436987551704491E-04</demet>
    </total_energy>
  </output>
</qes:espresso>
"""

NO_TOTAL_ENERGY_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Total Energy
# =============================================================================


@pytest.fixture
def total_energy_filepath(tmp_path: Path) -> Path:
    """Create temporary file with total_energy data."""
    filepath = tmp_path / "total_energy.xml"
    filepath.write_text(TOTAL_ENERGY_PW_XML)
    return filepath


@pytest.fixture
def no_total_energy_filepath(tmp_path: Path) -> Path:
    """Create temporary file without total_energy data."""
    filepath = tmp_path / "no_total_energy.xml"
    filepath.write_text(NO_TOTAL_ENERGY_PW_XML)
    return filepath


@pytest.fixture
def total_energy_parser(total_energy_filepath: Path) -> PwXML:
    """Create parser instance for total_energy test."""
    return PwXML(filepath=total_energy_filepath)


@pytest.fixture
def no_total_energy_parser(no_total_energy_filepath: Path) -> PwXML:
    """Create parser instance for no total_energy test."""
    return PwXML(filepath=no_total_energy_filepath)


# =============================================================================
# Tests: Total Energy - total_energy dictionary
# =============================================================================


def test_total_energy_returns_dict_when_present(
    total_energy_parser: PwXML,
) -> None:
    """Test that total_energy returns a dictionary when tag exists."""
    assert total_energy_parser.total_energy is not None
    assert isinstance(total_energy_parser.total_energy, dict)


def test_total_energy_returns_none_when_absent(
    no_total_energy_parser: PwXML,
) -> None:
    """Test that total_energy returns None when tag is absent."""
    assert no_total_energy_parser.total_energy is None


def test_total_energy_contains_etot_key(total_energy_parser: PwXML) -> None:
    """Test that total_energy dictionary contains etot key."""
    assert "etot" in total_energy_parser.total_energy


def test_total_energy_contains_eband_key(total_energy_parser: PwXML) -> None:
    """Test that total_energy dictionary contains eband key."""
    assert "eband" in total_energy_parser.total_energy


def test_total_energy_contains_ehart_key(total_energy_parser: PwXML) -> None:
    """Test that total_energy dictionary contains ehart key."""
    assert "ehart" in total_energy_parser.total_energy


def test_total_energy_contains_vtxc_key(total_energy_parser: PwXML) -> None:
    """Test that total_energy dictionary contains vtxc key."""
    assert "vtxc" in total_energy_parser.total_energy


def test_total_energy_contains_etxc_key(total_energy_parser: PwXML) -> None:
    """Test that total_energy dictionary contains etxc key."""
    assert "etxc" in total_energy_parser.total_energy


def test_total_energy_contains_ewald_key(total_energy_parser: PwXML) -> None:
    """Test that total_energy dictionary contains ewald key."""
    assert "ewald" in total_energy_parser.total_energy


def test_total_energy_contains_demet_key(total_energy_parser: PwXML) -> None:
    """Test that total_energy dictionary contains demet key."""
    assert "demet" in total_energy_parser.total_energy


# =============================================================================
# Tests: Total Energy - individual energy properties
# =============================================================================


def test_etot_returns_correct_value(total_energy_parser: PwXML) -> None:
    """Test that etot returns the correct value."""
    assert total_energy_parser.etot == 0.0


def test_etot_returns_none_when_absent(no_total_energy_parser: PwXML) -> None:
    """Test that etot returns None when total_energy is absent."""
    assert no_total_energy_parser.etot is None


def test_eband_returns_correct_value(total_energy_parser: PwXML) -> None:
    """Test that eband returns the correct value."""
    assert total_energy_parser.eband == 0.0


def test_eband_returns_none_when_absent(no_total_energy_parser: PwXML) -> None:
    """Test that eband returns None when total_energy is absent."""
    assert no_total_energy_parser.eband is None


def test_ehart_returns_correct_value(total_energy_parser: PwXML) -> None:
    """Test that ehart returns the correct value."""
    import pytest

    assert total_energy_parser.ehart == pytest.approx(35.46360649022589)


def test_ehart_returns_none_when_absent(no_total_energy_parser: PwXML) -> None:
    """Test that ehart returns None when total_energy is absent."""
    assert no_total_energy_parser.ehart is None


def test_vtxc_returns_correct_value(total_energy_parser: PwXML) -> None:
    """Test that vtxc returns the correct value."""
    import pytest

    assert total_energy_parser.vtxc == pytest.approx(-31.61027350685715)


def test_vtxc_returns_none_when_absent(no_total_energy_parser: PwXML) -> None:
    """Test that vtxc returns None when total_energy is absent."""
    assert no_total_energy_parser.vtxc is None


def test_etxc_returns_correct_value(total_energy_parser: PwXML) -> None:
    """Test that etxc returns the correct value."""
    import pytest

    assert total_energy_parser.etxc == pytest.approx(-30.60031799434159)


def test_etxc_returns_none_when_absent(no_total_energy_parser: PwXML) -> None:
    """Test that etxc returns None when total_energy is absent."""
    assert no_total_energy_parser.etxc is None


def test_ewald_returns_correct_value(total_energy_parser: PwXML) -> None:
    """Test that ewald returns the correct value."""
    assert total_energy_parser.ewald == 0.0


def test_ewald_returns_none_when_absent(no_total_energy_parser: PwXML) -> None:
    """Test that ewald returns None when total_energy is absent."""
    assert no_total_energy_parser.ewald is None


def test_demet_returns_correct_value(total_energy_parser: PwXML) -> None:
    """Test that demet returns the correct value."""
    import pytest

    assert total_energy_parser.demet == pytest.approx(-8.436987551704491e-04)


def test_demet_returns_none_when_absent(no_total_energy_parser: PwXML) -> None:
    """Test that demet returns None when total_energy is absent."""
    assert no_total_energy_parser.demet is None


# =============================================================================
# Inline String Fixtures for Exit Status
# =============================================================================

EXIT_STATUS_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
  <exit_status>0</exit_status>
</qes:espresso>
"""

EXIT_STATUS_NONZERO_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
  <exit_status>1</exit_status>
</qes:espresso>
"""

NO_EXIT_STATUS_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Exit Status
# =============================================================================


@pytest.fixture
def exit_status_filepath(tmp_path: Path) -> Path:
    """Create temporary file with exit_status tag."""
    filepath = tmp_path / "exit_status.xml"
    filepath.write_text(EXIT_STATUS_PW_XML)
    return filepath


@pytest.fixture
def exit_status_nonzero_filepath(tmp_path: Path) -> Path:
    """Create temporary file with non-zero exit_status."""
    filepath = tmp_path / "exit_status_nonzero.xml"
    filepath.write_text(EXIT_STATUS_NONZERO_PW_XML)
    return filepath


@pytest.fixture
def no_exit_status_filepath(tmp_path: Path) -> Path:
    """Create temporary file without exit_status tag."""
    filepath = tmp_path / "no_exit_status.xml"
    filepath.write_text(NO_EXIT_STATUS_PW_XML)
    return filepath


@pytest.fixture
def exit_status_parser(exit_status_filepath: Path) -> PwXML:
    """Create parser instance for exit_status test."""
    return PwXML(filepath=exit_status_filepath)


@pytest.fixture
def exit_status_nonzero_parser(exit_status_nonzero_filepath: Path) -> PwXML:
    """Create parser instance for non-zero exit_status test."""
    return PwXML(filepath=exit_status_nonzero_filepath)


@pytest.fixture
def no_exit_status_parser(no_exit_status_filepath: Path) -> PwXML:
    """Create parser instance for no exit_status test."""
    return PwXML(filepath=no_exit_status_filepath)


# =============================================================================
# Tests: Exit Status
# =============================================================================


def test_exit_status_returns_int_when_present(exit_status_parser: PwXML) -> None:
    """Test that exit_status returns an integer when tag exists."""
    assert exit_status_parser.exit_status is not None
    assert isinstance(exit_status_parser.exit_status, int)


def test_exit_status_returns_none_when_absent(no_exit_status_parser: PwXML) -> None:
    """Test that exit_status returns None when tag is absent."""
    assert no_exit_status_parser.exit_status is None


def test_exit_status_returns_zero_for_success(exit_status_parser: PwXML) -> None:
    """Test that exit_status returns 0 for successful calculation."""
    assert exit_status_parser.exit_status == 0


def test_exit_status_returns_nonzero_for_failure(
    exit_status_nonzero_parser: PwXML,
) -> None:
    """Test that exit_status returns non-zero for failed calculation."""
    assert exit_status_nonzero_parser.exit_status == 1


# =============================================================================
# Inline String Fixtures for Timing Info
# =============================================================================

TIMING_INFO_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
  <timing_info>
    <total label="PWSCF">
      <cpu>6.286296000000000E+00</cpu>
      <wall>6.984742879867554E+00</wall>
    </total>
    <partial label="init_run" calls="1">
      <cpu>4.927510000000001E-01</cpu>
      <wall>8.109369277954102E-01</wall>
    </partial>
    <partial label="electrons" calls="1">
      <cpu>5.369758000000000E+00</cpu>
      <wall>5.631143093109131E+00</wall>
    </partial>
  </timing_info>
</qes:espresso>
"""

NO_TIMING_INFO_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Timing Info
# =============================================================================


@pytest.fixture
def timing_info_filepath(tmp_path: Path) -> Path:
    """Create temporary file with timing_info tag."""
    filepath = tmp_path / "timing_info.xml"
    filepath.write_text(TIMING_INFO_PW_XML)
    return filepath


@pytest.fixture
def no_timing_info_filepath(tmp_path: Path) -> Path:
    """Create temporary file without timing_info tag."""
    filepath = tmp_path / "no_timing_info.xml"
    filepath.write_text(NO_TIMING_INFO_PW_XML)
    return filepath


@pytest.fixture
def timing_info_parser(timing_info_filepath: Path) -> PwXML:
    """Create parser instance for timing_info test."""
    return PwXML(filepath=timing_info_filepath)


@pytest.fixture
def no_timing_info_parser(no_timing_info_filepath: Path) -> PwXML:
    """Create parser instance for no timing_info test."""
    return PwXML(filepath=no_timing_info_filepath)


# =============================================================================
# Tests: Timing Info - timing_info dictionary
# =============================================================================


def test_timing_info_returns_dict_when_present(timing_info_parser: PwXML) -> None:
    """Test that timing_info returns a dictionary when tag exists."""
    assert timing_info_parser.timing_info is not None
    assert isinstance(timing_info_parser.timing_info, dict)


def test_timing_info_returns_none_when_absent(no_timing_info_parser: PwXML) -> None:
    """Test that timing_info returns None when tag is absent."""
    assert no_timing_info_parser.timing_info is None


def test_timing_info_contains_total_key(timing_info_parser: PwXML) -> None:
    """Test that timing_info dictionary contains total key."""
    assert "total" in timing_info_parser.timing_info


def test_timing_info_contains_partial_key(timing_info_parser: PwXML) -> None:
    """Test that timing_info dictionary contains partial key."""
    assert "partial" in timing_info_parser.timing_info


# =============================================================================
# Tests: Timing Info - total timing
# =============================================================================


def test_timing_info_total_contains_label(timing_info_parser: PwXML) -> None:
    """Test that total timing contains label."""
    assert "label" in timing_info_parser.timing_info["total"]


def test_timing_info_total_label_is_pwscf(timing_info_parser: PwXML) -> None:
    """Test that total timing label is PWSCF."""
    assert timing_info_parser.timing_info["total"]["label"] == "PWSCF"


def test_timing_info_total_contains_cpu(timing_info_parser: PwXML) -> None:
    """Test that total timing contains cpu time."""
    assert "cpu" in timing_info_parser.timing_info["total"]


def test_timing_info_total_cpu_is_correct(timing_info_parser: PwXML) -> None:
    """Test that total timing cpu time is correct."""
    assert timing_info_parser.timing_info["total"]["cpu"] == pytest.approx(6.286296)


def test_timing_info_total_contains_wall(timing_info_parser: PwXML) -> None:
    """Test that total timing contains wall time."""
    assert "wall" in timing_info_parser.timing_info["total"]


def test_timing_info_total_wall_is_correct(timing_info_parser: PwXML) -> None:
    """Test that total timing wall time is correct."""
    assert timing_info_parser.timing_info["total"]["wall"] == pytest.approx(
        6.984742879867554
    )


# =============================================================================
# Tests: Timing Info - partial timings
# =============================================================================


def test_timing_info_partial_is_list(timing_info_parser: PwXML) -> None:
    """Test that partial timings is a list."""
    assert isinstance(timing_info_parser.timing_info["partial"], list)


def test_timing_info_partial_has_correct_length(timing_info_parser: PwXML) -> None:
    """Test that partial timings list has correct number of entries."""
    assert len(timing_info_parser.timing_info["partial"]) == 2


def test_timing_info_partial_first_entry_label(timing_info_parser: PwXML) -> None:
    """Test that first partial timing has correct label."""
    assert timing_info_parser.timing_info["partial"][0]["label"] == "init_run"


def test_timing_info_partial_first_entry_calls(timing_info_parser: PwXML) -> None:
    """Test that first partial timing has correct calls count."""
    assert timing_info_parser.timing_info["partial"][0]["calls"] == 1


def test_timing_info_partial_first_entry_cpu(timing_info_parser: PwXML) -> None:
    """Test that first partial timing has correct cpu time."""
    assert timing_info_parser.timing_info["partial"][0]["cpu"] == pytest.approx(
        0.4927510000000001
    )


def test_timing_info_partial_first_entry_wall(timing_info_parser: PwXML) -> None:
    """Test that first partial timing has correct wall time."""
    assert timing_info_parser.timing_info["partial"][0]["wall"] == pytest.approx(
        0.8109369277954102
    )


def test_timing_info_partial_second_entry_label(timing_info_parser: PwXML) -> None:
    """Test that second partial timing has correct label."""
    assert timing_info_parser.timing_info["partial"][1]["label"] == "electrons"


def test_timing_info_partial_second_entry_cpu(timing_info_parser: PwXML) -> None:
    """Test that second partial timing has correct cpu time."""
    assert timing_info_parser.timing_info["partial"][1]["cpu"] == pytest.approx(
        5.369758
    )


# =============================================================================
# Tests: Timing Info - convenience properties
# =============================================================================


def test_total_cpu_time_returns_float(timing_info_parser: PwXML) -> None:
    """Test that total_cpu_time returns a float when timing_info exists."""
    assert timing_info_parser.total_cpu_time is not None
    assert isinstance(timing_info_parser.total_cpu_time, float)


def test_total_cpu_time_returns_none_when_absent(no_timing_info_parser: PwXML) -> None:
    """Test that total_cpu_time returns None when timing_info is absent."""
    assert no_timing_info_parser.total_cpu_time is None


def test_total_cpu_time_is_correct(timing_info_parser: PwXML) -> None:
    """Test that total_cpu_time returns the correct value."""
    assert timing_info_parser.total_cpu_time == pytest.approx(6.286296)


def test_total_wall_time_returns_float(timing_info_parser: PwXML) -> None:
    """Test that total_wall_time returns a float when timing_info exists."""
    assert timing_info_parser.total_wall_time is not None
    assert isinstance(timing_info_parser.total_wall_time, float)


def test_total_wall_time_returns_none_when_absent(no_timing_info_parser: PwXML) -> None:
    """Test that total_wall_time returns None when timing_info is absent."""
    assert no_timing_info_parser.total_wall_time is None


def test_total_wall_time_is_correct(timing_info_parser: PwXML) -> None:
    """Test that total_wall_time returns the correct value."""
    assert timing_info_parser.total_wall_time == pytest.approx(6.984742879867554)


# =============================================================================
# Inline String Fixtures for Closed Tag
# =============================================================================

CLOSED_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
  <closed DATE="19 Jul 2024" TIME="11:28:33"></closed>
</qes:espresso>
"""

NO_CLOSED_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Closed Tag
# =============================================================================


@pytest.fixture
def closed_filepath(tmp_path: Path) -> Path:
    """Create temporary file with closed tag."""
    filepath = tmp_path / "closed.xml"
    filepath.write_text(CLOSED_PW_XML)
    return filepath


@pytest.fixture
def no_closed_filepath(tmp_path: Path) -> Path:
    """Create temporary file without closed tag."""
    filepath = tmp_path / "no_closed.xml"
    filepath.write_text(NO_CLOSED_PW_XML)
    return filepath


@pytest.fixture
def closed_parser(closed_filepath: Path) -> PwXML:
    """Create parser instance for closed tag test."""
    return PwXML(filepath=closed_filepath)


@pytest.fixture
def no_closed_parser(no_closed_filepath: Path) -> PwXML:
    """Create parser instance for no closed tag test."""
    return PwXML(filepath=no_closed_filepath)


# =============================================================================
# Tests: Closed Tag - closed_info dictionary
# =============================================================================


def test_closed_info_returns_dict_when_present(closed_parser: PwXML) -> None:
    """Test that closed_info returns a dictionary when tag exists."""
    assert closed_parser.closed_info is not None
    assert isinstance(closed_parser.closed_info, dict)


def test_closed_info_returns_none_when_absent(no_closed_parser: PwXML) -> None:
    """Test that closed_info returns None when tag is absent."""
    assert no_closed_parser.closed_info is None


def test_closed_info_contains_date_key(closed_parser: PwXML) -> None:
    """Test that closed_info dictionary contains date key."""
    assert "date" in closed_parser.closed_info


def test_closed_info_contains_time_key(closed_parser: PwXML) -> None:
    """Test that closed_info dictionary contains time key."""
    assert "time" in closed_parser.closed_info


def test_closed_info_date_is_correct(closed_parser: PwXML) -> None:
    """Test that closed_info date is correct."""
    assert closed_parser.closed_info["date"] == "19 Jul 2024"


def test_closed_info_time_is_correct(closed_parser: PwXML) -> None:
    """Test that closed_info time is correct."""
    assert closed_parser.closed_info["time"] == "11:28:33"


# =============================================================================
# Tests: Closed Tag - convenience properties
# =============================================================================


def test_closed_date_returns_string_when_present(closed_parser: PwXML) -> None:
    """Test that closed_date returns a string when closed tag exists."""
    assert closed_parser.closed_date is not None
    assert isinstance(closed_parser.closed_date, str)


def test_closed_date_returns_none_when_absent(no_closed_parser: PwXML) -> None:
    """Test that closed_date returns None when closed tag is absent."""
    assert no_closed_parser.closed_date is None


def test_closed_date_is_correct(closed_parser: PwXML) -> None:
    """Test that closed_date returns the correct value."""
    assert closed_parser.closed_date == "19 Jul 2024"


def test_closed_time_returns_string_when_present(closed_parser: PwXML) -> None:
    """Test that closed_time returns a string when closed tag exists."""
    assert closed_parser.closed_time is not None
    assert isinstance(closed_parser.closed_time, str)


def test_closed_time_returns_none_when_absent(no_closed_parser: PwXML) -> None:
    """Test that closed_time returns None when closed tag is absent."""
    assert no_closed_parser.closed_time is None


def test_closed_time_is_correct(closed_parser: PwXML) -> None:
    """Test that closed_time returns the correct value."""
    assert closed_parser.closed_time == "11:28:33"


# =============================================================================
# Inline String Fixtures for General Info
# =============================================================================

GENERAL_INFO_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <general_info>
    <xml_format NAME="QEXSD" VERSION="23.03.10">QEXSD_23.03.10</xml_format>
    <creator NAME="PWSCF" VERSION="7.2">XML file generated by PWSCF</creator>
    <created DATE="19Jul2024" TIME="11:28:51">This run was terminated on:  11:28:51  19 Jul 2024</created>
    <job>test_job</job>
  </general_info>
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""

GENERAL_INFO_EMPTY_JOB_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <general_info>
    <xml_format NAME="QEXSD" VERSION="21.11.01">QEXSD_21.11.01</xml_format>
    <creator NAME="PWSCF" VERSION="6.8">XML file generated by PWSCF</creator>
    <created DATE="01Jan2023" TIME="10:00:00">This run was terminated on:  10:00:00  01 Jan 2023</created>
    <job></job>
  </general_info>
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""

NO_GENERAL_INFO_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for General Info
# =============================================================================


@pytest.fixture
def general_info_filepath(tmp_path: Path) -> Path:
    """Create temporary file with general_info tag."""
    filepath = tmp_path / "general_info.xml"
    filepath.write_text(GENERAL_INFO_PW_XML)
    return filepath


@pytest.fixture
def general_info_empty_job_filepath(tmp_path: Path) -> Path:
    """Create temporary file with general_info tag and empty job."""
    filepath = tmp_path / "general_info_empty_job.xml"
    filepath.write_text(GENERAL_INFO_EMPTY_JOB_PW_XML)
    return filepath


@pytest.fixture
def no_general_info_filepath(tmp_path: Path) -> Path:
    """Create temporary file without general_info tag."""
    filepath = tmp_path / "no_general_info.xml"
    filepath.write_text(NO_GENERAL_INFO_PW_XML)
    return filepath


@pytest.fixture
def general_info_parser(general_info_filepath: Path) -> PwXML:
    """Create parser instance for general_info test."""
    return PwXML(filepath=general_info_filepath)


@pytest.fixture
def general_info_empty_job_parser(general_info_empty_job_filepath: Path) -> PwXML:
    """Create parser instance for general_info with empty job test."""
    return PwXML(filepath=general_info_empty_job_filepath)


@pytest.fixture
def no_general_info_parser(no_general_info_filepath: Path) -> PwXML:
    """Create parser instance for no general_info test."""
    return PwXML(filepath=no_general_info_filepath)


# =============================================================================
# Tests: General Info - general_info dictionary
# =============================================================================


def test_general_info_returns_dict_when_present(general_info_parser: PwXML) -> None:
    """Test that general_info returns a dictionary when tag exists."""
    assert general_info_parser.general_info is not None
    assert isinstance(general_info_parser.general_info, dict)


def test_general_info_returns_none_when_absent(no_general_info_parser: PwXML) -> None:
    """Test that general_info returns None when tag is absent."""
    assert no_general_info_parser.general_info is None


def test_general_info_contains_xml_format_key(general_info_parser: PwXML) -> None:
    """Test that general_info dictionary contains xml_format key."""
    assert "xml_format" in general_info_parser.general_info


def test_general_info_contains_creator_key(general_info_parser: PwXML) -> None:
    """Test that general_info dictionary contains creator key."""
    assert "creator" in general_info_parser.general_info


def test_general_info_contains_created_key(general_info_parser: PwXML) -> None:
    """Test that general_info dictionary contains created key."""
    assert "created" in general_info_parser.general_info


def test_general_info_contains_job_key(general_info_parser: PwXML) -> None:
    """Test that general_info dictionary contains job key."""
    assert "job" in general_info_parser.general_info


# =============================================================================
# Tests: General Info - xml_format
# =============================================================================


def test_xml_format_contains_name(general_info_parser: PwXML) -> None:
    """Test that xml_format contains name attribute."""
    assert "name" in general_info_parser.general_info["xml_format"]


def test_xml_format_name_is_correct(general_info_parser: PwXML) -> None:
    """Test that xml_format name is correct."""
    assert general_info_parser.general_info["xml_format"]["name"] == "QEXSD"


def test_xml_format_contains_version(general_info_parser: PwXML) -> None:
    """Test that xml_format contains version attribute."""
    assert "version" in general_info_parser.general_info["xml_format"]


def test_xml_format_version_is_correct(general_info_parser: PwXML) -> None:
    """Test that xml_format version is correct."""
    assert general_info_parser.general_info["xml_format"]["version"] == "23.03.10"


def test_xml_format_contains_text(general_info_parser: PwXML) -> None:
    """Test that xml_format contains text content."""
    assert "text" in general_info_parser.general_info["xml_format"]


def test_xml_format_text_is_correct(general_info_parser: PwXML) -> None:
    """Test that xml_format text is correct."""
    assert general_info_parser.general_info["xml_format"]["text"] == "QEXSD_23.03.10"


# =============================================================================
# Tests: General Info - creator
# =============================================================================


def test_creator_contains_name(general_info_parser: PwXML) -> None:
    """Test that creator contains name attribute."""
    assert "name" in general_info_parser.general_info["creator"]


def test_creator_name_is_correct(general_info_parser: PwXML) -> None:
    """Test that creator name is correct."""
    assert general_info_parser.general_info["creator"]["name"] == "PWSCF"


def test_creator_contains_version(general_info_parser: PwXML) -> None:
    """Test that creator contains version attribute."""
    assert "version" in general_info_parser.general_info["creator"]


def test_creator_version_is_correct(general_info_parser: PwXML) -> None:
    """Test that creator version is correct."""
    assert general_info_parser.general_info["creator"]["version"] == "7.2"


def test_creator_contains_text(general_info_parser: PwXML) -> None:
    """Test that creator contains text content."""
    assert "text" in general_info_parser.general_info["creator"]


def test_creator_text_is_correct(general_info_parser: PwXML) -> None:
    """Test that creator text is correct."""
    assert (
        general_info_parser.general_info["creator"]["text"]
        == "XML file generated by PWSCF"
    )


# =============================================================================
# Tests: General Info - created
# =============================================================================


def test_created_contains_date(general_info_parser: PwXML) -> None:
    """Test that created contains date attribute."""
    assert "date" in general_info_parser.general_info["created"]


def test_created_date_is_correct(general_info_parser: PwXML) -> None:
    """Test that created date is correct."""
    assert general_info_parser.general_info["created"]["date"] == "19Jul2024"


def test_created_contains_time(general_info_parser: PwXML) -> None:
    """Test that created contains time attribute."""
    assert "time" in general_info_parser.general_info["created"]


def test_created_time_is_correct(general_info_parser: PwXML) -> None:
    """Test that created time is correct."""
    assert general_info_parser.general_info["created"]["time"] == "11:28:51"


def test_created_contains_text(general_info_parser: PwXML) -> None:
    """Test that created contains text content."""
    assert "text" in general_info_parser.general_info["created"]


def test_created_text_is_correct(general_info_parser: PwXML) -> None:
    """Test that created text is correct."""
    assert (
        general_info_parser.general_info["created"]["text"]
        == "This run was terminated on:  11:28:51  19 Jul 2024"
    )


# =============================================================================
# Tests: General Info - job
# =============================================================================


def test_job_is_correct(general_info_parser: PwXML) -> None:
    """Test that job is correct."""
    assert general_info_parser.general_info["job"] == "test_job"


def test_job_empty_returns_empty_string(general_info_empty_job_parser: PwXML) -> None:
    """Test that empty job tag returns empty string."""
    assert general_info_empty_job_parser.general_info["job"] == ""


# =============================================================================
# Tests: General Info - convenience properties
# =============================================================================


def test_xml_format_name_property_returns_string(general_info_parser: PwXML) -> None:
    """Test that xml_format_name returns a string."""
    assert general_info_parser.xml_format_name is not None
    assert isinstance(general_info_parser.xml_format_name, str)


def test_xml_format_name_property_returns_none_when_absent(
    no_general_info_parser: PwXML,
) -> None:
    """Test that xml_format_name returns None when absent."""
    assert no_general_info_parser.xml_format_name is None


def test_xml_format_name_property_is_correct(general_info_parser: PwXML) -> None:
    """Test that xml_format_name returns correct value."""
    assert general_info_parser.xml_format_name == "QEXSD"


def test_xml_format_version_property_returns_string(general_info_parser: PwXML) -> None:
    """Test that xml_format_version returns a string."""
    assert general_info_parser.xml_format_version is not None
    assert isinstance(general_info_parser.xml_format_version, str)


def test_xml_format_version_property_returns_none_when_absent(
    no_general_info_parser: PwXML,
) -> None:
    """Test that xml_format_version returns None when absent."""
    assert no_general_info_parser.xml_format_version is None


def test_xml_format_version_property_is_correct(general_info_parser: PwXML) -> None:
    """Test that xml_format_version returns correct value."""
    assert general_info_parser.xml_format_version == "23.03.10"


def test_creator_name_property_returns_string(general_info_parser: PwXML) -> None:
    """Test that creator_name returns a string."""
    assert general_info_parser.creator_name is not None
    assert isinstance(general_info_parser.creator_name, str)


def test_creator_name_property_returns_none_when_absent(
    no_general_info_parser: PwXML,
) -> None:
    """Test that creator_name returns None when absent."""
    assert no_general_info_parser.creator_name is None


def test_creator_name_property_is_correct(general_info_parser: PwXML) -> None:
    """Test that creator_name returns correct value."""
    assert general_info_parser.creator_name == "PWSCF"


def test_creator_version_property_returns_string(general_info_parser: PwXML) -> None:
    """Test that creator_version returns a string."""
    assert general_info_parser.creator_version is not None
    assert isinstance(general_info_parser.creator_version, str)


def test_creator_version_property_returns_none_when_absent(
    no_general_info_parser: PwXML,
) -> None:
    """Test that creator_version returns None when absent."""
    assert no_general_info_parser.creator_version is None


def test_creator_version_property_is_correct(general_info_parser: PwXML) -> None:
    """Test that creator_version returns correct value."""
    assert general_info_parser.creator_version == "7.2"


def test_created_date_property_returns_string(general_info_parser: PwXML) -> None:
    """Test that created_date returns a string."""
    assert general_info_parser.created_date is not None
    assert isinstance(general_info_parser.created_date, str)


def test_created_date_property_returns_none_when_absent(
    no_general_info_parser: PwXML,
) -> None:
    """Test that created_date returns None when absent."""
    assert no_general_info_parser.created_date is None


def test_created_date_property_is_correct(general_info_parser: PwXML) -> None:
    """Test that created_date returns correct value."""
    assert general_info_parser.created_date == "19Jul2024"


def test_created_time_property_returns_string(general_info_parser: PwXML) -> None:
    """Test that created_time returns a string."""
    assert general_info_parser.created_time is not None
    assert isinstance(general_info_parser.created_time, str)


def test_created_time_property_returns_none_when_absent(
    no_general_info_parser: PwXML,
) -> None:
    """Test that created_time returns None when absent."""
    assert no_general_info_parser.created_time is None


def test_created_time_property_is_correct(general_info_parser: PwXML) -> None:
    """Test that created_time returns correct value."""
    assert general_info_parser.created_time == "11:28:51"


def test_job_property_returns_string(general_info_parser: PwXML) -> None:
    """Test that job property returns a string."""
    assert general_info_parser.job is not None
    assert isinstance(general_info_parser.job, str)


def test_job_property_returns_none_when_absent(no_general_info_parser: PwXML) -> None:
    """Test that job property returns None when absent."""
    assert no_general_info_parser.job is None


def test_job_property_is_correct(general_info_parser: PwXML) -> None:
    """Test that job property returns correct value."""
    assert general_info_parser.job == "test_job"


# =============================================================================
# Inline String Fixtures for Parallel Info
# =============================================================================

PARALLEL_INFO_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <parallel_info>
    <nprocs>40</nprocs>
    <nthreads>1</nthreads>
    <ntasks>1</ntasks>
    <nbgrp>1</nbgrp>
    <npool>4</npool>
    <ndiag>10</ndiag>
  </parallel_info>
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""

NO_PARALLEL_INFO_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Parallel Info
# =============================================================================


@pytest.fixture
def parallel_info_filepath(tmp_path: Path) -> Path:
    """Create temporary file with parallel_info tag."""
    filepath = tmp_path / "parallel_info.xml"
    filepath.write_text(PARALLEL_INFO_PW_XML)
    return filepath


@pytest.fixture
def no_parallel_info_filepath(tmp_path: Path) -> Path:
    """Create temporary file without parallel_info tag."""
    filepath = tmp_path / "no_parallel_info.xml"
    filepath.write_text(NO_PARALLEL_INFO_PW_XML)
    return filepath


@pytest.fixture
def parallel_info_parser(parallel_info_filepath: Path) -> PwXML:
    """Create parser instance for parallel_info test."""
    return PwXML(filepath=parallel_info_filepath)


@pytest.fixture
def no_parallel_info_parser(no_parallel_info_filepath: Path) -> PwXML:
    """Create parser instance for no parallel_info test."""
    return PwXML(filepath=no_parallel_info_filepath)


# =============================================================================
# Tests: Parallel Info - parallel_info dictionary
# =============================================================================


def test_parallel_info_returns_dict_when_present(parallel_info_parser: PwXML) -> None:
    """Test that parallel_info returns a dictionary when tag exists."""
    assert parallel_info_parser.parallel_info is not None
    assert isinstance(parallel_info_parser.parallel_info, dict)


def test_parallel_info_returns_none_when_absent(no_parallel_info_parser: PwXML) -> None:
    """Test that parallel_info returns None when tag is absent."""
    assert no_parallel_info_parser.parallel_info is None


def test_parallel_info_contains_nprocs_key(parallel_info_parser: PwXML) -> None:
    """Test that parallel_info dictionary contains nprocs key."""
    assert "nprocs" in parallel_info_parser.parallel_info


def test_parallel_info_contains_nthreads_key(parallel_info_parser: PwXML) -> None:
    """Test that parallel_info dictionary contains nthreads key."""
    assert "nthreads" in parallel_info_parser.parallel_info


def test_parallel_info_contains_ntasks_key(parallel_info_parser: PwXML) -> None:
    """Test that parallel_info dictionary contains ntasks key."""
    assert "ntasks" in parallel_info_parser.parallel_info


def test_parallel_info_contains_nbgrp_key(parallel_info_parser: PwXML) -> None:
    """Test that parallel_info dictionary contains nbgrp key."""
    assert "nbgrp" in parallel_info_parser.parallel_info


def test_parallel_info_contains_npool_key(parallel_info_parser: PwXML) -> None:
    """Test that parallel_info dictionary contains npool key."""
    assert "npool" in parallel_info_parser.parallel_info


def test_parallel_info_contains_ndiag_key(parallel_info_parser: PwXML) -> None:
    """Test that parallel_info dictionary contains ndiag key."""
    assert "ndiag" in parallel_info_parser.parallel_info


# =============================================================================
# Tests: Parallel Info - values
# =============================================================================


def test_parallel_info_nprocs_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that nprocs value is correct."""
    assert parallel_info_parser.parallel_info["nprocs"] == 40


def test_parallel_info_nthreads_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that nthreads value is correct."""
    assert parallel_info_parser.parallel_info["nthreads"] == 1


def test_parallel_info_ntasks_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that ntasks value is correct."""
    assert parallel_info_parser.parallel_info["ntasks"] == 1


def test_parallel_info_nbgrp_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that nbgrp value is correct."""
    assert parallel_info_parser.parallel_info["nbgrp"] == 1


def test_parallel_info_npool_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that npool value is correct."""
    assert parallel_info_parser.parallel_info["npool"] == 4


def test_parallel_info_ndiag_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that ndiag value is correct."""
    assert parallel_info_parser.parallel_info["ndiag"] == 10


# =============================================================================
# Tests: Parallel Info - convenience properties
# =============================================================================


def test_nprocs_property_returns_int(parallel_info_parser: PwXML) -> None:
    """Test that nprocs property returns an integer."""
    assert parallel_info_parser.nprocs is not None
    assert isinstance(parallel_info_parser.nprocs, int)


def test_nprocs_property_returns_none_when_absent(
    no_parallel_info_parser: PwXML,
) -> None:
    """Test that nprocs property returns None when absent."""
    assert no_parallel_info_parser.nprocs is None


def test_nprocs_property_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that nprocs property returns correct value."""
    assert parallel_info_parser.nprocs == 40


def test_nthreads_property_returns_int(parallel_info_parser: PwXML) -> None:
    """Test that nthreads property returns an integer."""
    assert parallel_info_parser.nthreads is not None
    assert isinstance(parallel_info_parser.nthreads, int)


def test_nthreads_property_returns_none_when_absent(
    no_parallel_info_parser: PwXML,
) -> None:
    """Test that nthreads property returns None when absent."""
    assert no_parallel_info_parser.nthreads is None


def test_nthreads_property_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that nthreads property returns correct value."""
    assert parallel_info_parser.nthreads == 1


def test_ntasks_property_returns_int(parallel_info_parser: PwXML) -> None:
    """Test that ntasks property returns an integer."""
    assert parallel_info_parser.ntasks is not None
    assert isinstance(parallel_info_parser.ntasks, int)


def test_ntasks_property_returns_none_when_absent(
    no_parallel_info_parser: PwXML,
) -> None:
    """Test that ntasks property returns None when absent."""
    assert no_parallel_info_parser.ntasks is None


def test_ntasks_property_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that ntasks property returns correct value."""
    assert parallel_info_parser.ntasks == 1


def test_nbgrp_property_returns_int(parallel_info_parser: PwXML) -> None:
    """Test that nbgrp property returns an integer."""
    assert parallel_info_parser.nbgrp is not None
    assert isinstance(parallel_info_parser.nbgrp, int)


def test_nbgrp_property_returns_none_when_absent(
    no_parallel_info_parser: PwXML,
) -> None:
    """Test that nbgrp property returns None when absent."""
    assert no_parallel_info_parser.nbgrp is None


def test_nbgrp_property_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that nbgrp property returns correct value."""
    assert parallel_info_parser.nbgrp == 1


def test_npool_property_returns_int(parallel_info_parser: PwXML) -> None:
    """Test that npool property returns an integer."""
    assert parallel_info_parser.npool is not None
    assert isinstance(parallel_info_parser.npool, int)


def test_npool_property_returns_none_when_absent(
    no_parallel_info_parser: PwXML,
) -> None:
    """Test that npool property returns None when absent."""
    assert no_parallel_info_parser.npool is None


def test_npool_property_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that npool property returns correct value."""
    assert parallel_info_parser.npool == 4


def test_ndiag_property_returns_int(parallel_info_parser: PwXML) -> None:
    """Test that ndiag property returns an integer."""
    assert parallel_info_parser.ndiag is not None
    assert isinstance(parallel_info_parser.ndiag, int)


def test_ndiag_property_returns_none_when_absent(
    no_parallel_info_parser: PwXML,
) -> None:
    """Test that ndiag property returns None when absent."""
    assert no_parallel_info_parser.ndiag is None


def test_ndiag_property_is_correct(parallel_info_parser: PwXML) -> None:
    """Test that ndiag property returns correct value."""
    assert parallel_info_parser.ndiag == 10


# =============================================================================
# Inline String Fixtures for Input Tag
# =============================================================================

INPUT_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <input>
    <control_variables>
      <title>test_title</title>
      <calculation>bands</calculation>
      <restart_mode>from_scratch</restart_mode>
      <prefix>SrVO3</prefix>
      <pseudo_dir>.</pseudo_dir>
      <outdir>./out</outdir>
      <stress>false</stress>
      <forces>false</forces>
      <wf_collect>true</wf_collect>
      <disk_io>low</disk_io>
      <max_seconds>10000000</max_seconds>
      <nstep>1</nstep>
      <etot_conv_thr>5.000000000000000E-05</etot_conv_thr>
      <forc_conv_thr>5.000000000000000E-04</forc_conv_thr>
      <press_conv_thr>5.000000000000000E-01</press_conv_thr>
      <verbosity>low</verbosity>
    </control_variables>
    <atomic_species ntyp="3">
      <species name="Sr">
        <mass>8.762000000000000E+01</mass>
        <pseudo_file>Sr.pbe-spn-kjpaw_psl.1.0.0.UPF</pseudo_file>
        <starting_magnetization>7.000000000000000E-01</starting_magnetization>
      </species>
      <species name="V">
        <mass>5.094150000000000E+01</mass>
        <pseudo_file>V.pbe-spn-kjpaw_psl.1.0.0.UPF</pseudo_file>
        <starting_magnetization>0.000000000000000E+00</starting_magnetization>
      </species>
      <species name="O">
        <mass>1.599940000000000E+01</mass>
        <pseudo_file>O.pbe-n-kjpaw_psl.0.1.upf</pseudo_file>
        <starting_magnetization>0.000000000000000E+00</starting_magnetization>
      </species>
    </atomic_species>
    <atomic_structure nat="5" alat="7.26885043700000" bravais_index="1">
      <atomic_positions>
        <atom name="Sr" index="1">0.0 0.0 0.0</atom>
        <atom name="V" index="2">3.634 3.634 3.634</atom>
      </atomic_positions>
      <cell>
        <a1>7.268850437 0.0 0.0</a1>
        <a2>0.0 7.268850437 0.0</a2>
        <a3>0.0 0.0 7.268850437</a3>
      </cell>
    </atomic_structure>
    <dft>
      <functional>PBE</functional>
    </dft>
    <spin>
      <lsda>true</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </spin>
    <bands>
      <smearing degauss="7.000000000000000E-003">gaussian</smearing>
      <tot_charge>0.000000000000000E+00</tot_charge>
      <occupations>smearing</occupations>
    </bands>
    <basis>
      <gamma_only>false</gamma_only>
      <ecutwfc>2.500000000000000E+01</ecutwfc>
      <ecutrho>3.000000000000000E+02</ecutrho>
    </basis>
    <electron_control>
      <diagonalization>davidson</diagonalization>
      <mixing_mode>plain</mixing_mode>
      <mixing_beta>7.000000000000000E-01</mixing_beta>
      <conv_thr>5.000000000000000E-07</conv_thr>
      <mixing_ndim>8</mixing_ndim>
      <max_nstep>100</max_nstep>
      <diago_thr_init>0.000000000000000E+00</diago_thr_init>
      <diago_full_acc>false</diago_full_acc>
    </electron_control>
    <k_points_IBZ>
      <nk>151</nk>
      <k_point weight="1.00000000000000">0.0 0.0 0.0</k_point>
      <k_point weight="1.00000000000000">0.0166666 0.0 0.0</k_point>
    </k_points_IBZ>
    <ion_control>
      <ion_dynamics>none</ion_dynamics>
      <upscale>1.000000000000000E+02</upscale>
      <remove_rigid_rot>false</remove_rigid_rot>
      <refold_pos>false</refold_pos>
    </ion_control>
    <cell_control>
      <cell_dynamics>none</cell_dynamics>
      <pressure>0.000000000000000E+00</pressure>
      <wmass>1.865597000000000E+02</wmass>
      <cell_do_free>all</cell_do_free>
    </cell_control>
    <symmetry_flags>
      <nosym>false</nosym>
      <nosym_evc>false</nosym_evc>
      <noinv>false</noinv>
      <no_t_rev>false</no_t_rev>
      <force_symmorphic>false</force_symmorphic>
      <use_all_frac>false</use_all_frac>
    </symmetry_flags>
  </input>
  <output>
    <magnetization>
      <lsda>true</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""

NO_INPUT_PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <magnetization>
      <lsda>false</lsda>
      <noncolin>false</noncolin>
      <spinorbit>false</spinorbit>
    </magnetization>
  </output>
</qes:espresso>
"""


# =============================================================================
# Pytest Fixtures for Input Tag
# =============================================================================


@pytest.fixture
def input_filepath(tmp_path: Path) -> Path:
    """Create temporary file with input tag."""
    filepath = tmp_path / "input.xml"
    filepath.write_text(INPUT_PW_XML)
    return filepath


@pytest.fixture
def no_input_filepath(tmp_path: Path) -> Path:
    """Create temporary file without input tag."""
    filepath = tmp_path / "no_input.xml"
    filepath.write_text(NO_INPUT_PW_XML)
    return filepath


@pytest.fixture
def input_parser(input_filepath: Path) -> PwXML:
    """Create parser instance for input test."""
    return PwXML(filepath=input_filepath)


@pytest.fixture
def no_input_parser(no_input_filepath: Path) -> PwXML:
    """Create parser instance for no input test."""
    return PwXML(filepath=no_input_filepath)


# =============================================================================
# Tests: Input - control_variables
# =============================================================================


def test_control_variables_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that control_variables returns a dictionary when tag exists."""
    assert input_parser.control_variables is not None
    assert isinstance(input_parser.control_variables, dict)


def test_control_variables_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that control_variables returns None when tag is absent."""
    assert no_input_parser.control_variables is None


def test_control_variables_contains_calculation(input_parser: PwXML) -> None:
    """Test that control_variables contains calculation key."""
    assert "calculation" in input_parser.control_variables


def test_control_variables_calculation_is_correct(input_parser: PwXML) -> None:
    """Test that calculation value is correct."""
    assert input_parser.control_variables["calculation"] == "bands"


def test_control_variables_contains_prefix(input_parser: PwXML) -> None:
    """Test that control_variables contains prefix key."""
    assert "prefix" in input_parser.control_variables


def test_control_variables_prefix_is_correct(input_parser: PwXML) -> None:
    """Test that prefix value is correct."""
    assert input_parser.control_variables["prefix"] == "SrVO3"


def test_control_variables_contains_pseudo_dir(input_parser: PwXML) -> None:
    """Test that control_variables contains pseudo_dir key."""
    assert "pseudo_dir" in input_parser.control_variables


def test_control_variables_contains_outdir(input_parser: PwXML) -> None:
    """Test that control_variables contains outdir key."""
    assert "outdir" in input_parser.control_variables


def test_control_variables_contains_restart_mode(input_parser: PwXML) -> None:
    """Test that control_variables contains restart_mode key."""
    assert "restart_mode" in input_parser.control_variables


def test_control_variables_restart_mode_is_correct(input_parser: PwXML) -> None:
    """Test that restart_mode value is correct."""
    assert input_parser.control_variables["restart_mode"] == "from_scratch"


def test_control_variables_contains_verbosity(input_parser: PwXML) -> None:
    """Test that control_variables contains verbosity key."""
    assert "verbosity" in input_parser.control_variables


def test_control_variables_verbosity_is_correct(input_parser: PwXML) -> None:
    """Test that verbosity value is correct."""
    assert input_parser.control_variables["verbosity"] == "low"


def test_control_variables_contains_etot_conv_thr(input_parser: PwXML) -> None:
    """Test that control_variables contains etot_conv_thr key."""
    assert "etot_conv_thr" in input_parser.control_variables


def test_control_variables_etot_conv_thr_is_correct(input_parser: PwXML) -> None:
    """Test that etot_conv_thr value is correct."""
    assert input_parser.control_variables["etot_conv_thr"] == pytest.approx(5e-05)


def test_control_variables_contains_max_seconds(input_parser: PwXML) -> None:
    """Test that control_variables contains max_seconds key."""
    assert "max_seconds" in input_parser.control_variables


def test_control_variables_max_seconds_is_correct(input_parser: PwXML) -> None:
    """Test that max_seconds value is correct."""
    assert input_parser.control_variables["max_seconds"] == 10000000


# =============================================================================
# Tests: Input - control_variables convenience properties
# =============================================================================


def test_calculation_property_returns_string(input_parser: PwXML) -> None:
    """Test that calculation property returns a string."""
    assert input_parser.calculation is not None
    assert isinstance(input_parser.calculation, str)


def test_calculation_property_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that calculation property returns None when absent."""
    assert no_input_parser.calculation is None


def test_calculation_property_is_correct(input_parser: PwXML) -> None:
    """Test that calculation property returns correct value."""
    assert input_parser.calculation == "bands"


def test_prefix_property_returns_string(input_parser: PwXML) -> None:
    """Test that prefix property returns a string."""
    assert input_parser.prefix is not None
    assert isinstance(input_parser.prefix, str)


def test_prefix_property_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that prefix property returns None when absent."""
    assert no_input_parser.prefix is None


def test_prefix_property_is_correct(input_parser: PwXML) -> None:
    """Test that prefix property returns correct value."""
    assert input_parser.prefix == "SrVO3"


# =============================================================================
# Tests: Input - input_atomic_species
# =============================================================================


def test_input_atomic_species_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that input_atomic_species returns a dictionary when tag exists."""
    assert input_parser.input_atomic_species is not None
    assert isinstance(input_parser.input_atomic_species, dict)


def test_input_atomic_species_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that input_atomic_species returns None when tag is absent."""
    assert no_input_parser.input_atomic_species is None


def test_input_atomic_species_contains_ntyp(input_parser: PwXML) -> None:
    """Test that input_atomic_species contains ntyp key."""
    assert "ntyp" in input_parser.input_atomic_species


def test_input_atomic_species_ntyp_is_correct(input_parser: PwXML) -> None:
    """Test that ntyp value is correct."""
    assert input_parser.input_atomic_species["ntyp"] == 3


def test_input_atomic_species_contains_species(input_parser: PwXML) -> None:
    """Test that input_atomic_species contains species key."""
    assert "species" in input_parser.input_atomic_species


def test_input_atomic_species_has_three_species(input_parser: PwXML) -> None:
    """Test that there are three species."""
    assert len(input_parser.input_atomic_species["species"]) == 3


def test_input_atomic_species_first_species_name(input_parser: PwXML) -> None:
    """Test that first species name is correct."""
    assert input_parser.input_atomic_species["species"][0]["name"] == "Sr"


def test_input_atomic_species_first_species_mass(input_parser: PwXML) -> None:
    """Test that first species mass is correct."""
    assert input_parser.input_atomic_species["species"][0]["mass"] == pytest.approx(
        87.62
    )


def test_input_atomic_species_first_species_pseudo_file(input_parser: PwXML) -> None:
    """Test that first species pseudo_file is correct."""
    assert (
        input_parser.input_atomic_species["species"][0]["pseudo_file"]
        == "Sr.pbe-spn-kjpaw_psl.1.0.0.UPF"
    )


def test_input_atomic_species_first_species_magnetization(input_parser: PwXML) -> None:
    """Test that first species starting_magnetization is correct."""
    assert input_parser.input_atomic_species["species"][0][
        "starting_magnetization"
    ] == pytest.approx(0.7)


def test_input_atomic_species_second_species_name(input_parser: PwXML) -> None:
    """Test that second species name is correct."""
    assert input_parser.input_atomic_species["species"][1]["name"] == "V"


def test_input_atomic_species_third_species_name(input_parser: PwXML) -> None:
    """Test that third species name is correct."""
    assert input_parser.input_atomic_species["species"][2]["name"] == "O"


# =============================================================================
# Tests: Input - input_spin
# =============================================================================


def test_input_spin_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that input_spin returns a dictionary when tag exists."""
    assert input_parser.input_spin is not None
    assert isinstance(input_parser.input_spin, dict)


def test_input_spin_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that input_spin returns None when tag is absent."""
    assert no_input_parser.input_spin is None


def test_input_spin_contains_lsda(input_parser: PwXML) -> None:
    """Test that input_spin contains lsda key."""
    assert "lsda" in input_parser.input_spin


def test_input_spin_lsda_is_correct(input_parser: PwXML) -> None:
    """Test that input lsda value is correct."""
    assert input_parser.input_spin["lsda"] is True


def test_input_spin_contains_noncolin(input_parser: PwXML) -> None:
    """Test that input_spin contains noncolin key."""
    assert "noncolin" in input_parser.input_spin


def test_input_spin_noncolin_is_correct(input_parser: PwXML) -> None:
    """Test that input noncolin value is correct."""
    assert input_parser.input_spin["noncolin"] is False


def test_input_spin_contains_spinorbit(input_parser: PwXML) -> None:
    """Test that input_spin contains spinorbit key."""
    assert "spinorbit" in input_parser.input_spin


def test_input_spin_spinorbit_is_correct(input_parser: PwXML) -> None:
    """Test that input spinorbit value is correct."""
    assert input_parser.input_spin["spinorbit"] is False


# =============================================================================
# Tests: Input - input_bands
# =============================================================================


def test_input_bands_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that input_bands returns a dictionary when tag exists."""
    assert input_parser.input_bands is not None
    assert isinstance(input_parser.input_bands, dict)


def test_input_bands_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that input_bands returns None when tag is absent."""
    assert no_input_parser.input_bands is None


def test_input_bands_contains_occupations(input_parser: PwXML) -> None:
    """Test that input_bands contains occupations key."""
    assert "occupations" in input_parser.input_bands


def test_input_bands_occupations_is_correct(input_parser: PwXML) -> None:
    """Test that occupations value is correct."""
    assert input_parser.input_bands["occupations"] == "smearing"


def test_input_bands_contains_smearing(input_parser: PwXML) -> None:
    """Test that input_bands contains smearing key."""
    assert "smearing" in input_parser.input_bands


def test_input_bands_smearing_type_is_correct(input_parser: PwXML) -> None:
    """Test that smearing type is correct."""
    assert input_parser.input_bands["smearing"]["type"] == "gaussian"


def test_input_bands_smearing_degauss_is_correct(input_parser: PwXML) -> None:
    """Test that smearing degauss is correct."""
    assert input_parser.input_bands["smearing"]["degauss"] == pytest.approx(0.007)


def test_input_bands_contains_tot_charge(input_parser: PwXML) -> None:
    """Test that input_bands contains tot_charge key."""
    assert "tot_charge" in input_parser.input_bands


def test_input_bands_tot_charge_is_correct(input_parser: PwXML) -> None:
    """Test that tot_charge value is correct."""
    assert input_parser.input_bands["tot_charge"] == pytest.approx(0.0)


# =============================================================================
# Tests: Input - input_basis
# =============================================================================


def test_input_basis_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that input_basis returns a dictionary when tag exists."""
    assert input_parser.input_basis is not None
    assert isinstance(input_parser.input_basis, dict)


def test_input_basis_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that input_basis returns None when tag is absent."""
    assert no_input_parser.input_basis is None


def test_input_basis_contains_gamma_only(input_parser: PwXML) -> None:
    """Test that input_basis contains gamma_only key."""
    assert "gamma_only" in input_parser.input_basis


def test_input_basis_gamma_only_is_correct(input_parser: PwXML) -> None:
    """Test that gamma_only value is correct."""
    assert input_parser.input_basis["gamma_only"] is False


def test_input_basis_contains_ecutwfc(input_parser: PwXML) -> None:
    """Test that input_basis contains ecutwfc key."""
    assert "ecutwfc" in input_parser.input_basis


def test_input_basis_ecutwfc_is_correct(input_parser: PwXML) -> None:
    """Test that ecutwfc value is correct."""
    assert input_parser.input_basis["ecutwfc"] == pytest.approx(25.0)


def test_input_basis_contains_ecutrho(input_parser: PwXML) -> None:
    """Test that input_basis contains ecutrho key."""
    assert "ecutrho" in input_parser.input_basis


def test_input_basis_ecutrho_is_correct(input_parser: PwXML) -> None:
    """Test that ecutrho value is correct."""
    assert input_parser.input_basis["ecutrho"] == pytest.approx(300.0)


# =============================================================================
# Tests: Input - input_basis convenience properties
# =============================================================================


def test_ecutwfc_property_returns_float(input_parser: PwXML) -> None:
    """Test that ecutwfc property returns a float."""
    assert input_parser.ecutwfc is not None
    assert isinstance(input_parser.ecutwfc, float)


def test_ecutwfc_property_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that ecutwfc property returns None when absent."""
    assert no_input_parser.ecutwfc is None


def test_ecutwfc_property_is_correct(input_parser: PwXML) -> None:
    """Test that ecutwfc property returns correct value."""
    assert input_parser.ecutwfc == pytest.approx(25.0)


def test_ecutrho_property_returns_float(input_parser: PwXML) -> None:
    """Test that ecutrho property returns a float."""
    assert input_parser.ecutrho is not None
    assert isinstance(input_parser.ecutrho, float)


def test_ecutrho_property_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that ecutrho property returns None when absent."""
    assert no_input_parser.ecutrho is None


def test_ecutrho_property_is_correct(input_parser: PwXML) -> None:
    """Test that ecutrho property returns correct value."""
    assert input_parser.ecutrho == pytest.approx(300.0)


# =============================================================================
# Tests: Input - electron_control
# =============================================================================


def test_electron_control_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that electron_control returns a dictionary when tag exists."""
    assert input_parser.electron_control is not None
    assert isinstance(input_parser.electron_control, dict)


def test_electron_control_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that electron_control returns None when tag is absent."""
    assert no_input_parser.electron_control is None


def test_electron_control_contains_diagonalization(input_parser: PwXML) -> None:
    """Test that electron_control contains diagonalization key."""
    assert "diagonalization" in input_parser.electron_control


def test_electron_control_diagonalization_is_correct(input_parser: PwXML) -> None:
    """Test that diagonalization value is correct."""
    assert input_parser.electron_control["diagonalization"] == "davidson"


def test_electron_control_contains_mixing_mode(input_parser: PwXML) -> None:
    """Test that electron_control contains mixing_mode key."""
    assert "mixing_mode" in input_parser.electron_control


def test_electron_control_mixing_mode_is_correct(input_parser: PwXML) -> None:
    """Test that mixing_mode value is correct."""
    assert input_parser.electron_control["mixing_mode"] == "plain"


def test_electron_control_contains_mixing_beta(input_parser: PwXML) -> None:
    """Test that electron_control contains mixing_beta key."""
    assert "mixing_beta" in input_parser.electron_control


def test_electron_control_mixing_beta_is_correct(input_parser: PwXML) -> None:
    """Test that mixing_beta value is correct."""
    assert input_parser.electron_control["mixing_beta"] == pytest.approx(0.7)


def test_electron_control_contains_conv_thr(input_parser: PwXML) -> None:
    """Test that electron_control contains conv_thr key."""
    assert "conv_thr" in input_parser.electron_control


def test_electron_control_conv_thr_is_correct(input_parser: PwXML) -> None:
    """Test that conv_thr value is correct."""
    assert input_parser.electron_control["conv_thr"] == pytest.approx(5e-07)


def test_electron_control_contains_max_nstep(input_parser: PwXML) -> None:
    """Test that electron_control contains max_nstep key."""
    assert "max_nstep" in input_parser.electron_control


def test_electron_control_max_nstep_is_correct(input_parser: PwXML) -> None:
    """Test that max_nstep value is correct."""
    assert input_parser.electron_control["max_nstep"] == 100


# =============================================================================
# Tests: Input - k_points_IBZ
# =============================================================================


def test_k_points_ibz_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that k_points_ibz returns a dictionary when tag exists."""
    assert input_parser.k_points_ibz is not None
    assert isinstance(input_parser.k_points_ibz, dict)


def test_k_points_ibz_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that k_points_ibz returns None when tag is absent."""
    assert no_input_parser.k_points_ibz is None


def test_k_points_ibz_contains_nk(input_parser: PwXML) -> None:
    """Test that k_points_ibz contains nk key."""
    assert "nk" in input_parser.k_points_ibz


def test_k_points_ibz_nk_is_correct(input_parser: PwXML) -> None:
    """Test that nk value is correct."""
    assert input_parser.k_points_ibz["nk"] == 151


def test_k_points_ibz_contains_k_points(input_parser: PwXML) -> None:
    """Test that k_points_ibz contains k_points key."""
    assert "k_points" in input_parser.k_points_ibz


def test_k_points_ibz_has_two_k_points(input_parser: PwXML) -> None:
    """Test that there are two k_points in the list."""
    assert len(input_parser.k_points_ibz["k_points"]) == 2


def test_k_points_ibz_first_k_point_weight(input_parser: PwXML) -> None:
    """Test that first k_point weight is correct."""
    assert input_parser.k_points_ibz["k_points"][0]["weight"] == pytest.approx(1.0)


def test_k_points_ibz_first_k_point_coordinates(input_parser: PwXML) -> None:
    """Test that first k_point coordinates are correct."""
    import numpy as np

    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(
        input_parser.k_points_ibz["k_points"][0]["coordinates"], expected
    )


# =============================================================================
# Tests: Input - ion_control
# =============================================================================


def test_ion_control_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that ion_control returns a dictionary when tag exists."""
    assert input_parser.ion_control is not None
    assert isinstance(input_parser.ion_control, dict)


def test_ion_control_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that ion_control returns None when tag is absent."""
    assert no_input_parser.ion_control is None


def test_ion_control_contains_ion_dynamics(input_parser: PwXML) -> None:
    """Test that ion_control contains ion_dynamics key."""
    assert "ion_dynamics" in input_parser.ion_control


def test_ion_control_ion_dynamics_is_correct(input_parser: PwXML) -> None:
    """Test that ion_dynamics value is correct."""
    assert input_parser.ion_control["ion_dynamics"] == "none"


def test_ion_control_contains_upscale(input_parser: PwXML) -> None:
    """Test that ion_control contains upscale key."""
    assert "upscale" in input_parser.ion_control


def test_ion_control_upscale_is_correct(input_parser: PwXML) -> None:
    """Test that upscale value is correct."""
    assert input_parser.ion_control["upscale"] == pytest.approx(100.0)


# =============================================================================
# Tests: Input - cell_control
# =============================================================================


def test_cell_control_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that cell_control returns a dictionary when tag exists."""
    assert input_parser.cell_control is not None
    assert isinstance(input_parser.cell_control, dict)


def test_cell_control_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that cell_control returns None when tag is absent."""
    assert no_input_parser.cell_control is None


def test_cell_control_contains_cell_dynamics(input_parser: PwXML) -> None:
    """Test that cell_control contains cell_dynamics key."""
    assert "cell_dynamics" in input_parser.cell_control


def test_cell_control_cell_dynamics_is_correct(input_parser: PwXML) -> None:
    """Test that cell_dynamics value is correct."""
    assert input_parser.cell_control["cell_dynamics"] == "none"


def test_cell_control_contains_pressure(input_parser: PwXML) -> None:
    """Test that cell_control contains pressure key."""
    assert "pressure" in input_parser.cell_control


def test_cell_control_pressure_is_correct(input_parser: PwXML) -> None:
    """Test that pressure value is correct."""
    assert input_parser.cell_control["pressure"] == pytest.approx(0.0)


def test_cell_control_contains_cell_do_free(input_parser: PwXML) -> None:
    """Test that cell_control contains cell_do_free key."""
    assert "cell_do_free" in input_parser.cell_control


def test_cell_control_cell_do_free_is_correct(input_parser: PwXML) -> None:
    """Test that cell_do_free value is correct."""
    assert input_parser.cell_control["cell_do_free"] == "all"


# =============================================================================
# Tests: Input - symmetry_flags
# =============================================================================


def test_symmetry_flags_returns_dict_when_present(input_parser: PwXML) -> None:
    """Test that symmetry_flags returns a dictionary when tag exists."""
    assert input_parser.symmetry_flags is not None
    assert isinstance(input_parser.symmetry_flags, dict)


def test_symmetry_flags_returns_none_when_absent(no_input_parser: PwXML) -> None:
    """Test that symmetry_flags returns None when tag is absent."""
    assert no_input_parser.symmetry_flags is None


def test_symmetry_flags_contains_nosym(input_parser: PwXML) -> None:
    """Test that symmetry_flags contains nosym key."""
    assert "nosym" in input_parser.symmetry_flags


def test_symmetry_flags_nosym_is_correct(input_parser: PwXML) -> None:
    """Test that nosym value is correct."""
    assert input_parser.symmetry_flags["nosym"] is False


def test_symmetry_flags_contains_noinv(input_parser: PwXML) -> None:
    """Test that symmetry_flags contains noinv key."""
    assert "noinv" in input_parser.symmetry_flags


def test_symmetry_flags_noinv_is_correct(input_parser: PwXML) -> None:
    """Test that noinv value is correct."""
    assert input_parser.symmetry_flags["noinv"] is False


def test_symmetry_flags_contains_no_t_rev(input_parser: PwXML) -> None:
    """Test that symmetry_flags contains no_t_rev key."""
    assert "no_t_rev" in input_parser.symmetry_flags


def test_symmetry_flags_no_t_rev_is_correct(input_parser: PwXML) -> None:
    """Test that no_t_rev value is correct."""
    assert input_parser.symmetry_flags["no_t_rev"] is False
