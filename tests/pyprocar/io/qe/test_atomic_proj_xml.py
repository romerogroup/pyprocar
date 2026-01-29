"""Tests for AtomicProjXML parser."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.qe.projwfc import AtomicProjXML
from pyprocar.utils.units import RYDBERG_TO_EV

# =============================================================================
# Inline String Fixtures
# =============================================================================

NON_SPIN_POLARIZED_ATOMIC_PROJ_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PROJECTIONS>
  <HEADER NUMBER_OF_BANDS="2" NUMBER_OF_K-POINTS="2" NUMBER_OF_SPIN_COMPONENTS="1"
          NUMBER_OF_ATOMIC_WFC="2" NUMBER_OF_ELECTRONS="40" FERMI_ENERGY="0.372"/>
  <EIGENSTATES>
    <K-POINT Weight="0.001">0.0 0.0 0.0</K-POINT>
    <E>-0.5 -0.4</E>
    <PROJS>
      <ATOMIC_WFC index="1" spin="1">
0.1 0.0
0.2 0.0
      </ATOMIC_WFC>
      <ATOMIC_WFC index="2" spin="1">
0.15 0.0
0.25 0.0
      </ATOMIC_WFC>
    </PROJS>
    <K-POINT Weight="0.008">0.125 0.0 0.0</K-POINT>
    <E>-0.48 -0.38</E>
    <PROJS>
      <ATOMIC_WFC index="1" spin="1">
0.11 0.0
0.21 0.0
      </ATOMIC_WFC>
      <ATOMIC_WFC index="2" spin="1">
0.16 0.0
0.26 0.0
      </ATOMIC_WFC>
    </PROJS>
  </EIGENSTATES>
</PROJECTIONS>
"""

SPIN_POLARIZED_ATOMIC_PROJ_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PROJECTIONS>
  <HEADER NUMBER_OF_BANDS="2" NUMBER_OF_K-POINTS="2" NUMBER_OF_SPIN_COMPONENTS="2"
          NUMBER_OF_ATOMIC_WFC="2" NUMBER_OF_ELECTRONS="40" FERMI_ENERGY="0.373"/>
  <EIGENSTATES>
    <K-POINT Weight="0.001">0.0 0.0 0.0</K-POINT>
    <E>-0.5 -0.4</E>
    <PROJS>
      <ATOMIC_WFC index="1" spin="1">
0.1 0.0
0.2 0.0
      </ATOMIC_WFC>
      <ATOMIC_WFC index="2" spin="1">
0.15 0.0
0.25 0.0
      </ATOMIC_WFC>
    </PROJS>
    <K-POINT Weight="0.008">0.125 0.0 0.0</K-POINT>
    <E>-0.48 -0.38</E>
    <PROJS>
      <ATOMIC_WFC index="1" spin="1">
0.11 0.0
0.21 0.0
      </ATOMIC_WFC>
      <ATOMIC_WFC index="2" spin="1">
0.16 0.0
0.26 0.0
      </ATOMIC_WFC>
    </PROJS>
    <K-POINT Weight="0.001">0.0 0.0 0.0</K-POINT>
    <E>-0.52 -0.42</E>
    <PROJS>
      <ATOMIC_WFC index="1" spin="2">
0.08 0.0
0.18 0.0
      </ATOMIC_WFC>
      <ATOMIC_WFC index="2" spin="2">
0.13 0.0
0.23 0.0
      </ATOMIC_WFC>
    </PROJS>
    <K-POINT Weight="0.008">0.125 0.0 0.0</K-POINT>
    <E>-0.50 -0.40</E>
    <PROJS>
      <ATOMIC_WFC index="1" spin="2">
0.09 0.0
0.19 0.0
      </ATOMIC_WFC>
      <ATOMIC_WFC index="2" spin="2">
0.14 0.0
0.24 0.0
      </ATOMIC_WFC>
    </PROJS>
  </EIGENSTATES>
</PROJECTIONS>
"""

DATA_FILE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <general_info>
    <xml_format NAME="QEXSD" VERSION="21.11.01">QEXSD_21.11.01</xml_format>
  </general_info>
</qes:espresso>
"""

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def non_spin_filepath(tmp_path: Path) -> Path:
    """Create temporary file for non-spin-polarized test."""
    filepath = tmp_path / "atomic_proj.xml"
    filepath.write_text(NON_SPIN_POLARIZED_ATOMIC_PROJ_XML)
    return filepath


@pytest.fixture
def non_spin_parser(non_spin_filepath: Path) -> AtomicProjXML:
    """Create parser instance for non-spin-polarized test."""
    return AtomicProjXML(filepath=non_spin_filepath)


@pytest.fixture
def spin_polarized_filepath(tmp_path: Path) -> Path:
    """Create temporary file for spin-polarized test."""
    filepath = tmp_path / "atomic_proj.xml"
    filepath.write_text(SPIN_POLARIZED_ATOMIC_PROJ_XML)
    return filepath


@pytest.fixture
def spin_polarized_parser(spin_polarized_filepath: Path) -> AtomicProjXML:
    """Create parser instance for spin-polarized test."""
    return AtomicProjXML(filepath=spin_polarized_filepath)


# =============================================================================
# Tests: Header Parsing
# =============================================================================


def test_n_bands_returns_correct_count(non_spin_parser: AtomicProjXML) -> None:
    """Test that n_bands parses correctly from HEADER."""
    assert non_spin_parser.n_bands == 2


def test_n_kpoints_returns_correct_count(non_spin_parser: AtomicProjXML) -> None:
    """Test that n_kpoints parses correctly from HEADER."""
    assert non_spin_parser.n_kpoints == 2


def test_n_atomic_wfc_returns_correct_count(non_spin_parser: AtomicProjXML) -> None:
    """Test that n_atm_wfc parses correctly from HEADER."""
    assert non_spin_parser.n_atm_wfc == 2


def test_n_electrons_returns_correct_count(non_spin_parser: AtomicProjXML) -> None:
    """Test that n_electrons parses correctly from HEADER."""
    assert non_spin_parser.n_electrons == 40


def test_n_spins_non_spin_polarized(non_spin_parser: AtomicProjXML) -> None:
    """Test that n_spin_channels is 1 for non-spin-polarized."""
    assert non_spin_parser.n_spin_channels == 1


def test_n_spins_spin_polarized(spin_polarized_parser: AtomicProjXML) -> None:
    """Test that n_spin_channels is 2 for spin-polarized."""
    assert spin_polarized_parser.n_spin_channels == 2


# =============================================================================
# Tests: Fermi Energy
# =============================================================================


def test_fermi_energy_non_spin_polarized(non_spin_parser: AtomicProjXML) -> None:
    """Test that Fermi energy is parsed and converted to eV."""
    expected_fermi_ev = 0.372 * RYDBERG_TO_EV
    assert pytest.approx(non_spin_parser.fermi, rel=1e-3) == expected_fermi_ev


def test_fermi_energy_spin_polarized(spin_polarized_parser: AtomicProjXML) -> None:
    """Test that Fermi energy is parsed for spin-polarized case."""
    expected_fermi_ev = 0.373 * RYDBERG_TO_EV
    assert pytest.approx(spin_polarized_parser.fermi, rel=1e-3) == expected_fermi_ev


# =============================================================================
# Tests: Eigenvalue (bands) Shapes
# =============================================================================


def test_eigenvalues_shape_non_spin_polarized(non_spin_parser: AtomicProjXML) -> None:
    """Test that bands shape is (n_kpoints, n_bands, n_spin_channels) for non-spin-polarized."""
    bands = non_spin_parser.bands
    assert bands is not None
    assert bands.shape == (2, 2, 1)


def test_eigenvalues_shape_spin_polarized(spin_polarized_parser: AtomicProjXML) -> None:
    """Test that bands shape is (n_kpoints, n_bands, n_spin_channels) for spin-polarized."""
    bands = spin_polarized_parser.bands
    assert bands is not None
    assert bands.shape == (2, 2, 2)


def test_kpoints_returns_correct_shape(non_spin_parser: AtomicProjXML) -> None:
    """Test that kpoints has shape (n_kpoints, 3)."""
    kpoints = non_spin_parser.kpoints
    assert kpoints is not None
    assert kpoints.shape == (2, 3)


def test_kpoints_first_is_gamma(non_spin_parser: AtomicProjXML) -> None:
    """Test that first k-point is gamma point."""
    kpoints = non_spin_parser.kpoints
    assert kpoints is not None
    np.testing.assert_array_almost_equal(kpoints[0], [0.0, 0.0, 0.0])


def test_projections_shape_non_spin_polarized(non_spin_parser: AtomicProjXML) -> None:
    """Test that projections shape is (n_kpoints, n_bands, n_spin_projections, n_atm_wfc)."""
    projections = non_spin_parser.projections
    assert projections is not None
    # n_spin_projections = n_spin_channels for colinear case
    assert projections.shape == (2, 2, 1, 2)


def test_projections_are_complex(non_spin_parser: AtomicProjXML) -> None:
    """Test that projections are complex-valued."""
    projections = non_spin_parser.projections
    assert projections is not None
    assert np.iscomplexobj(projections)


def test_weights_returns_correct_shape(non_spin_parser: AtomicProjXML) -> None:
    """Test that weights has correct shape."""
    weights = non_spin_parser.weights
    assert weights is not None
    assert weights.shape == (2,)


def test_weights_values_match_fixture(non_spin_parser: AtomicProjXML) -> None:
    """Test that weight values match the fixture values."""
    weights = non_spin_parser.weights
    assert weights is not None
    assert pytest.approx(weights[0]) == 0.001
    assert pytest.approx(weights[1]) == 0.008
