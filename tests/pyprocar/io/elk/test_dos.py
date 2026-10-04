"""Tests for ElkDOS extractor."""

import logging

import numpy as np
import pytest

from pyprocar.io.elk.dos import ElkDOS
from pyprocar.utils.units import HARTREE_TO_EV
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


# Example TDOS.OUT content for non-spin-polarized case
# Format: energy (Hartree), DOS
TDOS_NON_SPIN = """ -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000       0.100000000
 -0.3000000000       0.500000000
 -0.2500000000       1.200000000
 -0.2000000000       2.500000000
 -0.1500000000       3.800000000
 -0.1000000000       4.200000000
 -0.0500000000       3.500000000
"""

# Example TDOS.OUT content for spin-polarized case
# Two blocks separated by blank line (spin up, spin down)
TDOS_SPIN = """ -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000       0.100000000
 -0.3000000000       0.500000000

 -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000      -0.080000000
 -0.3000000000      -0.400000000
"""

# Example PDOS file content (projected DOS per atom)
# Same format as TDOS but with orbital projections
PDOS_S01_A0001 = """ -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000       0.010000000
 -0.3000000000       0.050000000

 -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000       0.005000000
 -0.3000000000       0.025000000

"""

PDOS_S02_A0001 = """ -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000       0.020000000
 -0.3000000000       0.100000000

 -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000       0.015000000
 -0.3000000000       0.075000000

"""


@pytest.fixture
def dos_dir_non_spin(tmp_path):
    """Create a temporary directory with DOS files for non-spin case."""
    tdos_file = tmp_path / "TDOS.OUT"
    tdos_file.write_text(TDOS_NON_SPIN)
    return tmp_path


@pytest.fixture
def dos_dir_spin(tmp_path):
    """Create a temporary directory with DOS files for spin-polarized case."""
    tdos_file = tmp_path / "TDOS.OUT"
    tdos_file.write_text(TDOS_SPIN)
    # Add PDOS files
    pdos1 = tmp_path / "PDOS_S01_A0001.OUT"
    pdos1.write_text(PDOS_S01_A0001)
    pdos2 = tmp_path / "PDOS_S02_A0001.OUT"
    pdos2.write_text(PDOS_S02_A0001)
    return tmp_path


class TestElkDOSInit(BaseTest):
    def test_dos_from_dirpath(self, dos_dir_non_spin):
        """Test loading ElkDOS from directory."""
        dos = ElkDOS(dirpath=dos_dir_non_spin)
        assert dos is not None

    def test_dos_from_str(self):
        """Test loading ElkDOS from string content."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        assert dos is not None


class TestElkDOSDetection(BaseTest):
    def test_has_dos_true(self, dos_dir_non_spin):
        """Test has_dos is True when TDOS.OUT exists."""
        dos = ElkDOS(dirpath=dos_dir_non_spin)
        assert dos.has_dos is True

    def test_has_dos_false(self, tmp_path):
        """Test has_dos is False when no DOS files."""
        dos = ElkDOS(dirpath=tmp_path)
        assert dos.has_dos is False


class TestElkDOSEnergies(BaseTest):
    def test_energies_hartree_shape(self):
        """Test energies_hartree array shape."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        assert dos.energies_hartree is not None
        assert len(dos.energies_hartree) == 10

    def test_energies_shape(self):
        """Test energies array shape (converted to eV)."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        assert dos.energies is not None
        assert len(dos.energies) == 10

    def test_energies_conversion(self):
        """Test energies are converted from Hartree to eV."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        # First energy: -0.5 Hartree = -13.6 eV
        assert dos.energies[0] == pytest.approx(-0.5 * HARTREE_TO_EV)


class TestElkDOSSpin(BaseTest):
    def test_nspin_non_polarized(self):
        """Test nspin is 1 for non-spin-polarized."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        assert dos.nspin == 1

    def test_nspin_polarized(self):
        """Test nspin is 2 for spin-polarized."""
        dos = ElkDOS.from_str(tdos_content=TDOS_SPIN)
        assert dos.nspin == 2


class TestElkDOSTotal(BaseTest):
    def test_total_shape_non_spin(self):
        """Test total DOS shape for non-spin-polarized."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        assert dos.total is not None
        assert dos.total.shape == (10, 1)

    def test_total_shape_spin(self):
        """Test total DOS shape for spin-polarized."""
        dos = ElkDOS.from_str(tdos_content=TDOS_SPIN)
        assert dos.total is not None
        assert dos.total.shape == (5, 2)

    def test_total_values_non_spin(self):
        """Test total DOS values for non-spin-polarized, converted to states/eV."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        assert dos.total[3, 0] == pytest.approx(0.1 / HARTREE_TO_EV)

    def test_spin_down_positive(self):
        """Elk writes spin down negative; the core layout carries it positive."""
        dos = ElkDOS.from_str(tdos_content=TDOS_SPIN)
        assert dos.total[3, 1] == pytest.approx(0.08 / HARTREE_TO_EV)


class TestElkDOSProjected(BaseTest):
    def test_projected_none_without_pdos(self):
        """Test projected is None when no PDOS files."""
        dos = ElkDOS.from_str(tdos_content=TDOS_NON_SPIN)
        assert dos.projected is None

    def test_natoms_with_pdos(self, dos_dir_spin):
        """Test natoms count with PDOS files."""
        dos = ElkDOS(dirpath=dos_dir_spin)
        assert dos.natoms == 2

    def test_projected_shape(self, dos_dir_spin):
        """Test projected DOS shape."""
        dos = ElkDOS(dirpath=dos_dir_spin)
        assert dos.projected is not None
        assert dos.projected.shape == (5, 2, 2, 16)
