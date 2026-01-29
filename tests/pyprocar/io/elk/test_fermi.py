"""Tests for ElkFermi extractor."""

import logging
from pathlib import Path

import pytest

from pyprocar.io.elk.fermi import HARTREE_TO_EV, ElkFermi
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


# Example EFERMI.OUT content from non-spin-polarized calculation
EFERMI_NON_SPIN = """  0.3218543102
"""

# Example EFERMI.OUT content from spin-polarized calculation
EFERMI_SPIN_POLARIZED = """  0.3378291969
"""


@pytest.fixture
def efermi_non_spin(tmp_path: Path) -> Path:
    """Create a temporary EFERMI.OUT file for non-spin-polarized case."""
    efermi_file = tmp_path / "EFERMI.OUT"
    efermi_file.write_text(EFERMI_NON_SPIN)
    return efermi_file


@pytest.fixture
def efermi_spin_polarized(tmp_path: Path) -> Path:
    """Create a temporary EFERMI.OUT file for spin-polarized case."""
    efermi_file = tmp_path / "EFERMI.OUT"
    efermi_file.write_text(EFERMI_SPIN_POLARIZED)
    return efermi_file


class TestElkFermiInit(BaseTest):
    def test_fermi_from_filepath_non_spin(self, efermi_non_spin: Path) -> None:
        """Test loading ElkFermi from EFERMI.OUT file."""
        fermi = ElkFermi(efermi_non_spin)
        assert fermi is not None
        assert fermi.filepath == efermi_non_spin

    def test_fermi_from_filepath_spin_polarized(self, efermi_spin_polarized: Path) -> None:
        """Test loading ElkFermi from EFERMI.OUT file."""
        fermi = ElkFermi(efermi_spin_polarized)
        assert fermi is not None
        assert fermi.filepath == efermi_spin_polarized

    def test_fermi_from_str(self) -> None:
        """Test loading ElkFermi from string content."""
        fermi = ElkFermi.from_str(EFERMI_NON_SPIN)
        assert fermi is not None
        assert fermi.filepath is None


class TestElkFermiValues(BaseTest):
    def test_fermi_hartree_value_non_spin(self) -> None:
        """Test that Fermi energy in Hartree is parsed correctly."""
        fermi = ElkFermi.from_str(EFERMI_NON_SPIN)
        assert fermi.fermi_hartree == pytest.approx(0.3218543102)

    def test_fermi_hartree_value_spin_polarized(self) -> None:
        """Test that Fermi energy in Hartree is parsed correctly."""
        fermi = ElkFermi.from_str(EFERMI_SPIN_POLARIZED)
        assert fermi.fermi_hartree == pytest.approx(0.3378291969)

    def test_fermi_ev_conversion(self) -> None:
        """Test that Fermi energy is correctly converted to eV."""
        fermi = ElkFermi.from_str(EFERMI_NON_SPIN)
        expected_ev = 0.3218543102 * HARTREE_TO_EV
        assert fermi.fermi_ev == pytest.approx(expected_ev)

    def test_fermi_ev_value_non_spin(self) -> None:
        """Test specific eV value for non-spin-polarized case."""
        fermi = ElkFermi.from_str(EFERMI_NON_SPIN)
        # 0.3218543102 Hartree * 27.211386245988 = 8.757... eV
        assert fermi.fermi_ev == pytest.approx(8.7573, rel=1e-3)
