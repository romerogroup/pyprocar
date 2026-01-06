"""Tests for BXSF parser."""

from pathlib import Path

import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE
from pyprocar.io.bxsf import BxsfParser


BXSF_STR = """\
BEGIN_INFO
  Fermi Energy: 5.5000
END_INFO

BEGIN_BLOCK_BANDGRID_3D
  fermi_surface
  BEGIN_BANDGRID_3D_fermi
    1
    3 3 3
    0.0 0.0 0.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    BAND: 1
    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0

    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0

    1.0 2.0 3.0
    4.0 5.0 6.0
    7.0 8.0 9.0
  END_BANDGRID_3D_fermi
END_BLOCK_BANDGRID_3D
"""


@pytest.fixture
def bxsf_dir(tmp_path: Path) -> Path:
    """Create temporary directory with BXSF file."""
    filepath = tmp_path / "in.bxsf"
    filepath.write_text(BXSF_STR)
    return tmp_path


class TestBxsfParser:
    """Tests for BxsfParser class."""

    def test_parser_creates_ebs(self, bxsf_dir: Path) -> None:
        """Test parser creates ElectronicBandStructureMesh."""
        parser = BxsfParser(bxsf_dir)
        ebs = parser.ebs
        assert ebs is not None
        assert isinstance(ebs, ElectronicBandStructureMesh)

    def test_parser_kgrid_info(self, bxsf_dir: Path) -> None:
        """Test parser provides kgrid_info."""
        parser = BxsfParser(bxsf_dir)
        assert parser.kgrid_info is not None
        assert parser.kgrid_info.kgrid == (2, 2, 2)
        assert parser.kgrid_info.kgrid_mode == KGRID_MODE.GAMMA

    def test_parser_fermi_energy(self, bxsf_dir: Path) -> None:
        """Test EBS has correct Fermi energy."""
        parser = BxsfParser(bxsf_dir)
        ebs = parser.ebs
        assert ebs is not None
        assert ebs.fermi == pytest.approx(5.5)

    def test_parser_returns_none_properties(self, bxsf_dir: Path) -> None:
        """Test parser returns None for unsupported properties."""
        parser = BxsfParser(bxsf_dir)
        assert parser.kpath is None
        assert parser.structure is None
        assert parser.dos is None


class TestBxsfParserMissingFiles:
    """Tests for BxsfParser with missing files."""

    def test_missing_file_returns_none_ebs(self, tmp_path: Path) -> None:
        """Test parser handles missing file gracefully."""
        parser = BxsfParser(tmp_path, filepaths="nonexistent.bxsf")
        assert parser.ebs is None

    def test_missing_file_returns_none_kgrid_info(self, tmp_path: Path) -> None:
        """Test parser returns None kgrid_info for missing file."""
        parser = BxsfParser(tmp_path, filepaths="nonexistent.bxsf")
        assert parser.kgrid_info is None


class TestBxsfParserFromStr:
    """Tests for BxsfParser.from_str()."""

    def test_from_str_creates_ebs(self) -> None:
        """Test from_str creates working parser."""
        parser = BxsfParser.from_str(BXSF_STR)
        ebs = parser.ebs
        assert ebs is not None
        assert isinstance(ebs, ElectronicBandStructureMesh)

    def test_from_str_multiple_files(self) -> None:
        """Test from_str with multiple file strings."""
        parser = BxsfParser.from_str(BXSF_STR)
        assert len(parser._extractors) == 1


class TestBxsfParserCustomFilepath:
    """Tests for BxsfParser with custom filepaths."""

    def test_custom_filepath(self, tmp_path: Path) -> None:
        """Test parser with custom filepath."""
        filepath = tmp_path / "custom.bxsf"
        filepath.write_text(BXSF_STR)

        parser = BxsfParser(tmp_path, filepaths="custom.bxsf")
        assert parser.ebs is not None


class TestBxsfParserIntegration:
    """Integration tests via Parser factory."""

    def test_parser_factory(self, bxsf_dir: Path) -> None:
        """Test BXSF is accessible via Parser(code='bxsf', ...)."""
        from pyprocar.io import Parser

        parser = Parser(code="bxsf", dirpath=bxsf_dir)
        assert parser.ebs is not None
