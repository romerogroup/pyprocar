"""Tests for FrmSrf parser."""

from pathlib import Path

import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE
from pyprocar.io.frmsf import FrmsfParser


# FrmSrf file: 2x2x2 k-grid, 2 bands, method 1 (gamma-centered)
FRMSF_STR = """\
2 2 2
1
2
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0
"""


# FrmSrf file with Monkhorst-Pack method
FRMSF_STR_MP = """\
2 2 2
0
1
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
"""


@pytest.fixture
def frmsf_dir(tmp_path: Path) -> Path:
    """Create temporary directory with FrmSrf file."""
    filepath = tmp_path / "in.frmsf"
    filepath.write_text(FRMSF_STR)
    return tmp_path


@pytest.fixture
def frmsf_dir_mp(tmp_path: Path) -> Path:
    """Create temporary directory with Monkhorst-Pack FrmSrf file."""
    filepath = tmp_path / "in.frmsf"
    filepath.write_text(FRMSF_STR_MP)
    return tmp_path


class TestFrmsfParser:
    """Tests for FrmsfParser class."""

    def test_parser_creates_ebs(self, frmsf_dir: Path) -> None:
        """Test parser creates ElectronicBandStructureMesh."""
        parser = FrmsfParser(frmsf_dir)
        ebs = parser.ebs
        assert ebs is not None
        assert isinstance(ebs, ElectronicBandStructureMesh)

    def test_parser_kgrid_info_gamma(self, frmsf_dir: Path) -> None:
        """Test parser provides kgrid_info for gamma-centered grid."""
        parser = FrmsfParser(frmsf_dir)
        assert parser.kgrid_info is not None
        assert parser.kgrid_info.kgrid == (2, 2, 2)
        assert parser.kgrid_info.kgrid_mode == KGRID_MODE.GAMMA

    def test_parser_kgrid_info_monkhorst(self, frmsf_dir_mp: Path) -> None:
        """Test parser provides kgrid_info for Monkhorst-Pack grid."""
        parser = FrmsfParser(frmsf_dir_mp)
        assert parser.kgrid_info is not None
        assert parser.kgrid_info.kgrid_mode == KGRID_MODE.MONKHORST

    def test_parser_fermi_is_zero(self, frmsf_dir: Path) -> None:
        """Test EBS Fermi energy is 0.0 (FrmSrf doesn't provide it)."""
        parser = FrmsfParser(frmsf_dir)
        ebs = parser.ebs
        assert ebs is not None
        assert ebs.fermi == pytest.approx(0.0)

    def test_parser_returns_none_properties(self, frmsf_dir: Path) -> None:
        """Test parser returns None for unsupported properties."""
        parser = FrmsfParser(frmsf_dir)
        assert parser.kpath is None
        assert parser.structure is None
        assert parser.dos is None


class TestFrmsfParserMissingFiles:
    """Tests for FrmsfParser with missing files."""

    def test_missing_file_returns_none_ebs(self, tmp_path: Path) -> None:
        """Test parser handles missing file gracefully."""
        parser = FrmsfParser(tmp_path, filepath="nonexistent.frmsf")
        assert parser.ebs is None

    def test_missing_file_returns_none_kgrid_info(self, tmp_path: Path) -> None:
        """Test parser returns None kgrid_info for missing file."""
        parser = FrmsfParser(tmp_path, filepath="nonexistent.frmsf")
        assert parser.kgrid_info is None


class TestFrmsfParserFromStr:
    """Tests for FrmsfParser.from_str()."""

    def test_from_str_creates_ebs(self) -> None:
        """Test from_str creates working parser."""
        parser = FrmsfParser.from_str(FRMSF_STR)
        ebs = parser.ebs
        assert ebs is not None
        assert isinstance(ebs, ElectronicBandStructureMesh)

    def test_from_str_dirpath_is_empty(self) -> None:
        """Test from_str sets dirpath to empty Path."""
        parser = FrmsfParser.from_str(FRMSF_STR)
        assert parser.dirpath == Path("")


class TestFrmsfParserCustomFilepath:
    """Tests for FrmsfParser with custom filepath."""

    def test_custom_filepath(self, tmp_path: Path) -> None:
        """Test parser with custom filepath."""
        filepath = tmp_path / "custom.frmsf"
        filepath.write_text(FRMSF_STR)

        parser = FrmsfParser(tmp_path, filepath="custom.frmsf")
        assert parser.ebs is not None


class TestFrmsfParserIntegration:
    """Integration tests via Parser factory."""

    def test_parser_factory(self, frmsf_dir: Path) -> None:
        """Test FrmSrf is accessible via Parser(code='frmsf', ...)."""
        from pyprocar.io import Parser

        parser = Parser(code="frmsf", dirpath=frmsf_dir)
        assert parser.ebs is not None
