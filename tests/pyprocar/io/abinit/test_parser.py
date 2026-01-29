import logging
from pathlib import Path

import pytest

from tests.pyprocar.io.abinit import ABINIT_DATA_DIR, CALC_TYPES
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


def get_all_dirs() -> list[Path]:
    """Get all calculation directories for testing."""
    dirs: list[Path] = []
    for calc_type in CALC_TYPES:
        for mode in ["bands", "dos", "fermi"]:
            dirpath = ABINIT_DATA_DIR / calc_type / mode
            if dirpath.exists():
                dirs.append(dirpath)
    return dirs


@pytest.fixture(params=get_all_dirs(), ids=lambda p: f"{p.parent.name}/{p.name}")  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
def calc_dirpath(request: pytest.FixtureRequest) -> Path:
    return request.param  # type: ignore[no-any-return]


class TestAbinitParserInit(BaseTest):
    def test_init_from_dirpath(self, calc_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitParser

        parser = AbinitParser(calc_dirpath)
        assert parser is not None

    def test_detect_files_runs(self, calc_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitParser

        parser = AbinitParser(calc_dirpath)
        summary = parser.summary()
        assert "files" in summary


class TestAbinitParserOutput(BaseTest):
    def test_abinit_output_detected(self, calc_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitParser

        parser = AbinitParser(calc_dirpath)
        assert parser.abinit_output is not None


class TestAbinitParserStructure(BaseTest):
    def test_structure_returns_structure_object(self, calc_dirpath: Path) -> None:
        from pyprocar.core import Structure
        from pyprocar.io.abinit import AbinitParser

        parser = AbinitParser(calc_dirpath)
        structure = parser.structure
        assert isinstance(structure, Structure)


class TestAbinitParserEBS(BaseTest):
    def test_ebs_for_bands_calculation(self) -> None:
        from pyprocar.io.abinit import AbinitParser

        dirpath = ABINIT_DATA_DIR / "non-spin-polarized" / "bands"
        parser = AbinitParser(dirpath)
        ebs = parser.ebs
        # EBS should be available for bands calculation
        assert ebs is not None or parser.abinit_procar is None


class TestAbinitParserDOS(BaseTest):
    def test_dos_for_dos_calculation(self) -> None:
        from pyprocar.io.abinit import AbinitParser

        dirpath = ABINIT_DATA_DIR / "non-spin-polarized" / "dos"
        parser = AbinitParser(dirpath)
        dos = parser.dos
        # DOS should be available if DOS files exist
        has_dos_files = parser._detected.get("dos_total") is not None
        if has_dos_files:
            assert dos is not None
