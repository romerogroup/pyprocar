import logging
from pathlib import Path

import pytest

from tests.pyprocar.io.abinit import ABINIT_DATA_DIR, CALC_TYPES
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


def get_kpoints_files() -> list[Path]:
    """Get all KPOINTS files for testing."""
    files: list[Path] = []
    for calc_type in CALC_TYPES:
        for mode in ["bands", "dos", "fermi"]:
            filepath = ABINIT_DATA_DIR / calc_type / mode / "KPOINTS"
            if filepath.exists():
                files.append(filepath)
    return files


@pytest.fixture(params=get_kpoints_files(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
def kpoints_filepath(request: pytest.FixtureRequest) -> Path:
    return request.param  # type: ignore[no-any-return]


class TestAbinitKpointsInit(BaseTest):
    def test_init_from_filepath(self, kpoints_filepath: Path) -> None:
        from pyprocar.io.abinit import AbinitKpoints

        kpoints = AbinitKpoints(kpoints_filepath)
        assert kpoints is not None


class TestAbinitKpointsInheritance(BaseTest):
    def test_inherits_from_vasp_kpoints(self) -> None:
        from pyprocar.io.abinit import AbinitKpoints
        from pyprocar.io.vasp import Kpoints

        assert issubclass(AbinitKpoints, Kpoints)
