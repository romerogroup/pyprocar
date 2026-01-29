import logging
from pathlib import Path

import numpy as np
import pytest

from tests.pyprocar.io.abinit import ABINIT_DATA_DIR, CALC_TYPES
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


def get_dos_dirs() -> list[Path]:
    """Get all DOS directories for testing."""
    dirs: list[Path] = []
    for calc_type in CALC_TYPES:
        dirpath = ABINIT_DATA_DIR / calc_type / "dos"
        if dirpath.exists():
            dirs.append(dirpath)
    return dirs


@pytest.fixture(params=get_dos_dirs(), ids=lambda p: p.parent.name)  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
def dos_dirpath(request: pytest.FixtureRequest) -> Path:
    return request.param  # type: ignore[no-any-return]


class TestAbinitDOSInit(BaseTest):
    def test_init_from_dirpath(self, dos_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert dos is not None

    def test_total_dos_filepath_found(self, dos_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert dos.total_dos_filepath is not None


class TestAbinitDOSTotal(BaseTest):
    def test_dos_total_is_ndarray(self, dos_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert isinstance(dos.dos_total, np.ndarray)

    def test_energies_is_ndarray(self, dos_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert isinstance(dos.energies, np.ndarray)

    def test_fermi_is_float(self, dos_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert isinstance(dos.fermi, float)


class TestAbinitDOSProjected(BaseTest):
    def test_projected_is_ndarray_or_none(self, dos_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        projected = dos.projected
        assert projected is None or isinstance(projected, np.ndarray)

    def test_projected_shape_has_4_dimensions(self, dos_dirpath: Path) -> None:
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        projected = dos.projected
        if projected is not None:
            assert len(projected.shape) == 4
