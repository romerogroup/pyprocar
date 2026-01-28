import logging

import numpy as np
import pytest

from tests.pyprocar.io.abinit import ABINIT_DATA_DIR, CALC_TYPES
from tests.utils import BaseTest

logger = logging.getLogger(__name__)


def get_bands_dirs():
    """Get all bands directories for testing (contain PROCAR files)."""
    dirs = []
    for calc_type in CALC_TYPES:
        dirpath = ABINIT_DATA_DIR / calc_type / "bands"
        if dirpath.exists():
            dirs.append(dirpath)
    return dirs


@pytest.fixture(params=get_bands_dirs(), ids=lambda p: p.parent.name)
def bands_dirpath(request):
    return request.param


class TestAbinitProcarInit(BaseTest):
    def test_init_from_dirpath(self, bands_dirpath):
        from pyprocar.io.abinit import AbinitOutput, AbinitProcar

        output = AbinitOutput(bands_dirpath / "abinit.out")
        procar = AbinitProcar(dirpath=bands_dirpath, abinit_output=output)
        assert procar is not None


class TestAbinitProcarMerge(BaseTest):
    def test_merge_creates_procar_file(self, bands_dirpath):
        from pyprocar.io.abinit import AbinitOutput, AbinitProcar

        output = AbinitOutput(bands_dirpath / "abinit.out")
        procar = AbinitProcar(dirpath=bands_dirpath, abinit_output=output)

        merged_file = bands_dirpath / "PROCAR"
        assert merged_file.exists()

    def test_vasp_procar_is_parsed(self, bands_dirpath):
        from pyprocar.io.abinit import AbinitOutput, AbinitProcar

        output = AbinitOutput(bands_dirpath / "abinit.out")
        procar = AbinitProcar(dirpath=bands_dirpath, abinit_output=output)

        assert procar.vasp_procar is not None


class TestAbinitProcarData(BaseTest):
    def test_kpoints_is_ndarray(self, bands_dirpath):
        from pyprocar.io.abinit import AbinitOutput, AbinitProcar

        output = AbinitOutput(bands_dirpath / "abinit.out")
        procar = AbinitProcar(dirpath=bands_dirpath, abinit_output=output)

        assert isinstance(procar.vasp_procar.kpoints, np.ndarray)

    def test_bands_is_ndarray(self, bands_dirpath):
        from pyprocar.io.abinit import AbinitOutput, AbinitProcar

        output = AbinitOutput(bands_dirpath / "abinit.out")
        procar = AbinitProcar(dirpath=bands_dirpath, abinit_output=output)

        assert isinstance(procar.vasp_procar.bands, np.ndarray)
