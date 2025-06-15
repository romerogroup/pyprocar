import json
import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import vasp
from pyprocar.utils.log_utils import set_verbose_level
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)

OUTCAR_DATA_DIR = DATA_DIR / "io" / "vasp" / "outcar"


def get_test_id(outcar_filepath: Path) -> str:
    """Creates a nice, readable ID for each test run."""
    return f"{outcar_filepath.stem}"


outcar_files = []
for filepath in OUTCAR_DATA_DIR.glob("OUTCAR_*"):
    print(filepath)
    suffix = filepath.suffix
    if suffix in [".json", ".py"]:
        continue
    outcar_files.append(filepath)


@pytest.fixture(
    params=outcar_files,
    ids=get_test_id,
)
def outcar_filepath(request):
    """Fixture that provides OUTCAR file paths for testing."""
    return request.param


class TestOutcarBase(BaseTest):
    def test_symmetry_operations(self, outcar_filepath):
        outcar = vasp.Outcar(outcar_filepath)
        outcar_name = outcar_filepath.stem
        expected_filename = f"{outcar_name}_expected.json"

        outcar_dict = outcar.to_dict()
        with open(OUTCAR_DATA_DIR / expected_filename) as f:
            expected_outcar_dict = json.load(f)

        for key in expected_outcar_dict.keys():
            assert outcar_dict[key] == expected_outcar_dict[key]

    def test_attrs(self, outcar_filepath):
        outcar = vasp.Outcar(outcar_filepath)

        assert isinstance(outcar.version_tuple, tuple)
        assert len(outcar.version_tuple) == 3

        assert isinstance(outcar.efermi, float)
        assert isinstance(outcar.reciprocal_lattice, np.ndarray)
        assert isinstance(outcar.rotations, np.ndarray)
        assert isinstance(outcar.get_symmetry_operations(), list)
        assert isinstance(outcar.to_dict(), dict)
        assert "efermi" in outcar
        assert outcar["efermi"] == outcar.efermi


# class TestVaspVersions(TestOutcarBase):
#     """
#     This class inherits the tools from TestOutcarBase and uses them
#     to run tests on specific data sets.
#     """

#     @pytest.mark.parametrize("outcar_version", VASP_VERSIONS_TO_TEST)
#     def test_parser_for_version(self, outcar_version):
#         """
#         This single test method will be run multiple times by pytest, once for
#         each directory specified in VASP_VERSIONS_TO_TEST.
#         """

#         self.test_symmetry_operations(outcar_version)
#         self.test_attrs(outcar_version)
