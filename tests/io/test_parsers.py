import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from pyprocar import io
from pyprocar.core import DensityOfStates, ElectronicBandStructure, Structure
from pyprocar.utils.log_utils import set_verbose_level
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

VERBOSE = 2
set_verbose_level(VERBOSE)


class CalcType(Enum):
    DOS = "dos"
    BANDS = "bands"
    FERMI = "fermi"
    FERMI_PLANE = "fermi_plane"


class MagType(Enum):
    NON_COLINEAR = "non-colinear"
    SPIN_POLARIZED = "spin-polarized"
    NON_SPIN_POLARIZED = "non-spin-polarized"


class Code(Enum):
    VASP = "vasp"
    ABINIT = "abinit"
    ELK = "elk"
    QE = "qe"
    SIESTA = "siesta"


@dataclass
class CalcInfo:
    """A simple data class to hold the parameters for each test case."""

    path: Path
    mat_system: str
    code: str
    version: str
    mag_type: str
    calc_type: str

    def get_id(self) -> str:
        return f"{self.code}-{self.version}-{self.mat_system}-{self.mag_type}-{self.calc_type}"


def find_test_cases(data_root: Path) -> list[CalcInfo]:
    """
    Scans the data directory to find all valid test calculation paths
    and returns a list of structured CalcInfo objects.
    """
    test_cases = []
    # The glob pattern matches the 5 levels of your directory structure
    # {mat_system}/{code}/{version}/{mag_type}/{calc_type}
    for path in data_root.glob("*/*/*/*/*"):
        if path.is_dir():
            parts = path.relative_to(data_root).parts
            info = CalcInfo(
                path=path,
                code=parts[0],
                version=parts[1],
                mat_system=parts[2],
                mag_type=parts[3],
                calc_type=parts[4],
            )
            test_cases.append(info)
    return test_cases


ALL_TEST_CASES = find_test_cases(DATA_DIR / "codes")
EBS_CALC_TYPES = [
    CalcType.FERMI.value,
    CalcType.BANDS.value,
    CalcType.FERMI_PLANE.value,
]
DOS_CALC_TYPES = [
    CalcType.DOS.value,
]


def get_test_id(calc_info: CalcInfo) -> str:
    """Creates a nice, readable ID for each test run."""
    return f"{calc_info.code}-{calc_info.version}-{calc_info.mat_system}-{calc_info.mag_type}-{calc_info.calc_type}"


@pytest.fixture(params=ALL_TEST_CASES, ids=get_test_id)
def calc_test_case(request):
    """
    This is the parameterized fixture. Pytest will run any test that
    uses this fixture once for each item in ALL_TEST_CASES.

    The `request.param` object will be one CalcInfo instance at a time.
    """
    return request.param


class TestParsers(BaseTest):
    """Test class for VASP Procar file parsing."""

    def test_dos(self, calc_test_case: CalcInfo):
        """Test parsing of Procar file."""
        # You can now easily access all the parameters for the current test run
        print(
            f"\nTesting: {calc_test_case.mat_system} "
            f"({calc_test_case.code} v{calc_test_case.version}) - "
            f"{calc_test_case.mag_type}/{calc_test_case.calc_type}"
        )

        parser = io.Parser(
            code=calc_test_case.code,
            dirpath=calc_test_case.path,
        )
        dos_calc_types = [CalcType.DOS.value]
        if calc_test_case.calc_type not in dos_calc_types:
            return None

        dos = parser.dos
        structure = parser.structure

        assert dos is not None
        assert structure is not None

        expected_dos = DensityOfStates.load(calc_test_case.path / "dos.pkl")
        expected_structure = Structure.load(calc_test_case.path / "structure.pkl")

        assert dos == expected_dos
        assert structure == expected_structure

    def test_ebs(self, calc_test_case: CalcInfo):
        """Test parsing of Procar file."""
        # You can now easily access all the parameters for the current test run
        print(
            f"\nTesting: {calc_test_case.mat_system} "
            f"({calc_test_case.code} v{calc_test_case.version}) - "
            f"{calc_test_case.mag_type}/{calc_test_case.calc_type}"
        )

        parser = io.Parser(
            code=calc_test_case.code,
            dirpath=calc_test_case.path,
        )

        ebs_calc_types = [
            CalcType.FERMI.value,
            CalcType.BANDS.value,
            CalcType.FERMI_PLANE.value,
        ]
        if calc_test_case.calc_type not in ebs_calc_types:
            return None

        ebs = parser.ebs
        structure = parser.structure

        assert ebs is not None
        assert structure is not None

        expected_ebs = ElectronicBandStructure.load(calc_test_case.path / "ebs.pkl")
        expected_structure = Structure.load(calc_test_case.path / "structure.pkl")

        assert ebs == expected_ebs
        assert structure == expected_structure

        if calc_test_case.calc_type == CalcType.BANDS.value:
            assert ebs.kpath is not None

    def test_fermi2d(self, calc_test_case: CalcInfo):
        """Test parsing of Procar file."""
        # You can now easily access all the parameters for the current test run
        print(
            f"\nTesting: {calc_test_case.mat_system} "
            f"({calc_test_case.code} v{calc_test_case.version}) - "
            f"{calc_test_case.mag_type}/{calc_test_case.calc_type}"
        )
