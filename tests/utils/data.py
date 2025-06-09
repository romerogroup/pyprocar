import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from pyprocar import io
from pyprocar.utils.log_utils import set_verbose_level
from tests.utils import DATA_DIR, BaseTest

logger = logging.getLogger(__name__)

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
