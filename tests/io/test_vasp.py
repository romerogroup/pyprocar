import json
import logging
from pathlib import Path

import pytest

from pyprocar.io import vasp
from pyprocar.utils.log_utils import set_verbose_level

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTCAR_DATA_DIR = DATA_DIR / "io" / "vasp" / "outcar"


class TestOUTCAR:
    def test_outcar_v54(self):
        outcar = vasp.Outcar(OUTCAR_DATA_DIR / "OUTCAR_v54")

        outcar_dict = outcar.to_dict()
        with open(OUTCAR_DATA_DIR / "OUTCAR_v54_expected.json") as f:
            expected_outcar_dict = json.load(f)

        for key in expected_outcar_dict.keys():
            assert outcar_dict[key] == expected_outcar_dict[key]

    def test_outcar_v62(self):
        outcar = vasp.Outcar(OUTCAR_DATA_DIR / "OUTCAR_v62")

        outcar_dict = outcar.to_dict()
        with open(OUTCAR_DATA_DIR / "OUTCAR_v62_expected.json") as f:
            expected_outcar_dict = json.load(f)

        for key in expected_outcar_dict.keys():
            assert outcar_dict[key] == expected_outcar_dict[key]

    def test_outcar_v64(self):
        outcar = vasp.Outcar(OUTCAR_DATA_DIR / "OUTCAR_v64")

        outcar_dict = outcar.to_dict()
        with open(OUTCAR_DATA_DIR / "OUTCAR_v64_expected.json") as f:
            expected_outcar_dict = json.load(f)

        for key in expected_outcar_dict.keys():
            assert outcar_dict[key] == expected_outcar_dict[key]
