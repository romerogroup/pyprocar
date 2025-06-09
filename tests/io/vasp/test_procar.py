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

PROCAR_DATA_DIR = DATA_DIR / "io" / "vasp" / "procar"


@pytest.fixture(
    params=[
        PROCAR_DATA_DIR / "PROCAR_non-colinear",
        PROCAR_DATA_DIR / "PROCAR_non-spin-polarized",
        PROCAR_DATA_DIR / "PROCAR_spin-polarized",
    ]
)
def outcar_filepath(request):
    """Fixture that provides OUTCAR file paths for testing."""
    return request.param


class TestProcar(BaseTest):
    """Test class for VASP Procar file parsing."""

    def test_procar_parser(self):
        """Test parsing of Procar file."""
        procar = vasp.Procar(PROCAR_DATA_DIR / "PROCAR")
