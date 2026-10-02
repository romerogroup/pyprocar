from pathlib import Path

from tests.utils import ROOT_DIR

ABINIT_DATA_DIR = ROOT_DIR / "data" / "codes" / "abinit" / "9.6" / "Fe"

# Calculation types available for testing
CALC_TYPES = [
    "non-spin-polarized",
    "spin-polarized-colinear",
    "non-colinear",
]

# Calculation modes
CALC_MODES = ["bands", "dos", "fermi"]
