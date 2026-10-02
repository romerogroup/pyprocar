import logging
from pathlib import Path

from tests.utils.base_test import BaseTest

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)


ROOT_DIR = Path(__file__).parent.parent.parent
TEST_DIR = ROOT_DIR / "tests"
DATA_DIR = ROOT_DIR / "data"
