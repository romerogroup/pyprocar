import os
import shutil
import sys
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

CURRENT_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR = CURRENT_DIR.parent.parent
TEST_DIR = ROOT_DIR / "tests"
DATA_DIR = TEST_DIR / "data"

sys.path.append(str(ROOT_DIR))

CODES_DIR = ROOT_DIR / "codes"
