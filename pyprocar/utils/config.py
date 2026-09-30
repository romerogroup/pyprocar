import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


# Important directory paths
FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])  # pyprocar
ROOT = str(FILE.parents[2])  # PyProcar
LOG_DIR = os.path.join(ROOT, "logs")
DATA_DIR = os.getenv("DATA_DIR")

if DATA_DIR is None:
    DATA_DIR = os.path.join(ROOT, "data")


CONFIG_FILE = os.path.join(PKG_DIR, "cfg", "package.yml")

# Load config from yaml file
with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)
