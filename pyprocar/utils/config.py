from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


class ConfigManager:
    """Manages YAML-based configuration files."""

    config: dict[str, dict[str, object]]

    def __init__(self, config_path: str) -> None:
        self.config = self._load_config(config_path)

    def _load_config(self, file_path: str) -> dict[str, dict[str, object]]:
        """Load configuration from a YAML file."""
        with open(file_path) as file:
            return yaml.safe_load(file)

    def update_config(self, new_config: dict[str, object]) -> None:
        """Update the current configuration with the provided dictionary."""
        for key, value in new_config.items():
            self.config[key]["value"] = value

    def get_config(self) -> dict[str, dict[str, object]]:
        """Retrieve the current configuration."""
        return self.config


# Important directory paths
FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])  # pyprocar
ROOT = str(FILE.parents[2])  # PyProcar
LOG_DIR = str(Path(ROOT) / "logs")
data_dir: str | None = os.getenv("DATA_DIR")

if data_dir is None:
    data_dir = str(Path(ROOT) / "data")

DATA_DIR: str = data_dir


CONFIG_FILE = str(Path(PKG_DIR) / "cfg" / "package.yml")

# Load config from yaml file
with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)
