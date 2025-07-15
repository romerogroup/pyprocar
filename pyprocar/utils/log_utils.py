import logging
import logging.config
import os
from datetime import datetime


def set_verbose_level(verbose: int):
    user_logger = logging.getLogger("user")
    package_logger = logging.getLogger("pyprocar")

    if verbose == 0:
        user_logger.setLevel(logging.CRITICAL)
        package_logger.setLevel(logging.CRITICAL)
    elif verbose == 1:
        user_logger.setLevel(logging.DEBUG)
        package_logger.setLevel(logging.CRITICAL)
    elif verbose >= 2:
        user_logger.setLevel(logging.DEBUG)
        package_logger.setLevel(logging.DEBUG)


class UserFriendlyFormatter(logging.Formatter):
    """Custom formatter that makes warnings and errors more noticeable to users"""

    # ANSI color codes for terminal output
    YELLOW = "\033[93m"  # Warning
    RED = "\033[91m"  # Error/Critical
    BOLD = "\033[1m"  # Bold text
    RESET = "\033[0m"  # Reset formatting

    def format(self, record):
        # Default format for regular messages
        self._style._fmt = "%(message)s"

        # Special formatting for warnings and errors
        if record.levelno >= logging.ERROR:
            self._style._fmt = f"{self.RED}{self.BOLD}ERROR: %(message)s{self.RESET}"
        elif record.levelno >= logging.WARNING:
            self._style._fmt = (
                f"{self.YELLOW}{self.BOLD}WARNING: %(message)s{self.RESET}"
            )

        return super().format(record)


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(levelname)s] %(asctime)s - %(name)s[%(lineno)d][%(funcName)s] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "user": {
            "()": UserFriendlyFormatter,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "user_console": {
            "class": "logging.StreamHandler",
            "formatter": "user",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "simple",
            "filename": "pyprocar.log",
            "mode": "a",
        },
    },
    "loggers": {
        "pyprocar": {
            "level": "ERROR",
            "handlers": ["console", "file"],
            "propagate": False,
        },
        "user": {"level": "ERROR", "handlers": ["user_console"], "propagate": False},
        "tests": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
    },
}


def setup_logging():
    logging.config.dictConfig(logging_config)
