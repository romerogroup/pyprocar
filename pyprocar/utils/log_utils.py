import logging
import logging.config
import os
import sys
import sysconfig
import warnings
from types import FrameType

_PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__)) + os.sep
_STDLIB_DIRS = tuple({os.path.join(sysconfig.get_path(k), "") for k in ("stdlib", "platstdlib")})
_SITE_DIRS = tuple({os.path.join(sysconfig.get_path(k), "") for k in ("purelib", "platlib")})


def _is_library_file(filename: str) -> bool:
    if filename.startswith((_PACKAGE_DIR, "<frozen ", "<string>")):
        return True
    return filename.startswith(_STDLIB_DIRS) and not filename.startswith(_SITE_DIRS)


def warn_user(message: str) -> None:
    frame = sys._getframe(1)
    nearest_outside: FrameType | None = None
    while _is_library_file(filename := frame.f_code.co_filename):
        if nearest_outside is None and not filename.startswith((_PACKAGE_DIR, "<")):
            nearest_outside = frame
        if frame.f_back is None:
            # A thread's stack holds no caller frame; name the line that called into pyprocar.
            frame = nearest_outside or frame
            break
        frame = frame.f_back
    module_globals = frame.f_globals
    # No module_globals: CPython would load the source through __spec__.loader, which
    # IPython's namespace and a script's __main__ lack (ValueError, DeprecationWarning).
    warnings.warn_explicit(
        message,
        UserWarning,
        frame.f_code.co_filename,
        frame.f_lineno,
        module=module_globals.get("__name__", "<string>"),
        registry=module_globals.setdefault("__warningregistry__", {}),
    )


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


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(levelname)s] %(asctime)s - %(name)s[%(lineno)d][%(funcName)s] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "user": {"format": "%(message)s"},
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
            "level": "INFO",
            "handlers": ["file"],
            "propagate": False,
        },
        "user": {"level": "ERROR", "handlers": ["user_console"], "propagate": False},
        "tests": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
    },
}


def setup_logging():
    logging.config.dictConfig(logging_config)
