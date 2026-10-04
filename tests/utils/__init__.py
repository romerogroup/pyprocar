import logging
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

from tests.utils.base_test import BaseTest

logger = logging.getLogger("pyprocar")
logger.setLevel(logging.DEBUG)


ROOT_DIR = Path(__file__).parent.parent.parent
TEST_DIR = ROOT_DIR / "tests"
DATA_DIR = ROOT_DIR / "data"


def writable_copy(
    src: Path, dst: Path, ignore: Callable[[str, list[str]], set[str]] | None = None
) -> Path:
    """Copy a fixture file or tree to ``dst`` with user write permission on every entry.

    ``verify.sh fetch`` makes fixtures read-only, and ``copytree`` keeps their modes.
    """
    if not src.is_dir():
        return Path(shutil.copyfile(src, dst))
    shutil.copytree(src, dst, ignore=ignore)
    for path in [dst, *dst.rglob("*")]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    return dst
