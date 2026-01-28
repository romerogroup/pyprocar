__author__ = "Logan Lang"
__maintainer__ = "Logan Lang"
__email__ = "lllang@mix.wvu.edu"
__date__ = "March 31, 2020"

import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Any

from pyprocar.core.atomic_orbital_index import OrbitalIndexer
from pyprocar.io.qe.utils import parse_qe_input_cards

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

FLOAT_PATTERN = r"[-+]?\d+(?:\.\d+)?"
COORDS_PATTERN = rf"\s*({FLOAT_PATTERN})\s*({FLOAT_PATTERN})\s*({FLOAT_PATTERN})\s*"

ORBITAL_ORDERING = OrbitalIndexer()


def convert_lorbnum_to_letter(lorbnum: int) -> str:
    """A helper method to convert the lorb number to the letter format

    Parameters
    ----------
    lorbnum : int
        The number of the l orbital

    Returns
    -------
    str
        The l orbital name
    """
    lorb_mapping = {0: "s", 1: "p", 2: "d", 3: "f"}
    return lorb_mapping[lorbnum]


class ProjwfcIn:
    """Holds the projwfc input file."""

    @classmethod
    def is_file_of_type(cls, filepath: str | Path) -> bool:
        """Quickly determine if an input file looks like a projwfc.x input.

        Checks the beginning of the file for a 'projwfc' token (case-insensitive)
        in comments or the file body. Also accepts presence of typical projwfc
        variables like 'outdir' and 'prefix' in a minimal input.
        """
        try:
            p = Path(filepath)
            with p.open("r", errors="ignore") as f:
                head_lines = [f.readline() for _ in range(50)]
            head = "".join(head_lines)
            if not head:
                return False
            if re.search(r"projwfc", head, re.IGNORECASE):
                return True
            return False
        except Exception:
            return False

    _filepath: Path | None
    _text: str | None

    def __init__(self, filepath: str | Path) -> None:
        self._filepath = Path(filepath)
        self._text = self._read()

    def _read(self) -> str | None:
        if self.filepath is None:
            return None
        with open(self.filepath) as f:
            text = f.read()
        return text

    @property
    def filepath(self) -> Path | None:
        return self._filepath

    @property
    def text(self) -> str | None:
        return self._text

    @cached_property
    def data(self) -> dict[str, Any]:
        if self.text is None:
            return {}
        data = parse_qe_input_cards(self.text)
        logger.info(f"PROJWFC INPUT: {data}")
        return data

    @cached_property
    def is_kresolved(self) -> bool:
        if "kresolveddos" in self.data:
            return self.data["kresolveddos"] == True
        else:
            return False
