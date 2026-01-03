from abc import ABC
from pathlib import Path


class BaseParser(ABC):
    def __init__(self, dirpath: str | Path):
        self.dirpath = Path(dirpath).resolve()

    @property
    def ebs(self):
        pass

    @property
    def dos(self):
        pass

    @property
    def structure(self):
        pass

    @property
    def kpath(self):
        pass
