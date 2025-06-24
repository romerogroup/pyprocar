from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class BaseParser(ABC):
    def __init__(self, dirpath: Union[str, Path]):
        self.dirpath = Path(dirpath).resolve()
        
    @abstractmethod
    def version(self):
        pass
    
    @abstractmethod
    def version_tuple(self):
        pass

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

    
    
