import collections
import gzip
import logging
import re
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class Kpoints(collections.abc.Mapping):
    """
    A class to parse the KPOINTS file

    Parameters
    ----------
    filename : str, optional
        The KPOINTS filename, by default "KPOINTS"
    has_time_reversal : bool, optional
        A boolean vlaue to determine if the kpioints has time reversal symmetry,
        by default True
    """

    def __init__(self, filepath: Union[str, Path], has_time_reversal: bool = True):

        self.filepath = Path(filepath)
        self.file_str = None
        self.metadata = None
        self.mode = None
        self.kgrid = None
        self.kshift = None
        self.ngrids = None
        self.special_kpoints = None
        self.knames = None
        self.cartesian = False
        self.automatic = False
        self.has_time_reversal = has_time_reversal
        self._parse_kpoints()

    def _parse_kpoints(self):
        """A helper method to parse the KOINTS file"""
        with open(self.filepath, "r") as rf:
            self.comment = rf.readline()
            grids = rf.readline()
            grids = grids[: grids.find("!")]
            self.ngrids = [int(x) for x in grids.split()]
            if self.ngrids[0] == 0:
                self.automatic = True
            mode = rf.readline()
            if mode[0].lower() == "m":
                self.mode = "monkhorst-pack"
            elif mode[0].lower() == "g":
                self.mode = "gamma"
            elif mode[0].lower() == "l":
                self.mode = "line"
            if self.mode == "gamma" or self.mode == "monkhorst-pack":
                kgrid = rf.readline()
                self.kgrid = [int(x) for x in kgrid.split()]
                shift = rf.readline()
                self.kshift = [int(float(x)) for x in shift.split()]
                
                if len(self.kshift) == 0:
                    self.kshift = (0, 0, 0)

            elif self.mode == "line":
                if rf.readline()[0].lower() == "c":
                    self.cartesian = True
                else:
                    self.cartesian = False
                self.file_str = rf.read()

                temp = np.array(
                    re.findall(
                        r"([0-9.-]+)\s*([0-9.-]+)\s*([0-9.-]+)(.*)", self.file_str
                    )
                )
                temp_special_kp = temp[:, :3].astype(float)
                temp_knames = temp[:, -1]
                nsegments = temp_special_kp.shape[0] // 2
                if len(self.ngrids) == 1:
                    self.ngrids = [self.ngrids[0]] * nsegments
                self.knames = np.reshape(
                    [x.replace("!", "").strip() for x in temp_knames], (nsegments, 2)
                )
                self.special_kpoints = temp_special_kp.reshape(nsegments, 2, 3)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

