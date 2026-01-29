from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from . import dbCovalentBond


class DB:
    covalentRadii: dict[str, npt.NDArray[np.floating[Any]]]
    informed_radii: dict[str, float]

    def __init__(self, customRadii: dict[str, float] | None = None) -> None:
        cr = dbCovalentBond.covalent_radii
        cr = cr.replace("-", "nan")
        cr_names = [x.split()[1] for x in cr.split("\n")]
        cr_values = [x.split()[2:] for x in cr.split("\n")]
        # converting the values to an array, in Angstroms
        cr_values_arrays = [np.array(x, dtype=float) / 100 for x in cr_values]
        self.covalentRadii = dict(zip(cr_names, cr_values_arrays))
        # print(self.covalentRadii)

        if customRadii is not None:
            self.informed_radii = customRadii
        else:
            self.informed_radii = {}

        return

    def estimateBond(self, element1: str, element2: str) -> float:
        """Estimates the covalent bond by summing the larger covalent radius
        for each atoms
        """

        if element1 in self.informed_radii:
            radii1: float = self.informed_radii[element1]
        else:
            radii1 = float(np.nanmax(self.covalentRadii[element1]))

        if element2 in self.informed_radii:
            radii2: float = self.informed_radii[element2]
        else:
            radii2 = float(np.nanmax(self.covalentRadii[element2]))

        return radii1 + radii2

    def get_bandwidth(self, element1: str, element2: str) -> float:
        if (element1 == "H") or (element2 == "H"):
            return 0.02
        else:
            return 0.1


atomicDB = DB()
