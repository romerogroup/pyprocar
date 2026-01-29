#!/usr/bin/env python3
import re


class outcarParser:
    def __init__(self, OUTCAR: str) -> None:
        # May be some important values left adding for predicting user intention
        # Currently Finding: ISPIN, NKPOINTS, ORBTIAL_MAG, NBANDS, EMIN, EMAX

        outcar_path = str(OUTCAR)
        f = open(outcar_path)
        self.outcar: list[str] = f.readlines()
        # ISPIN
        self.ISPIN: int = int(re.findall(r"ISPIN\s*=\s*(\d*)", "".join(self.outcar))[0])
        print("ISPIN", self.ISPIN)

        # NKPOINTS
        self.NKPOINTS: int = int(re.findall(r"NKPTS\s*=\s*(\d*)", "".join(self.outcar))[0])
        print("NKPOINTS", self.NKPOINTS)

        # MAGMOM(?)

        # ORBITAL MAG(?)
        self.ORBITAL_MAG: str = re.findall(r"ORBITALMAG\s*=\s*(\S*)", "".join(self.outcar))[0]
        print("ORBITAL_MAG", self.ORBITAL_MAG)

        # NBANDS(?)
        self.NBANDS: int = int(re.findall(r"NBANDS\s*=\s*(\d*)", "".join(self.outcar))[0])
        print("NBANDS", self.NBANDS)

        # EMIN EMAX(?) dont know if to reverse the values for bandplots, this values are for DOS
        self.EMIN: int = int(re.findall(r"EMIN\s*=\s*(-*\d*)", "".join(self.outcar))[0])
        print("EMIN", self.EMIN)
        self.EMAX: int = int(re.findall(r"EMAX\s*=\s*(-*\d*)", "".join(self.outcar))[0])
        print("EMAX", self.EMAX)

    def kpointPlotState(self) -> str | None:
        if self.NKPOINTS == 0:
            state = None
        if self.NKPOINTS == 1:
            state = "atomic"

        # Here I should also check if it is a KPOINT mesh or a path
        # Need to know where to find path or mesh if added
        else:
            state = "parametric"

        return state

    def IsMagnetic(self) -> bool:
        if self.ISPIN == 0:
            value = False
        elif self.ISPIN == 1:
            value = True
        else:
            value = False
        return value
