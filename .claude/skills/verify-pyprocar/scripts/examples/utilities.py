# File utilities exported at top level: bandgap, kpath, filter, repair, cat, generate2dkmesh.
# Fixture: data/examples/bands/non-spin-polarized (PROCAR, POSCAR, vasprun.xml)
# Outputs land in the calc copy (generate2dkmesh writes ./Kgrid.dat to the CWD, so chdir first).
import os

import numpy as np
from verify_steps import CALC, finish, step

import pyprocar

os.chdir(CALC)


def head(path, n=4):
    return [l.rstrip() for l in open(path).readlines()[:n]]


@step("bandgap")
def _():
    gap = pyprocar.bandgap(dirname=str(CALC), code="vasp")
    return {"gap_eV": float(gap)}


@step("kpath")
def _():
    pyprocar.kpath(infile="POSCAR", outfile="KPOINTS_pyprocar", grid_size=40)
    lines = open("KPOINTS_pyprocar").read().split("\n")
    return {"head": lines[:4], "n_lines": len(lines)}


@step("filter_bands")
def _():
    pyprocar.filter(inFile="PROCAR", outFile="PROCAR_bands_1_5", bands=[1, 5])
    return {"in_bytes": os.path.getsize("PROCAR"), "out_bytes": os.path.getsize("PROCAR_bands_1_5"),
            "head": head("PROCAR_bands_1_5", 2)}


@step("repair")
def _():
    pyprocar.repair(infile="PROCAR", outfile="PROCAR_repaired")
    return {"in_bytes": os.path.getsize("PROCAR"), "out_bytes": os.path.getsize("PROCAR_repaired")}


@step("cat")
def _():
    pyprocar.cat(inFiles=["PROCAR", "PROCAR"], outFile="PROCAR_merged")
    return {"in_bytes": os.path.getsize("PROCAR"), "out_bytes": os.path.getsize("PROCAR_merged")}


@step("generate2dkmesh")
def _():
    k = pyprocar.generate2dkmesh(-0.5, -0.5, 0.5, 0.5, 0.0, 5, 4)
    return {"shape": list(np.shape(k)), "head": head("Kgrid.dat", 4)}


finish()
