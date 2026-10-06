# File utilities exported at top level: bandgap, kpath, filter, repair, cat, generate2dkmesh,
# spin_asymmetry, download_from_hf, and the scripts/procar.py CLI run in-process.
# Fixture: data/examples/bands/non-spin-polarized (PROCAR, POSCAR, vasprun.xml)
# Outputs land in the calc copy (generate2dkmesh writes ./Kgrid.dat to the CWD, so chdir first).
import os
import runpy
import sys

import numpy as np
from verify_steps import CALC, REPO, finish, step

import pyprocar
from pyprocar.utils.download_examples import download_from_hf  # pyprocar.download_from_hf

os.chdir(CALC)


def head(path, n=4):
    with open(path) as f:
        return [line.rstrip() for line in f.readlines()[:n]]


@step("bandgap")
def _():
    gap = pyprocar.bandgap(dirname=str(CALC), code="vasp")
    return {"gap_eV": float(gap)}


@step("kpath")
def _():
    pyprocar.kpath(infile="POSCAR", outfile="KPOINTS_pyprocar", grid_size=40)
    with open("KPOINTS_pyprocar") as f:
        lines = f.read().split("\n")
    return {"head": lines[:4], "n_lines": len(lines)}


@step("filter_bands")
def _():
    pyprocar.filter(inFile="PROCAR", outFile="PROCAR_bands_1_5", bands=[1, 5])
    return {
        "in_bytes": os.path.getsize("PROCAR"),
        "out_bytes": os.path.getsize("PROCAR_bands_1_5"),
        "head": head("PROCAR_bands_1_5", 2),
    }


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


@step("spin_asymmetry")
def _():
    pyprocar.spin_asymmetry()


@step("download_from_hf_str_output_path")  # the call SKILL.md's Known repo issues names
def _():
    # PROCAR exists in the CWD, so a working call returns early without any network access.
    return {"returned": str(download_from_hf("PROCAR", output_path="."))}


def cli(*argv):
    """Run `python scripts/procar.py <argv>` in this process, so a crash keeps its traceback."""
    saved = sys.argv
    sys.argv = ["procar.py", *argv]
    try:
        runpy.run_path(str(REPO / "scripts/procar.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise RuntimeError(f"procar.py {' '.join(argv)} exited {e.code}") from e
    finally:
        sys.argv = saved


@step("cli_cat_gz")
def _():
    cli("cat", "PROCAR", "PROCAR", "cli_merged", "--gz")
    return {"out_bytes": os.path.getsize("cli_merged.gz")}


@step("cli_filter_atoms")
def _():
    cli("filter", "PROCAR", "cli_atoms", "-a", "0", "1", "-a", "2")
    return {"head": head("cli_atoms", 2)}


@step("cli_bandgap")
def _():
    cli("bandgap", "PROCAR", "OUTCAR", "vasp", "5.3017")


@step("cli_generate2dkmesh")
def _():
    cli("generate2dkmesh", "-0.5", "-0.5", "0.5", "0.5", "0", "5", "4")


finish()
