# Parser layer: get_parser(code, dirpath) on the VASP bands fixture
# and on the local QE fixture in data/codes.
# Fixture: data/examples/bands/non-spin-polarized (QE dir is read in place; parsers write nothing)
import numpy as np
from verify_steps import CALC, REPO, finish, step

from pyprocar.io import get_parser


def facts(p):
    return {
        "bands_shape": list(np.shape(p.ebs.bands.to_array())),
        "ebs_fermi": float(p.ebs.fermi),
        "kpath_ticks": list(p.kpath.tick_names),
        "species": list(p.structure.species),
        "has_dos": p.dos is not None,
    }


@step("vasp")
def _():
    return facts(get_parser(code="vasp", dirpath=CALC))


@step("qe")
def _():
    return facts(
        get_parser(code="qe", dirpath=REPO / "data/codes/qe/7.2/SrVO3/non-spin-polarized/bands")
    )


finish()
