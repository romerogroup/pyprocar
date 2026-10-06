# Parser layer: get_parser(code, dirpath) on the VASP bands fixture
# and on the local QE, Elk and Abinit fixtures in data/codes.
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


CODES = REPO / "data/codes"


@step("qe_non_colinear")
def _():
    p = get_parser(code="qe", dirpath=CODES / "qe/7.2/SrVO3/non-colinear/bands")
    ebs = p.ebs
    assert ebs is not None and ebs.projected is not None and ebs.orbital_names is not None
    return {
        **facts(p),
        "projected_shape": list(np.shape(ebs.projected.to_array())),
        "orbital_names": list(ebs.orbital_names)[:9],
    }


@step("elk_bands")
def _():
    out = {}
    for spin in ("non-spin-polarized", "spin-polarized-colinear"):
        p = get_parser(code="elk", dirpath=CODES / f"elk/6.3/SrVO3/{spin}/bands")
        assert p.structure is not None and p.structure.lattice is not None
        diag = np.round(np.diag(p.structure.lattice), 6).tolist()
        out[spin] = {**facts(p), "lattice_diag": diag}
    return out


@step("abinit_bands")
def _():
    p = get_parser(code="abinit", dirpath=CODES / "abinit/9.6/Fe/non-spin-polarized/bands")
    ebs, kpath = p.ebs, p.kpath
    assert ebs is not None and ebs.bands is not None and kpath is not None
    ticks = zip(kpath.tick_names, kpath.tick_positions, strict=True)
    return {
        "ebs_type": type(ebs).__name__,
        "bands_shape": list(np.shape(ebs.bands.to_array())),
        "ticks": [f"{n}({i})" for n, i in ticks],
    }


finish()
