from pathlib import Path

import numpy as np
import pytest

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.io import CodeParser, get_parser
from tests.pyprocar.io.bxsf.test_parser import BXSF_STR
from tests.pyprocar.io.elk import test_parser as elk
from tests.pyprocar.io.frmsf.test_parser import FRMSF_STR
from tests.pyprocar.io.lobster.test_doscar_lobster import DOSCAR_CONTENT
from tests.pyprocar.io.lobster.test_fatband import FATBAND_CONTENT
from tests.pyprocar.io.lobster.test_lobsterout import LOBSTEROUT_CONTENT
from tests.pyprocar.io.siesta.test_parser import BANDS_STR, FDF_STR
from tests.utils import DATA_DIR

MEMBER_TYPES = {
    "ebs": ElectronicBandStructure,
    "dos": DensityOfStates,
    "structure": Structure,
    "kpath": KPath,
    "fermi": float,
    "version": str,
    "reciprocal_lattice": np.ndarray,
}

CODES_DIR = DATA_DIR / "codes"

LEGACY_DOS = pytest.mark.xfail(
    reason="adapter builds DOS in the legacy (nspin, nE) layout that DensityOfStates rejects",
    strict=True,
)
KGRID_MODE_ENUM = pytest.mark.xfail(
    reason="ElectronicBandStructureMesh passes a KGRID_MODE enum to get_kpoints_from_kgrid, "
    "which expects a str",
    strict=True,
)

CASES = [
    pytest.param(
        "vasp",
        CODES_DIR / "vasp/6.4/SrVO3/non-spin-polarized/bands",
        {"ebs", "dos", "structure", "kpath", "fermi", "version"},
        id="vasp-bands",
    ),
    pytest.param(
        "abinit",
        CODES_DIR / "abinit/9.6/Fe/non-spin-polarized/bands",
        {"ebs", "structure", "kpath", "version"},
        id="abinit-bands",
    ),
    pytest.param(
        "qe",
        CODES_DIR / "qe/7.2/SrVO3/non-spin-polarized/bands",
        {"ebs", "structure", "kpath", "fermi", "reciprocal_lattice"},
        id="qe-bands",
    ),
    pytest.param(
        "qe",
        CODES_DIR / "qe/7.2/SrVO3/non-spin-polarized/dos",
        {"ebs", "dos", "structure", "fermi", "reciprocal_lattice"},
        id="qe-dos",
        marks=[LEGACY_DOS, KGRID_MODE_ENUM],
    ),
    pytest.param(
        "elk",
        {
            "elk.in": elk.ELKIN_BANDS,
            "FERMI.OUT": elk.EFERMI_OUT,
            "GEOMETRY.OUT": elk.GEOMETRY_OUT,
            "BANDLINES.OUT": elk.BANDLINES_OUT,
            "BANDS.OUT": elk.BANDS_OUT,
            "BAND_S01_A0001.OUT": elk.BAND_S01_A0001,
            "BAND_S02_A0001.OUT": elk.BAND_S02_A0001,
        },
        {"ebs", "structure", "kpath", "fermi", "reciprocal_lattice"},
        id="elk-bands",
    ),
    pytest.param(
        "elk",
        {
            "elk.in": elk.ELKIN_DOS,
            "FERMI.OUT": elk.EFERMI_OUT,
            "GEOMETRY.OUT": elk.GEOMETRY_OUT,
            "TDOS.OUT": elk.TDOS_OUT,
        },
        {"dos", "structure", "fermi", "reciprocal_lattice"},
        id="elk-dos",
        marks=LEGACY_DOS,
    ),
    pytest.param(
        "siesta",
        {"silicon.fdf": FDF_STR, "silicon.bands": BANDS_STR},
        {"ebs", "structure", "fermi", "reciprocal_lattice"},
        id="siesta-bands",
    ),
    pytest.param(
        "lobster",
        {
            "lobsterout": LOBSTEROUT_CONTENT,
            "DOSCAR.lobster": DOSCAR_CONTENT,
            "FATBAND_Fe_s.lobster": FATBAND_CONTENT,
        },
        {"ebs", "dos"},
        id="lobster-bands",
        marks=LEGACY_DOS,
    ),
    pytest.param("bxsf", {"in.bxsf": BXSF_STR}, {"ebs"}, id="bxsf-mesh"),
    pytest.param("frmsf", {"in.frmsf": FRMSF_STR}, {"ebs"}, id="frmsf-mesh"),
]


def calc_dir(source: Path | dict[str, str], tmp_path: Path) -> Path:
    if isinstance(source, Path):
        return source
    for name, content in source.items():
        (tmp_path / name).write_text(content)
    return tmp_path


def test_every_registered_code_has_a_populated_case() -> None:
    assert {case.values[0] for case in CASES} == set(CodeParser.as_list())


@pytest.mark.parametrize("code", CodeParser.as_list())
def test_empty_directory_reports_every_member_as_none(
    code: str, tmp_path: Path
) -> None:
    parser = get_parser(code, tmp_path)

    assert {name: getattr(parser, name) for name in MEMBER_TYPES} == dict.fromkeys(
        MEMBER_TYPES
    )


@pytest.mark.parametrize(("code", "source", "expected_present"), CASES)
def test_populated_directory_returns_core_types(
    code: str, source: Path | dict[str, str], expected_present: set[str], tmp_path: Path
) -> None:
    parser = get_parser(code, calc_dir(source, tmp_path))

    values = {name: getattr(parser, name) for name in MEMBER_TYPES}

    assert {
        name for name, value in values.items() if value is not None
    } == expected_present
    for name, value in values.items():
        assert value is None or isinstance(value, MEMBER_TYPES[name]), name
