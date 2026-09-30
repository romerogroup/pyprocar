import re
from pathlib import Path

import numpy as np
import pytest

from pyprocar.core import DensityOfStates, ElectronicBandStructure, KPath, Structure
from pyprocar.core.ebs import ElectronicBandStructurePath
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

LEGACY_DOS_REASON = (
    "ElkDOS builds the legacy (nspin, nE) layout that DensityOfStates rejects"
)
KGRID_MODE_ENUM = pytest.mark.xfail(
    reason="ElectronicBandStructureMesh passes a KGRID_MODE enum to get_kpoints_from_kgrid, "
    "which expects a str",
    raises=AttributeError,
    strict=True,
)
ELK_LEGACY_DOS = pytest.mark.xfail(
    reason=LEGACY_DOS_REASON, raises=ValueError, strict=True
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
        marks=KGRID_MODE_ENUM,
    ),
    pytest.param(
        "elk",
        {
            "elk.in": elk.ELKIN_BANDS,
            "EFERMI.OUT": elk.EFERMI_OUT,
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
            "EFERMI.OUT": elk.EFERMI_OUT,
            "GEOMETRY.OUT": elk.GEOMETRY_OUT,
            "TDOS.OUT": elk.TDOS_OUT,
        },
        {"dos", "structure", "fermi", "reciprocal_lattice"},
        id="elk-dos",
        marks=ELK_LEGACY_DOS,
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


@pytest.mark.parametrize(
    "code, relpath",
    [
        ("vasp", "vasp/6.4/SrVO3/non-spin-polarized/bands"),
        ("qe", "qe/7.2/SrVO3/non-spin-polarized/bands"),
        pytest.param(
            "abinit",
            "abinit/9.6/Fe/non-spin-polarized/bands",
            marks=pytest.mark.xfail(
                reason="AbinitParser returns bands in Hartree and fermi in eV",
                raises=AssertionError,
                strict=True,
            ),
        ),
    ],
)
def test_real_ebs_bands_are_unshifted_around_fermi(code: str, relpath: str) -> None:
    ebs = get_parser(code, CODES_DIR / relpath).ebs

    assert ebs is not None and ebs.bands is not None
    bands = ebs.bands.to_array()
    assert bands.min() < ebs.fermi < bands.max()


@pytest.mark.parametrize(
    ("code", "source", "total_shape", "projected_shape", "energy_range", "fermi"),
    [
        pytest.param(
            "qe",
            CODES_DIR / "qe/7.2/SrVO3/non-spin-polarized/dos",
            (7434, 1),
            (7434, 1, 5, 16),
            (-54.82, 19.51),
            12.5491,
            id="qe-non-spin-polarized",
        ),
        pytest.param(
            "qe",
            CODES_DIR / "qe/7.2/SrVO3/spin-polarized-colinear/dos",
            (7433, 2),
            (7433, 2, 5, 16),
            (-54.808, 19.512),
            12.5465,
            id="qe-spin-polarized",
        ),
        pytest.param(
            "lobster",
            {"lobsterout": LOBSTEROUT_CONTENT, "DOSCAR.lobster": DOSCAR_CONTENT},
            (5, 1),
            None,
            (-10.0, 10.0),
            0.0,
            id="lobster",
        ),
    ],
)
def test_dos_is_in_core_layout_with_unshifted_energies(
    code: str,
    source: Path | dict[str, str],
    total_shape: tuple[int, ...],
    projected_shape: tuple[int, ...] | None,
    energy_range: tuple[float, float],
    fermi: float,
    tmp_path: Path,
) -> None:
    dos = get_parser(code, calc_dir(source, tmp_path)).dos

    assert dos is not None and dos.total is not None
    assert dos.total.to_array().shape == total_shape
    projected = dos.projected
    assert (
        None if projected is None else projected.to_array().shape
    ) == projected_shape
    assert (dos.energies[0], dos.energies[-1]) == pytest.approx(energy_range)
    assert dos.fermi == pytest.approx(fermi)


def test_from_code_names_the_member_the_adapter_could_not_provide(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match=re.escape(f"The elk parser found no ebs in {tmp_path}")
    ):
        ElectronicBandStructurePath.from_code(code="elk", dirpath=str(tmp_path))
    with pytest.raises(
        ValueError, match=re.escape(f"The elk parser found no dos in {tmp_path}")
    ):
        DensityOfStates.from_code(code="elk", dirpath=str(tmp_path))
