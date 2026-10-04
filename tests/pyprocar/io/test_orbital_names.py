"""Every parser names the orbitals it projects, so check_projected_layout compares them.

The expected names come from each code's own output for the same run: projwfc.x's
Lowdin charge labels, Abinit's PROCAR header and Elk's ELMIREP.OUT.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import get_parser
from pyprocar.io.vasp import Procar
from tests.pyprocar.io.lobster.test_doscar_lobster import DOSCAR_CONTENT
from tests.pyprocar.io.lobster.test_lobsterout import LOBSTEROUT_CONTENT
from tests.utils import DATA_DIR

CODES = DATA_DIR / "codes"

FILTERED_PROCAR = """PROCAR lm decomposed
# of k-points:    1         # of bands:    2         # of ions:    2

 k-point     1 :    0.00000000 0.00000000 0.00000000     weight = 1.00000000

band     1 # energy   -1.00000000 # occ.  1.00000000

ion o0 tot
    1  0.100  0.100
    2  0.200  0.200
tot    0.300  0.300

band     2 # energy    1.00000000 # occ.  0.00000000

ion o0 tot
    1  0.400  0.400
    2  0.500  0.500
tot    0.900  0.900

"""


def test_filtered_procar_keeps_the_column_name_its_header_gives() -> None:
    procar = Procar.from_str(FILTERED_PROCAR)

    assert procar.orbital_names == ["o0"]
    assert procar.projected is not None
    assert procar.projected[0, :, 0, :, 0].tolist() == [[0.1, 0.2], [0.4, 0.5]]


@pytest.mark.data
def test_vasp_band_structure_of_a_filtered_procar_is_named_o0() -> None:
    ebs = get_parser("vasp", DATA_DIR / "examples/bands/auto").ebs

    assert ebs is not None
    assert ebs.orbital_names == ["o0"]


def lowdin_labels(projwfc_output: Path) -> list[str]:
    """The s, then (l, m) labels projwfc.x prints for atom 1's Lowdin charges."""
    lines = [line for line in projwfc_output.read_text().splitlines() if "Atom #   1:" in line]
    labels = [label for line in lines for label in re.findall(r"(\w[\w-]*)\s*=", line)]
    return [label for label in labels if label not in {"charge", "p", "d", "f"}]


@pytest.mark.data
@pytest.mark.parametrize(
    ("relpath", "member", "projwfc_output"),
    [
        ("non-spin-polarized/bands", "ebs", "kpdos.out"),
        ("non-spin-polarized/dos", "dos", "pdos.out"),
    ],
)
def test_qe_names_its_orbitals_as_projwfc_labels_them(relpath, member, projwfc_output) -> None:
    calc = CODES / "qe/7.2/SrVO3" / relpath
    expected = lowdin_labels(calc / projwfc_output)

    names = getattr(get_parser("qe", calc), member).orbital_names

    assert expected == ["s", "pz", "px", "py", "dz2", "dxz", "dyz", "dx2-y2", "dxy"]
    assert names[:9] == expected


@pytest.mark.data
def test_abinit_dos_names_its_orbitals_as_the_same_runs_procar_does() -> None:
    calc = CODES / "abinit/9.6/Fe/non-spin-polarized/dos"
    header = next(line for line in (calc / "PROCAR").open() if line.startswith("ion"))

    dos = get_parser("abinit", calc).dos

    assert dos is not None
    assert dos.orbital_names == header.split()[1:-1]


@pytest.mark.data
def test_elk_dos_names_its_orbitals_by_the_l_m_slots_of_elmirep() -> None:
    calc = CODES / "elk/6.3/SrVO3/non-spin-polarized/dos"
    first_atom = (calc / "ELMIREP.OUT").read_text().split("\n \n")[0]
    expected = [f"Y{ang}{m}" for ang, m in re.findall(r"l =\s*(\d+), m =\s*(-?\d+)", first_atom)]

    dos = get_parser("elk", calc).dos

    assert dos is not None
    assert len(expected) == 16
    assert dos.orbital_names == expected


LOBSTER_PDOS_BLOCK = """ -10.0000  10.0000   5   0.0000   1.0000 ; Z= 26 ; s p_y p_z p_x
 -10.0000   0.0100   0.0200   0.0300   0.0400
  -5.0000   0.0500   0.0600   0.0700   0.0800
   0.0000   0.0900   0.1000   0.1100   0.1200
   5.0000   0.1300   0.1400   0.1500   0.1600
  10.0000   0.1700   0.1800   0.1900   0.2000
"""


def test_lobster_dos_names_the_slot_each_orbital_column_fills(tmp_path: Path) -> None:
    (tmp_path / "lobsterout").write_text(LOBSTEROUT_CONTENT)
    (tmp_path / "DOSCAR.lobster").write_text(DOSCAR_CONTENT + LOBSTER_PDOS_BLOCK)

    dos = get_parser("lobster", tmp_path).dos

    assert dos is not None and dos.projected is not None and dos.orbital_names is not None
    p_z = dos.projected.to_array()[:, 0, 0, dos.orbital_names.index("p_z")]
    np.testing.assert_allclose(p_z, [0.03, 0.07, 0.11, 0.15, 0.19])
