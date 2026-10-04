"""QE non-collinear band structures come out as total, Sx, Sy and Sz per real (l, m) orbital.

The atomic_proj.xml amplitudes below are written by hand from QE 7.2's own
definitions of the spinor states (upflib spinor.f90, sph_ind.f90 and the rot_ylm
matrix in init_us_1.f90), for four bands with a known spinor on one p shell:

    band 1: pz with spin along +x    band 2: pz with spin along +y
    band 3: px with spin down        band 4: py with spin up

Without spin-orbit coupling projwfc.x projects on |l m s_z> directly.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import get_parser
from tests.utils import DATA_DIR

R6, R3, R2 = 1 / math.sqrt(6), 1 / math.sqrt(3), 1 / math.sqrt(2)

# <state|band> for the states j=1/2 (m_j=-1/2, 1/2) then j=3/2 (m_j=-3/2..3/2).
SOC_STATES = [
    "l=1 j=0.5 m_j=-0.5",
    "l=1 j=0.5 m_j= 0.5",
    "l=1 j=1.5 m_j=-1.5",
    "l=1 j=1.5 m_j=-0.5",
    "l=1 j=1.5 m_j= 0.5",
    "l=1 j=1.5 m_j= 1.5",
]
SOC_WFC = [1, 1, 2, 2, 2, 2]
SOC_AMPLITUDES = [
    [-R6, R6, 0, R3, R3, 0],
    [-1j * R6, R6, 0, 1j * R3, R3, 0],
    [0, -R3, -R2, 0, R6, 0],
    [-1j * R3, 0, 0, -1j * R6, 0, -1j * R2],
]

# <state|band> for (m=1 pz, m=2 px, m=3 py) spin up, then the same spin down.
NOSOC_STATES = [f"l=1 m= {m} s_z= {s_z}" for s_z in ("0.5", "-0.5") for m in (1, 2, 3)]
NOSOC_WFC = [1] * 6
NOSOC_AMPLITUDES = [
    [R2, 0, 0, R2, 0, 0],
    [R2, 0, 0, 1j * R2, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
]

# (band, orbital index in projwfc.x order s, pz, px, py) -> (total, Sx, Sy, Sz)
EXPECTED = {
    (0, 1): [1.0, 1.0, 0.0, 0.0],
    (1, 1): [1.0, 0.0, 1.0, 0.0],
    (2, 2): [1.0, 0.0, 0.0, -1.0],
    (3, 3): [1.0, 0.0, 0.0, 1.0],
}

PW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">
  <output>
    <atomic_species ntyp="1">
      <species name="Fe"><mass>55.8</mass><pseudo_file>Fe.upf</pseudo_file></species>
    </atomic_species>
    <atomic_structure nat="1" alat="5.0">
      <atomic_positions>
        <atom name="Fe" index="1">0.0 0.0 0.0</atom>
      </atomic_positions>
      <cell>
        <a1>5.0 0.0 0.0</a1>
        <a2>0.0 5.0 0.0</a2>
        <a3>0.0 0.0 5.0</a3>
      </cell>
    </atomic_structure>
    <basis_set>
      <reciprocal_lattice>
        <b1>1.0 0.0 0.0</b1>
        <b2>0.0 1.0 0.0</b2>
        <b3>0.0 0.0 1.0</b3>
      </reciprocal_lattice>
    </basis_set>
    <band_structure>
      <noncolin>true</noncolin>
      <nbnd>4</nbnd>
      <fermi_energy>0.0</fermi_energy>
      <nks>1</nks>
      <ks_energies>
        <k_point weight="1.0">0.0 0.0 0.0</k_point>
        <npw>100</npw>
        <eigenvalues size="4">-0.2 -0.1 0.1 0.2</eigenvalues>
        <occupations size="4">1.0 1.0 0.0 0.0</occupations>
      </ks_energies>
    </band_structure>
  </output>
</qes:espresso>
"""


def projwfc_out(states: list[str], wfcs: list[int]) -> str:
    lines = [
        f"     state #{i:4d}: atom   1 (Fe ), wfc {wfc:2d} ({state})"
        for i, (state, wfc) in enumerate(zip(states, wfcs, strict=True), start=1)
    ]
    return (
        "\n     Program PROJWFC v.7.2 starts on  1Jan2026 at 12: 0: 0\n\n"
        + "     Atomic states used for projection\n\n"
        + "\n".join(lines)
        + "\n\n  natomwfc =            6\n  nbnd     =            4\n  nkstot   =            1\n"
    )


def atomic_proj_xml(amplitudes: list[list[complex]]) -> str:
    wfcs = "".join(
        f'      <ATOMIC_WFC index="{i + 1}" spin="1">\n'
        + "".join(f"  {complex(band[i]).real:.15f} {complex(band[i]).imag:.15f}\n" for band in amplitudes)
        + "      </ATOMIC_WFC>\n"
        for i in range(6)
    )
    return (
        "<PROJECTIONS>\n"
        + '  <HEADER NUMBER_OF_BANDS="4" NUMBER_OF_K-POINTS="1" NUMBER_OF_SPIN_COMPONENTS="1"'
        + ' NUMBER_OF_ATOMIC_WFC="6" NUMBER_OF_ELECTRONS="2.0" FERMI_ENERGY="0.0"/>\n'
        + "  <EIGENSTATES>\n"
        + '    <K-POINT Weight="1.0">0.0 0.0 0.0</K-POINT>\n'
        + "    <E>-0.2 -0.1 0.1 0.2</E>\n"
        + f"    <PROJS>\n{wfcs}    </PROJS>\n"
        + "  </EIGENSTATES>\n</PROJECTIONS>\n"
    )


@pytest.mark.parametrize(
    ("states", "wfcs", "amplitudes"),
    [
        pytest.param(SOC_STATES, SOC_WFC, SOC_AMPLITUDES, id="spin-orbit-j-basis"),
        pytest.param(NOSOC_STATES, NOSOC_WFC, NOSOC_AMPLITUDES, id="no-spin-orbit-s_z-basis"),
    ],
)
def test_spinor_projections_give_total_and_spin_per_real_orbital(
    tmp_path: Path, states, wfcs, amplitudes
) -> None:
    (tmp_path / "pw.xml").write_text(PW_XML)
    (tmp_path / "kpdos.out").write_text(projwfc_out(states, wfcs))
    (tmp_path / "out" / "fe.save").mkdir(parents=True)
    (tmp_path / "out" / "fe.save" / "atomic_proj.xml").write_text(atomic_proj_xml(amplitudes))

    ebs = get_parser("qe", tmp_path).ebs

    assert ebs is not None and ebs.projected is not None
    projected = ebs.projected.to_array()
    assert projected.shape == (1, 4, 4, 1, 16)
    expected = np.zeros((4, 4, 16))
    for (band, orbital), spin in EXPECTED.items():
        expected[band, :, orbital] = spin
    np.testing.assert_allclose(projected[0, :, :, 0, :], expected, atol=1e-12)
    assert ebs.orbital_names[:4] == ["s", "pz", "px", "py"]


QE_NONCOLLINEAR_BANDS = DATA_DIR / "codes/qe/7.2/SrVO3/non-colinear/bands"
D_ORBITALS = ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]


@pytest.mark.data
def test_qe_spin_orbit_band_structure_matches_projwfc_weights_at_gamma() -> None:
    """kpdos.out of the same run prints each band's weights on the j states at Gamma.

    Band 41 (11.583 eV) is V 0.377 [#28] + 0.285 [#35] + 0.251 [#33] + 0.071 [#30],
    all V d states, 0.984 in all. Band 47 (13.817 eV) is V 0.500 [#34] + 0.333 [#29]
    and O s 0.079 [#54] (atom 5) + 0.020 [#38] + 0.020 [#46], |psi|^2 = 0.951.
    Cubic SrVO3 puts the first in V t2g (dxz, dyz, dxy) and the second in V eg.
    """
    ebs = get_parser("qe", QE_NONCOLLINEAR_BANDS).ebs

    assert ebs is not None and ebs.projected is not None
    projected = ebs.projected.to_array()
    assert projected.shape == (155, 50, 4, 5, 16)
    assert ebs.orbital_names[4:9] == D_ORBITALS
    v_d = dict(zip(D_ORBITALS, projected[0, :, 0, 1, 4:9].T, strict=True))
    t2g = v_d["dxz"] + v_d["dyz"] + v_d["dxy"]
    eg = v_d["dz2"] + v_d["dx2-y2"]
    assert (t2g[40], eg[40]) == pytest.approx((0.984, 0.0), abs=2e-3)
    assert (t2g[46], eg[46]) == pytest.approx((0.0, 0.833), abs=2e-3)
    assert projected[0, 46, 0, 4, 0] == pytest.approx(0.079, abs=1e-3)
    assert projected[0, 46, 0].sum() == pytest.approx(0.951, abs=2e-3)
