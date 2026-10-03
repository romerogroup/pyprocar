from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.abinit import AbinitParser
from tests.pyprocar.io.abinit import ABINIT_DATA_DIR

GAMMA = (0.0, 0.0, 0.0)
H = (0.5, -0.5, 0.5)
N = (0.0, 0.0, 0.5)
P = (0.25, 0.25, 0.25)


def write_procar(path: Path, kpoints: np.ndarray) -> None:
    lines = [
        "PROCAR lm decomposed",
        f"# of k-points:  {len(kpoints)}         # of bands:   1         # of ions:   1",
        "",
    ]
    for ik, (kx, ky, kz) in enumerate(kpoints, start=1):
        lines += [
            f" k-point {ik:>5} :    {kx:.8f} {ky:.8f} {kz:.8f}     weight = 1.00000000",
            "",
            "band     1 # energy   -1.00000000 # occ.  2.00000000",
            "",
            "ion      s      py     pz     px    dxy    dyz    dz2    dxz    dx2    tot",
            "    1  1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000",
            "tot    1.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  1.000",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def abinit_path_dir(tmp_path: Path) -> Path:
    # Like data/codes/abinit/9.6/Fe/*/bands: P, the start after the
    # zero-division N-P jump, is not written.
    t = np.linspace(0, 1, 6)[:, None]
    gamma_h = np.array(GAMMA) + t * (np.array(H) - np.array(GAMMA))
    h_n = np.array(H) + t[1:] * (np.array(N) - np.array(H))
    p_gamma = np.array(P) + t[1:] * (np.array(GAMMA) - np.array(P))
    write_procar(tmp_path / "PROCAR", np.vstack([gamma_h, h_n, p_gamma]))
    kpoints_lines = [
        "KPOINTS",
        "5 ! Grid points",
        "Line_mode",
        "reciprocal",
        "0.0 0.0 0.0 ! GAMMA",
        "0.5 -0.5 0.5 ! H",
        "",
        "0.5 -0.5 0.5 ! H",
        "0.0 0.0 0.5 ! N",
        "",
        "0.25 0.25 0.25 ! P",
        "0.0 0.0 0.0 ! GAMMA",
    ]
    (tmp_path / "KPOINTS").write_text("\n".join(kpoints_lines) + "\n")
    return tmp_path


def test_kpath_ticks_follow_abinit_segment_boundaries(abinit_path_dir: Path):
    kpath = AbinitParser(abinit_path_dir).kpath

    assert kpath is not None
    assert list(zip(kpath.tick_positions, kpath.tick_names, strict=True)) == [
        (0, "$\\Gamma$"),
        (5, "H"),
        (10, "N|P"),
        (15, "$\\Gamma$"),
    ]


def test_kpath_distances_keep_steps_across_shared_boundaries(abinit_path_dir: Path):
    kpath = AbinitParser(abinit_path_dir).kpath

    assert kpath is not None
    distances = np.asarray(kpath.get_distances(as_segments=False, cartesian=False))
    gamma_h = np.linalg.norm(H)
    h_n = np.linalg.norm(np.subtract(N, H))
    assert distances[[5, 6, 10]] == pytest.approx([gamma_h, gamma_h + h_n / 5, gamma_h + h_n])


@pytest.mark.data
@pytest.mark.parametrize(
    "calc_type", ["non-spin-polarized", "spin-polarized-colinear", "non-colinear"]
)
def test_fe_bands_ticks_sit_on_the_high_symmetry_points(calc_type: str):
    kpath = AbinitParser(ABINIT_DATA_DIR / calc_type / "bands").kpath

    assert kpath is not None
    assert list(zip(kpath.tick_positions, kpath.tick_names, strict=True)) == [
        (0, "$\\Gamma$"),
        (50, "H"),
        (100, "N"),
        (150, "$\\Gamma$"),
        (200, "P"),
        (250, "H|P"),
        (300, "N"),
    ]


@pytest.mark.data
def test_fe_bands_ebs_is_a_path_with_cartesian_tick_distances():
    ebs = AbinitParser(ABINIT_DATA_DIR / "non-spin-polarized" / "bands").ebs
    assert ebs is not None
    bands = ebs.bands
    assert bands is not None
    kpath_meta = bands.metadata["kpath"]

    a = 2 * 1.420026
    gamma_h, h_n, gamma_p = 1 / a, 1 / (np.sqrt(2) * a), np.sqrt(3) / (2 * a)
    expected = np.cumsum([0, gamma_h, h_n, h_n, gamma_p, gamma_p])
    tick_x = kpath_meta["k_distances"][kpath_meta["tick_positions"]]
    assert kpath_meta["tick_names"] == ["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H|P", "N"]
    assert tick_x[:6] == pytest.approx(expected, rel=1e-4)
