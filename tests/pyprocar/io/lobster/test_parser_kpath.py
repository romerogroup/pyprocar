from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.lobster import LobsterParser


@pytest.fixture
def lobster_dir(tmp_path: Path) -> Path:
    (tmp_path / "lobsterout").write_text(
        "LOBSTER v4.1.0\ncalculating FatBand for Element: Fe Orbital(s): s\nfinished!\n"
    )
    fatband = ["# FATBAND for Fe (s)", "# NBANDS 1"]
    for ik, kx in enumerate(np.linspace(0, 0.5, 6), start=1):
        fatband += [
            f"# K-Point {ik:>3} :    {kx:.5f}    0.00000    0.00000",
            "   1   -5.00000    0.10000",
        ]
    (tmp_path / "FATBAND_Fe_s.lobster").write_text("\n".join(fatband) + "\n")
    (tmp_path / "scf.in").write_text(
        "K_POINTS crystal_b\n2\n  0.0 0.0 0.0 5 !G\n  0.5 0.0 0.0 1 !X\n"
    )
    return tmp_path


def test_kpath_distances_use_the_parser_reciprocal_lattice(lobster_dir: Path, monkeypatch):
    monkeypatch.setattr(LobsterParser, "reciprocal_lattice", property(lambda _: np.eye(3) / 4))
    kpath = LobsterParser(lobster_dir).kpath

    assert kpath is not None
    assert kpath.k_distances[-1] == pytest.approx(0.5 / 4)
