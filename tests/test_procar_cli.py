import inspect
import runpy
import sys

import pytest

import pyprocar
from tests.utils import ROOT_DIR

PROCAR_CLI = ROOT_DIR / "scripts" / "procar.py"


def run_fermi2D(monkeypatch: pytest.MonkeyPatch, *cli_args: str) -> list[dict[str, object]]:
    signature = inspect.signature(pyprocar.fermi2D)
    calls: list[dict[str, object]] = []

    def record(*args: object, **kwargs: object) -> None:
        calls.append(dict(signature.bind(*args, **kwargs).arguments))

    monkeypatch.setattr(pyprocar, "fermi2D", record)
    monkeypatch.setattr(sys, "argv", ["procar.py", "fermi2D", *cli_args])
    runpy.run_path(str(PROCAR_CLI), run_name="__main__")
    return calls


def test_fermi2D_subcommand_passes_keywords_fermi2D_accepts(monkeypatch: pytest.MonkeyPatch):
    calls = run_fermi2D(monkeypatch, "calc_dir", "-e", "0.1")

    assert calls == [
        {
            "code": "vasp",
            "dirname": "calc_dir",
            "mode": "plain",
            "fermi": None,
            "spins": None,
            "atoms": None,
            "orbitals": None,
            "energy": 0.1,
            "savefig": None,
            "plot_arrows": True,
        }
    ]


def test_fermi2D_subcommand_maps_every_option(monkeypatch: pytest.MonkeyPatch):
    calls = run_fermi2D(
        monkeypatch,
        "calc_dir",
        "--code", "qe",
        "--mode", "spin_texture",
        "-s", "1", "3",
        "-a", "0", "2",
        "-o", "4", "5",
        "-e", "-0.2",
        "-f", "5.5",
        "--savefig", "fs.png",
        "--noarrow",
    )  # fmt: skip

    assert calls == [
        {
            "code": "qe",
            "dirname": "calc_dir",
            "mode": "spin_texture",
            "fermi": 5.5,
            "spins": [1, 3],
            "atoms": [0, 2],
            "orbitals": [4, 5],
            "energy": -0.2,
            "savefig": "fs.png",
            "plot_arrows": False,
        }
    ]
