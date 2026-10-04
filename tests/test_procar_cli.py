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


ORBITALS = "ion      s     py     pz     px    dxy    dyz    dz2    dxz  x2-y2    tot"


def _two_ion_procar(weights: tuple[float, float]) -> str:
    rows = [
        f"    {ion}  {w:.3f}" + "  0.000" * 8 + f"  {w:.3f}" for ion, w in enumerate(weights, 1)
    ]
    total = sum(weights)
    rows.append(f"tot    {total:.3f}" + "  0.000" * 8 + f"  {total:.3f}")
    lines = [
        "PROCAR lm decomposed",
        "# of k-points:  1         # of bands:   1         # of ions:    2",
        "",
        " k-point     1 :    0.00000000 0.00000000 0.00000000     weight = 1.00000000",
        "",
        "band     1 # energy  -1.00000000 # occ.  1.00000000",
        " ",
        ORBITALS,
        *rows,
        "",
    ]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("cli_atoms", [["-a", "0", "1"], ["-a", "1", "2", "--human"]])
def test_filter_subcommand_groups_atoms(tmp_path, monkeypatch: pytest.MonkeyPatch, cli_atoms):
    (tmp_path / "PROCAR").write_text(_two_ion_procar((0.1, 0.2)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["procar.py", "filter", "PROCAR", "out", *cli_atoms])

    runpy.run_path(str(PROCAR_CLI), run_name="__main__")

    lines = (tmp_path / "out").read_text().splitlines()
    assert lines[1].split() == [
        "#",
        "of",
        "k-points:",
        "1",
        "#",
        "of",
        "bands:",
        "1",
        "#",
        "of",
        "ions:",
        "1",
    ]
    grouped = next(line.split() for line in lines if line.split()[:1] == ["1"])
    assert float(grouped[1]) == pytest.approx(0.1 + 0.2)
    assert float(grouped[-1]) == pytest.approx(0.1 + 0.2)


@pytest.mark.guards_existing_behaviour(
    reason="#285 asked to check the cat subcommand; it works, and this run of it keeps it working"
)
def test_cat_subcommand_joins_procars(tmp_path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "PROCAR_1").write_text(_two_ion_procar((0.1, 0.2)))
    (tmp_path / "PROCAR_2").write_text(_two_ion_procar((0.3, 0.4)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["procar.py", "cat", "PROCAR_1", "PROCAR_2", "merged"])

    runpy.run_path(str(PROCAR_CLI), run_name="__main__")

    text = (tmp_path / "merged").read_text()
    assert text.splitlines()[1].split()[:4] == ["#", "of", "k-points:", "2"]
    assert text.count("k-point ") == 2
    assert "    2  0.200" in text and "    2  0.400" in text
