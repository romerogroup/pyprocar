import shutil
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from pyprocar.core import ElectronicBandStructure
from pyprocar.scripts import scriptAutoBandsplot
from tests.utils import DATA_DIR, writable_copy


class ReachedStructure(Exception):
    pass


class FakeEBS:
    fermi: float = 9.13835

    @property
    def structure(self):
        raise ReachedStructure


def fake_parser(*_: object) -> SimpleNamespace:
    return SimpleNamespace()


def fake_from_code(*_: object, **_kwargs: object) -> FakeEBS:
    return FakeEBS()


@pytest.mark.parametrize("code", ["abinit", "qe", "elk"])
def test_autobandsplot_without_fermi_gets_past_the_fermi_lookup(code, monkeypatch):
    monkeypatch.setattr(scriptAutoBandsplot, "get_parser", fake_parser)
    monkeypatch.setattr(ElectronicBandStructure, "from_code", fake_from_code)

    with pytest.raises(ReachedStructure):
        scriptAutoBandsplot.autobandsplot(code=code, dirname="unused")


AUTO = DATA_DIR / "examples" / "bands" / "auto"


@pytest.fixture
def auto_calc(tmp_path, monkeypatch):
    if not AUTO.exists():
        pytest.skip("fixture bands/auto not downloaded")
    calc = tmp_path / "calc"
    writable_copy(AUTO, calc, ignore=shutil.ignore_patterns("report.txt", "*.pkl"))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    yield calc, cwd
    plt.close("all")


@pytest.mark.data
def test_autobandsplot_writes_no_file_by_default(auto_calc):
    calc, cwd = auto_calc

    scriptAutoBandsplot.autobandsplot(code="vasp", dirname=str(calc), fermi=3.0667)

    assert list(cwd.iterdir()) == []
    assert not (calc / "report.txt").exists()


@pytest.mark.data
def test_autobandsplot_writes_the_report_where_asked(auto_calc, tmp_path):
    calc, cwd = auto_calc
    report = tmp_path / "analysis.txt"

    scriptAutoBandsplot.autobandsplot(code="vasp", dirname=str(calc), fermi=3.0667, report=report)

    assert list(cwd.iterdir()) == []
    assert report.read_text() == (AUTO / "report.txt").read_text()
