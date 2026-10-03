from types import SimpleNamespace

import pytest

from pyprocar.core import ElectronicBandStructure
from pyprocar.scripts import scriptAutoBandsplot


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
