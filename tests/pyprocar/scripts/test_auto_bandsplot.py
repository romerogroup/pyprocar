from types import SimpleNamespace

import pytest

from pyprocar.core import ElectronicBandStructure
from pyprocar.scripts import scriptAutoBandsplot


class ReachedStructure(Exception):
    pass


class FakeEBS:
    fermi = 9.13835

    @property
    def structure(self):
        raise ReachedStructure


@pytest.mark.parametrize("code", ["abinit", "qe", "elk"])
def test_autobandsplot_without_fermi_gets_past_the_fermi_lookup(code, monkeypatch):
    monkeypatch.setattr(scriptAutoBandsplot, "get_parser", lambda *_: SimpleNamespace())
    monkeypatch.setattr(ElectronicBandStructure, "from_code", lambda *_, **__: FakeEBS())

    with pytest.raises(ReachedStructure):
        scriptAutoBandsplot.autobandsplot(code=code, dirname="unused")
