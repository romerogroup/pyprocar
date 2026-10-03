from types import SimpleNamespace

import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructure
from pyprocar.scripts import scriptBandGap


def _use_bands(monkeypatch, bands):
    bands = np.asarray(bands, dtype=float)
    kpoints = np.column_stack([np.linspace(0, 0.5, bands.shape[0]), np.zeros((bands.shape[0], 2))])
    ebs = ElectronicBandStructure(kpoints=kpoints, bands=bands, fermi=0.0)

    def get_parser(_code: str, _dirpath: str) -> SimpleNamespace:
        return SimpleNamespace(ebs=ebs)

    monkeypatch.setattr(scriptBandGap, "get_parser", get_parser)


def _two_band_insulator(vbm, cbm, nkpoints=4):
    k = np.linspace(0, 1, nkpoints)
    return np.stack([vbm - 0.5 * k, cbm + 0.5 * k], axis=1)


def test_non_spin_polarized_gap(monkeypatch):
    _use_bands(monkeypatch, _two_band_insulator(-0.5, 1.0)[..., np.newaxis])

    gap = scriptBandGap.bandgap(dirname=".", fermi=0.0)

    assert type(gap) is float
    assert gap == pytest.approx(1.5)


def test_spin_polarized_gap_spans_both_channels(monkeypatch):
    up = _two_band_insulator(-0.5, 1.0)
    down = _two_band_insulator(-0.2, 0.8)
    _use_bands(monkeypatch, np.stack([up, down], axis=-1))

    gap = scriptBandGap.bandgap(dirname=".", fermi=0.0)

    assert type(gap) is float
    assert gap == pytest.approx(1.0)


def test_spin_polarized_metal_in_one_channel(monkeypatch):
    up = _two_band_insulator(-0.5, 1.0)
    down = _two_band_insulator(-0.2, 0.8)
    down[:, 0] = np.linspace(-0.3, 0.3, down.shape[0])
    _use_bands(monkeypatch, np.stack([up, down], axis=-1))

    gap = scriptBandGap.bandgap(dirname=".", fermi=0.0)

    assert type(gap) is float
    assert gap == 0.0


def test_band_crossing_fermi_below_the_highest_occupied_state_is_metal(monkeypatch):
    _use_bands(monkeypatch, np.array([[-1.0, -0.5], [-0.01, 0.5]])[..., np.newaxis])

    assert scriptBandGap.bandgap(dirname=".", fermi=0.0) == 0.0
