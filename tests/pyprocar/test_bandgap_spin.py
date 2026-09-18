"""Regression tests for pyprocar.bandgap with spin-polarized bands."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyprocar.scripts import scriptBandGap


def _patch_parser(monkeypatch, bands, efermi=0.0):
    ebs = SimpleNamespace(bands=np.asarray(bands, dtype=float), efermi=efermi)

    class FakeParser:
        def __init__(self, code, dirpath):
            self.ebs = ebs

    monkeypatch.setattr(scriptBandGap.io, "Parser", FakeParser)


def _two_band_insulator(vbm, cbm, nkpoints=4):
    """Bands of shape (nkpoints, 2) with a flat valence and conduction band."""
    k = np.linspace(0, 1, nkpoints)
    valence = vbm - 0.5 * k
    conduction = cbm + 0.5 * k
    return np.stack([valence, conduction], axis=1)


def test_bandgap_non_spin_polarized(monkeypatch):
    bands = _two_band_insulator(-0.5, 1.0)[..., np.newaxis]
    _patch_parser(monkeypatch, bands)

    gap = scriptBandGap.bandgap(dirname=".", fermi=0.0)

    assert isinstance(gap, float)
    assert gap == pytest.approx(1.5)


def test_bandgap_spin_polarized_uses_both_channels(monkeypatch):
    up = _two_band_insulator(-0.5, 1.0)
    down = _two_band_insulator(-0.2, 0.8)
    bands = np.stack([up, down], axis=-1)
    _patch_parser(monkeypatch, bands)

    gap = scriptBandGap.bandgap(dirname=".", fermi=0.0)

    # VBM = max(-0.5, -0.2) = -0.2 ; CBM = min(1.0, 0.8) = 0.8
    assert isinstance(gap, float)
    assert gap == pytest.approx(1.0)


def test_bandgap_spin_polarized_metal_in_one_channel(monkeypatch):
    up = _two_band_insulator(-0.5, 1.0)
    down = _two_band_insulator(-0.2, 0.8)
    # Make the spin-down valence band cross the Fermi level
    down[:, 0] = np.linspace(-0.3, 0.3, down.shape[0])
    bands = np.stack([up, down], axis=-1)
    _patch_parser(monkeypatch, bands)

    gap = scriptBandGap.bandgap(dirname=".", fermi=0.0)

    assert isinstance(gap, float)
    assert gap == 0.0
