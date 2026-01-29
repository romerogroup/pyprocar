"""Tests for DoscarLobster extractor."""

import pytest

from pyprocar.io.lobster import DoscarLobster

# Minimal DOSCAR.lobster content (non-spin-polarized)
DOSCAR_CONTENT = """   2   2   0   1
  10.0000   3.0000   3.0000   3.0000   0.0010
  0.00000
 CAR
 Fe2O3
 -10.0000  10.0000   5   0.0000   1.0000
 -10.0000   0.1000   0.0500
  -5.0000   0.5000   0.3000
   0.0000   1.0000   0.8000
   5.0000   0.5000   0.9500
  10.0000   0.1000   1.0000
"""


class TestDoscarLobster:
    def test_from_str(self) -> None:
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert extractor is not None

    def test_nedos(self) -> None:
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert extractor.nedos == 5

    def test_energies(self) -> None:
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert len(extractor.energies) == 5
        assert extractor.energies[0] == pytest.approx(-10.0)
        assert extractor.energies[-1] == pytest.approx(10.0)

    def test_total_dos_shape(self) -> None:
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert extractor.total_dos.shape == (5, 1)  # (nedos, n_spins)

    def test_is_spin_polarized(self) -> None:
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert not extractor.is_spin_polarized

    def test_mapping_protocol(self) -> None:
        extractor = DoscarLobster.from_str(DOSCAR_CONTENT)
        assert "energies" in extractor
        assert len(extractor) == 5
