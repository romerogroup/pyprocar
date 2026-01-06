"""Tests for Fatband extractor."""

import numpy as np
import pytest

from pyprocar.io.lobster import Fatband


FATBAND_CONTENT = """# FATBAND for Fe (s)
# NBANDS 4
# K-Point   1 :    0.00000    0.00000    0.00000
   1   -5.00000    0.10000
   2   -2.00000    0.20000
   3    1.00000    0.30000
   4    4.00000    0.40000
# K-Point   2 :    0.50000    0.00000    0.00000
   1   -4.50000    0.15000
   2   -1.50000    0.25000
   3    1.50000    0.35000
   4    4.50000    0.45000
"""


class TestFatband:
    def test_from_str(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor is not None

    def test_element(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor.element == "Fe"

    def test_orbital(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor.orbital == "s"

    def test_n_bands(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor.n_bands == 4

    def test_n_kpoints(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor.n_kpoints == 2

    def test_kpoints_shape(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor.kpoints.shape == (2, 3)

    def test_bands_shape(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor.bands.shape == (2, 4, 1)  # (n_kpoints, n_bands, n_spins)

    def test_projections_shape(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert extractor.projections.shape == (2, 4, 1)

    def test_mapping_protocol(self):
        extractor = Fatband.from_str(FATBAND_CONTENT)
        assert "element" in extractor
        assert len(extractor) == 6
