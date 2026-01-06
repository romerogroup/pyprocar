"""Tests for SIESTA Bands extractor."""

import numpy as np

from pyprocar.io.siesta import Bands


# Minimal .bands file: 2 k-points, 3 bands, 1 spin
BANDS_STR = """
-5.5000
0.0000 1.0000
-10.0000 5.0000
3 1 2
0.0000 -8.5 -4.2 1.3
1.0000 -7.8 -3.9 2.1
"""


class TestBands:
    def test_fermi_energy(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        assert bands.fermi_energy == -5.5

    def test_k_path_bounds(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        assert bands.k_path_bounds == (0.0, 1.0)

    def test_energy_bounds(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        assert bands.energy_bounds == (-10.0, 5.0)

    def test_dimensions(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        assert bands.n_bands == 3
        assert bands.n_spins == 1
        assert bands.n_kpoints == 2

    def test_k_distances(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        assert bands.k_distances.shape == (2,)
        assert np.allclose(bands.k_distances, [0.0, 1.0])

    def test_bands_shape(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        assert bands.bands.shape == (2, 3, 1)

    def test_bands_values(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        # First k-point
        assert np.allclose(bands.bands[0, :, 0], [-8.5, -4.2, 1.3])
        # Second k-point
        assert np.allclose(bands.bands[1, :, 0], [-7.8, -3.9, 2.1])

    def test_mapping_interface(self) -> None:
        bands = Bands.from_str(BANDS_STR)
        assert bands["fermi_energy"] == -5.5
        assert len(bands) == 2  # Number of k-points


# Spin-polarized test
BANDS_STR_SPIN = """
-5.5000
0.0000 1.0000
-10.0000 5.0000
2 2 1
0.0000 -8.5 -4.2 -8.0 -3.8
"""


class TestBandsSpinPolarized:
    def test_n_spins(self) -> None:
        bands = Bands.from_str(BANDS_STR_SPIN)
        assert bands.n_spins == 2

    def test_bands_shape(self) -> None:
        bands = Bands.from_str(BANDS_STR_SPIN)
        assert bands.bands.shape == (1, 2, 2)

    def test_bands_values(self) -> None:
        bands = Bands.from_str(BANDS_STR_SPIN)
        # Spin up
        assert np.allclose(bands.bands[0, :, 0], [-8.5, -4.2])
        # Spin down
        assert np.allclose(bands.bands[0, :, 1], [-8.0, -3.8])
