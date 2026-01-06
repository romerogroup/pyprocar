"""Tests for SIESTA FDF extractor."""

import numpy as np

from pyprocar.io.siesta import FDF

FDF_STR = """
SystemLabel silicon

LatticeConstant 5.43 Ang

%block LatticeVectors
  0.5  0.5  0.0
  0.0  0.5  0.5
  0.5  0.0  0.5
%endblock LatticeVectors

%block ChemicalSpeciesLabel
  1  14  Si
%endblock ChemicalSpeciesLabel

AtomicCoordinatesFormat Fractional

%block AtomicCoordinatesAndAtomicSpecies
  0.00  0.00  0.00  1
  0.25  0.25  0.25  1
%endblock AtomicCoordinatesAndAtomicSpecies

%block BandLines
  1  0.500  0.500  0.500  L
 20  0.000  0.000  0.000  G
 20  0.500  0.000  0.500  X
%endblock BandLines
"""


class TestFDF:
    def test_system_label(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.system_label == "silicon"

    def test_lattice_constant(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.lattice_constant == 5.43

    def test_lattice_vectors(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.lattice_vectors.shape == (3, 3)
        expected = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ])
        assert np.allclose(fdf.lattice_vectors, expected)

    def test_atomic_coords_format(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.atomic_coords_format == "Fractional"

    def test_species_labels(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.species_labels == {"1": "Si"}

    def test_atoms(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.atoms == ["Si", "Si"]

    def test_atomic_positions(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.atomic_positions.shape == (2, 3)
        assert np.allclose(fdf.atomic_positions[0], [0.0, 0.0, 0.0])
        assert np.allclose(fdf.atomic_positions[1], [0.25, 0.25, 0.25])

    def test_has_band_lines(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf.has_band_lines is True

    def test_band_lines(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        band_lines = fdf.band_lines
        assert band_lines is not None
        assert len(band_lines) == 3
        assert band_lines[0]["label"] == "L"
        assert band_lines[1]["label"] == "G"
        assert band_lines[2]["label"] == "X"

    def test_mapping_interface(self) -> None:
        fdf = FDF.from_str(FDF_STR)
        assert fdf["system_label"] == "silicon"
        assert len(fdf) == 2  # Number of atoms
        assert list(fdf) == ["Si", "Si"]
