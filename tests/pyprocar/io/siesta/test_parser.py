"""Tests for SIESTA Parser."""

from pathlib import Path

import numpy as np
import pytest

from pyprocar.io.siesta import FDF, Bands, SiestaParser

FDF_STR = """
SystemLabel silicon

LatticeConstant 5.43 Ang

%block LatticeVectors
  1.00  0.00  0.00
  0.00  1.00  0.00
  0.00  0.00  1.00
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
  1  0.5  0.5  0.5  L
  1  0.0  0.0  0.0  G
%endblock BandLines
"""

# Bands file with 2 k-points, 3 bands, 1 spin (BandLines: L, then 1 point to G)
BANDS_STR = """
-5.5000
0.0000 1.0000
-10.0000 5.0000
3 1 2
0.0000 -8.5 -4.2 1.3
1.0000 -7.8 -3.9 2.1
"""


@pytest.fixture
def siesta_dir(tmp_path: Path) -> Path:
    """Create a temporary SIESTA calculation directory."""
    fdf_file = tmp_path / "silicon.fdf"
    fdf_file.write_text(FDF_STR)

    bands_file = tmp_path / "silicon.bands"
    bands_file.write_text(BANDS_STR)

    return tmp_path


class TestSiestaParser:
    def test_auto_detect_fdf(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir)
        assert parser._fdf is not None
        assert parser._fdf.system_label == "silicon"

    def test_auto_detect_bands(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir)
        assert parser._bands is not None
        assert parser._bands.n_kpoints == 2

    def test_fermi(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir)
        assert parser.fermi == -5.5

    def test_structure(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir)
        structure = parser.structure
        assert structure is not None
        atoms = structure.atoms
        assert atoms is not None
        assert len(atoms) == 2
        assert list(atoms) == ["Si", "Si"]

    def test_reciprocal_lattice(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir)
        recip = parser.reciprocal_lattice
        assert recip is not None
        assert np.allclose(recip, np.eye(3) / 5.43)

    def test_kpath(self, siesta_dir: Path) -> None:
        kpath = SiestaParser(siesta_dir).kpath

        assert kpath is not None
        assert list(zip(kpath.tick_positions, kpath.tick_names, strict=True)) == [
            (0, "L"),
            (1, "Γ"),
        ]

    @pytest.mark.parametrize(
        ("scale_line", "gamma_l"),
        [
            ("", np.sqrt(3) / 4 / 5.43),
            ("BandLinesScale pi/a\n", np.sqrt(3) / 4 / 5.43),
            ("BandLinesScale ReciprocalLatticeVectors\n", np.sqrt(3) / 2 / 5.43),
        ],
    )
    def test_kpath_distances_follow_band_lines_scale(
        self, siesta_dir: Path, scale_line: str, gamma_l: float
    ) -> None:
        (siesta_dir / "silicon.fdf").write_text(FDF_STR + scale_line)
        kpath = SiestaParser(siesta_dir).kpath

        assert kpath is not None
        assert kpath.k_distances == pytest.approx([0.0, gamma_l])

    def test_ebs(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir)
        ebs = parser.ebs
        assert ebs is not None

    def test_dos_returns_none(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir)
        assert parser.dos is None

    def test_custom_fdf_path(self, siesta_dir: Path) -> None:
        parser = SiestaParser(siesta_dir, fdf="silicon.fdf")
        assert parser._fdf is not None

    def test_pre_instantiated_extractors(self, siesta_dir: Path) -> None:
        fdf = FDF(siesta_dir / "silicon.fdf")
        bands = Bands(siesta_dir / "silicon.bands")

        parser = SiestaParser(siesta_dir, fdf=fdf, bands=bands)
        assert parser._fdf is fdf
        assert parser._bands is bands


class TestSiestaParserMissingFiles:
    def test_no_fdf_file(self, tmp_path: Path) -> None:
        parser = SiestaParser(tmp_path)
        assert parser._fdf is None
        assert parser.structure is None
        assert parser.ebs is None

    def test_no_bands_file(self, tmp_path: Path) -> None:
        fdf_file = tmp_path / "test.fdf"
        fdf_file.write_text(FDF_STR)

        parser = SiestaParser(tmp_path)
        assert parser._fdf is not None
        assert parser._bands is None
        assert parser.fermi is None


FCC_FDF_STR = """
SystemLabel fcc
LatticeConstant 5.43 Ang
%block LatticeVectors
  0.5  0.5  0.0
  0.0  0.5  0.5
  0.5  0.0  0.5
%endblock LatticeVectors
%block ChemicalSpeciesLabel
  1  14  Si
%endblock ChemicalSpeciesLabel
{format_line}
%block AtomicCoordinatesAndAtomicSpecies
  0.5  0.0  0.0  1
%endblock AtomicCoordinatesAndAtomicSpecies
%block BandLines
  1  0.000  0.000  0.000  G
 20  0.500  0.000  0.500  X
 20  0.500  0.250  0.750  W
%endblock BandLines
BandLinesScale ReciprocalLatticeVectors
"""


@pytest.mark.parametrize(
    ("format_line", "cartesian"),
    [
        ("AtomicCoordinatesFormat Fractional", [1.3575, 1.3575, 0.0]),
        ("AtomicCoordinatesFormat ScaledByLatticeVectors", [1.3575, 1.3575, 0.0]),
        ("AtomicCoordinatesFormat ScaledCartesian", [2.715, 0.0, 0.0]),
        ("AtomicCoordinatesFormat Ang", [0.5, 0.0, 0.0]),
        ("AtomicCoordinatesFormat NotScaledCartesianAng", [0.5, 0.0, 0.0]),
        ("AtomicCoordinatesFormat Bohr", [0.26458860533560, 0.0, 0.0]),
        ("AtomicCoordinatesFormat NotScaledCartesianBohr", [0.26458860533560, 0.0, 0.0]),
        ("", [0.26458860533560, 0.0, 0.0]),
    ],
)
def test_structure_reads_each_atomic_coordinates_format(
    tmp_path: Path, format_line: str, cartesian: list[float]
) -> None:
    (tmp_path / "fcc.fdf").write_text(FCC_FDF_STR.format(format_line=format_line))

    structure = SiestaParser(tmp_path).structure

    assert structure is not None
    assert structure.cartesian_coordinates is not None
    assert np.allclose(structure.cartesian_coordinates, [cartesian])


def test_kpath_has_siestas_band_line_point_count(tmp_path: Path) -> None:
    (tmp_path / "fcc.fdf").write_text(FCC_FDF_STR.format(format_line=""))

    kpath = SiestaParser(tmp_path).kpath

    assert kpath is not None
    assert kpath.n_kpoints == 41
    assert list(zip(kpath.tick_positions, kpath.tick_names, strict=True)) == [
        (0, "Γ"),
        (20, "X"),
        (40, "W"),
    ]


# si.fdf of a Siesta 5.4.2 run: simple cubic, a = 2.6 Ang, a one-point M-R row.
SIESTA_542_FDF = """
System-Label   si
%block chemical_species_label
 1 14 Si
%endblock chemical_species_label
LATTICE_CONSTANT 2.6 Ang
%block lattice-vectors
  1.0 0.0 0.0
  0.0 1.0 0.0
  0.0 0.0 1.0
%endblock lattice-vectors
atomic.coordinates.format NotScaledCartesianBohr
%block AtomicCoordinatesAndAtomicSpecies
  0.10 0.20 0.30 1
%endblock AtomicCoordinatesAndAtomicSpecies
%block BandLines
  1  0.0 0.0 0.0  \\Gamma
 20  1.0 0.0 0.0  X
 20  1.0 1.0 0.0  M
  1  1.0 1.0 1.0  R
 10  0.0 0.0 0.0  \\Gamma
%endblock BandLines
"""
# Tick x values that run wrote to si.bands, in 1/Bohr with the 2 pi factor.
SIESTA_542_TICK_X = [0.0, 0.639407, 1.278815, 1.918222, 3.025708]


def test_kpath_matches_siesta_542_ticks_with_a_one_point_row(tmp_path: Path) -> None:
    (tmp_path / "si.fdf").write_text(SIESTA_542_FDF)

    kpath = SiestaParser(tmp_path).kpath

    assert kpath is not None
    assert kpath.n_kpoints == 52
    assert kpath.tick_positions == [0, 20, 40, 41, 51]
    assert kpath.tick_names == ["$\\Gamma$", "X", "M", "R", "$\\Gamma$"]
    tick_x = np.asarray(kpath.k_distances)[kpath.tick_positions]
    siesta_x = np.array(SIESTA_542_TICK_X) / (2 * np.pi * 0.52917721067121)
    assert tick_x == pytest.approx(siesta_x, abs=1e-6)


def test_bands_and_fermi_survive_a_bad_unrelated_block(siesta_dir: Path) -> None:
    (siesta_dir / "silicon.fdf").write_text(FDF_STR + "%block Unrelated\n 1 2 3\n")

    parser = SiestaParser(siesta_dir)

    assert parser._bands is not None
    assert parser.fermi == -5.5
    assert parser.structure is not None


def test_auto_detect_skips_an_fdf_another_fdf_redirects_to(tmp_path: Path) -> None:
    # Layout of a Siesta 5.4.2 run: si.fdf reads its lattice from lv.fdf.
    start = SIESTA_542_FDF.index("%block lattice-vectors")
    end = SIESTA_542_FDF.index("%endblock lattice-vectors") + len("%endblock lattice-vectors")
    main = SIESTA_542_FDF[:start] + "%block LatticeVectors < lv.fdf" + SIESTA_542_FDF[end:]
    (tmp_path / "lv.fdf").write_text("  1.0 0.0 0.0\n  0.0 1.1 0.0\n  0.0 0.0 1.2\n")
    (tmp_path / "si.fdf").write_text(main)

    parser = SiestaParser(tmp_path)

    assert parser._fdf is not None
    assert parser._fdf.filepath == tmp_path / "si.fdf"
    structure = parser.structure
    assert structure is not None
    assert structure.lattice is not None
    assert np.allclose(np.diag(structure.lattice), [2.6, 2.86, 3.12])
