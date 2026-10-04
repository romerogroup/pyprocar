"""Tests for ElkParser integration."""

import logging
import shutil
import warnings

import numpy as np
import pytest

from pyprocar.core import DensityOfStates, Structure
from pyprocar.core.atomic_orbital_index import orbital_shells
from pyprocar.core.ebs import ElectronicBandStructure, ElectronicBandStructurePath
from pyprocar.core.kpoints import KPath
from pyprocar.io.elk import ElkParser
from tests.utils import DATA_DIR, BaseTest
from tests.utils.user_warning import user_warning

logger = logging.getLogger(__name__)


# Minimal elk.in for bands calculation
ELKIN_BANDS = """tasks
  0
  22

scale
  7.2589

avec
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

atoms
  2                                    : nspecies
  'Sr.in'                              : spfname
  1                                    : natoms
  0.0  0.0  0.0  0. 0. 0.
  'V.in'
  1
  0.5 0.5 0.5 0.0 0.0 1.0

plot1d
  3 10
   0.0  0.0  0.0 : G
   0.5  0.0  0.0 : X
   0.5  0.0625  0.0 : A
"""

# Minimal elk.in for DOS calculation (no plot1d)
ELKIN_DOS = """tasks
  0
  10

scale
  7.2589

avec
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0

atoms
  2                                    : nspecies
  'Sr.in'                              : spfname
  1                                    : natoms
  0.0  0.0  0.0  0. 0. 0.
  'V.in'
  1
  0.5 0.5 0.5 0.0 0.0 1.0
"""

EFERMI_OUT = """  0.3218543102
"""

GEOMETRY_OUT = """
scale
 1.0

avec
   7.258900000       0.000000000       0.000000000
   0.000000000       7.258900000       0.000000000
   0.000000000       0.000000000       7.258900000

atoms
   2                                    : nspecies
'Sr.in'                                 : spfname
   1                                    : natoms
    0.00000000    0.00000000    0.00000000    0.00000000  0.00000000  0.00000000
'V.in'                                  : spfname
   1                                    : natoms
    0.50000000    0.50000000    0.50000000    0.00000000  0.00000000  1.00000000
"""

BANDLINES_OUT = """   0.000000000      -4.832432739
   0.000000000       2.461158505

  0.4327918353      -4.832432739
  0.4327918353       2.461158505

  0.8655836707      -4.832432739
  0.8655836707       2.461158505

"""

# Minimal BAND.OUT - 10 k-points, 2 bands
BAND_OUT = """   0.000000000      -2.401220419        0.000002    0.000000
  0.0540989794      -2.401219000        0.000002    0.000000
  0.1081979588      -2.401218000        0.000002    0.000000
  0.1622969382      -2.401217000        0.000001    0.000000
  0.2163959176      -2.401216000        0.000001    0.000000
  0.2704948970      -2.401215000        0.000001    0.000000
  0.3245938764      -2.401214000        0.000000    0.000000
  0.3786928558      -2.401213000        0.000000    0.000001
  0.4327918353      -2.401212000        0.000000    0.000001
  0.8655836707      -2.401211000        0.000000    0.000001

   0.000000000      -1.452042593        0.000000    0.000009
  0.0540989794      -1.452100000        0.000010    0.000008
  0.1081979588      -1.452200000        0.000050    0.000007
  0.1622969382      -1.452300000        0.000100    0.000006
  0.2163959176      -1.452400000        0.000150    0.000005
  0.2704948970      -1.452500000        0.000200    0.000004
  0.3245938764      -1.452600000        0.000250    0.000003
  0.3786928558      -1.452700000        0.000300    0.000002
  0.4327918353      -1.452800000        0.000330    0.000001
  0.8655836707      -1.452900000        0.000350    0.000000

"""

# Projection file for Sr atom
BAND_S01_A0001 = """   0.000000000      -2.401220419        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.0540989794      -2.401219000        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.1081979588      -2.401218000        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.1622969382      -2.401217000        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2163959176      -2.401216000        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2704948970      -2.401215000        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3245938764      -2.401214000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3786928558      -2.401213000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.4327918353      -2.401212000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.8655836707      -2.401211000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000

   0.000000000      -1.452042593        0.000000    0.000005    0.000000    0.000005    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.0540989794      -1.452100000        0.000005    0.000004    0.000000    0.000004    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.1081979588      -1.452200000        0.000025    0.000004    0.000000    0.000004    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.1622969382      -1.452300000        0.000050    0.000003    0.000000    0.000003    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2163959176      -1.452400000        0.000075    0.000002    0.000000    0.000002    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2704948970      -1.452500000        0.000100    0.000002    0.000000    0.000002    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3245938764      -1.452600000        0.000125    0.000001    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3786928558      -1.452700000        0.000150    0.000001    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.4327918353      -1.452800000        0.000165    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.8655836707      -1.452900000        0.000175    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000

"""

# Projection file for V atom
BAND_S02_A0001 = """   0.000000000      -2.401220419        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.0540989794      -2.401219000        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.1081979588      -2.401218000        0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.1622969382      -2.401217000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2163959176      -2.401216000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2704948970      -2.401215000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3245938764      -2.401214000        0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3786928558      -2.401213000        0.000000    0.000001    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.4327918353      -2.401212000        0.000000    0.000001    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.8655836707      -2.401211000        0.000000    0.000001    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000

   0.000000000      -1.452042593        0.000000    0.000004    0.000000    0.000004    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.0540989794      -1.452100000        0.000005    0.000004    0.000000    0.000004    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.1081979588      -1.452200000        0.000025    0.000003    0.000000    0.000003    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000001
  0.1622969382      -1.452300000        0.000050    0.000003    0.000000    0.000003    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2163959176      -1.452400000        0.000075    0.000002    0.000000    0.000002    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.2704948970      -1.452500000        0.000100    0.000002    0.000000    0.000002    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3245938764      -1.452600000        0.000125    0.000001    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.3786928558      -1.452700000        0.000150    0.000001    0.000000    0.000001    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.4327918353      -1.452800000        0.000165    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000
  0.8655836707      -1.452900000        0.000175    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000

"""

# TDOS for DOS calculation
TDOS_OUT = """ -0.5000000000       0.000000000
 -0.4500000000       0.000000000
 -0.4000000000       0.000000000
 -0.3500000000       0.100000000
 -0.3000000000       0.500000000
"""


@pytest.fixture
def bands_calc_dir(tmp_path):
    """Create a temporary directory with all files for bands calculation."""
    (tmp_path / "elk.in").write_text(ELKIN_BANDS)
    (tmp_path / "FERMI.OUT").write_text(EFERMI_OUT)  # Parser expects FERMI.OUT
    (tmp_path / "GEOMETRY.OUT").write_text(GEOMETRY_OUT)
    (tmp_path / "BANDLINES.OUT").write_text(BANDLINES_OUT)
    (tmp_path / "BAND_S01_A0001.OUT").write_text(BAND_S01_A0001)
    (tmp_path / "BAND_S02_A0001.OUT").write_text(BAND_S02_A0001)
    return tmp_path


@pytest.fixture
def dos_calc_dir(tmp_path):
    """Create a temporary directory with all files for DOS calculation."""
    (tmp_path / "elk.in").write_text(ELKIN_DOS)
    (tmp_path / "FERMI.OUT").write_text(EFERMI_OUT)  # Parser expects FERMI.OUT
    (tmp_path / "GEOMETRY.OUT").write_text(GEOMETRY_OUT)
    (tmp_path / "TDOS.OUT").write_text(TDOS_OUT)
    return tmp_path


class TestElkParserInit(BaseTest):
    def test_parser_init_bands(self, bands_calc_dir):
        """Test ElkParser initialization for bands calculation."""
        parser = ElkParser(bands_calc_dir)
        assert parser is not None

    def test_parser_init_dos(self, dos_calc_dir):
        """Test ElkParser initialization for DOS calculation."""
        parser = ElkParser(dos_calc_dir)
        assert parser is not None


class TestElkParserStructure(BaseTest):
    def test_structure_type(self, bands_calc_dir):
        """Test that structure returns a Structure object."""
        parser = ElkParser(bands_calc_dir)
        assert isinstance(parser.structure, Structure)

    def test_structure_natoms(self, bands_calc_dir):
        """Test structure has correct number of atoms."""
        parser = ElkParser(bands_calc_dir)
        assert parser.structure.natoms == 2

    def test_structure_atoms(self, bands_calc_dir):
        """Test structure has correct atom list."""
        parser = ElkParser(bands_calc_dir)
        assert list(parser.structure.atoms) == ["Sr", "V"]


class TestElkParserFermi(BaseTest):
    def test_fermi_value(self, bands_calc_dir):
        """Test Fermi energy value."""
        parser = ElkParser(bands_calc_dir)
        # 0.3218543102 Hartree * 27.211386 = 8.757 eV
        assert parser.fermi == pytest.approx(8.757, rel=0.01)

    def test_nspin(self, bands_calc_dir):
        """Test nspin value."""
        parser = ElkParser(bands_calc_dir)
        assert parser.nspin == 1


class TestElkParserBands(BaseTest):
    def test_is_bands_calculation(self, bands_calc_dir):
        """Test is_bands_calculation is True."""
        parser = ElkParser(bands_calc_dir)
        assert parser.is_bands_calculation is True

    def test_is_bands_calculation_false_for_dos(self, dos_calc_dir):
        """Test is_bands_calculation is False for DOS calc."""
        parser = ElkParser(dos_calc_dir)
        assert parser.is_bands_calculation is False

    def test_ebs_type(self, bands_calc_dir):
        """Test that ebs returns ElectronicBandStructure."""
        parser = ElkParser(bands_calc_dir)
        assert isinstance(parser.ebs, ElectronicBandStructure)

    def test_ebs_none_for_dos(self, dos_calc_dir):
        """Test that ebs is None for DOS calculation."""
        parser = ElkParser(dos_calc_dir)
        assert parser.ebs is None

    def test_kpath_type(self, bands_calc_dir):
        """Test that kpath returns KPath."""
        parser = ElkParser(bands_calc_dir)
        assert isinstance(parser.kpath, KPath)

    def test_kpath_none_for_dos(self, dos_calc_dir):
        """Test that kpath is None for DOS calculation."""
        parser = ElkParser(dos_calc_dir)
        assert parser.kpath is None


class TestElkParserDOS(BaseTest):
    @pytest.mark.skip(
        reason="ElkDOS.total shape (nspin, nenergies) doesn't match DensityOfStates expected (n_energies, n_spin)"
    )
    def test_dos_type(self, dos_calc_dir):
        """Test that dos returns DensityOfStates."""
        parser = ElkParser(dos_calc_dir)
        assert isinstance(parser.dos, DensityOfStates)

    def test_dos_none_for_bands(self, bands_calc_dir):
        """Test that dos is None for bands calculation (no TDOS.OUT)."""
        parser = ElkParser(bands_calc_dir)
        assert parser.dos is None


ELK_DOS_DIR = DATA_DIR / "codes" / "elk" / "6.3" / "SrVO3"


@pytest.mark.parametrize(
    ("mag", "fermi", "dos_at_fermi"),
    [
        ("non-spin-polarized", 9.10040, [1.42933]),
        ("spin-polarized-colinear", 9.19282, [0.71666, 0.82630]),
    ],
)
@pytest.mark.data
def test_real_dos_is_in_core_layout_in_ev(
    mag: str, fermi: float, dos_at_fermi: list[float]
) -> None:
    dos = ElkParser(ELK_DOS_DIR / mag / "dos").dos

    assert dos is not None and dos.projected is not None
    nspin = len(dos_at_fermi)
    total = dos.total.to_array()
    assert total.shape == (500, nspin)
    assert dos.projected.to_array().shape == (500, nspin, 5, 16)
    assert dos.fermi == pytest.approx(fermi, abs=1e-4)
    assert (dos.energies[0], dos.energies[-1]) == pytest.approx(
        (fermi - 13.60569, fermi + 13.55127), abs=1e-4
    )
    ifermi = int(np.argmin(np.abs(dos.energies - fermi)))
    assert total[ifermi] == pytest.approx(dos_at_fermi, abs=1e-4)
    occupied = dos.energies <= fermi
    electrons = total[occupied].sum() * (dos.energies[1] - dos.energies[0])
    assert electrons == pytest.approx(18.8, abs=0.2)
    muffin_tin_sum = dos.projected.to_array().sum(axis=(2, 3))
    assert (muffin_tin_sum <= total + 1e-9).all()


@pytest.mark.parametrize("with_geometry_out", [True, False])
def test_lattice_is_angstrom_and_reciprocal_lattice_is_inverse_angstrom_without_two_pi(
    bands_calc_dir, with_geometry_out: bool
):
    if not with_geometry_out:
        (bands_calc_dir / "GEOMETRY.OUT").unlink()
    parser = ElkParser(bands_calc_dir)
    structure, reciprocal_lattice, kpath = (
        parser.structure,
        parser.reciprocal_lattice,
        parser.kpath,
    )
    assert structure is not None and reciprocal_lattice is not None and kpath is not None
    lattice = structure.lattice
    assert lattice is not None
    assert np.allclose(lattice, np.eye(3) * 3.841244, atol=1e-6)
    assert np.allclose(reciprocal_lattice, np.eye(3) * 0.260332, atol=1e-6)
    assert np.allclose(reciprocal_lattice, np.linalg.inv(lattice).T)
    assert np.allclose(
        kpath.kpoints[:4], [[0, 0, 0], [0.0625, 0, 0], [0.125, 0, 0], [0.1875, 0, 0]]
    )
    assert kpath.get_distances(as_segments=False, cartesian=True)[2] == pytest.approx(
        0.125 * 0.260332, abs=1e-6
    )


SCALED_ELKIN = """tasks
  0

scale
  2.0

scale1
  1.5

scale3
  3.0

scalex
  2.0

avec
  1.0 1.0 0.0
  0.0 1.0 0.0
  0.0 0.0 1.0

atoms
  1                                    : nspecies
  'Si.in'                              : spfname
  2                                    : natoms; atpos, bfcmt below
  0.0 0.0 0.0
  0.25 0.75 0.5    0.0 0.0 0.0
"""

MOLECULE_ELKIN = """tasks
  0

molecule
  .true.

avec
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 20.0

atoms
  1
  'N.in'
  2
  2.5 0.0 -1.0
  2.5 0.0  1.0
"""

MOLECULE_GEOMETRY_OUT = """
scale
 1.0

scale1
 1.0

scale2
 1.0

scale3
 1.0

avec
   10.00000000       0.000000000       0.000000000
   0.000000000       10.00000000       0.000000000
   0.000000000       0.000000000       20.00000000

molecule
 T

atoms
   1                                    : nspecies
'N.in'                                  : spfname
   2                                    : natoms; atpos, bfcmt below
    2.50000000    0.00000000   -1.00000000    0.00000000  0.00000000  0.00000000
    2.50000000    0.00000000    1.00000000    0.00000000  0.00000000  0.00000000
"""


def test_elkin_fallback_applies_scale_scale123_and_scalexyz_to_the_lattice(tmp_path):
    (tmp_path / "elk.in").write_text(SCALED_ELKIN)
    structure = ElkParser(tmp_path).structure

    assert structure is not None
    lattice, fractional = structure.lattice, structure.fractional_coordinates
    assert lattice is not None and fractional is not None
    assert np.allclose(
        lattice,
        [[3.175063, 1.587532, 0.0], [0.0, 1.058354, 0.0], [0.0, 0.0, 3.175063]],
        atol=1e-6,
    )
    assert np.allclose(fractional, [[0, 0, 0], [0.25, 0.75, 0.5]])


@pytest.mark.parametrize(
    ("filename", "content"),
    [("elk.in", MOLECULE_ELKIN), ("GEOMETRY.OUT", MOLECULE_GEOMETRY_OUT)],
    ids=["elk.in", "GEOMETRY.OUT"],
)
def test_molecule_positions_are_cartesian_bohr(tmp_path, filename, content):
    (tmp_path / filename).write_text(content)
    structure = ElkParser(tmp_path).structure

    assert structure is not None
    lattice, fractional = structure.lattice, structure.fractional_coordinates
    cartesian = structure.cartesian_coordinates
    assert lattice is not None and fractional is not None and cartesian is not None
    assert np.allclose(lattice, np.diag([5.291772, 5.291772, 10.583544]), atol=1e-6)
    assert np.allclose(fractional, [[0.25, 0, -0.05], [0.25, 0, 0.05]])
    assert np.allclose(cartesian, [[1.322943, 0, -0.529177], [1.322943, 0, 0.529177]], atol=1e-6)


def test_ebs_kpoints_are_fractional_and_kdirect_is_gone(bands_calc_dir):
    with pytest.raises(TypeError):
        ElkParser(bands_calc_dir, kdirect=False)  # pyright: ignore[reportCallIssue]
    ebs = ElkParser(bands_calc_dir).ebs

    assert isinstance(ebs, ElectronicBandStructurePath)
    assert np.allclose(ebs.kpath.kpoints[:2], [[0, 0, 0], [0.0625, 0, 0]])
    assert np.allclose(ebs.kpoints_cartesian[:2], [[0, 0, 0], [0.0625 * 0.260332, 0, 0]], atol=1e-7)


@pytest.mark.parametrize(
    ("task", "files"),
    [
        ("22", {"BAND_S01_A0001.OUT": BAND_S01_A0001, "BAND_S02_A0001.OUT": BAND_S02_A0001}),
        ("20", {"BAND.OUT": BAND_OUT}),
    ],
    ids=["task22-BAND_S", "task20-BAND.OUT"],
)
def test_bands_come_from_the_files_elk_writes(tmp_path, task, files):
    (tmp_path / "elk.in").write_text(ELKIN_BANDS.replace("  22\n", f"  {task}\n"))
    (tmp_path / "FERMI.OUT").write_text(EFERMI_OUT)
    (tmp_path / "BANDLINES.OUT").write_text(BANDLINES_OUT)
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    ebs = ElkParser(tmp_path).ebs

    assert isinstance(ebs, ElectronicBandStructurePath) and ebs.bands is not None
    bands = ebs.bands.to_array()
    assert bands.shape == (11, 2, 1)
    assert bands[0, :, 0] == pytest.approx([-56.582434, -30.753990], abs=1e-5)


@pytest.mark.data
@pytest.mark.parametrize(
    ("mag", "shape", "first_band_at_gamma"),
    [
        ("non-spin-polarized", (54, 41, 1), [-56.582434]),
        ("spin-polarized-colinear", (44, 71, 2), [-55.780498, -56.116906]),
    ],
)
def test_real_elk_bands_read_band_s_files(mag, shape, first_band_at_gamma):
    ebs = ElkParser(ELK_DOS_DIR / mag / "bands").ebs

    assert isinstance(ebs, ElectronicBandStructurePath) and ebs.bands is not None
    bands = ebs.bands.to_array()
    assert bands.shape == shape
    assert bands[0, 0, :] == pytest.approx(first_band_at_gamma, abs=1e-5)
    assert ebs.kpath.tick_names == ["Γ", "X", "M", "Γ", "R", "X"]
    kpoints = np.asarray(ebs.kpath.kpoints)
    assert np.allclose(
        kpoints[ebs.kpath.tick_positions],
        [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
    )


def test_repeated_block_keeps_the_last_copy_like_elk(tmp_path):
    (tmp_path / "elk.in").write_text(
        "avec\n1 0 0\n0 1 0\n0 0 1\n\navec\n2 0 0\n0 2 0\n0 0 2\n\n"
        + "atoms\n1\n'Si.in'\n1\n0 0 0\n"
    )
    structure = ElkParser(tmp_path).structure

    assert structure is not None and structure.lattice is not None
    assert np.allclose(structure.lattice, np.eye(3) * 1.058354, atol=1e-6)


def test_inline_comments_on_keyword_lines_are_ignored(tmp_path):
    (tmp_path / "elk.in").write_text(
        ELKIN_BANDS.replace("scale\n", "scale : global\n")
        .replace("plot1d\n", "plot1d : path\n")
        .replace("tasks\n", "tasks : run\n")
        + "\nspinpol : collinear\n  .true.\n"
    )
    parser = ElkParser(tmp_path)

    assert parser.nspin == 2
    assert parser.is_bands_calculation
    assert parser.elkin is not None
    assert parser.elkin.nkpoints == 10
    assert parser.elkin.high_symmetry_points.tolist() == [[0, 0, 0], [0.5, 0, 0], [0.5, 0.0625, 0]]


def test_missing_plot1d_uses_the_elk_default_path_and_warns(tmp_path):
    (tmp_path / "elk.in").write_text(ELKIN_BANDS.split("plot1d")[0])
    with user_warning(__file__, match="elk.in has no plot1d block"):
        elkin = ElkParser(tmp_path).elkin
        assert elkin is not None
        assert elkin.nkpoints == 200

    assert elkin.high_symmetry_points.tolist() == [[0, 0, 0], [1, 1, 1]]


def test_structure_from_elk_in_warns_that_geometry_out_is_missing(tmp_path):
    (tmp_path / "elk.in").write_text(ELKIN_BANDS)
    with user_warning(__file__, match="No GEOMETRY.OUT in"):
        assert ElkParser(tmp_path).structure is not None


def test_band_path_repeats_each_inner_vertex_so_kpath_finds_every_segment(bands_calc_dir):
    ebs = ElkParser(bands_calc_dir).ebs

    assert isinstance(ebs, ElectronicBandStructurePath) and ebs.bands is not None
    kpoints = np.asarray(ebs.kpath.kpoints)
    assert len(kpoints) == 11 and ebs.bands.to_array().shape == (11, 2, 1)
    assert np.allclose(
        kpoints[[0, 8, 9, 10]], [[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0.5, 0.0625, 0]]
    )


def _band_dir(tmp_path, task, files):
    (tmp_path / "elk.in").write_text(ELKIN_BANDS.replace("  22\n", f"  {task}\n"))
    (tmp_path / "FERMI.OUT").write_text(EFERMI_OUT)
    (tmp_path / "BANDLINES.OUT").write_text(BANDLINES_OUT)
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    return tmp_path


def test_band_file_follows_the_elk_in_task_and_warns_about_the_other(tmp_path):
    stale = BAND_OUT.replace("-2.401220419", "-9.000000000")
    calc_dir = _band_dir(
        tmp_path,
        "22",
        {
            "BAND.OUT": stale,
            "BAND_S01_A0001.OUT": BAND_S01_A0001,
            "BAND_S02_A0001.OUT": BAND_S02_A0001,
        },
    )
    with user_warning(__file__, match="Both BAND.OUT and BAND_S01_A0001.OUT are in"):
        ebs = ElkParser(calc_dir).ebs

    assert isinstance(ebs, ElectronicBandStructurePath) and ebs.bands is not None
    assert ebs.bands.to_array()[0, 0, 0] == pytest.approx(-56.582434, abs=1e-5)


def test_vertex_labels_keep_latex_that_kpath_does_not_alias(tmp_path):
    calc_dir = _band_dir(
        tmp_path, "22", {"BAND_S01_A0001.OUT": BAND_S01_A0001, "BAND_S02_A0001.OUT": BAND_S02_A0001}
    )
    (calc_dir / "elk.in").write_text(
        (calc_dir / "elk.in").read_text().replace(": G", ": \\Gamma").replace(": A", ": \\Sigma_1")
    )
    ebs = ElkParser(calc_dir).ebs

    assert isinstance(ebs, ElectronicBandStructurePath)
    assert ebs.kpath.tick_names == ["Γ", "X", "$\\Sigma_1$"]


def test_task_20_bands_carry_no_projections_from_stale_band_s_files(tmp_path):
    calc_dir = _band_dir(
        tmp_path,
        "20",
        {
            "BAND.OUT": BAND_OUT,
            "BAND_S01_A0001.OUT": BAND_S01_A0001,
            "BAND_S02_A0001.OUT": BAND_S02_A0001,
        },
    )
    ebs = ElkParser(calc_dir).ebs

    assert isinstance(ebs, ElectronicBandStructurePath) and ebs.bands is not None
    assert ebs.bands.to_array().shape == (11, 2, 1)
    assert ebs.projected is None


DISTANCES = [line.split()[0] for line in BAND_OUT.splitlines()[:10]]


def _band_s(states: list[tuple[float, list[float]]]) -> str:
    lines = []
    for energy, characters in states:
        for ik, distance in enumerate(DISTANCES):
            values = "".join(f"{c + 0.001 * ik:12.6f}" for c in characters)
            lines.append(f"{distance:>18}{energy:18.10f}{values}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _band_task_dir(tmp_path, task: int, spinpol: bool, atom_states) -> ElkParser:
    elkin = ELKIN_BANDS.replace("  22\n", f"  {task}\n")
    if spinpol:
        elkin += "\nspinpol\n  .true.\n"
    (tmp_path / "elk.in").write_text(elkin)
    (tmp_path / "FERMI.OUT").write_text(EFERMI_OUT)
    (tmp_path / "GEOMETRY.OUT").write_text(GEOMETRY_OUT)
    (tmp_path / "BANDLINES.OUT").write_text(BANDLINES_OUT)
    for name, states in zip(("BAND_S01_A0001.OUT", "BAND_S02_A0001.OUT"), atom_states, strict=True):
        (tmp_path / name).write_text(_band_s(states))
    return ElkParser(tmp_path)


def _first_and_last_kpoint(ebs: ElectronicBandStructure) -> np.ndarray:
    projected = ebs.projected
    assert projected is not None
    return projected.to_array()[[0, -1]]


def test_task_23_spin_characters_go_to_each_state_spin_channel(tmp_path):
    parser = _band_task_dir(
        tmp_path,
        23,
        spinpol=True,
        atom_states=[
            [(-0.30, [0.100, 0.000]), (-0.28, [0.000, 0.200])],
            [(-0.30, [0.300, 0.000]), (-0.28, [0.000, 0.400])],
        ],
    )

    ebs = parser.ebs
    assert ebs is not None and ebs.bands is not None
    assert ebs.bands.to_array()[0, 0] == pytest.approx(
        [(-0.30 + 0.3218543102) * 27.211386245988, (-0.28 + 0.3218543102) * 27.211386245988]
    )
    assert ebs.orbital_names == ["spin"]
    ends = _first_and_last_kpoint(ebs)
    assert ends.shape == (2, 1, 2, 2, 1)
    assert ends[0, 0, :, :, 0].transpose() == pytest.approx(
        np.array([[0.100, 0.200], [0.300, 0.400]])
    )
    assert ends[1, 0, :, :, 0].transpose() == pytest.approx(
        np.array([[0.109, 0.209], [0.309, 0.409]])
    )


def test_task_24_moment_character_is_read_per_atom(tmp_path):
    parser = _band_task_dir(
        tmp_path,
        24,
        spinpol=True,
        atom_states=[
            [(-0.30, [0.300]), (-0.28, [-0.250])],
            [(-0.30, [0.050]), (-0.28, [-0.020])],
        ],
    )

    ebs = parser.ebs
    assert ebs is not None
    assert ebs.orbital_names == ["moment"]
    ends = _first_and_last_kpoint(ebs)
    assert ends[0, 0, :, :, 0].transpose() == pytest.approx(
        np.array([[0.300, -0.250], [0.050, -0.020]])
    )


def test_task_21_reads_l_characters_without_the_sum_column(tmp_path):
    parser = _band_task_dir(
        tmp_path,
        21,
        spinpol=False,
        atom_states=[
            [(-0.30, [0.600, 0.100, 0.200, 0.300, 0.000])],
            [(-0.30, [0.040, 0.010, 0.000, 0.030, 0.000])],
        ],
    )

    ebs = parser.ebs
    assert ebs is not None
    assert ebs.orbital_names == ["s", "p", "d", "f"]
    ends = _first_and_last_kpoint(ebs)
    assert ends[0, 0, 0, 0, :] == pytest.approx([0.100, 0.200, 0.300, 0.000])
    assert ends[0, 0, 0, 1, :] == pytest.approx([0.010, 0.000, 0.030, 0.000])


ELK_BANDS_SP = DATA_DIR / "codes" / "elk" / "6.3" / "SrVO3" / "spin-polarized-colinear" / "bands"


@pytest.mark.data
def test_real_spin_polarized_task_22_reads_the_spin_down_states():
    ebs = ElkParser(ELK_BANDS_SP).ebs

    assert ebs is not None and ebs.projected is not None
    projected = ebs.projected.to_array()
    assert projected.shape == (44, 71, 2, 5, 16)
    assert projected[0, 70, 0, 1, :].sum() == pytest.approx(0.135255, abs=1e-6)
    assert projected[0, 70, 1, 1, :].sum() == pytest.approx(0.132893, abs=1e-6)
    assert projected[0, 0, :, 1, 0] == pytest.approx([0.989874, 0.989950])
    assert ebs.orbital_names is not None and len(ebs.orbital_names) == 16


def test_task_20_reads_bands_from_band_s_files_without_their_characters(tmp_path):
    calc_dir = _band_dir(
        tmp_path, "20", {"BAND_S01_A0001.OUT": BAND_S01_A0001, "BAND_S02_A0001.OUT": BAND_S02_A0001}
    )

    with user_warning(__file__, match="elk.in lists no task 21 to 24"):
        ebs = ElkParser(calc_dir).ebs
        assert isinstance(ebs, ElectronicBandStructurePath) and ebs.bands is not None
        assert ebs.projected is None

    assert ebs.bands.to_array()[0, 0, 0] == pytest.approx(
        (-2.401220419 + 0.3218543102) * 27.211386245988
    )


YLM_NAMES = [f"Y{ang}{m}" for ang in range(4) for m in range(-ang, ang + 1)]
IRREP_NAMES = [f"Y{ang}_ir{i}" for ang in range(4) for i in range(1, 2 * ang + 2)]
ELMIREP_OUT = "\nSpecies :    1 (Sr), atom :    1\n l =  0, m =  0, lm=   1 :    528.0\n"


def elk_11_info(version):
    """The INFO.OUT header that writeinfo.f90 of Elk 11.2.3 writes."""
    label = f"Elk code version {version}"
    rule = "─" * (len(label) + 2)
    return f"\n┌{rule}┐\n│ {label} │\n└{rule}┘\n"


ELK_6_3_INFO = (
    "+----------------------------+\r\n| Elk version 6.3.02 started |\r\n"
    "+----------------------------+\r\n"
)


def _task_22_dir(tmp_path, info=None, elmirep=False, extra_elkin="", tasks="  0\n  22\n"):
    calc_dir = _band_dir(
        tmp_path, "22", {"BAND_S01_A0001.OUT": BAND_S01_A0001, "BAND_S02_A0001.OUT": BAND_S02_A0001}
    )
    (calc_dir / "elk.in").write_text(ELKIN_BANDS.replace("  0\n  22\n", tasks) + extra_elkin)
    if info is not None:
        (calc_dir / "INFO.OUT").write_text(info, encoding="utf-8")
    if elmirep:
        (calc_dir / "ELMIREP.OUT").write_text(ELMIREP_OUT)
    return calc_dir


@pytest.mark.parametrize(
    "setup",
    [
        {"info": elk_11_info("10.7.8")},
        {"info": elk_11_info("11.2.3"), "elmirep": True},
    ],
    ids=["10.7.8", "11.2.3"],
)
def test_task_22_characters_from_elk_10_7_8_are_named_by_irreducible_representation(
    tmp_path, setup
):
    ebs = ElkParser(_task_22_dir(tmp_path, **setup)).ebs

    assert ebs is not None
    assert ebs.orbital_names == IRREP_NAMES
    assert [letter for letter, _ in orbital_shells(ebs.orbital_names)] == ["s", "p", "d", "f"]


@pytest.mark.guards_existing_behaviour(
    reason="Ylm-basis runs keep the names they had; the irrep-basis tests are the red ones"
)
@pytest.mark.parametrize(
    "setup",
    [
        {"info": elk_11_info("10.7.8"), "extra_elkin": "\nlmirep\n  .false.\n"},
        {"info": elk_11_info("10.7.7")},
        {"info": ELK_6_3_INFO, "elmirep": True},
        {},
    ],
    ids=["lmirep-false", "10.7.7", "6.3-with-task-10-elmirep", "no-version-no-elmirep"],
)
def test_task_22_characters_in_the_ylm_basis_keep_their_l_m_names(tmp_path, setup):
    calc_dir = _task_22_dir(tmp_path, **setup)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*INFO.OUT")
        ebs = ElkParser(calc_dir).ebs

    assert ebs is not None
    assert ebs.orbital_names == YLM_NAMES


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        ({"elmirep": True}, "no INFO.OUT"),
        ({"elmirep": True, "tasks": "  0\n  10\n  22\n"}, "no INFO.OUT"),
        ({"elmirep": True, "info": "\n| Ground-state run |\n"}, "INFO.OUT names no Elk version"),
    ],
    ids=["no-info", "no-info-task-10", "info-without-version"],
)
def test_task_22_basis_that_cannot_be_told_apart_keeps_l_m_names_and_warns(tmp_path, setup, match):
    calc_dir = _task_22_dir(tmp_path, **setup)

    with user_warning(__file__, match=match):
        ebs = ElkParser(calc_dir).ebs
        assert ebs is not None
        assert ebs.orbital_names == YLM_NAMES


@pytest.mark.data
@pytest.mark.guards_existing_behaviour(
    reason="Elk 6.3 always wrote Ylm characters; #285's first head renamed them by irrep"
)
def test_real_elk_6_3_bands_beside_a_task_10_elmirep_keep_their_l_m_names(tmp_path):
    calc_dir = tmp_path / "bands"
    shutil.copytree(
        ELK_BANDS_SP,
        calc_dir,
        ignore=shutil.ignore_patterns("EVEC*.OUT", "STATE.OUT", "VARIABLES.OUT"),
    )
    shutil.copy(ELK_BANDS_SP.parent / "dos" / "ELMIREP.OUT", calc_dir)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*INFO.OUT")
        ebs = ElectronicBandStructurePath.from_code("elk", calc_dir)

    assert ebs.orbital_names == YLM_NAMES
