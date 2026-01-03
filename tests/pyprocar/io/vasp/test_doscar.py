import logging
from pathlib import Path

import numpy as np
import pytest

from pyprocar.io import vasp
from pyprocar.utils.log_utils import set_verbose_level

logger = logging.getLogger(__name__)

VERBOSE = 2
set_verbose_level(VERBOSE)


NON_COLINEAR_DOSCAR = """   2   2   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1200  0.0120
    -0.5000  0.2200  0.0220
     0.0000  0.3200  0.0320
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0100  0.0200  0.0300  0.0400  0.0500  0.0600  0.0700  0.0800  0.0900  0.1000  0.1100  0.1200
    -0.5000  0.0110  0.0210  0.0310  0.0410  0.0510  0.0610  0.0710  0.0810  0.0910  0.1010  0.1110  0.1210
     0.0000  0.0120  0.0220  0.0320  0.0420  0.0520  0.0620  0.0720  0.0820  0.0920  0.1020  0.1120  0.1220
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0130  0.0230  0.0330  0.0430  0.0530  0.0630  0.0730  0.0830  0.0930  0.1030  0.1130  0.1230
    -0.5000  0.0140  0.0240  0.0340  0.0440  0.0540  0.0640  0.0740  0.0840  0.0940  0.1040  0.1140  0.1240
     0.0000  0.0150  0.0250  0.0350  0.0450  0.0550  0.0650  0.0750  0.0850  0.0950  0.1050  0.1150  0.1250
"""


NON_SPIN_POLARIZED_DOSCAR = """   2   2   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1000  0.0100
    -0.5000  0.2000  0.0200
     0.0000  0.3000  0.0300
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0100  0.0200  0.0300
    -0.5000  0.0400  0.0500  0.0600
     0.0000  0.0700  0.0800  0.0900
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0150  0.0250  0.0350
    -0.5000  0.0450  0.0550  0.0650
     0.0000  0.0750  0.0850  0.0950
"""


SPIN_POLARIZED_DOSCAR = """   2   2   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1500 -0.1500  0.0150 -0.0150
    -0.5000  0.2500 -0.2500  0.0250 -0.0250
     0.0000  0.3500 -0.3500  0.0350 -0.0350
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0100  0.0200  0.0300  0.0400
    -0.5000  0.0500  0.0600  0.0700  0.0800
     0.0000  0.0900  0.1000  0.1100  0.1200
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.0120  0.0220  0.0320  0.0420
    -0.5000  0.0520  0.0620  0.0720  0.0820
     0.0000  0.0920  0.1020  0.1120  0.1220
"""


TOTAL_DOS_ONLY = """   1   1   1   0
  1.00000000  0.00000000  0.00000000  0.00000000  1.00000000
  0.00000000
  CAR 
 Test system                                
    1.00000000    -1.00000000 3      0.50000000      1.00000000
    -1.0000  0.1000  0.0100
    -0.5000  0.2000  0.0200
     0.0000  0.3000  0.0300
"""


@pytest.fixture
def non_spin_doscar_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_non_spin"
    filepath.write_text(NON_SPIN_POLARIZED_DOSCAR)
    return filepath


@pytest.fixture
def spin_doscar_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_spin"
    filepath.write_text(SPIN_POLARIZED_DOSCAR)
    return filepath


@pytest.fixture
def non_colinear_doscar_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_non_colinear"
    filepath.write_text(NON_COLINEAR_DOSCAR)
    return filepath


@pytest.fixture
def non_spin_doscar(non_spin_doscar_filepath: Path):
    return vasp.Doscar(non_spin_doscar_filepath)


@pytest.fixture
def spin_doscar(spin_doscar_filepath: Path):
    return vasp.Doscar(spin_doscar_filepath)


@pytest.fixture
def non_colinear_doscar(non_colinear_doscar_filepath: Path):
    return vasp.Doscar(non_colinear_doscar_filepath)


@pytest.fixture
def total_dos_only_filepath(tmp_path: Path) -> Path:
    filepath = tmp_path / "DOSCAR_total_only"
    filepath.write_text(TOTAL_DOS_ONLY)
    return filepath


@pytest.fixture
def total_dos_only_doscar(total_dos_only_filepath: Path):
    return vasp.Doscar(total_dos_only_filepath)


class TestDoscar:
    def test_doscar_from_str_sets_natoms(self) -> None:
        doscar = vasp.Doscar.from_str(SPIN_POLARIZED_DOSCAR)
        assert doscar.natoms == 2

    def test_doscar_is_spin_polarized(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.is_spin_pol is True

    def test_doscar_is_not_spin_polarized(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.is_spin_pol is False

    def test_doscar_energies_match_nedos(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.energies.shape[0] == spin_doscar.nedos

    def test_doscar_total_dos_spin_shape(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.total_dos.shape == (spin_doscar.nedos, 2)

    def test_doscar_total_dos_non_spin_shape(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.total_dos.shape == (non_spin_doscar.nedos, 1)

    def test_doscar_integrated_dos_spin_shape(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.integrated_dos.shape == (spin_doscar.nedos, 2)

    def test_doscar_integrated_dos_non_spin_shape(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.integrated_dos.shape == (non_spin_doscar.nedos, 1)

    def test_doscar_projected_dos_spin_available(self, spin_doscar: vasp.Doscar) -> None:
        assert spin_doscar.projected_dos is not None

    def test_doscar_projected_dos_non_spin_available(self, non_spin_doscar: vasp.Doscar) -> None:
        assert non_spin_doscar.projected_dos is not None

    def test_doscar_projected_dos_non_colinear_available(
        self, non_colinear_doscar: vasp.Doscar
    ) -> None:
        pdos = non_colinear_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos is not None

    def test_doscar_projected_dos_spin_channels(self, spin_doscar: vasp.Doscar) -> None:
        pdos = spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[1] == 2

    def test_doscar_projected_dos_non_spin_channels(self, non_spin_doscar: vasp.Doscar) -> None:
        pdos = non_spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[1] == 1

    def test_doscar_projected_dos_non_colinear_channels(
        self, non_colinear_doscar: vasp.Doscar
    ) -> None:
        pdos = non_colinear_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[1] == 4

    def test_doscar_projected_dos_matches_atom_count(self, spin_doscar: vasp.Doscar) -> None:
        pdos = spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[2] == spin_doscar.natoms

    def test_doscar_projected_dos_matches_energy_count(self, non_spin_doscar: vasp.Doscar) -> None:
        pdos = non_spin_doscar.projected_dos
        assert isinstance(pdos, np.ndarray)
        assert pdos.shape[0] == non_spin_doscar.nedos

    def test_doscar_projected_dos_missing_returns_none(
        self, total_dos_only_doscar: vasp.Doscar
    ) -> None:
        assert total_dos_only_doscar.projected_dos is None
