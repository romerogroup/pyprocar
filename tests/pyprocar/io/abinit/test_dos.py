import logging

import numpy as np
import pytest

from tests.pyprocar.io.abinit import ABINIT_DATA_DIR, CALC_TYPES
from tests.utils import BaseTest

pytestmark = pytest.mark.data

logger = logging.getLogger(__name__)


def get_dos_dirs():
    """Get all DOS directories for testing."""
    dirs = []
    for calc_type in CALC_TYPES:
        dirpath = ABINIT_DATA_DIR / calc_type / "dos"
        if dirpath.exists():
            dirs.append(dirpath)
    return dirs


@pytest.fixture(params=get_dos_dirs(), ids=lambda p: p.parent.name)
def dos_dirpath(request):
    return request.param


class TestAbinitDOSInit(BaseTest):
    def test_init_from_dirpath(self, dos_dirpath):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert dos is not None

    def test_total_dos_filepath_found(self, dos_dirpath):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert dos.total_dos_filepath is not None


class TestAbinitDOSTotal(BaseTest):
    def test_dos_total_is_ndarray(self, dos_dirpath):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert isinstance(dos.dos_total, np.ndarray)

    def test_energies_is_ndarray(self, dos_dirpath):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert isinstance(dos.energies, np.ndarray)

    def test_fermi_is_float(self, dos_dirpath):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        assert isinstance(dos.fermi, float)


class TestAbinitDOSProjected(BaseTest):
    def test_projected_is_ndarray_or_none(self, dos_dirpath):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        projected = dos.projected
        assert projected is None or isinstance(projected, np.ndarray)

    def test_projected_shape_has_4_dimensions(self, dos_dirpath):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(dos_dirpath)
        projected = dos.projected
        if projected is not None:
            # Shape: (n_energies, n_spins, n_atoms, n_orbitals)
            assert len(projected.shape) == 4


NSP_DOS = ABINIT_DATA_DIR / "non-spin-polarized" / "dos"
SP_DIR = ABINIT_DATA_DIR / "spin-polarized-colinear"


class TestAbinitDOSUnits(BaseTest):
    """Expected values are the abinito_DOS_TOTAL / DOS_AT0001 columns times 27.211386245988 eV/Ha."""

    def test_energies_are_absolute_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(NSP_DOS)
        assert dos.energies.shape == (13201,)
        # line 16: -2.90000 Ha; line 6001: 0.09250 Ha
        assert dos.energies[0] == pytest.approx(-78.9130201134)
        assert dos.energies[5985] == pytest.approx(2.5170532278)

    def test_fermi_is_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        # header "Fermi energy :       0.37035511"
        assert AbinitDOS(NSP_DOS).fermi == pytest.approx(10.0778759464)

    def test_total_dos_is_states_per_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        # line 6001: 3.0621 electrons/Ha
        assert AbinitDOS(NSP_DOS).dos_total[5985, 0] == pytest.approx(0.1125300994)

    def test_spin_down_block_is_states_per_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        # -2.86550 Ha: spin-up 72.8724, spin-down 72.8771 electrons/Ha
        dos = AbinitDOS(SP_DIR / "dos")
        assert dos.energies[69] == pytest.approx(-77.9742272879)
        assert dos.dos_total[69] == pytest.approx([2.6780113053, 2.6781840271])

    def test_projected_dos_is_states_per_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        # DOS_AT0001 at 0.09250 Ha: lm=1 0 column is 0.11, lm=0 0 is 1.05 electrons/Ha
        projected = AbinitDOS(NSP_DOS).projected
        assert projected is not None
        assert projected[5985, 0, 0, 0] == pytest.approx(0.0385867883)
        assert projected[5985, 0, 0, 2] == pytest.approx(0.0040424254)

    def test_dos_fermi_matches_bands_fermi(self):
        from pyprocar.io.abinit import AbinitParser

        # abinit.out "Fermi (or HOMO) energy (eV) =   9.11796"; DOS header 0.33507858 Ha
        bands_output = AbinitParser(SP_DIR / "bands").abinit_output
        dos = AbinitParser(SP_DIR / "dos").dos
        assert bands_output is not None and dos is not None
        assert bands_output.fermi == pytest.approx(9.11796)
        assert dos.fermi == pytest.approx(bands_output.fermi, abs=1e-4)


def test_dosplot_puts_abinit_fermi_at_zero():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pyprocar

    fig, ax = pyprocar.dosplot(code="abinit", dirname=str(NSP_DOS), orientation="vertical", show=False)
    energies = np.asarray(ax.get_lines()[0].get_ydata())
    plt.close(fig)
    # (-2.9 Ha - 0.37035511 Ha) and (3.7 Ha - 0.37035511 Ha), in eV
    assert energies.min() == pytest.approx(-88.9908960598)
    assert energies.max() == pytest.approx(90.6042531638)
