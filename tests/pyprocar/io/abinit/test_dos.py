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
    def test_energies_are_absolute_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(NSP_DOS)
        assert dos.energies.shape == (13201,)
        assert dos.energies[0] == pytest.approx(-78.9130201134)
        assert dos.energies[5985] == pytest.approx(2.5170532278)

    def test_fermi_is_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        assert AbinitDOS(NSP_DOS).fermi == pytest.approx(10.0778759464)

    def test_total_dos_is_states_per_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        assert AbinitDOS(NSP_DOS).dos_total[5985, 0] == pytest.approx(0.1125300994)

    def test_spin_down_block_is_states_per_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        dos = AbinitDOS(SP_DIR / "dos")
        assert dos.energies[69] == pytest.approx(-77.9742272879)
        assert dos.dos_total[69] == pytest.approx([2.6780113053, 2.6781840271])

    def test_projected_dos_is_states_per_ev(self):
        from pyprocar.io.abinit import AbinitDOS

        projected = AbinitDOS(NSP_DOS).projected
        assert projected is not None
        assert projected[5985, 0, 0, 0] == pytest.approx(0.0385867883)
        assert projected[5985, 0, 0, 2] == pytest.approx(0.0040424254)

    def test_dos_fermi_matches_bands_fermi(self):
        from pyprocar.io.abinit import AbinitParser

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

    fig, ax = pyprocar.dosplot(
        code="abinit", dirname=str(NSP_DOS), orientation="vertical", show=False
    )
    energies = np.asarray(ax.get_lines()[0].get_ydata())
    plt.close(fig)
    assert energies.min() == pytest.approx(-88.9908960598)
    assert energies.max() == pytest.approx(90.6042531638)


NCL_DOS = ABINIT_DATA_DIR / "non-colinear" / "dos"
NCL_DOS_FILES = ("abinit.out", "abinito_DOS_TOTAL", "abinito_DOS_AT0001")


def test_non_magnetic_spinor_dos_has_total_and_zero_spin_channels():
    """abinit.in sets nspinor 2 and nspden 1, so the run carries no magnetization.

    The l = 0, 1, 2 columns (2 to 4) of abinito_DOS_AT0001 integrate to 21.833
    electrons by the trapezoid rule on the file's own energy grid.
    """
    from pyprocar.io import get_parser

    dos = get_parser("abinit", NCL_DOS).dos

    assert dos is not None and dos.projected is not None
    projected = dos.projected.to_array()
    assert projected.shape == (9401, 4, 1, 9)
    assert dos.is_non_collinear
    assert np.trapezoid(projected[:, 0].sum(axis=(1, 2)), dos.energies) == pytest.approx(
        21.833, abs=0.005
    )
    assert np.all(projected[:, 1:] == 0.0)


def test_magnetic_spinor_dos_keeps_only_the_total_and_says_why(tmp_path, caplog):
    for name in NCL_DOS_FILES:
        (tmp_path / name).write_bytes((NCL_DOS / name).read_bytes())
    out = tmp_path / "abinit.out"
    out.write_text(out.read_text().replace("nspden =       1", "nspden =       4"))
    from pyprocar.io import get_parser

    user_logger = logging.getLogger("user")
    user_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="user"):
            dos = get_parser("abinit", tmp_path).dos
    finally:
        user_logger.removeHandler(caplog.handler)

    assert dos is not None and dos.projected is not None
    assert dos.projected.to_array().shape == (9401, 1, 1, 9)
    assert "hold no magnetization components" in caplog.text
