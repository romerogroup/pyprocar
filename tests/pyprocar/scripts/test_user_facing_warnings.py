"""The one-call functions show their warnings and progress in a default call (#284)."""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyprocar
from pyprocar.core.ebs import ElectronicBandStructureMesh
from tests.pyprocar.core.test_ebs import make_hexagonal_ebs_path
from tests.pyprocar.core.test_fermisurface_noncollinear import sphere_mesh
from tests.pyprocar.scripts.test_fermi2d_spins import spin_polarized_mesh


class _Messages(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def user_output():
    """What the user logger emits, at whatever level the code under test leaves it."""
    loggers = [logging.getLogger("user"), logging.getLogger("pyprocar")]
    levels = [logger.level for logger in loggers]
    handler = _Messages()
    loggers[0].addHandler(handler)
    yield handler.messages
    loggers[0].removeHandler(handler)
    for logger, level in zip(loggers, levels, strict=True):
        logger.setLevel(level)
    plt.close("all")


def test_bandsplot_without_fermi_warns_that_the_bands_are_not_shifted(tmp_path):
    make_hexagonal_ebs_path(kpath_has_lattice=True).save(tmp_path / "ebs.pkl")

    with pytest.warns(UserWarning, match=r"`fermi` is not set! Set `fermi=\{value\}`"):
        _, ax = pyprocar.bandsplot(
            code="vasp", dirname=str(tmp_path), use_cache=True, show=False
        )

    assert ax.get_ylabel() == "E (eV)"
    plt.close("all")


@pytest.mark.parametrize(
    "method",
    ["plot_fermi_surface", "plot_fermi_cross_section", "plot_fermi_cross_section_box_widget"],
)
def test_fermi_handler_without_a_crossing_warns_and_plots_nothing(monkeypatch, user_output, method):
    ebs = sphere_mesh(1, np.full((2, 1, 2, 1), 0.5))

    def from_code(_cls: type[ElectronicBandStructureMesh], *_args: object, **_kwargs: object):
        return ebs

    monkeypatch.setattr(ElectronicBandStructureMesh, "from_code", classmethod(from_code))
    # Band 0 spans 0 to 0.75 eV and band 1 sits at 5 eV, so nothing crosses 50 eV.
    handler = pyprocar.FermiHandler(code="vasp", dirname="calc", fermi=50.0)

    with pytest.warns(UserWarning, match="No Fermi surface found for the given parameters"):
        result = getattr(handler, method)(mode="plain", show=False)

    assert result is None


def test_fermi2d_honours_its_verbose_argument(tmp_path, user_output):
    spin_polarized_mesh(2.0).save(tmp_path / "ebs.pkl")

    pyprocar.fermi2D(code="vasp", dirname=str(tmp_path), use_cache=True, show=False, verbose=0)
    silent = list(user_output)
    pyprocar.fermi2D(code="vasp", dirname=str(tmp_path), use_cache=True, show=False, verbose=1)

    assert silent == []
    assert "### Parameters ###" in user_output
    assert "k_z_plane       : 0.0" in user_output
