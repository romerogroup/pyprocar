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
from tests.pyprocar.io.elk.test_parser import (
    BAND_S01_A0001,
    BAND_S02_A0001,
    BANDLINES_OUT,
    EFERMI_OUT,
    ELKIN_BANDS,
)
from tests.pyprocar.scripts.test_fermi2d_spins import spin_polarized_mesh
from tests.utils.user_warning import user_warning


@pytest.fixture
def live_user_log(caplog):
    loggers = [logging.getLogger("user"), logging.getLogger("pyprocar")]
    levels = [logger.level for logger in loggers]
    loggers[0].addHandler(caplog.handler)
    yield caplog
    loggers[0].removeHandler(caplog.handler)
    for logger, level in zip(loggers, levels, strict=True):
        logger.setLevel(level)
    plt.close("all")


def test_bandsplot_without_fermi_warns_that_the_bands_are_not_shifted(tmp_path):
    make_hexagonal_ebs_path(kpath_has_lattice=True).save(tmp_path / "ebs.pkl")

    with user_warning(__file__, match=r"`fermi` is not set! Set `fermi=\{value\}`"):
        _, ax = pyprocar.bandsplot(code="vasp", dirname=str(tmp_path), use_cache=True, show=False)

    assert ax.get_ylabel() == "E (eV)"
    plt.close("all")


def test_bandsplot_names_the_callers_line_for_a_warning_from_a_cached_parser_member(tmp_path):
    (tmp_path / "elk.in").write_text(ELKIN_BANDS)
    (tmp_path / "FERMI.OUT").write_text(EFERMI_OUT)
    (tmp_path / "BANDLINES.OUT").write_text(BANDLINES_OUT)
    (tmp_path / "BAND_S01_A0001.OUT").write_text(BAND_S01_A0001)
    (tmp_path / "BAND_S02_A0001.OUT").write_text(BAND_S02_A0001)

    with user_warning(__file__, match="No GEOMETRY.OUT in"):
        _, ax = pyprocar.bandsplot(code="elk", dirname=str(tmp_path), fermi=0.0, show=False)

    assert ax.get_ylabel() == r"E - E$_F$ (eV)"
    plt.close("all")


def test_band_structure_2d_without_fermi_warns_that_the_bands_are_not_shifted(monkeypatch):
    ebs = spin_polarized_mesh(2.0)

    def from_code(_cls: type[ElectronicBandStructureMesh], *_args: object, **_kwargs: object):
        return ebs

    monkeypatch.setattr(ElectronicBandStructureMesh, "from_code", classmethod(from_code))
    handler = pyprocar.BandStructure2DHandler(code="vasp", dirname="calc")

    with user_warning(__file__, match=r"`fermi` is not set! Set `fermi=\{value\}`"):
        handler.plot_band_structure(mode="plain", show=False, render_offscreen=True)


@pytest.mark.parametrize(
    "method",
    ["plot_fermi_surface", "plot_fermi_cross_section", "plot_fermi_cross_section_box_widget"],
)
@pytest.mark.usefixtures("live_user_log")
@pytest.mark.guards_existing_behaviour(
    reason="FermiSurface.from_ebs raises before the no-surface warning, which no"
    + " public call reaches; this pins the loud error"
)
def test_fermi_handler_without_a_crossing_raises(monkeypatch, method):
    ebs = sphere_mesh(1, np.full((2, 1, 2, 1), 0.5))

    def from_code(_cls: type[ElectronicBandStructureMesh], *_args: object, **_kwargs: object):
        return ebs

    monkeypatch.setattr(ElectronicBandStructureMesh, "from_code", classmethod(from_code))
    above_every_band = 50.0
    handler = pyprocar.FermiHandler(code="vasp", dirname="calc", fermi=above_every_band)

    with pytest.raises(ValueError, match="No Fermi surfaces were generated"):
        getattr(handler, method)(mode="plain", show=False)


def test_fermi2d_honours_its_verbose_argument(tmp_path, live_user_log):
    spin_polarized_mesh(2.0).save(tmp_path / "ebs.pkl")

    pyprocar.fermi2D(code="vasp", dirname=str(tmp_path), use_cache=True, show=False, verbose=0)
    silent = live_user_log.messages
    pyprocar.fermi2D(code="vasp", dirname=str(tmp_path), use_cache=True, show=False, verbose=1)

    assert silent == []
    assert "### Parameters ###" in live_user_log.messages
    assert "k_z_plane       : 0.0" in live_user_log.messages
