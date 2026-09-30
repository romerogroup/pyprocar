import numpy as np
import pytest

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.property_store import Property
from pyprocar.plotter.fs_plot import FermiPlotter
from tests.utils import DATA_DIR

MESH_DIR = DATA_DIR / "examples" / "fermi3d" / "non-spin-polarized"


@pytest.fixture
def fs():
    return FermiSurface.from_code(code="vasp", dirpath=MESH_DIR)


@pytest.fixture
def bs2d():
    return BandStructure2D.from_code(
        code="vasp", dirpath=MESH_DIR, grid_interpolation=(20, 20), padding=5
    )


def test_fs_get_property_zeroes_values_off_each_band_surface(fs):
    speed = fs.get_property("fermi_speed").value

    for (iband, ispin), mask in fs.band_spin_mask.items():
        assert np.all(speed[~mask, iband, ispin] == 0)
        assert np.any(speed[mask, iband, ispin] != 0)


def test_fs_set_values_leaves_the_property_unchanged(fs):
    prop = fs.get_property("fermi_speed")
    before = prop.value.copy()

    fs.set_values("fermi_speed", prop.value)

    np.testing.assert_array_equal(prop.value, before)


def test_fs_compute_projected_sum_agrees_with_get_property(fs):
    direct = fs.compute_projected_sum(atoms=[0]).value
    cached = fs.get_property("projected_sum", atoms=[0]).value

    np.testing.assert_allclose(cached, direct)


def test_fs_compute_projected_sum_does_not_leak_into_by_name_lookups(fs):
    full = fs.compute_projected_sum().value
    fs.compute_projected_sum(atoms=[0])

    np.testing.assert_allclose(fs.get_property("projected_sum").value, full)
    series = FermiPlotter(off_screen=True)._to_series_list(fs, scalars_data="projected_sum")
    (iband, ispin), mask = next(iter(fs.band_spin_mask.items()))
    np.testing.assert_allclose(series[0].scalars, full[mask])


def test_fs_include_normal_label_prefixes_label_without_selection(fs):
    prop = fs.compute_projected_sum(norm_mode="max", include_normal_label=True)

    assert prop.metadata["label"] == "Max-Normed Projected Sum"
    assert prop.metadata["label_plain"] == "Max-Normed Projected Sum"


def test_fs_get_property_unknown_name_raises_key_error(fs):
    with pytest.raises(KeyError):
        fs.get_property("no_such_property")


def test_bs2d_get_property_returns_property(bs2d):
    prop = bs2d.get_property("bands")

    assert isinstance(prop, Property)
    assert prop.value.shape[0] == bs2d.n_grid_points
    assert bs2d.get_property("bands") is prop


def test_bs2d_get_property_unknown_name_raises_key_error(bs2d):
    with pytest.raises(KeyError):
        bs2d.get_property("no_such_property")
