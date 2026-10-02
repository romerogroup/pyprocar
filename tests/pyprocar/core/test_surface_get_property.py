import numpy as np
import pytest

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.property_store import Property
from pyprocar.plotter.bs_2d_plot import BS2DPlotter
from pyprocar.plotter.fs_plot import FermiPlotter
from tests.utils import DATA_DIR

pytestmark = pytest.mark.data

MESH_DIR = DATA_DIR / "examples" / "fermi3d" / "non-spin-polarized"
GRAPHENE_DIR = DATA_DIR / "examples" / "bands" / "2d-bands" / "graphene"


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
    assert series[0].scalars is not None
    np.testing.assert_allclose(series[0].scalars, full[mask, iband, ispin])


def test_fs_include_normal_label_prefixes_label_without_selection(fs):
    prop = fs.compute_projected_sum(norm_mode="max", include_normal_label=True)

    assert prop.metadata["label"] == "Max-Normed Projected Sum"
    assert prop.metadata["label_plain"] == "Max-Normed Projected Sum"


def test_fs_get_property_unknown_name_raises_key_error(fs):
    with pytest.raises(KeyError):
        fs.get_property("no_such_property")


def test_bs2d_get_property_returns_surface_basis_property(bs2d):
    prop = bs2d.get_property("bands_speed")

    assert isinstance(prop, Property)
    assert prop.value.shape == (bs2d.n_points,)
    series = BS2DPlotter(bs2d, off_screen=True)._to_series_list(bs2d, scalars_data="bands_speed")
    for s, mask in zip(series, bs2d.band_spin_mask.values(), strict=True):
        np.testing.assert_array_equal(s.scalars, prop.value[mask])


def test_bs2d_get_property_unknown_name_raises_key_error(bs2d):
    with pytest.raises(KeyError):
        bs2d.get_property("no_such_property")


def test_bs2d_set_values_colours_each_point_with_its_own_band():
    bs2d = BandStructure2D.from_code(code="vasp", dirpath=GRAPHENE_DIR, grid_interpolation=(20, 20))

    bs2d.set_values("bands", bs2d.get_property("bands").value)

    colours, heights = bs2d.point_data["bands"], bs2d.points[:, 2]
    finite = np.isfinite(colours) & np.isfinite(heights)
    assert finite.sum() > 100
    assert np.corrcoef(colours[finite], heights[finite])[0, 1] > 0.999


def test_fs_band_masks_select_each_isosurface_points(fs):
    for key, mask in fs.band_spin_mask.items():
        np.testing.assert_allclose(fs.points[mask], fs.band_isosurfaces[key].points)
