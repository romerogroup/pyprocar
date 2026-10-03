import numpy as np
import pytest
import pyvista as pv

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

N_K = 10
ATOM_0_WEIGHT = (0.3, 0.6)
ATOM_1_WEIGHT = 0.1


@pytest.fixture(scope="module")
def two_sphere_surface() -> FermiSurface:
    frac = np.arange(N_K) / N_K
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    k2 = np.sum(((kpoints + 0.5) % 1.0 - 0.5) ** 2, axis=1)
    projected = np.zeros((len(kpoints), 2, 1, 2, 1))
    for iband, weight in enumerate(ATOM_0_WEIGHT):
        projected[:, iband, 0, :, 0] = [weight, ATOM_1_WEIGHT]
    ebs = ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, N_K), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=kpoints,
        bands=np.stack([k2, 2 * k2], axis=1)[..., np.newaxis],
        projected=projected,
        fermi=0.1,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
    )
    return FermiSurface.from_ebs(ebs)


@pytest.mark.parametrize("key", [(0, 0), (1, 0)])
def test_select_bands_keeps_the_selected_surface_and_its_projection(two_sphere_surface, key):
    fs = two_sphere_surface
    iband, ispin = key

    selected = fs.select_bands([key])
    values = selected.compute_projected_sum(atoms=[0]).value

    assert list(selected.band_spin_mask) == [key]
    assert selected.n_points == fs.band_spin_mask[key].sum()
    np.testing.assert_allclose(selected.points, fs.points[fs.band_spin_mask[key]])
    np.testing.assert_allclose(values[:, iband, ispin], ATOM_0_WEIGHT[iband])
    np.testing.assert_array_equal(values[:, 1 - iband, ispin], 0.0)


def test_select_bands_integral_normalizes_over_the_selected_surface(two_sphere_surface):
    selected = two_sphere_surface.select_bands([(1, 0)])

    values = selected.compute_projected_sum(atoms=[0], norm_mode="integral").value

    mesh = pv.PolyData(np.asarray(selected.points), np.asarray(selected.faces))
    mesh.point_data["normed"] = values[:, 1, 0]
    integral = np.asarray(mesh.integrate_data().point_data["normed"])[0]
    assert integral == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(values[:, 1, 0], 1 / mesh.area)


@pytest.mark.parametrize("norm_mode", ["raw", "max", "total", "integral"])
def test_empty_selection_gives_an_empty_projection(two_sphere_surface, norm_mode):
    empty = two_sphere_surface.select_bands([])

    values = empty.compute_projected_sum(atoms=[0], norm_mode=norm_mode).value

    assert values.shape == (0, 2, 1)


def test_select_bands_keeps_several_surfaces_in_surface_order(two_sphere_surface):
    fs = two_sphere_surface

    selected = fs.select_bands([(1, 0), (0, 0)])
    values = selected.compute_projected_sum(atoms=[0]).value

    assert list(selected.band_spin_mask) == [(0, 0), (1, 0)]
    for key in [(0, 0), (1, 0)]:
        mask = selected.band_spin_mask[key]
        np.testing.assert_allclose(selected.points[mask], fs.points[fs.band_spin_mask[key]])
        np.testing.assert_allclose(values[mask, key[0], 0], ATOM_0_WEIGHT[key[0]])
