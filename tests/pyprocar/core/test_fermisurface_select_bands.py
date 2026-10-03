import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

N_K = 10


@pytest.fixture(scope="module")
def two_sphere_surface() -> FermiSurface:
    """Bands 0 and 1 are nested spheres around Gamma, both crossing E_F = 0.1.

    Atom 0 holds 0.3 on band 0 and 0.6 on band 1; atom 1 holds 0.1 on both.
    """
    frac = np.arange(N_K) / N_K
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    k2 = np.sum(((kpoints + 0.5) % 1.0 - 0.5) ** 2, axis=1)
    projected = np.zeros((len(kpoints), 2, 1, 2, 1))
    projected[:, 0, 0, :, 0] = [0.3, 0.1]
    projected[:, 1, 0, :, 0] = [0.6, 0.1]
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
    np.testing.assert_allclose(values[:, iband, ispin], [0.3, 0.6][iband])
    np.testing.assert_array_equal(values[:, 1 - iband, ispin], 0.0)


def test_select_bands_integral_normalizes_over_the_selected_surface(two_sphere_surface):
    selected = two_sphere_surface.select_bands([(1, 0)])

    values = selected.compute_projected_sum(atoms=[0], norm_mode="integral").value

    sphere_area = 4 * np.pi * 0.05
    np.testing.assert_allclose(values[:, 1, 0], 1 / sphere_area, rtol=0.05)


@pytest.mark.parametrize("norm_mode", ["raw", "max", "total", "integral"])
def test_empty_selection_gives_an_empty_projection(two_sphere_surface, norm_mode):
    empty = two_sphere_surface.select_bands([])

    values = empty.compute_projected_sum(atoms=[0], norm_mode=norm_mode).value

    assert values.shape == (0, 2, 1)
