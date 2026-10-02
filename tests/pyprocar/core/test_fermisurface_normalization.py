import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo


@pytest.fixture(scope="module")
def sphere_surface() -> FermiSurface:
    """Band 0 is a sphere around Gamma with weights (0.3, 0.1) on atoms (0, 1).

    Band 1 lies far above the Fermi level, so it has no surface, and carries a
    larger weight (0.9) on atom 0.
    """
    n = 10
    frac = np.arange(n) / n
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    bands = np.stack([np.sum(centred**2, axis=1), np.full(len(kpoints), 5.0)], axis=1)
    projected = np.zeros((len(kpoints), 2, 1, 2, 1))
    projected[:, 0, 0, :, 0] = [0.3, 0.1]
    projected[:, 1, 0, :, 0] = [0.9, 0.0]
    ebs = ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=(n, n, n), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)),
        kpoints=kpoints,
        bands=bands[..., np.newaxis],
        projected=projected,
        fermi=0.1,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
    )
    return FermiSurface.from_ebs(ebs)


@pytest.mark.parametrize(
    ("norm_mode", "on_surface"),
    [("raw", 0.3), ("max", 1.0), ("total", 0.75)],
)
def test_projected_sum_normalizes_over_plotted_points(sphere_surface, norm_mode, on_surface):
    fs = sphere_surface

    values = fs.get_property("projected_sum", atoms=[0], norm_mode=norm_mode).value

    assert list(fs.band_spin_mask) == [(0, 0)]
    assert values.shape == (fs.n_points, 2, 1)
    np.testing.assert_allclose(values[:, 0, 0], on_surface)
    np.testing.assert_array_equal(values[:, 1, 0], 0.0)


def test_projected_sum_integral_normalizes_by_surface_area_integral(sphere_surface):
    fs = sphere_surface

    values = fs.get_property("projected_sum", atoms=[0], norm_mode="integral").value

    sphere_area = 4 * np.pi * 0.1
    np.testing.assert_allclose(values[:, 0, 0], 1 / sphere_area, rtol=0.05)
    np.testing.assert_array_equal(values[:, 1, 0], 0.0)
