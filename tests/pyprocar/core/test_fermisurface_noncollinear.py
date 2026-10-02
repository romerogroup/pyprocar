import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo


@pytest.fixture(scope="module")
def noncollinear_surface() -> FermiSurface:
    """A sphere around Gamma laid out as the VASP parser loads a non-collinear run.

    The energies repeat across 4 spin channels, and the projections hold the
    total (0.4) and the Sx, Sy, Sz components (0.1, 0.2, 0.3) on atom 0.
    """
    n = 10
    frac = np.arange(n) / n
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    bands = np.repeat(np.sum(centred**2, axis=1)[:, np.newaxis, np.newaxis], 4, axis=2)
    projected = np.zeros((len(kpoints), 1, 4, 2, 1))
    projected[:, 0, :, 0, 0] = [0.4, 0.1, 0.2, 0.3]
    ebs = ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=(n, n, n), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)),
        kpoints=kpoints,
        bands=bands,
        projected=projected,
        fermi=0.1,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
    )
    return FermiSurface.from_ebs(ebs)


def test_noncollinear_surface_has_one_spin_channel(noncollinear_surface):
    assert list(noncollinear_surface.band_spin_mask) == [(0, 0)]


def test_noncollinear_projected_sum_takes_the_total_component(noncollinear_surface):
    fs = noncollinear_surface

    values = fs.get_property("projected_sum", atoms=[0], spins=[0]).value

    assert values.shape == (fs.n_points, 1, 1)
    np.testing.assert_allclose(values[:, 0, 0], 0.4)


def test_noncollinear_projected_spin_texture_holds_sx_sy_sz(noncollinear_surface):
    fs = noncollinear_surface

    values = fs.get_property("projected_sum_spin_texture", atoms=[0]).value

    assert values.shape == (fs.n_points, 1, 1, 3)
    np.testing.assert_allclose(values[:, 0, 0, :], np.tile([0.1, 0.2, 0.3], (fs.n_points, 1)))
