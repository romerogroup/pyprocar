import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import pyprocar
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo
from tests.utils.user_warning import user_warning

N_K = 12
FERMI = 0.1


def spin_polarized_mesh(spin_down_scale: float | None) -> ElectronicBandStructureMesh:
    """Spin up is a cylinder of radius sqrt(FERMI) along kz.

    Spin down is the same cylinder with energies scaled by ``spin_down_scale``, or a
    flat band at 5 eV that never crosses E_F when the scale is None.
    """
    frac = np.arange(N_K) / N_K
    kpoints = np.stack(np.meshgrid(frac, frac, [0.0], indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    up = np.sum(centred[:, :2] ** 2, axis=1)
    down = np.full(len(kpoints), 5.0) if spin_down_scale is None else spin_down_scale * up
    bands = np.stack([up, down], axis=1)[:, np.newaxis, :]
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=(N_K, N_K, 1), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0, 0, 0)),
        kpoints=kpoints,
        bands=bands,
        projected=np.ones((len(kpoints), 1, 2, 1, 1)),
        fermi=FERMI,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
    )


def contour_radii(tmp_path, spin_down_scale, spins):
    spin_polarized_mesh(spin_down_scale).save(tmp_path / "ebs.pkl")
    _, ax = pyprocar.fermi2D(
        code="vasp", dirname=str(tmp_path), spins=spins, use_cache=True, show=False
    )
    segments = [c.get_segments() for c in ax.collections if isinstance(c, LineCollection)]
    plt.close("all")
    points = [np.concatenate(s) for s in segments if len(s)]
    return np.linalg.norm(np.concatenate(points), axis=1) if points else np.empty(0)


@pytest.mark.parametrize(("spins", "radius"), [([0], np.sqrt(FERMI)), ([1], np.sqrt(FERMI / 2))])
def test_fermi2d_draws_only_the_requested_spin_channel(tmp_path, spins, radius):
    radii = contour_radii(tmp_path, 2.0, spins)

    assert radii.size > 0
    np.testing.assert_allclose(radii, radius, atol=0.02)


def test_fermi2d_draws_both_channels_by_default(tmp_path):
    radii = contour_radii(tmp_path, 2.0, None)

    assert np.isclose(radii, np.sqrt(FERMI), atol=0.02).any()
    assert np.isclose(radii, np.sqrt(FERMI / 2), atol=0.02).any()


@pytest.mark.guards_existing_behaviour(
    reason="#281 used stacklevel=2, which already names this file one frame up;"
    " this pins the line through warn_user"
)
def test_fermi2d_spin_channel_without_a_crossing_warns_and_draws_nothing(tmp_path):
    with user_warning(__file__, match=r"no band of spin channel\(s\) \[1\] crosses"):
        radii = contour_radii(tmp_path, None, [1])

    assert radii.size == 0
