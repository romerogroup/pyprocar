import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import pyprocar
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

N_K = 12
N_KZ = 3
FERMI = 0.1


@pytest.fixture
def cylinder_calc(tmp_path):
    frac = np.arange(N_K) / N_K
    kz = np.arange(N_KZ) / N_KZ
    kpoints = np.stack(np.meshgrid(frac, frac, kz, indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    kx2_plus_ky2 = np.sum(centred[:, :2] ** 2, axis=1)
    energies = np.stack([kx2_plus_ky2, np.full(len(kpoints), 5.0)], axis=1)
    projected = np.zeros((len(kpoints), 2, 1, 1, 1))
    projected[:, :, 0, 0, 0] = 0.2 + 0.6 * np.abs(np.cos(2 * np.pi * kpoints[:, :1]))
    ebs = ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, N_KZ), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=kpoints,
        bands=energies[..., np.newaxis],
        projected=projected,
        fermi=FERMI,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
    )
    ebs.save(tmp_path / "ebs.pkl")
    return tmp_path


@pytest.mark.parametrize("mode", ["plain", "plain_bands", "parametric"])
def test_fewer_kz_points_than_kx_gives_the_right_contour(cylinder_calc, mode):
    _, ax = pyprocar.fermi2D(
        code="vasp", dirname=str(cylinder_calc), mode=mode, use_cache=True, show=False
    )
    segments = np.concatenate(
        [np.concatenate(c.get_segments()) for c in ax.collections if isinstance(c, LineCollection)]
    )
    plt.close("all")

    radii = np.linalg.norm(segments, axis=1)
    np.testing.assert_allclose(radii, np.sqrt(FERMI), atol=0.02)
