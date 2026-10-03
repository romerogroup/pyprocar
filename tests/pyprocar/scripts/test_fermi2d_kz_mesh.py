import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

import pyprocar
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

N_K = 12
FERMI = 0.1
FILL_TOL = 0.1


def cylinder_mesh(n_kz: int) -> ElectronicBandStructureMesh:
    frac = np.arange(N_K) / N_K
    kz = np.arange(n_kz) / n_kz
    kpoints = np.stack(np.meshgrid(frac, frac, kz, indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    kx2_plus_ky2 = np.sum(centred[:, :2] ** 2, axis=1)
    energies = np.stack([kx2_plus_ky2, np.full(len(kpoints), 5.0)], axis=1)
    projected = np.zeros((len(kpoints), 2, 1, 1, 1))
    projected[:, :, 0, 0, 0] = 0.2 + 0.6 * np.abs(np.cos(2 * np.pi * kpoints[:, :1]))
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, n_kz), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=kpoints,
        bands=energies[..., np.newaxis],
        projected=projected,
        fermi=FERMI,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
    )


@pytest.mark.parametrize("n_kz", [1, 3])
@pytest.mark.parametrize("mode", ["plain", "plain_bands", "parametric"])
def test_fermi2d_contour_on_meshes_thinner_in_kz(tmp_path, n_kz, mode):
    cylinder_mesh(n_kz).save(tmp_path / "ebs.pkl")

    _, ax = pyprocar.fermi2D(
        code="vasp", dirname=str(tmp_path), mode=mode, use_cache=True, show=False
    )
    segments = np.concatenate(
        [np.concatenate(c.get_segments()) for c in ax.collections if isinstance(c, LineCollection)]
    )
    plt.close("all")

    radii = np.linalg.norm(segments, axis=1)
    np.testing.assert_allclose(radii, np.sqrt(FERMI), atol=0.02)


def test_single_kz_surface_spans_only_the_filled_slab():
    fs = FermiSurface.from_ebs(cylinder_mesh(1))

    z = fs.points[:, 2]
    assert (z.min(), z.max()) == pytest.approx((-FILL_TOL, FILL_TOL))
    assert fs.area == pytest.approx(2 * np.pi * np.sqrt(FERMI) * 2 * FILL_TOL, rel=0.02)
