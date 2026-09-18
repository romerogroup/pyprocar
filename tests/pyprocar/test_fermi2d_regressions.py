"""Regression tests for pyprocar.fermi2D using a tiny synthetic band structure.

Covers:
- kz-plane slicing must keep kpoints consistent with bands/projected
  (regression from 880640d8, ebs_sum reshape to nkpoints).
- mode='parametric' with the default clim (min(None, float) TypeError and
  unsupported ``norm`` argument in FermiSurface.show_colorbar).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import pyprocar
from pyprocar.core import ElectronicBandStructure
from pyprocar.utils import data_utils


@pytest.fixture
def synthetic_ebs_dir(tmp_path):
    """Write a cached ebs.pkl with a 12x12x3 k-mesh (only 1/3 in the kz=0 plane)."""
    n = 12
    grid = np.arange(n) / n - 0.5
    kz = np.array([-1 / 3, 0.0, 1 / 3])
    kpoints = np.array(np.meshgrid(grid, grid, kz, indexing="ij")).reshape(3, -1).T
    kcart = 2 * np.pi * kpoints
    band = (kcart[:, 0] ** 2 + kcart[:, 1] ** 2) / 10 - 1.0
    bands = np.stack([band, band + 5.0], axis=1)[..., np.newaxis]
    # projected shape: (nkpoints, nbands, natoms, nprincipals, norbitals, nspins)
    projected = np.ones((kpoints.shape[0], 2, 1, 1, 1, 1))
    projected[:, :, 0, 0, 0, 0] = np.abs(np.cos(kcart[:, 0]))[:, np.newaxis]

    ebs = ElectronicBandStructure(
        kpoints=kpoints,
        bands=bands,
        projected=projected,
        efermi=0.0,
        reciprocal_lattice=np.eye(3),
        shift_to_efermi=False,
    )
    data_utils.save_pickle(ebs, str(tmp_path / "ebs.pkl"))
    return tmp_path


@pytest.mark.parametrize("mode", ["plain", "plain_bands", "parametric"])
def test_fermi2d_modes_with_kz_slice(synthetic_ebs_dir, mode):
    fs = pyprocar.fermi2D(
        code="vasp",
        dirname=str(synthetic_ebs_dir),
        fermi=0.0,
        mode=mode,
        use_cache=True,
        show=False,
    )
    try:
        assert fs is not None
        assert len(fs.handles) > 0
    finally:
        plt.close("all")


def test_fermi2d_parametric_default_clim_colorbar(synthetic_ebs_dir):
    fs = pyprocar.fermi2D(
        code="vasp",
        dirname=str(synthetic_ebs_dir),
        fermi=0.0,
        mode="parametric",
        use_cache=True,
        show=False,
        show_colorbar=True,
    )
    try:
        assert fs.colorbar is not None
    finally:
        plt.close("all")
