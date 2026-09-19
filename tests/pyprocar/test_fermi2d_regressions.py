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
from matplotlib.collections import LineCollection
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


def _line_collection_scalars(fs):
    arrays = [
        np.asarray(collection.get_array(), dtype=float).ravel()
        for collection in fs.ax.collections
        if isinstance(collection, LineCollection) and collection.get_array() is not None
    ]
    values = np.concatenate(arrays)
    return values[np.isfinite(values)]


def test_fermi2d_parametric_default_clim_colorbar(synthetic_ebs_dir):
    """With clim=(None, None) the norm, line colors and colorbar must use the
    range of the projection data, not the (-0.5, 0.5) fallback."""
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
        scalars = _line_collection_scalars(fs)
        assert scalars.size > 0
        # The synthetic projections lie in [0, 1], so the fallback would differ.
        assert fs.norm.vmin == pytest.approx(scalars.min())
        assert fs.norm.vmax == pytest.approx(scalars.max())
        assert fs.norm.vmin >= 0.0

        for collection in fs.ax.collections:
            if isinstance(collection, LineCollection) and collection.get_array() is not None:
                assert collection.norm.vmin == pytest.approx(scalars.min())
                assert collection.norm.vmax == pytest.approx(scalars.max())

        assert fs.colorbar is not None
        assert fs.colorbar.mappable.norm.vmin == pytest.approx(scalars.min())
        assert fs.colorbar.mappable.norm.vmax == pytest.approx(scalars.max())
    finally:
        plt.close("all")


def test_fermi2d_parametric_explicit_clim_is_respected(synthetic_ebs_dir):
    fs = pyprocar.fermi2D(
        code="vasp",
        dirname=str(synthetic_ebs_dir),
        fermi=0.0,
        mode="parametric",
        use_cache=True,
        show=False,
        show_colorbar=True,
        clim=(0.2, 0.9),
    )
    try:
        assert fs.norm.vmin == pytest.approx(0.2)
        assert fs.norm.vmax == pytest.approx(0.9)
        assert fs.colorbar.mappable.norm.vmin == pytest.approx(0.2)
        assert fs.colorbar.mappable.norm.vmax == pytest.approx(0.9)
    finally:
        plt.close("all")


def test_fermi2d_parametric_partial_clim(synthetic_ebs_dir):
    """Only the missing clim entry is taken from the data."""
    fs = pyprocar.fermi2D(
        code="vasp",
        dirname=str(synthetic_ebs_dir),
        fermi=0.0,
        mode="parametric",
        use_cache=True,
        show=False,
        clim=(None, 0.9),
    )
    try:
        scalars = _line_collection_scalars(fs)
        assert fs.norm.vmin == pytest.approx(scalars.min())
        assert fs.norm.vmax == pytest.approx(0.9)
    finally:
        plt.close("all")
