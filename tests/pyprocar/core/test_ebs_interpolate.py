import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo, generate_gamma_centered_kpoints, sort_kpoints


def band_energy(kpoints: np.ndarray) -> np.ndarray:
    """A band with a few low Fourier components, which FFT interpolation reproduces exactly."""
    k = 2 * np.pi * kpoints
    return np.cos(k[:, 0]) + 0.5 * np.cos(k[:, 1]) + 0.25 * np.cos(k[:, 2])


def cosine_mesh(kgrid: tuple[int, int, int]) -> ElectronicBandStructureMesh:
    kpoints = generate_gamma_centered_kpoints(kgrid)
    energy = band_energy(kpoints)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=kgrid, kgrid_mode=KGRID_MODE.GAMMA, kshift=(0, 0, 0)),
        kpoints=kpoints,
        bands=np.stack([energy, -0.5 * energy], axis=1)[:, :, None],
        reciprocal_lattice=np.eye(3),
    )


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("kgrid", [(4, 6, 2), (6, 6, 6)])
def test_interpolated_bands_match_the_band_at_each_returned_kpoint(kgrid, factor):
    new_kgrid = np.array(kgrid) * factor

    mesh = cosine_mesh(kgrid).interpolate(interpolation_factor=factor, inplace=False)

    assert np.array_equal(mesh.kgrid, new_kgrid)
    # One period of the uniform grid with step 1/(n*f): every k*(n*f) is an integer, and the
    # n*f residues of each axis appear once each.
    steps = mesh.kpoints * new_kgrid
    assert np.allclose(steps, np.round(steps), atol=1e-9)
    cells = {tuple(c) for c in np.mod(np.round(steps).astype(int), new_kgrid)}
    assert len(cells) == mesh.n_kpoints == np.prod(new_kgrid)
    # Sorted with kx fastest, the order every Mesh array is reshaped with.
    assert np.array_equal(mesh.kpoints, sort_kpoints(mesh.kpoints, order="F"))
    assert np.allclose(mesh.kgrid_spacing, 1 / new_kgrid)

    energy = band_energy(mesh.kpoints)
    error = np.abs(mesh.bands.value[:, :, 0] - np.stack([energy, -0.5 * energy], axis=1))
    assert error.max() < 1e-9
