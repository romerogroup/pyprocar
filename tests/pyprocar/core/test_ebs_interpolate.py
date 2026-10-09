import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.kpoints import (
    KGRID_MODE,
    KGridInfo,
    generate_gamma_centered_kpoints,
    sort_kpoints,
)


def band_energies(kpoints: np.ndarray) -> np.ndarray:
    """Two bands with a few low Fourier components, which FFT interpolation reproduces exactly."""
    k = 2 * np.pi * kpoints
    energy = np.cos(k[:, 0]) + 0.5 * np.cos(k[:, 1]) + 0.25 * np.cos(k[:, 2])
    return np.stack([energy, -0.5 * energy], axis=1)


def cosine_mesh(kgrid: tuple[int, int, int]) -> ElectronicBandStructureMesh:
    kpoints = generate_gamma_centered_kpoints(kgrid)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=kgrid, kgrid_mode=KGRID_MODE.GAMMA, kshift=(0, 0, 0)),
        kpoints=kpoints,
        bands=band_energies(kpoints)[:, :, None],
        reciprocal_lattice=np.eye(3),
    )


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("kgrid", [(4, 6, 2), (6, 6, 6), (5, 3, 4)])
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

    error = np.abs(mesh.property_store["bands"].value[:, :, 0] - band_energies(mesh.kpoints))
    assert error.max() < 1e-9


def test_a_single_kpoint_axis_stays_a_single_kpoint():
    # A 2D calculation samples one kz plane; interpolation has nothing to resolve along kz.
    mesh = cosine_mesh((6, 4, 1)).interpolate(interpolation_factor=2, inplace=False)

    assert np.array_equal(mesh.kgrid, [12, 8, 1])
    assert mesh.n_kpoints == 12 * 8
    assert np.all(mesh.kpoints[:, 2] == 0)
    assert np.allclose(mesh.kgrid_spacing, [1 / 12, 1 / 8, 1])
    error = np.abs(mesh.property_store["bands"].value[:, :, 0] - band_energies(mesh.kpoints))
    assert error.max() < 1e-9


def test_a_complex_property_keeps_its_imaginary_part():
    # exp(2 pi i kx) times a real cosine: complex, and band-limited on a (4, 6, 2) grid.
    def phase(kpoints: np.ndarray) -> np.ndarray:
        values = np.exp(2j * np.pi * kpoints[:, 0]) * np.cos(2 * np.pi * kpoints[:, 1])
        return values[:, None, None, None, None]

    mesh = cosine_mesh((4, 6, 2))
    mesh.add_property(name="projected_phase", value=phase(mesh.kpoints))

    mesh = mesh.interpolate(interpolation_factor=2, inplace=False)

    phase_values = mesh.property_store["projected_phase"].value
    assert np.iscomplexobj(phase_values)
    assert np.abs(phase_values - phase(mesh.kpoints)).max() < 1e-9
    assert not np.iscomplexobj(mesh.property_store["bands"].value)


def test_interpolating_a_padded_mesh_halves_its_kpoint_spacing():
    # A 5-point axis padded by 2 has 9 points 1/5 apart; doubling gives 18 points 1/10 apart.
    padded = cosine_mesh((5, 5, 5)).pad(padding=2, inplace=False)

    mesh = padded.interpolate(interpolation_factor=2, inplace=False)

    assert np.array_equal(mesh.kgrid, [18, 18, 18])
    assert np.allclose(mesh.kgrid_spacing, 0.1)
    for axis in range(3):
        assert np.allclose(np.diff(np.unique(mesh.kpoints[:, axis].round(9))), 0.1)
    # FFT interpolation passes through its samples: every second point is a padded point.
    coarse_steps = (mesh.kpoints - padded.kpoints[0]) / 0.2
    kept = np.all(np.isclose(coarse_steps, np.round(coarse_steps)), axis=1)
    assert np.allclose(mesh.kpoints[kept], padded.kpoints)
    assert np.allclose(
        mesh.property_store["bands"].value[kept], padded.property_store["bands"].value, atol=1e-9
    )
