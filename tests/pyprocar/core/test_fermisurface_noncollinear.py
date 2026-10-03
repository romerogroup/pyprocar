import numpy as np
import pytest

from pyprocar.core.bandstructure2D import BandStructure2D
from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

N_K = 10


def sphere_mesh(n_channels: int, projected: np.ndarray) -> ElectronicBandStructureMesh:
    """Band 0 is a sphere around Gamma crossing E_F = 0.1; band 1 sits at 5 eV.

    Every spin channel repeats these energies, as the VASP parser stores them.
    """
    frac = np.arange(N_K) / N_K
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    energies = np.stack([np.sum(centred**2, axis=1), np.full(len(kpoints), 5.0)], axis=1)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, N_K), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=kpoints,
        bands=np.repeat(energies[..., np.newaxis], n_channels, axis=2),
        projected=np.broadcast_to(projected, (len(kpoints), *projected.shape)).copy(),
        fermi=0.1,
        reciprocal_lattice=np.eye(3),
        orbital_names=["s"],
    )


@pytest.fixture(scope="module")
def noncollinear_ebs() -> ElectronicBandStructureMesh:
    """Atom 0 of band 0 holds the total (0.4) and Sx, Sy, Sz (0.1, 0.2, 0.3).

    Band 1 holds total 0.9 and no spin. Atom 1 holds nothing.
    """
    projected = np.zeros((2, 4, 2, 1))
    projected[0, :, 0, 0] = [0.4, 0.1, 0.2, 0.3]
    projected[1, 0, 0, 0] = 0.9
    return sphere_mesh(4, projected)


@pytest.fixture(scope="module")
def spin_polarized_ebs() -> ElectronicBandStructureMesh:
    """Atom 0 holds 0.3 spin-up and 0.7 spin-down on band 0, 0.9 and 0.1 on band 1."""
    projected = np.zeros((2, 2, 2, 1))
    projected[0, :, 0, 0] = [0.3, 0.7]
    projected[1, :, 0, 0] = [0.9, 0.1]
    return sphere_mesh(2, projected)


@pytest.fixture(scope="module")
def noncollinear_surface(noncollinear_ebs) -> FermiSurface:
    return FermiSurface.from_ebs(noncollinear_ebs)


def test_noncollinear_surface_has_one_spin_channel(noncollinear_surface):
    assert list(noncollinear_surface.band_spin_mask) == [(0, 0)]


def test_noncollinear_projected_sum_takes_the_total_component(noncollinear_surface):
    fs = noncollinear_surface

    values = fs.get_property("projected_sum", atoms=[0], spins=[0]).value

    assert values.shape == (fs.n_points, 2, 1)
    np.testing.assert_allclose(values[:, 0, 0], 0.4)


def test_noncollinear_projected_spin_texture_holds_sx_sy_sz(noncollinear_surface):
    fs = noncollinear_surface

    values = fs.get_property("projected_sum_spin_texture", atoms=[0]).value

    assert values.shape == (fs.n_points, 2, 1, 3)
    np.testing.assert_allclose(values[:, 0, 0, :], np.tile([0.1, 0.2, 0.3], (fs.n_points, 1)))


def test_noncollinear_ebs_projected_sum_defaults_to_the_total(noncollinear_ebs):
    values = noncollinear_ebs.compute_projected_sum(atoms=[0]).value

    assert values.shape == (noncollinear_ebs.n_kpoints, 2, 1)
    np.testing.assert_allclose(values[:, 0, 0], 0.4)
    np.testing.assert_allclose(values[:, 1, 0], 0.9)


def test_noncollinear_surface_projected_sum_defaults_to_the_total(noncollinear_ebs):
    fs = FermiSurface.from_ebs(noncollinear_ebs)

    values = fs.get_property("projected_sum", atoms=[0]).value

    np.testing.assert_allclose(values[:, 0, 0], 0.4)


def test_noncollinear_band_structure_2d_has_one_spin_channel(noncollinear_ebs):
    bs2d = BandStructure2D.from_ebs(noncollinear_ebs, grid_interpolation=(10, 10), padding=2)

    values = bs2d.get_property("projected_sum", atoms=[0], spins=[0]).value

    assert list(bs2d.band_spin_surface_map) == [(0, 0), (1, 0)]
    np.testing.assert_allclose(np.unique(values.round(6)), [0.4, 0.9])


def test_noncollinear_projected_sum_defaults_to_the_total_without_a_selection(
    noncollinear_ebs,
):
    values = noncollinear_ebs.compute_projected_sum().value

    np.testing.assert_allclose(values[:, 0, 0], 0.4)
    np.testing.assert_allclose(values[:, 1, 0], 0.9)


def test_noncollinear_total_normalized_projection_is_one(noncollinear_ebs):
    values = noncollinear_ebs.compute_projected_sum(atoms=[0], norm_mode="total").value

    np.testing.assert_allclose(values, 1.0)


def test_noncollinear_surface_total_normalized_projection_is_one(noncollinear_surface):
    fs = noncollinear_surface

    values = fs.compute_projected_sum(atoms=[0], norm_mode="total").value

    np.testing.assert_allclose(values[fs.band_spin_mask[(0, 0)], 0, 0], 1.0)


def test_spin_polarized_projected_sum_default_keeps_both_channels(spin_polarized_ebs):
    values = spin_polarized_ebs.compute_projected_sum().value

    np.testing.assert_allclose(values[:, 0, 0], 0.3)
    np.testing.assert_allclose(values[:, 0, 1], 0.7)


def test_spin_polarized_band_structure_2d_keeps_both_channels(spin_polarized_ebs):
    bs2d = BandStructure2D.from_ebs(spin_polarized_ebs, grid_interpolation=(10, 10), padding=2)

    assert list(bs2d.band_spin_surface_map) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_spin_polarized_surface_keeps_both_spin_channels(spin_polarized_ebs):
    fs = FermiSurface.from_ebs(spin_polarized_ebs)

    assert list(fs.band_spin_mask) == [(0, 0), (0, 1)]
