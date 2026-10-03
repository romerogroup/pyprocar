import itertools

import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

RADIUS = 0.15
FCC_RECIPROCAL = np.array([[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]) / 4.0


def sphere_ebs(kgrid: tuple[int, int, int]) -> ElectronicBandStructureMesh:
    """One band E(k) = |k|^2 in Cartesian 1/Angstrom, periodic over the fcc reciprocal lattice."""
    axes = [np.arange(n) / n for n in kgrid]
    frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    images = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    centred = frac - np.round(frac)
    cart = (centred[:, None, :] + images[None, :, :]) @ FCC_RECIPROCAL
    energies = np.min(np.sum(cart**2, axis=-1), axis=1)
    return ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(kgrid=kgrid, kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)),
        kpoints=frac,
        bands=energies[:, None, None],
        projected=np.ones((len(frac), 1, 1, 1, 1)),
        fermi=RADIUS**2,
        reciprocal_lattice=FCC_RECIPROCAL,
        orbital_names=["s"],
    )


@pytest.mark.parametrize("kgrid", [(16, 16, 16), (16, 20, 24)])
def test_sphere_in_a_non_orthogonal_cell_keeps_its_radius(kgrid: tuple[int, int, int]) -> None:
    fs = FermiSurface.from_ebs(sphere_ebs(kgrid))

    radii = np.linalg.norm(fs.points, axis=1)

    assert radii.min() == pytest.approx(RADIUS, abs=0.005)
    assert radii.max() == pytest.approx(RADIUS, abs=0.005)


def test_band_energy_interpolated_onto_the_surface_equals_the_isovalue() -> None:
    fs = FermiSurface.from_ebs(sphere_ebs((16, 20, 24)))

    energies = fs.get_property("bands").to_array()[:, 0, 0]

    assert energies.min() == pytest.approx(RADIUS**2, abs=0.005)
    assert energies.max() == pytest.approx(RADIUS**2, abs=0.005)
