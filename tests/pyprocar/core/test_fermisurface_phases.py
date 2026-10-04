import numpy as np

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from tests.pyprocar.core.test_fermisurface_without_projections import N_K, _sphere_energies
from tests.pyprocar.core.test_ibz2fbz import gamma_info


def test_fermi_surface_pads_the_projections_but_not_the_phases() -> None:
    kpoints, energies = _sphere_energies()
    ebs = ElectronicBandStructureMesh(
        kgrid_info=gamma_info((N_K, N_K, N_K)),
        kpoints=kpoints,
        bands=energies.reshape(-1, 1, 1),
        projected=np.ones((len(kpoints), 1, 1, 1, 1)),
        projected_phase=np.exp(2j * np.pi * kpoints[:, 0]).reshape(-1, 1, 1, 1, 1),
        orbital_names=["s"],
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
    )

    fs = FermiSurface.from_ebs(ebs, padding=2)

    assert fs.ebs.projected_phase is None
    assert fs.ebs.projected is not None
    assert fs.ebs.projected.value.shape == (14**3, 1, 1, 1, 1)
    assert ebs.projected_phase is not None
    np.testing.assert_allclose(
        ebs.projected_phase.value.ravel(), np.exp(2j * np.pi * ebs.kpoints[:, 0]), atol=1e-12
    )
