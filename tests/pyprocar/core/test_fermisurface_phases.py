import numpy as np

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

N_K = 10
PADDING = 2


def test_fermi_surface_pads_the_projections_but_not_the_phases() -> None:
    frac = np.arange(N_K) / N_K
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    energies = np.sum(centred**2, axis=1) - 0.1
    ebs = ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, N_K), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=kpoints,
        bands=energies.reshape(-1, 1, 1),
        projected=np.ones((len(kpoints), 1, 1, 1, 1)),
        projected_phase=np.exp(2j * np.pi * kpoints[:, 0]).reshape(-1, 1, 1, 1, 1),
        orbital_names=["s"],
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
    )

    fs = FermiSurface.from_ebs(ebs, padding=PADDING)

    assert fs.ebs.projected_phase is None
    assert fs.ebs.projected is not None
    assert fs.ebs.projected.value.shape == (14**3, 1, 1, 1, 1)
    assert ebs.projected_phase is not None
    np.testing.assert_allclose(
        ebs.projected_phase.value.ravel(), np.exp(2j * np.pi * ebs.kpoints[:, 0]), atol=1e-12
    )
