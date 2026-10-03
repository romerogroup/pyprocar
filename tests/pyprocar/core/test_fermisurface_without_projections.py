"""Fermi surfaces from band structures that carry energies only, as BXSF and FRMSF files do."""

from pathlib import Path

import numpy as np

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.fermisurface import FermiSurface
from pyprocar.core.kpoints import KGRID_MODE, KGridInfo

N_K = 10


def _sphere_energies() -> tuple[np.ndarray, np.ndarray]:
    frac = np.arange(N_K) / N_K
    kpoints = np.stack(np.meshgrid(frac, frac, frac, indexing="ij"), axis=-1).reshape(-1, 3)
    centred = (kpoints + 0.5) % 1.0 - 0.5
    return kpoints, np.sum(centred**2, axis=1) - 0.1


def test_fermi_surface_from_frmsf_file(tmp_path: Path) -> None:
    _, energies = _sphere_energies()
    lines = [f"{N_K} {N_K} {N_K}", "1", "1", "1.0 0.0 0.0", "0.0 1.0 0.0", "0.0 0.0 1.0"]
    lines.append(" ".join(f"{e:.6f}" for e in energies))
    (tmp_path / "in.frmsf").write_text("\n".join(lines) + "\n")

    fs = FermiSurface.from_code("frmsf", str(tmp_path))

    assert fs.ebs.n_spins == 1
    assert fs.ebs.spin_projection_names == ["Spin-up"]
    assert list(fs.band_spin_mask) == [(0, 0)]


def test_spin_polarized_mesh_without_projections_keeps_both_spin_surfaces() -> None:
    kpoints, energies = _sphere_energies()
    ebs = ElectronicBandStructureMesh(
        kgrid_info=KGridInfo(
            kgrid=(N_K, N_K, N_K), kgrid_mode=KGRID_MODE.GAMMA, kshift=(0.0, 0.0, 0.0)
        ),
        kpoints=kpoints,
        bands=np.stack([energies, energies + 0.05], axis=1)[:, np.newaxis, :],
        projected=None,
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
    )

    fs = FermiSurface.from_ebs(ebs)

    assert ebs.n_spins == 2
    assert ebs.spin_projection_names == ["Spin-up", "Spin-down"]
    assert list(fs.band_spin_mask) == [(0, 0), (0, 1)]
