"""POSCAR and EIGENVAL readers shared by the references, numpy only.

`Poscar.lattice` (3, 3) direct vectors as rows in Angstrom, `Poscar.positions` (n_atoms, 3)
fractional, `Poscar.species` one name per atom. `Eigenval.kpoints` (n_k, 3) fractional,
`Eigenval.energies` (n_k, n_bands) eV.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Poscar:
    lattice: np.ndarray
    positions: np.ndarray
    species: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Eigenval:
    kpoints: np.ndarray
    energies: np.ndarray


def read_poscar(path: Path) -> Poscar:
    """Read a VASP 5 POSCAR with Direct or Cartesian coordinates and optional Selective dynamics."""
    lines = path.read_text(encoding="utf-8").splitlines()
    scale = float(lines[1].split()[0])
    if scale <= 0:
        raise ValueError(f"{path}: a non-positive POSCAR scale (volume form) is not supported")
    lattice = scale * np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    species = tuple(name for name, count in zip(names, counts, strict=True) for _ in range(count))
    mode_line = 7
    if lines[mode_line].strip().lower().startswith("s"):
        mode_line += 1
    cartesian = lines[mode_line].strip().lower()[0] in "ck"
    n_atoms = sum(counts)
    raw = np.array(
        [[float(x) for x in lines[mode_line + 1 + i].split()[:3]] for i in range(n_atoms)]
    )
    positions = raw * scale @ np.linalg.inv(lattice) if cartesian else raw
    return Poscar(lattice=lattice, positions=positions, species=species)


def read_eigenval(path: Path) -> Eigenval:
    """Parse a non-spin-polarized EIGENVAL into k-points (n_k, 3) and energies (n_k, n_bands)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    ispin = int(lines[0].split()[3])
    if ispin != 1:
        raise ValueError(f"{path}: ISPIN = {ispin}; only ISPIN = 1 is supported")
    n_k, n_bands = (int(x) for x in lines[5].split()[1:3])
    tokens = np.array(" ".join(lines[6:]).split(), dtype=float)
    blocks = tokens.reshape(n_k, 4 + 3 * n_bands)
    return Eigenval(kpoints=blocks[:, :3], energies=blocks[:, 4:].reshape(n_k, n_bands, 3)[:, :, 1])
