"""QE phases unfold to the full k grid with the orbital rotations of the core.

QE's real harmonics (upflib/ylmr2.f90) carry the Condon-Shortley sign (-1)^m, so px, py,
dxz, dyz, fxz2, fyz2, fx3-3xy2 and f3yx2-y3 have the opposite sign to the core's. The
synthetic band below has the coefficients Y_lm(v(k)) of the standard real harmonics at
v(k) = sin(2 pi k), a field each cubic operation carries along, so its coefficients at
every k of the full grid are known without unfolding.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pyprocar.core.ebs import ElectronicBandStructureMesh, orbital_rotation
from pyprocar.io.qe.parser import QEParser
from tests.pyprocar.core.test_ebs_unfold_exact_grid import CUBIC_OPERATIONS
from tests.pyprocar.core.test_ibz2fbz import gamma_grid, gamma_info, key, single_site, wedge
from tests.utils import DATA_DIR

# projwfc.x's labels (PP/src/projwfc.f90, lm_label_global_frame)
QE_NAMES = ["s", "pz", "px", "py", "dz2", "dxz", "dyz", "dx2-y2", "dxy"] + [
    "fz3",
    "fxz2",
    "fyz2",
    "fzx2-zy2",
    "fxyz",
    "fx3-3xy2",
    "f3yx2-y3",
]
QE_SIGNED = {"px", "py", "dxz", "dyz", "fxz2", "fyz2", "fx3-3xy2", "f3yx2-y3"}
SHELLS = [slice(0, 1), slice(1, 4), slice(4, 9), slice(9, 16)]
CUBIC = np.array(CUBIC_OPERATIONS, dtype=float)
STATES = "\n".join(
    f"     state #{i + 1:4d}: atom   1 (X  ), wfc {deg + 1:2d} (l={deg} m={m:2d})"
    for i, (deg, m) in enumerate((deg, m) for deg in range(4) for m in range(1, 2 * deg + 2))
)
PROJWFC_OUT = f"""
     Program PROJWFC v.7.5 starts on  1Jan2026 at 12: 0: 0

     Atomic states used for projection
     (read from pseudopotential files):

{STATES}

     natomwfc =   16
     nbnd     =    1
     nspin    =    1

=------------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------------=
"""


def real_harmonics(v: np.ndarray) -> np.ndarray:
    """Standard real harmonics times r^l, in projwfc.x's order s, pz px py, dz2 ... dxy, fz3 ..."""
    x, y, z = v.T
    r2 = x**2 + y**2 + z**2
    d, f = np.sqrt(15 / np.pi), np.sqrt(35 / (2 * np.pi))
    return np.stack(
        [
            np.full_like(x, 0.5 / np.sqrt(np.pi)),
            np.sqrt(3 / (4 * np.pi)) * z,
            np.sqrt(3 / (4 * np.pi)) * x,
            np.sqrt(3 / (4 * np.pi)) * y,
            np.sqrt(5 / np.pi) / 4 * (3 * z**2 - r2),
            d / 2 * x * z,
            d / 2 * y * z,
            d / 4 * (x**2 - y**2),
            d / 2 * x * y,
            np.sqrt(7 / np.pi) / 4 * z * (5 * z**2 - 3 * r2),
            np.sqrt(21 / (2 * np.pi)) / 4 * x * (5 * z**2 - r2),
            np.sqrt(21 / (2 * np.pi)) / 4 * y * (5 * z**2 - r2),
            np.sqrt(105 / np.pi) / 4 * z * (x**2 - y**2),
            np.sqrt(105 / np.pi) / 2 * x * y * z,
            f / 4 * x * (x**2 - 3 * y**2),
            f / 4 * y * (3 * x**2 - y**2),
        ],
        axis=-1,
    )


def field(k: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * np.asarray(k, dtype=float))


def test_qe_phases_unfold_to_the_coefficients_at_each_grid_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "projwfc.out").write_text(PROJWFC_OUT)
    ibz = np.asarray(wedge(gamma_grid((4, 4, 4)), CUBIC), dtype=float)
    qe_signs = np.array([-1.0 if name in QE_SIGNED else 1.0 for name in QE_NAMES])
    qe_phase = (real_harmonics(field(ibz)) * qe_signs).astype(complex)
    qe_phase = qe_phase[:, np.newaxis, np.newaxis, np.newaxis, :]

    parser = QEParser(tmp_path)
    monkeypatch.setattr(QEParser, "kpoints", property(lambda _: ibz))
    parser.__dict__.update(
        spd_phase=qe_phase,
        spd=np.abs(qe_phase) ** 2,
        bands=np.zeros((len(ibz), 1, 1)),
        fermi=0.0,
        reciprocal_lattice=np.eye(3),
        structure=single_site(rotations=CUBIC),
        kpath=None,
        kgrid_info=gamma_info((4, 4, 4)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ebs = parser.ebs

    assert ebs is not None and ebs.projected is not None and ebs.projected_phase is not None
    assert ebs.orbital_names == QE_NAMES
    expected = real_harmonics(field(ebs.kpoints))
    assert len({key(k) for k in ebs.kpoints}) == 64
    np.testing.assert_allclose(ebs.projected.to_array()[:, 0, 0, 0], expected**2, atol=1e-12)
    unfolded = ebs.projected_phase.to_array()[:, 0, 0, 0]
    for shell in SHELLS:
        np.testing.assert_allclose(
            np.einsum("ki,kj->kij", unfolded[:, shell], unfolded[:, shell].conj()),
            np.einsum("ki,kj->kij", expected[:, shell], expected[:, shell]),
            atol=1e-12,
        )


@pytest.mark.data
@pytest.mark.parametrize(
    ("mag", "keeps_phase"),
    [("non-spin-polarized", True), ("spin-polarized-colinear", True), ("non-colinear", False)],
)
def test_qe_mesh_rotates_every_orbital_and_keeps_collinear_phases(
    mag: str, keeps_phase: bool
) -> None:
    calc = DATA_DIR / f"codes/qe/7.2/SrVO3/{mag}/fermi"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ebs = ElectronicBandStructureMesh.from_code("qe", str(calc))

    assert [str(w.message) for w in caught if issubclass(w.category, UserWarning)] == []
    assert (ebs.projected_phase is not None) == keeps_phase
    assert ebs.n_kpoints == 16**3


@pytest.mark.data
@pytest.mark.parametrize("mag", ["non-spin-polarized", "spin-polarized-colinear"])
def test_qe_mesh_bands_are_eigenvectors_of_their_little_group(mag: str) -> None:
    """A nondegenerate band at k is an eigenvector of each orbital rotation whose R fixes k.

    Every cubic operation maps V onto itself up to a lattice vector, so lattice and Bloch factors
    are one common phase. QE's harmonic signs break this wherever an operation mixes a signed
    orbital with an unsigned one.
    """
    ebs = ElectronicBandStructureMesh.from_code(
        "qe", str(DATA_DIR / f"codes/qe/7.2/SrVO3/{mag}/fermi")
    )
    phase, bands, structure, names = (
        ebs.projected_phase,
        ebs.bands,
        ebs.structure,
        ebs.orbital_names,
    )
    assert phase is not None and bands is not None and structure is not None
    assert structure.atoms is not None and names is not None
    vanadium = [str(atom) for atom in structure.atoms].index("V")
    phases = np.asarray(phase.value)[:, :, :, vanadium]
    gaps = np.pad(
        np.diff(np.asarray(bands.value), axis=1),
        ((0, 0), (1, 1), (0, 0)),
        constant_values=np.inf,
    )
    norms = np.linalg.norm(phases, axis=-1)
    selected = (np.minimum(gaps[:, :-1], gaps[:, 1:]) > 1e-3) & (norms**2 > 0.05)

    residuals = []
    for rotation in CUBIC:
        steps = ebs.kpoints @ rotation.T - ebs.kpoints
        fixed = np.all(np.abs(steps - np.round(steps)) < 1e-6, axis=1)
        mask = selected & fixed[:, np.newaxis, np.newaxis]
        coefficients, norm = phases[mask], norms[mask]
        turned = coefficients @ orbital_rotation(names, rotation).T
        overlap = np.sum(coefficients.conj() * turned, axis=-1) / norm**2
        residual = turned - overlap[:, np.newaxis] * coefficients
        residuals.append(np.linalg.norm(residual, axis=-1) / norm)
    residuals = np.concatenate(residuals)

    assert len(residuals) > 1000
    assert residuals.max() < 0.05
