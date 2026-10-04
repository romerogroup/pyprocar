# Popescu-Zunger reference for band unfolding: the reference weights from the
# raw PROCAR phases against ebs.unfold, the test literals, and the primitive EIGENVAL bands.
# Fixture: data/examples/bands/unfolding (uses its supercell/ and primitive/ subdirs)
import ast
import functools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from references.unfold import (
    Procar,
    atom_permutations,
    primitive_translations,
    read_procar_phases,
    translation_overlaps,
    unfolding_weights,
    validate,
)
from references.validation import require
from references.vasp import Eigenval, read_eigenval, read_poscar
from verify_steps import CALC, REPO, finish, png, step

from pyprocar.core import ElectronicBandStructurePath

MATRIX = np.diag([2, 2, 2])
TEST_TOLERANCE = 1e-5
"""The abs tolerance at which tests/pyprocar/scripts/test_unfold.py compares its literals."""
SUPERCELL, PRIMITIVE = CALC / "supercell", CALC / "primitive"


@dataclass(frozen=True, slots=True)
class FixtureReference:
    procar: Procar
    eigenval: Eigenval
    translations: np.ndarray
    permutations: np.ndarray
    weights: np.ndarray


@functools.cache
def _reference() -> FixtureReference:
    procar = read_procar_phases(SUPERCELL / "PROCAR")
    poscar = read_poscar(SUPERCELL / "POSCAR")
    translations = primitive_translations(MATRIX)
    perms = atom_permutations(poscar.positions, poscar.species, translations)
    return FixtureReference(
        procar=procar,
        eigenval=read_eigenval(PRIMITIVE / "EIGENVAL"),
        translations=translations,
        permutations=perms,
        weights=unfolding_weights(procar.phases, perms),
    )


def _band_matching(
    ref: FixtureReference, weights: np.ndarray
) -> tuple[list[float], list[float], float]:
    """Unmatched visible supercell bands, unmatched primitive bands, worst visible distance."""
    energies, primitive = ref.procar.energies, ref.eigenval.energies
    unmatched_supercell: list[float] = []
    unmatched_primitive: list[float] = []
    worst = 0.0
    for e, w, p in zip(energies, weights, primitive, strict=True):
        visible = e[(w > 0.25) & (e > -10) & (e < 12)]
        distance = np.abs(visible[:, None] - p[None, :]).min(axis=1)
        worst = max(worst, float(distance.max(initial=0.0)))
        unmatched_supercell += visible[distance > 0.5].tolist()
        reference = p[(p > -9.5) & (p < 11.5)]
        unmatched_primitive += [float(x) for x in reference if np.abs(visible - x).min() > 0.5]
    return unmatched_supercell, unmatched_primitive, worst


@step("analytic_validation")
def _():
    return validate()


@step("reference_weights")
def _():
    ref = _reference()
    require(ref.procar.phases.shape == (150, 80, 24, 9), str(ref.procar.phases.shape))
    require(ref.eigenval.energies.shape == (150, 15), str(ref.eigenval.energies.shape))
    require(len(ref.translations) == 8, f"{len(ref.translations)} translations, not 8")
    return {
        "phases_shape": list(ref.procar.phases.shape),
        "primitive_bands_shape": list(ref.eigenval.energies.shape),
        "n_translations": len(ref.translations),
    }


@step("reference_vs_pyprocar")
def _():
    weights = _reference().weights
    ebs = ElectronicBandStructurePath.from_code("vasp", str(SUPERCELL))
    ebs.unfold(transformation_matrix=MATRIX, structure=ebs.structure)
    assert ebs.weights is not None
    library = np.asarray(ebs.weights.value)[..., 0]
    require(library.shape == weights.shape == (150, 80), str(library.shape))
    diff = float(np.abs(library - weights).max())
    require(diff < 1e-9, f"pyprocar weights differ from the reference by {diff:.2e}")
    return {"shape": list(weights.shape), "max_abs_diff": diff}


@step("reference_vs_test_literals")
def _():
    weights = _reference().weights
    tree = ast.parse((REPO / "tests/pyprocar/scripts/test_unfold.py").read_text(encoding="utf-8"))
    literal = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "INDEPENDENT_WEIGHTS" for t in node.targets)
    )
    expected: dict[tuple[int, int], float] = ast.literal_eval(literal)
    errors = {str(k): abs(float(weights[k]) - v) for k, v in expected.items()}
    worst = max(errors.values())
    require(len(expected) == 7, str(expected))
    require(
        worst < TEST_TOLERANCE,
        f"reference differs from INDEPENDENT_WEIGHTS by {worst:.2e}: {errors}",
    )
    return {"n_literals": len(expected), "max_abs_diff": worst, "abs_diff": errors}


@step("physics_vs_primitive")
def _():
    ref = _reference()
    energies, weights = ref.procar.energies, ref.weights
    kpoints = ref.procar.kpoints @ np.linalg.inv(MATRIX).T
    k_error = float(np.abs(kpoints - ref.eigenval.kpoints).max())
    require(k_error < 1e-4, f"supercell k-points are not the primitive ones ({k_error:.2e})")
    semicore = energies < -20
    require(
        set((ref.eigenval.energies < -20).sum(axis=1).tolist()) == {3},
        "3 primitive Mg 2p bands below -20 eV at every k",
    )
    require(
        set(semicore.sum(axis=1).tolist()) == {24},
        "24 supercell Mg 2p bands below -20 eV at every k",
    )
    semicore_sum = (weights * semicore).sum(axis=1)
    semicore_error = float(np.abs(semicore_sum - 3.0).max())
    require(semicore_error < 5e-3, f"Mg 2p weights sum off 3 by {semicore_error:.3e}")
    gamma = np.sort(weights[0][semicore[0]])
    gamma_error = float(np.abs(gamma - np.array([0.0] * 21 + [1.0] * 3)).max())
    require(gamma_error < 1e-3, str(gamma.tolist()))
    require(
        weights.min() > -1e-9 and weights.max() < 1 + 1e-9,
        f"weights span {weights.min()} to {weights.max()}, outside [0, 1]",
    )

    unmatched_supercell, unmatched_primitive, worst = _band_matching(ref, weights)
    require(unmatched_supercell == [], str(unmatched_supercell[:10]))
    require(unmatched_primitive == [], str(unmatched_primitive[:10]))

    phase = np.exp(-2j * np.pi * ref.procar.kpoints @ ref.translations.T)
    overlaps = translation_overlaps(ref.procar.phases, ref.permutations)
    kphase = (overlaps * phase[:, None, :]).mean(axis=-1).real
    control_supercell, control_primitive, control_worst = _band_matching(ref, kphase)
    require(
        bool(control_supercell or control_primitive), "k-phase convention passed the band match"
    )
    return {
        "kpoint_map_error": k_error,
        "semicore_weight_sum_min": float(semicore_sum.min()),
        "semicore_weight_sum_max": float(semicore_sum.max()),
        "gamma_semicore_error": gamma_error,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "worst_visible_distance_eV": worst,
        "unmatched_supercell": len(unmatched_supercell),
        "unmatched_primitive": len(unmatched_primitive),
        "kphase_control_unmatched_supercell": len(control_supercell),
        "kphase_control_unmatched_primitive": len(control_primitive),
        "kphase_control_worst_distance_eV": control_worst,
        "kphase_control_weight_min": float(kphase.min()),
    }


@step("plot_reference_over_primitive")
def _():
    ref = _reference()
    energies, weights = ref.procar.energies, ref.weights
    n_k, n_bands = energies.shape
    order = np.argsort(weights.ravel())
    x = np.repeat(np.arange(n_k), n_bands)[order]
    w = weights.ravel()[order]
    fig, (ax, core) = plt.subplots(
        1, 2, figsize=(11, 6), sharex=True, gridspec_kw={"width_ratios": [3, 1]}
    )
    points = None
    for axis, window in ((ax, (-10.0, 12.0)), (core, (-37.6, -36.6))):
        axis.plot(ref.eigenval.energies, color="0.55", lw=1.0, zorder=1)
        points = axis.scatter(
            x,
            energies.ravel()[order],
            c=w,
            s=1 + 12 * np.clip(w, 0, 1),
            cmap="Blues",
            vmin=0,
            vmax=1,
            edgecolors="none",
            zorder=2,
        )
        axis.set_ylim(*window)
        axis.set_xticks([0, 49, 99, 149], ["Γ", "M", "K", "Γ"])
        axis.set_xlim(0, n_k - 1)
        axis.grid(axis="x", color="0.85", lw=0.8)
    assert points is not None
    fig.colorbar(points, ax=core, label="reference weight W")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Reference W (dots) over primitive EIGENVAL bands (gray lines)")
    core.set_title("Mg 2p semicore")
    fig.tight_layout()
    return {"png_bytes": png("ref_unfold_weights", fig)}


finish()
