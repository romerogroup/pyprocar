# Non-collinear spin under ibz2fbz, against the spin_ibz reference.
# Fixture: data/examples/fermi2d/bisb_monolayer, the full 60x60x1 SOC mesh (ISYM=-1), the truth.
# The reference reduces it to one k per orbit of P3m1 times time reversal. The IBZ rows go back
# through the public ElectronicBandStructureMesh constructor, which unfolds them with ibz2fbz.
import matplotlib.pyplot as plt
import numpy as np
from references.spin_ibz import (
    IMAGE_CLASS_LABELS,
    burnside_orbit_count,
    grid_cells,
    point_group,
    read_cell,
    reduce_grid,
    rotate_spin,
    validate,
)
from references.validation import require
from verify_steps import CALC, finish, png, step

from pyprocar.core.ebs import ElectronicBandStructureMesh
from pyprocar.core.structure import Structure

N_BANDS = 60
DEGENERATE_EV = 5e-3
CLEAR_GAP_EV = 5e-2
# Worst case of PROCAR's 3-decimal rounding: 18 entries per component, two spins compared.
ROUNDING_BOUND = 2 * np.sqrt(3) * 18 * 5e-4
SPIN: dict[str, np.ndarray] = {}


@step("analytic_validation")
def _():
    return validate()


loaded = ElectronicBandStructureMesh.from_code("vasp", str(CALC))
assert isinstance(loaded, ElectronicBandStructureMesh)
assert loaded.bands is not None and loaded.projected is not None
assert loaded.structure is not None
full = loaded
STRUCTURE = loaded.structure
N_PARSER_ROTATIONS = 0 if STRUCTURE.rotations is None else len(STRUCTURE.rotations)
RAW_BANDS = loaded.bands.to_array()
FULL_BANDS = RAW_BANDS[..., 0] if RAW_BANDS.ndim == 3 else RAW_BANDS
FULL_PROJECTED = loaded.projected.to_array()
MESH = np.array(full.kgrid_info.kgrid)
GROUP = point_group(read_cell(CALC / "POSCAR"), symprec=1e-3)
REDUCTION = reduce_grid(full.kpoints, MESH, GROUP)


@step("full_mesh")
def _():
    require(full.n_kpoints == 3600 and MESH.tolist() == [60, 60, 1], "the full 60x60x1 mesh")
    require(full.is_non_collinear and FULL_PROJECTED.shape[2] == 4, "four spin channels")
    return {
        "n_kpoints": int(full.n_kpoints),
        "kgrid": MESH.tolist(),
        "bands_shape": list(RAW_BANDS.shape),
        "projected_shape": list(FULL_PROJECTED.shape),
        "fermi": float(full.fermi),
        "parser_rotations": N_PARSER_ROTATIONS,
    }


@step("reduce")
def _():
    n_ibz = len(REDUCTION.ibz_indices)
    burnside = burnside_orbit_count(MESH, GROUP)
    require(len(GROUP.kspace) == 6 and int(np.sum(GROUP.det < 0)) == 3, "P3m1 at symprec 1e-3")
    by_hand = (3600 + 4 + 2 * 3 + 2 * 1 + 6 * 60) // 12
    require(n_ibz == by_hand == burnside, f"{by_hand} orbits expected, got {n_ibz}, {burnside}")
    require(int(REDUCTION.orbit_size.sum()) == full.n_kpoints, "the orbits cover the grid once")
    counts = np.bincount(REDUCTION.image_class, minlength=len(IMAGE_CLASS_LABELS))
    return {
        "n_ibz": n_ibz,
        "burnside_orbits": burnside,
        "kspace_operations": GROUP.kspace.tolist(),
        "image_class_counts": dict(zip(IMAGE_CLASS_LABELS, counts.tolist(), strict=True)),
    }


def nearest_band_gap(bands: np.ndarray) -> np.ndarray:
    """Distance from each (k, band) energy to the closest other band, shape (n_k, n_bands)."""
    steps = np.diff(np.sort(bands, axis=1), axis=1)
    padded = np.pad(steps, ((0, 0), (1, 1)), constant_values=np.inf)
    return np.minimum(padded[:, :-1], padded[:, 1:])


def error_stats(error: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    picked = error[mask]
    return {"entries": int(picked.size), "mean": float(picked.mean()), "max": float(picked.max())}


@step("unfold_spin")
def _():
    ibz = REDUCTION.ibz_indices
    structure = Structure(
        atoms=STRUCTURE.atoms,
        fractional_coordinates=STRUCTURE.fractional_coordinates,
        lattice=STRUCTURE.lattice,
        rotations=GROUP.kspace.astype(float),
    )
    unfolded = ElectronicBandStructureMesh(
        kgrid_info=full.kgrid_info,
        kpoints=full.kpoints[ibz],
        bands=RAW_BANDS[ibz],
        projected=FULL_PROJECTED[ibz],
        fermi=full.fermi,
        reciprocal_lattice=full.reciprocal_lattice,
        orbital_names=full.orbital_names,
        structure=structure,
    )
    require(unfolded.n_kpoints == full.n_kpoints, "ibz2fbz restores every grid point")
    assert unfolded.projected is not None and unfolded.bands is not None
    projected = unfolded.projected.to_array()

    row_of_cell = np.empty(full.n_kpoints, dtype=int)
    row_of_cell[grid_cells(full.kpoints, MESH)] = np.arange(full.n_kpoints)
    target = row_of_cell[grid_cells(unfolded.kpoints, MESH)]
    require(np.array_equal(np.sort(target), np.arange(full.n_kpoints)), "one row per grid node")
    source = ibz[REDUCTION.source[target]]
    require(
        np.array_equal(projected[:, :, 0], FULL_PROJECTED[source][:, :, 0]),
        "channel 0 is copied unchanged from the IBZ source",
    )
    require(
        np.array_equal(unfolded.bands.to_array(), RAW_BANDS[source]), "bands come from the source"
    )

    spin_full = FULL_PROJECTED[:, :N_BANDS, 1:].sum(axis=(-2, -1))
    expected_full = rotate_spin(REDUCTION, spin_full[ibz])
    truth = spin_full[target]
    expected = expected_full[target]
    out = projected[:, :N_BANDS, 1:].sum(axis=(-2, -1))
    out_truth = np.linalg.norm(out - truth, axis=-1)
    out_expected = np.linalg.norm(out - expected, axis=-1)
    expected_truth = np.linalg.norm(expected - truth, axis=-1)

    gap = nearest_band_gap(FULL_BANDS)[target, :N_BANDS]
    nondegenerate = gap > DEGENERATE_EV
    clear = gap > CLEAR_GAP_EV
    generic = np.broadcast_to(
        (REDUCTION.orbit_size[REDUCTION.source[target]] == 12)[:, np.newaxis], gap.shape
    )
    everything = np.ones_like(nondegenerate)

    operation = REDUCTION.operation[target]
    mirror = GROUP.det[operation] < 0
    polar = REDUCTION.spin_rotation[target] * np.where(mirror, -1.0, 1.0)[:, None, None]
    polar_truth = np.linalg.norm(
        np.einsum("kij,kbj->kbi", polar, spin_full[source]) - truth, axis=-1
    )
    mirror_clean = mirror[:, np.newaxis] & nondegenerate

    image_class = REDUCTION.image_class[target]
    per_class: dict[str, dict[str, object]] = {}
    for code, label in enumerate(IMAGE_CLASS_LABELS):
        rows = np.broadcast_to((image_class == code)[:, np.newaxis], gap.shape)
        per_class[label] = {
            "n_kpoints": int((image_class == code).sum()),
            "mean_abs_spin_truth": float(np.linalg.norm(truth, axis=-1)[rows].mean()),
            "pyprocar_vs_truth": error_stats(out_truth, rows),
            "pyprocar_vs_truth_nondegenerate": error_stats(out_truth, rows & nondegenerate),
            "reference_vs_truth_nondegenerate": error_stats(expected_truth, rows & nondegenerate),
            "pyprocar_vs_reference": error_stats(out_expected, rows),
            "pyprocar_vs_reference_generic_orbits": error_stats(out_expected, rows & generic),
        }

    matches_reference = error_stats(out_expected, generic)
    matches_truth = error_stats(out_truth, nondegenerate)
    matches_truth_clear = error_stats(out_truth, clear)
    polar_control = error_stats(polar_truth, mirror_clean)
    # The lattice is hexagonal to ~1e-6, so B.T R inv(B.T) and A.T W inv(A.T) differ at 1e-6.
    require(matches_reference["max"] < 1e-4, "pyprocar rotates spin as the reference")
    # Random rounding of 2 x 18 entries per component alone gives a mean near 0.003.
    require(matches_truth["mean"] < 0.01, "the unfolded spin matches the full mesh")
    require(
        matches_truth_clear["max"] < ROUNDING_BOUND,
        "well-separated bands match the full mesh within PROCAR rounding",
    )
    require(polar_control["mean"] > 0.05, "a polar spin on the mirror images misses the full mesh")

    SPIN["truth"] = spin_full
    SPIN["expected"] = expected_full
    SPIN["out"] = np.empty_like(out)
    SPIN["out"][target] = out
    return {
        "n_unfolded": int(unfolded.n_kpoints),
        "n_bands_compared": N_BANDS,
        "nondegenerate_fraction": float(nondegenerate.mean()),
        "band_energy_max_error_ev": float(
            np.abs(FULL_BANDS[source, :N_BANDS] - FULL_BANDS[target, :N_BANDS]).max()
        ),
        "pyprocar_vs_truth": error_stats(out_truth, everything),
        "pyprocar_vs_truth_nondegenerate": matches_truth,
        "pyprocar_vs_truth_gap_over_50mev": matches_truth_clear,
        "reference_vs_truth_nondegenerate": error_stats(expected_truth, nondegenerate),
        "pyprocar_vs_reference": error_stats(out_expected, everything),
        "pyprocar_vs_reference_generic_orbits": matches_reference,
        "pyprocar_vs_reference_special_orbits_nondegenerate": error_stats(
            out_expected, ~generic & nondegenerate
        ),
        "rounding_bound": float(ROUNDING_BOUND),
        "negative_polar_vs_truth_mirror_nondegenerate": polar_control,
        "per_class": per_class,
    }


@step("spin_texture_png")
def _():
    energies = FULL_BANDS[:, :N_BANDS] - full.fermi
    band = int(np.argmin(np.abs(energies).min(axis=0)))
    nodes = np.rint(full.kpoints * MESH).astype(int) % MESH
    k_cart = (full.kpoints - np.rint(full.kpoints)) @ np.asarray(full.reciprocal_lattice)
    shown = (nodes[:, 0] % 2 == 0) & (nodes[:, 1] % 2 == 0)
    limit = float(np.abs(SPIN["truth"][:, band, 2]).max()) or 1.0
    arrow_step = 2 * float(np.linalg.norm(np.asarray(full.reciprocal_lattice)[0])) / MESH[0]
    longest = float(np.linalg.norm(SPIN["truth"][:, band, :2], axis=-1).max()) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    panels = {
        "truth": "full mesh (truth)",
        "expected": "reference: rotated IBZ spin",
        "out": "pyprocar ibz2fbz",
    }
    quiver = None
    for ax, (key, title) in zip(axes, panels.items(), strict=True):
        spin = SPIN[key][shown, band]
        quiver = ax.quiver(
            k_cart[shown, 0],
            k_cart[shown, 1],
            spin[:, 0],
            spin[:, 1],
            spin[:, 2],
            cmap="coolwarm",
            clim=(-limit, limit),
            pivot="middle",
            angles="xy",
            scale_units="xy",
            scale=longest / arrow_step,
            width=0.004,
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("k_x (1/Angstrom)")
    assert quiver is not None
    axes[0].set_ylabel("k_y (1/Angstrom)")
    fig.colorbar(quiver, ax=axes, label="S_z", shrink=0.8)
    distance = float(np.abs(energies[:, band]).min())
    fig.suptitle(f"BiSb monolayer spin texture, band {band}, min |E - E_F| = {distance:.3f} eV")
    size = png("spin_texture", fig)
    require(size > 20_000, "the spin texture PNG has content")
    return {"band": band, "min_abs_e_minus_ef_ev": distance, "png_bytes": size}


finish()
