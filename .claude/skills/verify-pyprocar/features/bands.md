# Band structure along a k-path

Plot E(k) along a high-symmetry path from a non-SCF bands calculation. Optionally, color or size the bands by atomic/orbital projections, draw band-velocity arrows, or overlay per-species weights.

## Sub-features

- Plain bands: `scalars_mode="none"`.
- Projected bands: `ebs.compute_projected_sum(atoms=, orbitals=)` with `scalars_mode="scatter"` or `"parametric"`.
- Band velocity vectors: `ebs.get_property("bands_velocity")`, drawn with `p.plot_quiver(ebs.kpath, ebs.bands.to_array(), velocity.to_array())`.
- Overlays: `ebs.build_overlay_species_weights(...)` / `build_overlay_orbitals_weights(...)` with `p.plot_overlay(ebs.kpath, ebs.bands, weights, labels=...)`.
- Spin: `spin-polarized` and `non-colinear` fixtures; `channel_mode="flip"` (on the non-spin fixture it renders the same as `normal`).
- Bands + DOS side by side: legacy `pyprocar.bandsdosplot(bands_settings=dict(...), dos_settings=dict(...))`.
- Other legacy modes and notebooks: `ipr`, `atomic`, `overlay*`, `autobandsplot`; fixtures `bands/{ipr,atomic_levels,auto,compare_bands}`. Not driven.
- Band unfolding: `pyprocar.unfold(code="vasp", dirname=<bands/unfolding/supercell copy>, mode=, unfold_mode=, transformation_matrix=np.diag([2, 2, 2]), fermi=5.2182, show=False, savefig=...)`, or `ebs.unfold(transformation_matrix=, structure=ebs.structure)` for the weights alone (#277). Check the weights against `bands/unfolding/primitive` (the same Γ M K Γ path, 150 k-points): every k has 24 supercell bands below -20 eV whose weights sum to 3.00 (the three primitive Mg 2p bands), and every supercell band with weight above 0.25 between -10 and 12 eV lies within 0.5 eV of a primitive band. The primitive run was self-consistent on its line-mode k-points, so the two calculations differ by 0.1 to 0.4 eV. Convention trap: adding a k-dependent phase exp(i k.r) to the weight breaks this (4 eV mismatches, negative weights). `tests/pyprocar/scripts/test_unfold.py` runs these checks.
- Unfolding reference: `scripts/lib/references/unfold.py` computes Popescu-Zunger weights from the raw PROCAR phases and POSCAR with numpy only. Compare it with `ebs.unfold` and the primitive bands:
  ```bash
  $H run ref-unfold data/examples/bands/unfolding .claude/skills/verify-pyprocar/scripts/examples/ref_unfold.py
  ```
  - `validate()` unfolds plane-wave states in supercells of a two-atom cell under diag(2,2,2) and three sheared matrices, one of them not normal, to exactly 0 or 1 (error 1.1e-16, tolerance 1e-12). The same states with an exp(-2 pi i k.t) factor miss by at least 0.259.
  - At d6d4aaa7 the reference matches `ebs.unfold` to 8.9e-16 over all 150 x 80 weights, and the `INDEPENDENT_WEIGHTS` literals to 2.1e-6.
  - Against `primitive/EIGENVAL`, the Mg 2p weights sum to between 2.999 and 3.002 at every k. No band with weight above 0.25 lies more than 0.375 eV from a primitive band.
  - With the k-phase factor, 806 such bands miss by up to 4.2 eV.

## How to get to it (user POV)

1. **Object API:** `ElectronicBandStructurePath.from_code(code="vasp", dirpath=...)`, then `BandStructurePlotter().plot(ebs.bands, ...)`. More calls are in `examples/general/new_bands_examples.py`.
2. **Legacy one-call:** `pyprocar.bandsplot(code="vasp", dirname=..., mode="plain"|"parametric"|"scatter"|..., fermi=5.3017, show=False, savefig=...)`. It returns `(fig, ax)`. This is what `examples/00-band_structure/*.ipynb` use.

## Driving it with verify.sh

```bash
$H run bands data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/bands.py
$H run bands-plain data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/bands_plain.py   # single proven step, exits 0
```

`bands.py` runs every object-API call and the legacy calls as separate steps. At d6d4aaa7 it exits 1 because of the two object-API crashes listed below.

The proven end state (SrVO3, non-spin-polarized, d6d4aaa7) is:
- `obj_plain`: 32 lines, xticklabels `Γ X M Γ R X`, ylim about `[-32.0, 13.6]` eV, meaning energies are already Fermi-shifted (by the parsed `ebs.fermi`, 4.9992). The PNG shows bands crossing 0 eV near Γ–M.
- `obj_parametric` / `obj_scatter`: 20 colored collections plus a colorbar axis. V-d weight is highest in the bands above 5 eV.
- `obj_quiver_plot_quiver`: 20 quiver collections and a colorbar. The x axis shows raw k-distance, with no high-symmetry tick labels.
- `legacy_bandsplot_plain`: 33 lines. `parametric` and `scatter`: 20 collections each. `overlay_species`: 60 collections, one band of O, Sr and V weight per band.
- `legacy_bandsdosplot`: the band panel with ticks `Γ X M Γ R X` and a total-DOS panel on the same energy axis.
- `side_effects.txt` is empty.

## Gotchas

These crash at d6d4aaa7. They are product gaps: record them as failures, and don't route around them.
- **Object `plot_overlay`**: `AttributeError: 'Property' object has no attribute 'ndim'` at `plotter/bs_plot.py:986`. The legacy `overlay_species` mode works.
- **`plot(..., vectors_data=..., vectors_mode="quiver")`**, as written in `new_bands_examples.py`: `vectors_mode` falls through to `Line2D.set()` and raises `AttributeError`. Use `plot_quiver` instead.

Other notes:
- `BandStructurePlotter.plot` draws on the current matplotlib figure. Grab it with `plt.gcf()` to save.
- The fixture ships a prebuilt `ebs.pkl`/`kpath.pkl`, which only `use_cache=True` reads.
