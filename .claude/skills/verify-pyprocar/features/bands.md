# Band structure along a k-path

Plot E(k) along a high-symmetry path from a non-SCF bands calculation. Optionally, color or size the bands by atomic/orbital projections, draw band-velocity arrows, or overlay per-species weights.

## Sub-features

- Plain bands: `scalars_mode="none"`.
- Projected bands: `ebs.compute_projected_sum(atoms=, orbitals=)` with `scalars_mode="scatter"` or `"parametric"`.
- Band velocity vectors: `ebs.get_property("bands_velocity")`, drawn with `p.plot_quiver(ebs.kpath, ebs.bands.to_array(), velocity.to_array())`.
- Overlays: `ebs.build_overlay_species_weights(...)` / `build_overlay_orbitals_weights(...)` with `p.plot_overlay(ebs.kpath, ebs.bands, weights, labels=...)`.
- Spin: `spin-polarized` and `non-colinear` fixtures; `channel_mode="flip"` (on the non-spin fixture it renders the same as `normal`).
- Bands + DOS side by side: legacy `pyprocar.bandsdosplot(bands_settings=dict(...), dos_settings=dict(...))`.
- Other legacy modes and notebooks: `ipr`, `atomic`, `overlay*`, `autobandsplot`, `unfold`; fixtures `bands/{ipr,atomic_levels,auto,unfolding,compare_bands}`. Not driven.

## How to get to it (user POV)

1. **Object API:** `ElectronicBandStructurePath.from_code(code="vasp", dirpath=...)`, then `BandStructurePlotter().plot(ebs.bands, ...)`. More calls are in `examples/general/new_bands_examples.py`.
2. **Legacy one-call:** `pyprocar.bandsplot(code="vasp", dirname=..., mode="plain"|"parametric"|"scatter"|..., fermi=5.3017, show=False, savefig=...)`. It returns `(fig, ax)`. This is what `examples/00-band_structure/*.ipynb` use.

## Driving it with verify.sh

```bash
$H run bands data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/bands.py
$H run bands-plain data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/bands_plain.py   # single proven step, exits 0
```

`bands.py` runs every object-API call and the legacy calls as separate steps. At 1f36aae1 it exits 1 because of the crashes listed below.

The proven end state (SrVO3, non-spin-polarized) is:
- `obj_plain`: 30 artists, xticklabels `Γ M Γ R X`, ylim about `[-32, 13.6]` eV, meaning energies are already Fermi-shifted (by the parsed `ebs.fermi`, 4.9992). The PNG shows bands crossing 0 eV near Γ–M.
- `obj_parametric` / `obj_scatter`: 20 colored collections plus a colorbar axis. V-d weight is highest in the bands above 5 eV.
- `obj_quiver_plot_quiver`: 20 quiver collections and a colorbar. The x axis shows raw k-distance, with no high-symmetry tick labels.
- `side_effects.txt` lists `./ebs.pkl`.

## Gotchas

These crash at 1f36aae1. They are product gaps: record them as failures, and don't route around them.
- **Legacy `bandsplot` plain** and **`bandsdosplot`**: `AttributeError: 'KPath' object has no attribute 'metadata'` at `plotter/bs_plot.py:346`.
- **Legacy `bandsplot` parametric/overlay_species** and **object `plot_overlay`**: `AttributeError: 'Property' object has no attribute 'ndim'` at `plotter/bs_plot.py:1551`.
- **Legacy `bandsplot(mode="scatter")`**: `ValueError: Invalid mode: scatter`, even though the error message lists `scatter`. The enum member is misspelled `SACATTER` in `scripts/scriptBandsplot.py`.
- **`plot(..., vectors_data=..., vectors_mode="quiver")`**, as written in `new_bands_examples.py`: `vectors_mode` falls through to `Line2D.set()` and raises `AttributeError`. Use `plot_quiver` instead.

Other notes:
- `BandStructurePlotter.plot` draws on the current matplotlib figure. Grab it with `plt.gcf()` to save.
- `from_code` always writes `ebs.pkl` into the calc dir. The fixture also ships a prebuilt `ebs.pkl`/`kpath.pkl`, which only `use_cache=True` would read.
