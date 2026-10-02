# Density of states

Plot total and projected DOS versus energy. The `dos-rewrite` branch is rebuilding this feature.

## Sub-features

- Total DOS: `dos.total`, a `Property`; use `.to_array()` to get the values.
- Projected sums: `dos.compute_projected_sum(atoms=, orbitals=, spins=, norm_mode=)`. `norm_mode` is one of `raw` (the default), `max`, `integral`, `electrons`, `total`, `total_projection`, `spin_magnitude` or `magnetization`.
- Total colored by projection: `plotter.plot(total, scalars_data=projected_sum, scalars_mode="line")`.
- Orientation: `DOSPlotter(orientation="horizontal"|"vertical")`.
- Spin: `channel_mode="flip"|"normal"`, plus `scalars_show_colorbar="per_channel"`, using the `spin-polarized` fixture. Not driven.

## How to get to it (user POV)

1. **Object API:** `DensityOfStates.from_code(code="vasp", dirpath=...)`, then `DOSPlotter(...).plot(...)` and `plotter.legend()`. The plotter owns `.fig`/`.ax`. See `examples/general/new_dos_examples.py`.
2. **Legacy one-call:** `pyprocar.dosplot(code="vasp", dirname=..., mode=..., fermi=5.3017, elimit=[-6, 4], show=False, savefig=...)`. The modes are `plain`, `parametric`, `parametric_line`, `stack`, `stack_species`, `stack_orbitals`, `overlay`, `overlay_species` and `overlay_orbitals`. See `examples/01-dos/*.ipynb`.

## Driving it with verify.sh

```bash
$H run dos data/examples/dos/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/dos.py
```

The proven end state (SrVO3, non-spin-polarized, 1f36aae1):
- `obj_total`: `total_shape` is `[6000, 1]` and `total_max` is 62.91. The energy range is about `[-31.1, 12.6]` eV, already Fermi-shifted. The PNG shows a non-zero curve with peaks near -29, -13 and 0 eV.
- `obj_total_colored_by_projection`: `proj_le_total` is true, with 1 colored collection plus a colorbar axis.
- `obj_vertical_projected_legend`: the legend reads `Total`, `O_{2-4}-(p)`.
- `obj_new_files_in_calc` is `[]`. `from_code` writes `dos.pkl` only when `use_cache=True`.

## Gotchas

- **Legacy `dosplot` crashes in every mode** at 1f36aae1, with `AttributeError: property 'energies' of 'DensityOfStates' object has no setter` at `scripts/scriptDosplot.py:338`. This is a product gap.
- Before that crash, legacy `dosplot` with the default `use_cache=False` **deletes and rewrites** the calc dir's `dos.pkl` and `structure.pkl`. They show up in `side_effects.txt`. This is one reason the harness works on a copy.
- The fixture ships `dos.pkl`, `structure.pkl` and PDFs from old runs. Ignore them; they are not evidence.
