# Density of states

Plot total and projected DOS versus energy.

## Sub-features

- Total DOS: `dos.total`, a `Property`; use `.to_array()` to get the values.
- Projected sums: `dos.compute_projected_sum(atoms=, orbitals=, spins=, species=, norm_mode=)`. `norm_mode` is one of `raw` (the default), `max`, `integral`, `electrons`, `total`, `total_projection`, `spin_magnitude` or `magnetization`.
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

The proven end state (SrVO3, non-spin-polarized, c13166ce):
- `obj_total`: `total_shape` is `[6000, 1]` and `total_max` is 62.91. The energy range is `[-31.06, 12.62]` eV, the raw vasprun grid: `from_code` does not shift by the Fermi energy, and only `dosplot` subtracts `fermi`. The PNG shows a non-zero curve with peaks near -29, -13, -11, 0 and 6.5 eV.
- `obj_total_colored_by_projection`: `proj_le_total` is true, with 1 colored collection plus a colorbar axis.
- `obj_vertical_projected_legend`: the legend texts are `Total` and `$\mathrm{O}_{2-4}-(p)$` (labels are LaTeX strings).
- Vertical spin-polarized total (#304, issue #285; `test_vertical_spin_polarized_total_labels_and_flips_each_channel` builds a synthetic `DensityOfStates`): `DOSPlotter.plot` in vertical orientation labels the channels `$Total - \uparrow$` and `$Total - \downarrow$` and mirrors the down channel to negative DOS. Before #304 both read `Total - ↑` and the down channel was not mirrored. No driver runs this on `data/examples/dos/spin-polarized`.
- `obj_new_files_in_calc` is `[]`. `from_code` writes `dos.pkl` only when `use_cache=True`.
- `legacy_dosplot_<mode>`, all with xlim `[-6, 4]`: `plain` 3 lines and legend `Total`; `parametric` 2 lines; `parametric_line` 1 colored collection; `stack_species` and `stack_orbitals` (atom 1) 3 filled collections under the total; `overlay` with `items={'O': [1, 2, 3], 'V': [4..8]}` 7 lines; `overlay_species` and `overlay_orbitals` 9 lines. The `stack_*` and `overlay*` legend entries end in `[\uparrow]` on this non-spin-polarized fixture, while the total reads `Total` (see Gotchas).
- `side_effects.txt` is empty.

## Gotchas

- **`dosplot(mode="stack")` without `items` crashes** at c13166ce with `IndexError: list index out of range` at `pyprocar/scripts/scriptDosplot.py:427`: it builds no components, and `_stack` reads `components[0]`. The docstring says it falls back to `stack_species`. This is a product gap; pass `items=` or use `stack_species`.
- Projection legends on a one-channel DOS name the channel: `$\mathrm{O}_{2-4}[\uparrow]$` beside a `Total` without an arrow. The spin label comes from the channel name `spin-up` (`pyprocar/core/atomic_orbital_index.py:709`).

- The fixture ships `dos.pkl`, `structure.pkl` and PDFs from old runs. Ignore them; they are not evidence.

## Expected step status

`$H compare <run-dir>` checks a run of this driver against this table (see SKILL.md, Drive).

| Driver | Step | Status | Site |
|---|---|---|---|
| `dos.py` | `obj_total` | ok | |
| `dos.py` | `obj_total_colored_by_projection` | ok | |
| `dos.py` | `obj_vertical_projected_legend` | ok | |
| `dos.py` | `legacy_dosplot_plain` | ok | |
| `dos.py` | `legacy_dosplot_parametric` | ok | |
| `dos.py` | `legacy_dosplot_parametric_line` | ok | |
| `dos.py` | `legacy_dosplot_stack_species` | ok | |
| `dos.py` | `legacy_dosplot_stack_orbitals` | ok | |
| `dos.py` | `legacy_dosplot_stack_no_items` | known-defect | `IndexError` at `pyprocar/scripts/scriptDosplot.py:427` |
| `dos.py` | `legacy_dosplot_overlay_items` | ok | |
| `dos.py` | `legacy_dosplot_overlay_species` | ok | |
| `dos.py` | `legacy_dosplot_overlay_orbitals` | ok | |
| `dos.py` | `side_effects.txt` | ok | |
