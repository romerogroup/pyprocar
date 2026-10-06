# 2D band structure surface

E(k_x, k_y) surfaces for 2D materials (graphene Dirac cones, BiSb Rashba bands), rendered in 3D with PyVista.

## Sub-features

- Plain energy surfaces near E_F, with an optional Fermi plane.
- Scalar coloring, e.g. `bands` or `bands_speed`. Parametric projections and spin texture are not driven.
- Brillouin zone: `plot(show_brillouin_zone=True, clip_brillouin_zone=...)`, both on by default, draws the `BrillouinZone2D` prism.
- Box and plane widgets: `BS2DPlotter(bs, off_screen=True).add_box_slicer(surface, normal=..., origin=..., cross_section_area=True)` (not `show_cross_section_area`, as in `FermiPlotter`) or `add_slicer(...)` on one of `bs.band_surfaces`. The surface needs active scalars (see Gotchas). The slice is the `slice` actor and the area text is `p.actors["area_text"].GetText(2)`. Not driven.

## How to get to it (user POV)

1. **Object API:** `BandStructure2D.from_code(code="vasp", dirpath=..., grid_interpolation=(nx, ny))` (`pyprocar/core/bandstructure2D.py`), then `BS2DPlotter(bs, off_screen=True).plot(scalars_data="bands", show_brillouin_zone=False)`, then `p.screenshot(...)`.
2. **Legacy:** `pyprocar.BandStructure2DHandler(code="vasp", dirname=..., fermi=...).plot_band_structure(mode="plain"|"parametric"|"spin_texture", show=False, render_offscreen=True, save_2d=...)`. `mode` is not validated: any value other than `parametric` and `spin_texture` draws plain surfaces. The notebook's `energy_lim=` is accepted and ignored; only the config field at `pyprocar/cfg/band_structure_2d.py:240` names it (source only). See `examples/00-band_structure/07-Plotting 2D Band Structure.ipynb`.

## Driving it with verify.sh

```bash
$H run bs2d data/examples/bands/2d-bands .claude/skills/verify-pyprocar/scripts/examples/bs2d.py
```

The fixture dir holds `graphene/` (Fermi -0.795606) and `bisb_monolayer/`. The driver uses `graphene/`.

The proven end state (c13166ce):
- `obj_surface`: `n_points` is 800 at a 20×20 grid, with about 11.2k distinct colors. The screenshot shows two separated sheets (π and π*) colored by energy from -6.5 to 4.9 eV.
- Grid coverage (#285, `tests/pyprocar/core/test_bandstructure2d_grid.py`): in both modes (`as_cartesian=False`, which `from_code` uses, and `True`, the `from_ebs` default) the uv grid is regular in the two fractional coordinates that span the plane, pulled in from the edges by 1e-12 of the span, so no point falls outside graphene's rhombic k-patch and no BiSb grid corner is lost to rounding. With `as_cartesian=True` the grid used to be the Cartesian bounding box: 440 of 800 graphene points and 1056 of 2400 BiSb points NaN at 20×20; now none. Before #285, 440 of 800 points (20×20) and 8280 of 16200 (90×90) were NaN, and the apparent Dirac gap was 1.22 and 0.267 eV. Now no point is NaN, and at 90×90, where every vertex is a DFT k-point, the gap is 2.511e-5 eV, the EIGENVAL value at K. Each band value stays on its own point: `bs.get_property('bands').value` equals `bs.points[:, 2]`. The ripples left on the upper sheet are real crossings with the σ and 2.2 eV bands, because bands are indexed by sorted energy.
- `obj_surface_with_bz`: the actors are `surface_0_0`, `surface_1_0`, a `BrillouinZone2D` and the scalar bar; the screenshot shows the hexagonal prism wireframe around the two sheets.
- `legacy_handler_plain` and `legacy_handler_plain_notebook_kwargs`: about 1.4k and 1.5k distinct colors. The screenshot shows the two sheets inside the hexagonal BZ prism with an `E - E_F (eV)` axis.
- `side_effects.txt` is empty.

## Gotchas

- `add_box_slicer` needs a surface with active scalars: `bs.band_surfaces` carry no point data, so set one first, for example `s["energy"] = s.points[:, 2]`.

## Expected step status

`$H compare <run-dir>` checks a run of this driver against this table (see SKILL.md, Drive).

| Driver | Step | Status | Site |
|---|---|---|---|
| `bs2d.py` | `obj_surface` | ok | |
| `bs2d.py` | `obj_surface_with_bz` | ok | |
| `bs2d.py` | `legacy_handler_plain` | ok | |
| `bs2d.py` | `legacy_handler_plain_notebook_kwargs` | ok | |
| `bs2d.py` | `side_effects.txt` | ok | |
