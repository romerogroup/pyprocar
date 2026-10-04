# 2D band structure surface

E(k_x, k_y) surfaces for 2D materials (graphene Dirac cones, BiSb Rashba bands), rendered in 3D with PyVista.

## Sub-features

- Plain energy surfaces near E_F, with an optional Fermi plane.
- Scalar coloring, e.g. `bands` or `bands_speed`. Parametric projections and spin texture are not driven.
- Box and plane widgets: `BS2DPlotter(bs, off_screen=True).add_box_slicer(surface, normal=..., origin=..., cross_section_area=True)` on one of `bs.band_surfaces`. The surface needs active scalars. The slice is the `slice` actor and the area text is `p.actors["area_text"].GetText(2)`.

## How to get to it (user POV)

1. **Object API:** `BandStructure2D.from_code(code="vasp", dirpath=..., grid_interpolation=(nx, ny))` (`pyprocar/core/bandstructure2D.py`), then `BS2DPlotter(bs, off_screen=True).plot(scalars_data="bands", show_brillouin_zone=False)`, then `p.screenshot(...)`.
2. **Legacy:** `pyprocar.BandStructure2DHandler(code="vasp", dirname=..., fermi=...).plot_band_structure(mode="plain"|"parametric"|"spin_texture", show=False, render_offscreen=True, save_2d=...)`. See `examples/00-band_structure/07-Plotting 2D Band Structure.ipynb`.

## Driving it with verify.sh

```bash
$H run bs2d data/examples/bands/2d-bands .claude/skills/verify-pyprocar/scripts/examples/bs2d.py
```

The fixture dir holds `graphene/` (Fermi -0.795606) and `bisb_monolayer/`. The driver uses `graphene/`.

The proven end state (d6d4aaa7):
- `obj_surface`: `n_points` is 800 at a 20×20 grid, with about 8.9k distinct colors. The screenshot shows two separated sheets (π and π*) colored by energy from -6.5 to 4.9 eV. They are jagged at this coarse grid.
- `legacy_handler_plain` and `legacy_handler_plain_notebook_kwargs`: about 1.2k distinct colors each. The screenshot shows the two sheets inside the hexagonal BZ prism with an `E - E_F (eV)` axis.
- `side_effects.txt` is empty.

## Gotchas

- `BS2DPlotter.plot(show_brillouin_zone=True)` raised `PyVistaAttributeError: Attribute 'brillouin_zone' does not exist` at 1f36aae1. The legacy handler draws the zone at d6d4aaa7, but the driver passes `False`, so that call is not re-driven.
