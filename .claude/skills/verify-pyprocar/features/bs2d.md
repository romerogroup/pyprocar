# 2D band structure surface

E(k_x, k_y) surfaces for 2D materials (graphene Dirac cones, BiSb Rashba bands), rendered in 3D with PyVista.

## Sub-features

- Plain energy surfaces near E_F, with an optional Fermi plane.
- Scalar coloring, e.g. `bands` or `bands_speed`. Parametric projections and spin texture are not driven.

## How to get to it (user POV)

1. **Object API:** `BandStructure2D.from_code(code="vasp", dirpath=..., grid_interpolation=(nx, ny))` (`pyprocar/core/bandstructure2D.py`), then `BS2DPlotter(bs, off_screen=True).plot(scalars_data="bands", show_brillouin_zone=False)`, then `p.screenshot(...)`.
2. **Legacy:** `pyprocar.BandStructure2DHandler(code="vasp", dirname=..., fermi=...).plot_band_structure(mode="plain"|"parametric"|"spin_texture", show=False, render_offscreen=True, save_2d=...)`. See `examples/00-band_structure/07-Plotting 2D Band Structure.ipynb`.

## Driving it with verify.sh

```bash
$H run bs2d data/examples/bands/2d-bands .claude/skills/verify-pyprocar/scripts/examples/bs2d.py
```

The fixture dir holds `graphene/` (Fermi -0.795606) and `bisb_monolayer/`. The driver uses `graphene/`.

The proven end state (1f36aae1):
- `obj_surface`: `n_points` is 800 at a 20×20 grid, with about 900 distinct colors. The screenshot shows two separated sheets (π and π*). They are jagged at this coarse grid.
- `side_effects.txt` lists `./graphene/ebs.pkl`.

## Gotchas

- **Legacy `BandStructure2DHandler.plot_band_structure` crashes** at 1f36aae1. This is a product gap.
  - With plain kwargs it raises `PyVistaAttributeError: Attribute 'brillouin_zone' does not exist ... 'BS2DPlotter'` at `plotter/bs_2d_plot.py:364`.
  - With the notebook's kwargs (`add_fermi_plane`, `fermi_plane_size`, `energy_lim`) it raises `TypeError: Plotter.__init__() got an unexpected keyword argument` at `bs_2d_plot.py:76`.
- `BS2DPlotter.plot(show_brillouin_zone=True)` hits the same `brillouin_zone` attribute error. Pass `False`.
