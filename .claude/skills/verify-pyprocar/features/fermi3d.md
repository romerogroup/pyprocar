# 3D Fermi surface

Build isosurfaces of E(k)=E_F on a full k-mesh inside the Brillouin zone. They can be colored by projections, spin texture or band velocity.

## Sub-features

- Plain surface with the BZ: `FermiPlotter.plot(fs, show_brillouin_zone=True)`. It returns a dict of meshes keyed by `(band, spin)`.
- Scalar coloring: compute the scalar first with `fs.get_property("projected_sum", atoms=, orbitals=, spins=)`, then pass `scalars_data="projected_sum"`, or pass a `Property`.
- Vector arrows (spin texture): `vectors_data=...`. Use the `non-colinear` fixture. Not driven.
- Band selection: `fs.select_bands([(band, spin), ...])` returns a new surface with only those sheets. On `fermi3d/non-spin-polarized` at E_F = 5.3017, `select_bands([(17, 0)])` keeps the 270 points of band 17, and its V d `projected_sum` spans about 0.794 to 0.803. `select_bands([])` gives an empty surface whose projections have shape `(0, n_bands, n_spins)`.
- Box and plane widgets: `FermiHandler.plot_fermi_cross_section_box_widget(mode=..., show=False, off_screen=True)`, or `FermiPlotter.add_box_slicer(fs, show_cross_section_area=True)`. The cross-section text is the `area_text` actor; read it with `p.actors["area_text"].GetText(2)`. `save_2d=` writes a screenshot of the 3D view (the handler then renders off screen and does not show), and `save_2d_slice=` writes the cross section as a matplotlib plot through `FermiPlotter.save_slice_2d`.
- Isovalue GIF: `add_isovalue_gif(e_surfaces, save_gif, ...)`. Not driven.
- de Haas–van Alphen: `van-alphen` fixture. Not driven.

## How to get to it (user POV)

1. **Object API:** `FermiSurface.from_code(code="vasp", dirpath=..., fermi=5.3017)`, then `FermiPlotter(off_screen=True).plot(fs)`, then `p.screenshot(...)` or `p.savefig(...)`.
2. **Legacy:** `pyprocar.FermiHandler(code="vasp", dirname=..., fermi=5.3017, use_cache=False).plot_fermi_surface(mode=..., show=False, save_2d=..., off_screen=True)`. See `examples/04-fermi3d/*.ipynb`. `save_gif`, `save_mp4` and `save_3d` only log "not yet implemented".

## Driving it with verify.sh

```bash
$H run fermi3d data/examples/fermi3d/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/fermi3d.py
```

The proven end state (SrVO3, non-spin-polarized, 1f36aae1):
- `obj_surface_with_bz` reports `n_points` 1908, `n_cells` 3576, `band_indices` `[[16,0],[17,0],[18,0]]`, 3 meshes and about 435 distinct colors. The screenshot shows closed cylinder-like sheets along the axes, inside the cubic BZ wireframe.
- `obj_scalars_projected_sum`: about 1.4k distinct colors.
- `legacy_handler_parametric` and `legacy_handler_fermi_speed`: more than 9k distinct colors each.
- `side_effects.txt` lists `./ebs.pkl`.

## Gotchas

- **Legacy `FermiHandler.plot_fermi_surface(mode="plain")` crashes** at 1f36aae1, with `KeyError: 'Data array (spin_band_index) not present'` at `plotter/fs_plot.py:382`, reached from `scriptFermiHandler.py:260`. This is a product gap. The parametric and fermi_speed modes work.
- `FermiHandler` re-parses from `dirname` on every `plot_fermi_surface` call, so the calc dir must still exist at plot time.
- Offscreen VTK works here, but it prints a `vtkEGLRenderWindow ... OpenGL 3.2` WARN line. Ignore it. Check the screenshot is not blank: about 1 distinct color means blank.
- `FermiPlotter` subclasses `pv.Plotter`, so pass `off_screen=True` to it. `PYVISTA_OFF_SCREEN=true` is set by the harness as a backstop.
