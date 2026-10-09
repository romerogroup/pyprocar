# 3D Fermi surface

Build isosurfaces of E(k)=E_F on a full k-mesh inside the Brillouin zone. They can be colored by projections, spin texture or band velocity.

## Sub-features

- Plain surface with the BZ: `FermiPlotter.plot(fs, show_brillouin_zone=True)`. It returns a dict of meshes keyed by `(band, spin)`.
- Scalar coloring: compute the scalar first with `fs.get_property("projected_sum", atoms=, orbitals=, spins=)`, then pass `scalars_data="projected_sum"`, or pass a `Property`.
- Vector arrows (spin texture): `vectors_data=...`. Use the `non-colinear` fixture. `vectors_data="fermi_velocity"` was driven on `van-alphen` (#286): the longest arrow is 0.1 times the zone's largest extent, 0.0509 of 0.5086 1/A. The arrow actors are named `vectors_<band>_<spin>`.
- Band selection: `fs.select_bands([(band, spin), ...])` returns a new surface with only those sheets. On `fermi3d/non-spin-polarized` at E_F = 5.3017, `select_bands([(17, 0)])` keeps the 270 points of band 17, and its V d `projected_sum` spans about 0.794 to 0.803. `select_bands([])` gives an empty surface whose projections have shape `(0, n_bands, n_spins)`.
- Box and plane widgets: `FermiHandler.plot_fermi_cross_section_box_widget(mode=..., show=False, off_screen=True)`, or `FermiPlotter.add_box_slicer(fs, show_cross_section_area=True)`. The cross-section text is the `area_text` actor; read it with `p.actors["area_text"].GetText(2)`. `save_2d=` writes a screenshot of the 3D view (the handler then renders off screen and does not show), and `save_2d_slice=` writes the cross section as a matplotlib plot through `FermiPlotter.save_slice_2d`.
- Isovalue GIF: `add_isovalue_gif(e_surfaces, save_gif, ...)`. Not driven.
- de Haas–van Alphen: `van-alphen` fixture (fcc Au, band 5 crosses E_F = 8.5642 from OUTCAR). Compare the `area_text` of `add_box_slicer(fs, show_cross_section_area=True)` and `add_slicer(fs, show_van_alphen_frequency=True)` with the reference `scripts/lib/references/orbits.py` (numpy, scipy and pyvista only):
  ```bash
  $H run ref-orbits data/examples/fermi3d/van-alphen .claude/skills/verify-pyprocar/scripts/examples/ref_orbits.py
  ```
  - The reference unfolds EIGENVAL with the 48 lattice operations, marches one period, slices its translates within 8 or 16 cells, and counts each closed loop through the first zone once per lattice translation.
  - `validate()` reproduces pi r^2 / cos t for tilted cylinder cuts (one ellipse 3.7 cells long) and pi (r^2 - d^2) for spheres on an fcc lattice within 1.0% at N = 32. One sphere cut lies near the X face of the fcc zone, outside the unit cube, so it fails a zone test that uses fractional G. It rejects the untiled single cell.
  - At 5884b801 all 25 cuts agree, with areas equal to the 4 printed decimals and frequencies within 1.5e-6 relative. They include the belly 4.1375 Ang^-2 on [111] through Γ, the neck 0.1025 Ang^-2 on [111] through L, the dog's bone 1.7003 Ang^-2 on [110] (4 translates counted once), and one random cut with 2 open curves.
  - A normal built exactly from the POSCAR lattice also prints `(normal snapped to [u v w])`, because pyprocar's reciprocal lattice is 4.6e-10 off POSCAR's. The areas are unaffected.
  - A normal within 3e-4 rad of a direction with given-basis indices up to 4 (the rule before #302), or of a primitive real-space lattice vector no longer than 4 times the sum of the lattice's successive minima, snaps to it, and the note gives [u v w] in the given basis. `ref_orbits.py` builds both sets by brute force in the given basis, without spglib or pyprocar.

## How to get to it (user POV)

1. **Object API:** `FermiSurface.from_code(code="vasp", dirpath=..., fermi=5.3017)`, then `FermiPlotter(off_screen=True).plot(fs)`, then `p.screenshot(...)` or `p.savefig(...)`.
2. **Legacy:** `pyprocar.FermiHandler(code="vasp", dirname=..., fermi=5.3017, use_cache=False).plot_fermi_surface(mode=..., show=False, save_2d=..., off_screen=True)`. See `examples/04-fermi3d/*.ipynb`. `save_gif`, `save_mp4` and `save_3d` only log "not yet implemented".

## Driving it with verify.sh

```bash
$H run fermi3d data/examples/fermi3d/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/fermi3d.py
```

The proven end state (SrVO3, non-spin-polarized, d6d4aaa7):
- `obj_surface_with_bz` reports `n_points` 1908, `n_cells` 3576, `band_indices` `[[16,0],[17,0],[18,0]]`, 3 meshes and about 430 distinct colors. The screenshot shows closed cylinder-like sheets along the axes, inside the cubic BZ wireframe.
- `obj_scalars_projected_sum`: about 16.6k distinct colors.
- `legacy_handler_plain`: about 290 distinct colors, a single-color set of cylinders along the axes inside the BZ. `legacy_handler_parametric` and `legacy_handler_fermi_speed`: more than 16k distinct colors each.
- `side_effects.txt` is empty.
- Peak memory (#305): `ebs.pad` drops `projected_phase`, which nothing reads on a padded mesh. `FermiSurface.from_code("qe", data/codes/qe/7.2/SrVO3/spin-polarized-colinear/fermi)` peaks at 3.8 GiB (`ru_maxrss`); padding the kept QE phases took it to 9.2 GiB.

## Gotchas

- `FermiHandler` re-parses from `dirname` on every `plot_fermi_surface` call, so the calc dir must still exist at plot time.
- Since #286 the drawn zone (`BrillouinZone`, `BrillouinZone2D`) and the cross-section zone come from the Delaunay-reduced basis. A sheared reciprocal basis of the same lattice, such as b2' = b2 + 3 b1, gives the zone of volume |det B|, and `BrillouinZone.points` holds only the zone's vertices. Since #302 the Fermi surface, its cross sections and `BandStructure2D` re-index the k-grid into a reduced basis of the k-point lattice (`pyprocar/core/_periodic_grid.py`) and draw a box that covers the zone, so a sheared basis gives the reduced cell's surface and orbits. The sphere |k|^2=0.1 on b2=(3,1,0) at 16x48x16 draws 1.2424 (0.8273 before), Au with b2 + 3 b1 cuts within 3e-17 of the reduced build, and orbit reach counts reduced cells. A basis already reduced within 1e-4 takes the old pad path bit for bit. A band whose surface lies only outside the zone is dropped with a `warn_user` naming it, instead of raising `Surface is empty after clipping`. A k-grid whose period in a reduced basis exceeds 16 source grids (`CUT_TILE_BUDGET`) gets unjoined slice loops and a note.
- Slice normals snap the same way on any basis and orientation of the lattice, because the candidates are fixed by the lattice alone: on the cubic lattice given with a3' = a3 + 6 a1, a (111) normal typed to 4 digits snaps to [-5 1 1] and the cut through Gamma counts the orbit around M once. Every direction with given indices up to 4 still snaps, also on a cell that is not reduced: on the Bi2Se3 rhombohedral primitive cell (alpha = 24 degrees) a normal typed near hexagonal [1 0 3] snaps to [4 2 3] and counts the orbit through Gamma once. On the fcc primitive basis a normal typed near cubic [113] snaps to [3 3 -1] and counts the orbit through Gamma once (4.137305), where indices up to 4 in spglib's Delaunay-reduced basis counted it twice (#302).
- Offscreen VTK works here, but it prints a `vtkEGLRenderWindow ... OpenGL 3.2` WARN line. Ignore it. Check the screenshot is not blank: about 1 distinct color means blank.
- `FermiPlotter` subclasses `pv.Plotter`, so pass `off_screen=True` to it. `PYVISTA_OFF_SCREEN=true` is set by the harness as a backstop.
