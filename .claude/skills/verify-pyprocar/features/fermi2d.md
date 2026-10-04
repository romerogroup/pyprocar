# 2D Fermi slice

A constant-energy contour on one k-plane (e.g. k_z=0), with optional projection coloring or spin-texture arrows.

## Sub-features

- Plain contours at `energy=0.0` relative to E_F.
- Parametric coloring by `atoms` and `orbitals`.
- Spin texture (`non-colinear` fixture) and Rashba splitting (`bisb_monolayer` fixture). Not driven.
- Spin unfolding reference: `scripts/lib/references/spin_ibz.py` reduces the full 60x60x1 SOC mesh to its orbits under P3m1 and time reversal with spglib and numpy only. Compare it with `ibz2fbz`, which the public `ElectronicBandStructureMesh` constructor runs on the IBZ rows:
  ```bash
  $H run ref-spin-ibz data/examples/fermi2d/bisb_monolayer .claude/skills/verify-pyprocar/scripts/examples/ref_spin_ibz.py
  ```
  - The structure gets the reference's k-space operations through `Structure(..., rotations=...)`.
  - `validate()` rebuilds a periodic C3v Rashba texture on a 12x12x1 grid from its 19 orbits to 3e-15. It rejects a polar spin (error 3.5 on the mirror images) and a time reversal that keeps the spin.
  - At 5884b801 the mesh has 331 orbits. The unfolded spin (bands 0-59) equals the reference to 2.8e-6 on generic orbits.
  - Against the full mesh, the mean error on non-degenerate entries is 0.0032, and at most 0.011 for bands more than 50 meV apart, inside the 0.031 PROCAR rounding bound. A polar spin raises the mirror-image error to 0.42.
  - The bands are copied unchanged. Since #299 channel 0 is permuted over atoms and rotated over orbitals, and its total per band equals the IBZ source to 2e-15.
- Line and arrow styles: `plot_line_kwargs` goes to the `LineCollection` and `plot_arrows_kwargs` to `quiver`. Either can carry `cmap`, which overrides the `cmap` argument for that artist.
- Symmetry unfolding: every fixture here is an irreducible wedge (`fermi2d/non-spin-polarized` holds 496 of the 61x61x1 points), and `ElectronicBandStructureMesh` rebuilds the full grid with `ibz2fbz`. Since #283 the images carry permuted atoms and rotated orbitals. Check it with the cubic relation P_dyz(C4 k) = P_dxz(k) on V for bands at least 10 meV from their neighbours: it holds to 1e-15 (dev: up to 0.92). A slice colored by V d_xz then lights only the two sheets at k_x = ±0.037 1/Å, the d_xz band that disperses along k_x alone; before #283 it lit the square around Γ instead. QE meshes (`data/codes/qe/7.2/SrVO3/*/fermi`) come back with `projected_phase` None and two `UserWarning`s, the dropped phases and the unrotated `orbital <i>` columns, because QE orbitals carry no names until #293. A group with QE time-reversal flags warns if plain time reversal fills grid points; on the SrVO3 spin-orbit fixture its 16 operations reach every point, so it stays silent.
- Collinear spin selection: in `fermi2d/spin-polarized` and `fermi3d/spin-polarized` no spin-down band crosses E_F = 5.3017 (spin-down gap about 3.3 to 6.5 eV), so every sheet is spin up. `spins=[0]` matches no selection and `spins=[1]` is all zeros. That is correct, not a bug.

## How to get to it (user POV)

1. **Legacy one-call:** `pyprocar.fermi2D(code="vasp", dirname=..., mode="plain"|"plain_bands"|"parametric"|"spin_texture", fermi=5.3017, energy=0.0, k_z_plane=0.0, show=False, savefig=...)`. It returns `(fig, ax)`. See `examples/02-fermi2d/*.ipynb`.
2. **Object API:** `FermiSlicePlotter(fs, normal=(0,0,1), origin=(0,0,0))` in `pyprocar/plotter/fs_slice_plot.py`. It takes the full, unsliced `FermiSurface.from_code(...)` and slices it internally. `plot(...)` returns a dict of artists. `plot_lines`, `plot_points` and `plot_arrows` are thin wrappers over it, and `p.fig`/`p.ax` hold the figure.

## Driving it with verify.sh

```bash
$H run fermi2d data/examples/fermi2d/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/fermi2d.py
```

The driver exits 1 at d6d4aaa7 because of the `scalars_data` gap below. The proven end state (SrVO3, non-spin-polarized, d6d4aaa7):
- `obj_slice_no_scalars`: one uncolored LineCollection. The PNG shows a closed rounded square around Γ inside a second closed curve, plus four open sheets.
- `obj_slice_projected`: after `prop = fs.get_property("projected_sum", atoms=[1], orbitals=[4..8], spins=[0])` and `fs.set_values("projected_sum", prop.value)`, the call `plot(scalars_name="projected_sum")` draws one LineCollection with 340 segments, colored from 0.79 to 0.84 and with a colorbar. Without `set_values` the name is not on the surface, so the plotter warns and draws uncolored lines.
- `legacy_fermi2D_str_plain`, `legacy_fermi2D_enum_plain` and `legacy_fermi2D_str_parametric`: one LineCollection each. The parametric PNG matches `obj_slice_projected`.
- `side_effects.txt` is empty.

## Gotchas

- **`FermiSlicePlotter.plot(scalars_data=<Property>)` draws uncolored lines** at d6d4aaa7. It looks the Property up by its name in the surface's point data, finds nothing, warns `Scalars name projected_sum not found in slice data`, and falls back. This is a product gap. Attach the values with `fs.set_values` and pass `scalars_name`, as `scripts/scriptFermi2D.py` does.
- `examples/general/new_ebs_examples.py` builds `FermiSlicePlotter` without a surface and calls a nonexistent `.scatter`. The example is stale; don't copy it.
