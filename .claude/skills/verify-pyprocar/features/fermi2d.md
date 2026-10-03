# 2D Fermi slice

A constant-energy contour on one k-plane (e.g. k_z=0), with optional projection coloring or spin-texture arrows.

## Sub-features

- Plain contours at `energy=0.0` relative to E_F.
- Parametric coloring by `atoms` and `orbitals`.
- Spin texture (`non-colinear` fixture) and Rashba splitting (`bisb_monolayer` fixture). Not driven.
- Line and arrow styles: `plot_line_kwargs` goes to the `LineCollection` and `plot_arrows_kwargs` to `quiver`. Either can carry `cmap`, which overrides the `cmap` argument for that artist.
- Collinear spin selection: in `fermi2d/spin-polarized` and `fermi3d/spin-polarized` no spin-down band crosses E_F = 5.3017 (spin-down gap about 3.3 to 6.5 eV), so every sheet is spin up. `spins=[0]` matches no selection and `spins=[1]` is all zeros. That is correct, not a bug.

## How to get to it (user POV)

1. **Legacy one-call:** `pyprocar.fermi2D(code="vasp", dirname=..., mode="plain"|"plain_bands"|"parametric"|"spin_texture", fermi=5.3017, energy=0.0, k_z_plane=0.0, show=False, savefig=...)`. It returns `(fig, ax)`. See `examples/02-fermi2d/*.ipynb`.
2. **Object API:** `FermiSlicePlotter(fs, normal=(0,0,1), origin=(0,0,0))` in `pyprocar/plotter/fs_slice_plot.py`. It takes the full, unsliced `FermiSurface.from_code(...)` and slices it internally. `plot(...)` returns a dict of artists. `plot_lines`, `plot_points` and `plot_arrows` are thin wrappers over it, and `p.fig`/`p.ax` hold the figure.

## Driving it with verify.sh

```bash
$H run fermi2d data/examples/fermi2d/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/fermi2d.py
```

The proven end state (SrVO3, non-spin-polarized, 1f36aae1):
- `obj_slice_projected`: after `fs.get_property("projected_sum", atoms=[1], orbitals=[4..8], spins=[0])`, the call `plot(scalars_name="projected_sum")` draws one LineCollection with 658 segments, colored and with a colorbar. The PNG shows a closed rounded square around Γ, plus four open sheets.

## Gotchas

These are product gaps at 1f36aae1. Record them; don't route around them.
- **Legacy `fermi2D` is unreachable in every mode.** A string `mode="plain"` raises `AttributeError: 'str' object has no attribute 'value'` at `scripts/scriptFermi2D.py:214`. The enum `Fermi2DMode.plain` gets past that line but raises `ValueError: Unknown mode` at `:264`.
- **`FermiSlicePlotter.plot()` with no scalars draws nothing.** It returns `{}`, and the axes are blank at about ±0.05. Plain contours need a scalar today.
- `examples/general/new_ebs_examples.py` builds `FermiSlicePlotter` without a surface and calls a nonexistent `.scatter`. The example is stale; don't copy it.
