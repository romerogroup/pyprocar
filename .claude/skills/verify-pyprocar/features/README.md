# pyprocar feature map

One file per user-facing feature. Each lists every entry point a user can reach it by; a proof that covers one entry point is incomplete when the file lists others.

Each feature has a reusable driver in `scripts/examples/<feature>.py`. It runs every entry point as an isolated step, records each step's result or crash in `summary.json`, and exits 1 if any step failed. Read `summary.json` per step instead of trusting the exit code alone.

| Feature | File | Driver / fixture | Status (dev @ d6d4aaa7, re-driven 2026-10-04) |
|---|---|---|---|
| Band structure along a k-path | [bands.md](bands.md) | `bands.py` / `data/examples/bands/non-spin-polarized` | Object API plain/parametric/scatter/`plot_quiver`/flip and legacy `bandsplot` plain/parametric/scatter/overlay_species and `bandsdosplot` work; `plot(vectors_mode=)` and `plot_overlay` crash |
| Density of states | [dos.md](dos.md) | `dos.py` / `data/examples/dos/non-spin-polarized` | Object API and legacy `dosplot` plain/parametric work; other `dosplot` modes not driven |
| 3D Fermi surface | [fermi3d.md](fermi3d.md) | `fermi3d.py` / `data/examples/fermi3d/non-spin-polarized` | Object API and legacy `FermiHandler` plain/parametric/fermi_speed work |
| 2D Fermi slice | [fermi2d.md](fermi2d.md) | `fermi2d.py` / `data/examples/fermi2d/non-spin-polarized` | Legacy `fermi2D` plain/parametric and `FermiSlicePlotter` with and without scalars work; `plot(scalars_data=<Property>)` draws uncolored lines |
| 2D band structure surface | [bs2d.md](bs2d.md) | `bs2d.py` / `data/examples/bands/2d-bands` | Object API and legacy `BandStructure2DHandler` work |
| Code parsers (VASP, QE, Elk, Abinit, Siesta, Lobster, BXSF, FRMSF) | [parsers.md](parsers.md) | `parsers.py` / `data/examples/bands/non-spin-polarized` + `data/codes/qe` | VASP, QE and Elk bands proven; others not driven |
| File utilities (`bandgap`, `kpath`, `filter`, `repair`, `cat`, `generate2dkmesh`) | [utilities.md](utilities.md) | `utilities.py` / `data/examples/bands/non-spin-polarized` | All six work |

Update the Status column whenever a run changes what is known.

## Independent references

A reference computes an expected value without pyprocar, so a verifier compares the library against it instead of writing a new one. Each lives in `scripts/lib/references/`, imports only numpy, scipy, pyvista or spglib, and has a `validate()` that checks it on an analytic case and rejects a wrong-convention control. Its driver runs `validate()` as the step `analytic_validation` before it touches the fixture. Run all validations without a fixture:

```bash
$H exec python .claude/skills/verify-pyprocar/scripts/lib/references/check.py
```

| Reference | Driver / fixture | Checks | Feature file |
|---|---|---|---|
| `unfold.py`: Popescu-Zunger weights from the raw PROCAR phases and POSCAR | `ref_unfold.py` / `data/examples/bands/unfolding` | `ebs.unfold` weights | [bands.md](bands.md) |
| `spin_ibz.py`: symmetry-reduced non-collinear mesh (spglib operations plus time reversal) and the axial spin transform of each unfolded k | `ref_spin_ibz.py` / `data/examples/fermi2d/bisb_monolayer` | spin under `ibz2fbz` | [fermi2d.md](fermi2d.md) |
| `orbits.py`: closed orbits of a plane through a 16-cell tiled marching-cubes surface | `ref_orbits.py` / `data/examples/fermi3d/van-alphen` | cross-section area and van Alphen frequency text | [fermi3d.md](fermi3d.md) |

A reference whose `validate()` fails is not evidence. Fix the reference before you compare against it.
