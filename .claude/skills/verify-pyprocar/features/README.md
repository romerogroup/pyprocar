# pyprocar feature map

One file per user-facing feature. Each lists every entry point a user can reach it by; a proof that covers one entry point is incomplete when the file lists others.

Each feature has a reusable driver in `scripts/examples/<feature>.py`. It runs every entry point as an isolated step, records each step's result or crash in `summary.json`, and exits 1 if any step failed. Each feature file ends with an `## Expected step status` table, one row per driver step: `ok`, or `known-defect` with the error type and crash site that `verify_steps` records as `where`. A `side_effects.txt` row lists the files the run must leave in the calc copy. `$H compare <run-dir>...` checks runs against these tables and exits 1 on any deviation, so read its output instead of the driver's exit code. When a deviation is a real change, update the row and the prose in the same edit.

| Feature | File | Driver / fixture | Status (dev @ c13166ce, re-driven 2026-10-05) |
|---|---|---|---|
| Band structure along a k-path | [bands.md](bands.md) | `bands.py` / `data/examples/bands/non-spin-polarized` | Object API plain/parametric/scatter/`plot_quiver`/flip/`plot_overlay` with an array and legacy `bandsplot` plain/parametric/scatter/overlay_species/overlay_orbitals and `bandsdosplot` work; `plot(vectors_mode=)` and `plot_overlay`/`plot_quiver` given `ebs.bands` crash |
| Density of states | [dos.md](dos.md) | `dos.py` / `data/examples/dos/non-spin-polarized` | Object API and every legacy `dosplot` mode work; `stack` without `items` crashes |
| 3D Fermi surface | [fermi3d.md](fermi3d.md) | `fermi3d.py` / `data/examples/fermi3d/non-spin-polarized` | Object API and legacy `FermiHandler` plain/parametric/fermi_speed, `save_3d` and the box widget work |
| 2D Fermi slice | [fermi2d.md](fermi2d.md) | `fermi2d.py` / `data/examples/fermi2d/non-spin-polarized` | Legacy `fermi2D` plain/plain_bands/parametric and `FermiSlicePlotter` with and without scalars work; `plot(scalars_data=<Property>)` draws uncolored lines |
| 2D band structure surface | [bs2d.md](bs2d.md) | `bs2d.py` / `data/examples/bands/2d-bands` | Object API with and without the zone and legacy `BandStructure2DHandler` work |
| Code parsers (VASP, QE, Elk, Abinit, Siesta, Lobster, BXSF, FRMSF) | [parsers.md](parsers.md) | `parsers.py` / `data/examples/bands/non-spin-polarized` + `data/codes` | VASP, QE (collinear and non-collinear), Elk and Abinit bands work; Siesta, Lobster, BXSF and FRMSF have no fixture |
| File utilities, test-data download and the `procar.py` CLI | [utilities.md](utilities.md) | `utilities.py` / `data/examples/bands/non-spin-polarized` | `bandgap`/`kpath`/`filter`/`repair`/`cat`/`generate2dkmesh` and `procar.py cat`/`filter` work; `spin_asymmetry` is a stub; `download_from_hf` with a str path and `procar.py bandgap`/`generate2dkmesh` crash |

Update the Status column whenever a run changes what is known.

## Entry points outside the map

`import pyprocar` and `pyproject.toml` expose these too. No feature file covers them:
- `pyprocar.pyposcar` (POSCAR reading, defects, clusters, RDF) is in `docs/source/api/pyposcar/`, with fixtures in `data/examples/pyposcar/`. It needs `import pyprocar.pyposcar`; `import pyprocar` does not load it. Not driven.
- `calculate_band_velocity`, `calculate_band_speed` and `calculate_avg_inv_effective_mass` are what `get_property("bands_velocity" | "bands_speed" | "avg_inv_effective_mass")` calls, which `bands.md`, `bs2d.md` and `fermi3d.md` drive.
- `welcome` prints the banner that the legacy functions print. `Settings` has no caller.
- `scripts/poscar.py`, `scripts/dftb+2procar.py` and `scripts/tmp.py` are uninstalled developer scripts.

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
