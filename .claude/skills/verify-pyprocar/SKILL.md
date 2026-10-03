---
name: verify-pyprocar
description: Prove a pyprocar change works by running the library the way a user does — load a real DFT calculation directory (VASP fixtures from the HF test dataset), plot bands / DOS / Fermi surfaces headlessly, and capture PNGs, summaries, logs and side effects. Use after changing parsers (pyprocar/io), core objects (pyprocar/core) or plotters (pyprocar/plotter), or whenever asked to verify/screenshot pyprocar output.
---

# verify-pyprocar

pyprocar is a Python library, not a server. The user surface is Python calls in a notebook or script:

- **Object API (current, dos-rewrite):** `ElectronicBandStructurePath/Mesh.from_code`, `DensityOfStates.from_code`, `FermiSurface.from_code` → `BandStructurePlotter` / `DOSPlotter` / `FermiPlotter` / `FermiSlicePlotter`. See `examples/general/new_*_examples.py`.
- **Legacy one-call functions:** `pyprocar.bandsplot`, `bandsdosplot`, `dosplot`, `fermi2D`, `FermiHandler`, `BandStructure2DHandler`, which the `examples/*/*.ipynb` notebooks use. They are mid-migration on this branch, and most of them crash at 1f36aae1. The per-feature status is in `features/README.md`.
- **File utilities:** `pyprocar.bandgap`, `kpath`, `filter`, `repair`, `cat`, `generate2dkmesh`. These read and write VASP files instead of plotting; see `features/utilities.md`.

"Driving" means: write a small driver script that makes those calls on a real calc dir, run it through the harness, and read the evidence.

All commands run from the repo root. `H=.claude/skills/verify-pyprocar/scripts/verify.sh`.

## Launch

There is nothing to keep alive. "Launch" means the env and fixtures exist:

```bash
$H fetch data/examples/bands/non-spin-polarized     # idempotent; ~20-35 MB each, into ./data (gitignored)
```

Fixture relpaths (HF dataset `lllangWV/pyprocar_test_data`): `data/examples/{bands,dos,fermi3d,fermi2d}/{non-spin-polarized,spin-polarized,non-colinear}`, plus `bands/{atomic_levels,auto,compare_bands,ipr,unfolding,2d-bands}`, `fermi2d/bisb_monolayer`, `fermi3d/van-alphen`. All are VASP; Fermi energy for the SrVO3 sets is `5.3017`.

Run the gates with the `pixi run --locked ...` commands in `.github/workflows/ci.yml`, as written. Replace `"$BASE"` with `origin/dev`. `pixi run -e dev typecheck` is the same type check. The `test` task also runs the tests marked `data`, and the `lint` task applies `ruff --fix` to every file, so neither is a CI gate.

`--locked` fails when `pixi.lock` does not match `pixi.toml`, in CI and locally. After you edit `pixi.toml`, run `pixi lock` and commit `pixi.lock` with it.

Never call an env binary such as `.pixi/envs/dev/bin/basedpyright` directly. Without the env on `PATH` it reports phantom errors that `pixi run` and CI do not.

## Doctor

```bash
$H doctor
```

Read-only. It prints the branch/commit, the count of dirty tracked files, the python/pyprocar/pyvista/vtk versions, the import path (it must be this checkout's `pyprocar/`), and the fixtures present. It is worth driving when pyprocar imports from this repo and the fixture you need is listed. A `dirty` count you didn't cause means someone is mid-edit, so say so before trusting results.

## Drive

```bash
$H run <name> <fixture-relpath> <driver.py>
```

What the harness does:
1. Creates `data/verify-runs/<timestamp>-<name>/`.
2. Copies the fixture to `work/calc` and the driver to `evidence/driver.py`.
3. Runs the driver with `TMPDIR=work/tmp`, `CALC=<calc copy>`, `EVIDENCE=<evidence dir>`, `REPO=<repo root>`, `MPLBACKEND=Agg`, `PYVISTA_OFF_SCREEN=true`, and `scripts/lib` on `PYTHONPATH`.
4. Records the exit code, `run.log` (stdout+stderr), and `side_effects.txt`, which lists files created or modified inside the calc copy.

The harness exits with the driver's exit code. Runs are isolated per directory, so parallel runs are safe.

Several agents in separate worktrees may verify at once. Never symlink the shared `data/` into a worktree and drive the library on it directly. The library has written into its input directory (`ebs.pkl` caches, a merged Abinit `PROCAR`). Run every sweep through `$H run` so it works on the per-run `work/calc` copy. Keep temp files off the shared tmpfs `/tmp`; the harness does this, and outside it set `TMPDIR` (or pytest `--basetemp`) to a dir under your run.

A driver reads `CALC` and `EVIDENCE` from the environment, calls the public API, saves the figure into `EVIDENCE`, writes a `summary.json` of observable facts, and `assert`s the end state. The worked example `scripts/examples/bands_plain.py` is proven and passes:

```bash
$H run bands-plain data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/bands_plain.py
```

Per-feature recipes are in `features/`. Each feature has a ready driver at `scripts/examples/<feature>.py` that exercises every entry point the feature file lists. It uses `scripts/lib/verify_steps.py`:
- `@step(name)` runs one entry point in isolation and records `{ok, ...facts}` or `{ok: false, error, where}` into `summary.json`, so one crash doesn't hide the rest.
- `finish()` exits 1 if any step failed. On this branch the legacy steps fail, so read `summary.json` per step and compare it against the feature file's proven end state.

To drive the whole map:

```bash
E=.claude/skills/verify-pyprocar/scripts/examples
for f in bands:bands/non-spin-polarized dos:dos/non-spin-polarized fermi3d:fermi3d/non-spin-polarized \
         fermi2d:fermi2d/non-spin-polarized bs2d:bands/2d-bands parsers:bands/non-spin-polarized \
         utilities:bands/non-spin-polarized; do
  $H run ${f%%:*} data/examples/${f#*:} $E/${f%%:*}.py; done
```

Headless rules:
- Matplotlib: always `savefig` into `EVIDENCE`. Pass `show=False` to legacy functions.
- PyVista: `pv.Plotter(off_screen=True)` + `screenshot(EVIDENCE/...)`. It works on this machine. The `vtkEGLRenderWindow ... OpenGL 3.2` WARN line is noise. A blank screenshot has about 1 distinct color; `distinct_colors()` checks for that.

## Evidence

Evidence lives at `data/verify-runs/<run>/evidence/` and survives cleanup. That directory is gitignored via `/data`. A proof includes:
- **The rendered image.** Open it with the Read tool and look at it. Check for correct k-path labels, sensible energy window and non-empty curves. A blank axes still has a nonzero PNG size.
- **`summary.json` with numbers that tie the image to the data.** Examples: array shapes, number of plotted artists, axis limits, tick labels, and projection sums.
- **`side_effects.txt`.** `DensityOfStates.from_code` writes nothing, while legacy `dosplot` deletes and rewrites `dos.pkl` and `structure.pkl`. Bands and Fermi surfaces also list `ebs.pkl`, a known defect (see Known repo issues). Anything else written there is a finding.
- **Exit code and `run.log`**, including warnings.

Proof standards:
- Exercise the public entry point a user calls: `from_code` + plotter, or the legacy function. Never hand-build `Property` arrays or call parser internals to skip a broken step. If the user path crashes, that crash is the result.
- Capture before/after when verifying a change. Run the same driver on the base commit (`git stash` or a worktree) and on the change, and compare the summaries and images.
- `use_cache=True` loads the fixture's shipped `ebs.pkl`, which may be stale. Verify parser changes with `use_cache=False`, the default.
- No mocks. The fixtures are real VASP output.

## Cleanup

```bash
$H clean data/verify-runs/<run>
```

This removes only that run's `work/` scratch copy (including its `TMPDIR`) and keeps `evidence/`. It refuses paths outside `data/verify-runs/`. Clean every run you made before hand-back, and leave other agents' runs alone. There are no processes to kill. Fetched fixtures in `data/examples/` are shared cache; leave them.

## Helpers

- `scripts/verify.sh`: `doctor | fetch <relpath>... | run <name> <fixture> <driver.py> | clean <run-dir>`
- `scripts/examples/bands_plain.py`: the minimal single-call template, which exits 0. Copy it for a one-off driver.
- `scripts/examples/{bands,dos,fermi3d,fermi2d,bs2d,parsers,utilities}.py`: full per-feature drivers. Run them as shown under Drive.
- `scripts/lib/verify_steps.py`: `step`, `png`, `distinct_colors` and `finish` for multi-step drivers. It is importable because the harness puts `scripts/lib` on `PYTHONPATH`.

Known repo issues that affect verification (as of dos-rewrite @ 1f36aae1):
- `pyprocar.download_from_hf(relpath, output_path=".")` crashes when given a str; it needs a `Path`. The harness passes a `Path`.
- The EBS-based `from_code` calls and legacy `bandsplot` write `ebs.pkl` into the calc dir even with `use_cache=False`. This is a library defect, not expected behaviour; [PR #245](https://github.com/romerogroup/pyprocar/pull/245) fixes it. Once it lands, `ebs.pkl` in `side_effects.txt` without `use_cache=True` is a regression.

A PR that changes a documented side effect updates this skill in the same PR.
