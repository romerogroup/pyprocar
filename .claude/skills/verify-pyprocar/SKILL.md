---
name: verify-pyprocar
description: Prove a pyprocar change works by running the library the way a user does. Load a real DFT calculation directory (VASP, QE, Elk or Abinit fixtures), plot bands / DOS / Fermi surfaces headlessly, and capture PNGs, summaries, logs and side effects. Use after changing parsers (pyprocar/io), core objects (pyprocar/core) or plotters (pyprocar/plotter), whenever asked to verify/screenshot pyprocar output, or before testing or gating pyprocar from a git worktree.
---

# verify-pyprocar

pyprocar is a Python library, not a server. The user surface is Python calls in a notebook or script:

- **Object API (current):** `ElectronicBandStructurePath/Mesh.from_code`, `DensityOfStates.from_code`, `FermiSurface.from_code` → `BandStructurePlotter` / `DOSPlotter` / `FermiPlotter` / `FermiSlicePlotter`. See `examples/general/new_*_examples.py`.
- **Legacy one-call functions:** `pyprocar.bandsplot`, `bandsdosplot`, `dosplot`, `fermi2D`, `FermiHandler`, `BandStructure2DHandler`, which the `examples/*/*.ipynb` notebooks use. They are mid-migration. The per-feature status is in `features/README.md`.
- **File utilities:** `pyprocar.bandgap`, `kpath`, `filter`, `repair`, `cat`, `generate2dkmesh`. These read and write VASP files instead of plotting; see `features/utilities.md`.

"Driving" means: write a small driver script that makes those calls on a real calc dir, run it through the harness, and read the evidence.

All commands run from the repo root. `H=.claude/skills/verify-pyprocar/scripts/verify.sh`.

## Launch

There is nothing to keep alive. "Launch" means the env and fixtures exist:

```bash
$H fetch data/examples/bands/non-spin-polarized     # idempotent; ~20-35 MB each, into ./data (gitignored)
```

`fetch` downloads a missing fixture, then makes it read-only, so a stray write into a fixture fails with `PermissionError` whatever the code path. Rerun it on an existing fixture to lock it; that needs no download and no env, in a linked worktree too. It takes a fixture root, `data/examples/<category>/<name>` or `data/<name>`, spelled in plain names (letters, digits, `_`, `.` and `-`), and refuses a path whose real location is not a fixture inside `data/`. Code that must write next to a fixture works on a writable copy: `$H run` makes one, drivers call `writable_copy` from `verify_steps`, and tests call `writable_copy` from `tests.utils`.

Fixture relpaths (HF dataset `lllangWV/pyprocar_test_data`): `data/examples/{bands,dos,fermi3d,fermi2d}/{non-spin-polarized,spin-polarized,non-colinear}`, plus `bands/{atomic_levels,auto,compare_bands,ipr,unfolding,2d-bands}`, `fermi2d/bisb_monolayer`, `fermi3d/van-alphen`. All are VASP; Fermi energy for the SrVO3 sets is `5.3017`.

Other codes are already extracted under `data/codes/`: `qe/7.2/SrVO3`, `elk/6.3/SrVO3` and `vasp/6.4/SrVO3` (spin variants), and `abinit/9.6/Fe`. Siesta has no fixture; its tests build synthetic dirs. `features/parsers.md` lists the proven end states. For an independent reference from another code, run it with `pixi exec -s qe` (or `-s abinit`, `-s gfortran`), and keep its inputs and outputs outside any worktree, because `git worktree remove --force` deletes them.

Run the gates with the commands in `.github/workflows/ci.yml`, as written, with `pixi run --locked -e <env>` replaced by `$H exec` and `"$BASE"` by `origin/dev`. In the main checkout `$H exec` is `pixi run -q --locked -e dev`; in a linked worktree it uses the main checkout's env (see below). The pixi `typecheck` task runs the same type check. The `test` task also runs the tests marked `data`, and the `lint` task applies `ruff --fix` to every file, so neither is a CI gate.

`$H exec python .github/scripts/red_green.py origin/dev` is the `red-green` job: it fails on each added or changed test that passes on the base code without `@pytest.mark.guards_existing_behaviour(reason="...")`.

`--locked` fails when `pixi.lock` does not match `pixi.toml`, in CI and locally. After you edit `pixi.toml`, run `pixi lock` and commit `pixi.lock` with it.

Run env binaries through `$H exec`. Called by path, without its env's `bin/` on `PATH`, `.pixi/envs/dev/bin/basedpyright` reports phantom errors that CI does not.

### Agents in worktrees

Several agents may work in separate git worktrees at once. A fresh worktree lacks the gitignored `data/` and `pyprocar/_version.py` (`No module named 'pyprocar._version'`), and `pixi run` there builds a multi-GB env. Set the worktree up once, from its root:

```bash
$H worktree-setup
```

It copies `_version.py` from the main checkout and links `data` to the main checkout's `data/` with `ln -sfn` (a rerun is safe). After that, `$H exec`, `doctor`, `fetch` and `run` use the main checkout's `.pixi/envs/dev` with this worktree first on `PYTHONPATH`, and `$H exec` sets `PYTHONDONTWRITEBYTECODE=1` and `TMPDIR=.tmp`. They exit 2 when that env is missing. `worktree-setup` then prints at most one lock warning, and each asks for one action:
- "this branch changes pixi.lock or pixi.toml relative to origin/dev": the shared env does not match the branch. Run the gates with `pixi run --locked` in this worktree and accept the build.
- "the main checkout's pixi.lock differs from this branch's": the main checkout is on another branch. Keep using `$H exec`, and treat CI as the final gate if a local result disagrees with it.
- "no merge base with origin/dev": run `git fetch origin dev` and rerun `$H worktree-setup`.

In every worktree:

- `data/` is shared and holds `.py` files. Give pytest an explicit test path, so it never collects them, and `-p no:cacheprovider`. Write under `data/` only through `$H fetch`, which adds fixtures to the shared cache, and `$H run`, which works on a per-run copy in `data/verify-runs/`.
- Before `git worktree remove --force`, run `chmod -R u+w` on read-only copies inside the worktree (a private data copy you locked, pytest temp copies of locked fixtures under `.tmp`), or the removal fails with `Permission denied`.
- The `data`-marked tests take 10-12 minutes per checkout. Run the modules for the packages you touch in the background, at your head and at `origin/dev`, so a new failure stands apart from an old one.
- CI has no `data/`. The conftest guard fails an unmarked test that opens or lists `data/`, but a read at import or collection time escapes it: it passes locally and fails in CI. Mark every test that needs a fixture `data`.
- CI runs the lint and format gates on the PR merged into `dev`. Judge `ruff_new_violations.py` on your head merged with current `origin/dev` (in a detached scratch worktree), because on a head behind `dev` it also flags files that only `dev` changed.
- basedpyright runs in lock mode against `.basedpyright/baseline.json`: a new error or warning fails, and so does a baseline entry whose diagnostic is gone. When your change deletes or rewrites code that has baseline entries, delete exactly those entries in their own commit. Never add or regenerate entries. basedpyright checks only the paths in the `include` list of `pyrightconfig.json`, minus hidden directories below them, so add a new top-level Python directory or root-level `.py` file to that list.

## Doctor

```bash
$H doctor
```

Read-only. It prints the branch/commit, the count of dirty tracked files, the python/pyprocar/pyvista/vtk versions, the import path (it must be this checkout's `pyprocar/`), and the fixtures present. `writable fixture paths` counts the files and dirs in each fixture root that still have a write bit, and a root that is a symlink counts its target; above 0, lock those fixtures with `$H fetch <relpath>`. It is worth driving when pyprocar imports from this repo and the fixture you need is listed. A `dirty` count you didn't cause means someone is mid-edit, so say so before trusting results.

## Drive

```bash
$H run <name> <fixture-relpath> <driver.py>
```

What the harness does:
1. Creates `data/verify-runs/<timestamp>-<name>/`.
2. Copies the fixture to `work/calc`, following symlinks (a reflink copy where the filesystem supports it), makes the copy writable, and copies the driver to `evidence/driver.py`. `<name>` is one plain name, and the fixture must resolve to a fixture directory inside `data/`.
3. Runs the driver with `TMPDIR=work/tmp`, `CALC=<calc copy>`, `EVIDENCE=<evidence dir>`, `REPO=<repo root>`, `MPLBACKEND=Agg`, `PYVISTA_OFF_SCREEN=true`, and `scripts/lib` on `PYTHONPATH`. pytest puts its `--basetemp` root under `TMPDIR`, so a driver that starts pytest also writes under the run.
4. Records the exit code, `run.log` (stdout+stderr), and `side_effects.txt`, which lists files created or modified inside the calc copy.

Before step 1, the harness exits 3 when `data/verify-runs/` or `$TMPDIR` (default `/tmp`) has less free space than the fixture size plus `VERIFY_MIN_FREE_MB` (default 2048). The message prints the free space. The harness exits with the driver's exit code. Runs are isolated per directory, so parallel runs are safe, except Fermi-surface builds (see Headless rules).

Every worktree's `data` links to the one shared `data/`. Run every sweep through `$H run` so it works on the per-run `work/calc` copy, never on the shared fixtures. The library has written into its input directory before (`ebs.pkl` caches, a merged Abinit `PROCAR`, both stopped in #245). Keep temp files off the shared tmpfs `/tmp`; the harness does this, and outside it set `TMPDIR` (or pytest `--basetemp`) to a dir under your run.

A driver reads `CALC` and `EVIDENCE` from the environment, calls the public API, saves the figure into `EVIDENCE`, writes a `summary.json` of observable facts, and `assert`s the end state. The worked example `scripts/examples/bands_plain.py` is proven and passes:

```bash
$H run bands-plain data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/bands_plain.py
```

Per-feature recipes are in `features/`. Each feature has a ready driver at `scripts/examples/<feature>.py` that exercises every entry point the feature file lists. It uses `scripts/lib/verify_steps.py`:
- `@step(name)` runs one entry point in isolation and records `{ok, ...facts}` or `{ok: false, error, where}` into `summary.json`, so one crash doesn't hide the rest.
- `finish()` exits 1 if any step failed. A step that records a known crash fails the run too, so read `summary.json` per step and compare it against the feature file's proven end state.

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
- Warnings: pyprocar reports what a user must see as a `UserWarning` (the `user` logger carries only `verbose` progress). `verify_steps` records each step's warnings under `"warnings"` in `summary.json`. A driver without it wraps the call in `warnings.catch_warnings(record=True)` with `warnings.simplefilter("always")` before it claims that a warning did or did not fire, because Python prints a repeated warning only once.
- Fermi surfaces: run one build at a time. A VASP fixture load peaks near 8.7 GB RSS, and parallel builds were OOM-killed (exit 137).
- `autobandsplot` writes its analysis report only to the path given as `report=`; pass a path inside `EVIDENCE`.

## Evidence

Evidence lives at `data/verify-runs/<run>/evidence/` and survives cleanup. That directory is gitignored via `/data`. A proof includes:
- **The rendered image.** Open it with the Read tool and look at it. Check for correct k-path labels, sensible energy window and non-empty curves. A blank axes still has a nonzero PNG size.
- **`summary.json` with numbers that tie the image to the data.** Examples: array shapes, number of plotted artists, axis limits, tick labels, and projection sums.
- **`side_effects.txt`.** Since #245, `from_code` and the legacy plotting functions write nothing into the calc dir without `use_cache=True`; the bands, dos, fermi3d, fermi2d and bs2d drivers listed nothing at d6d4aaa7. An `ebs.pkl` or any other file listed there is a regression. Only the file utilities write files, as `features/utilities.md` documents.
- **Exit code and `run.log`**, including warnings.

Proof standards:
- Exercise the public entry point a user calls: `from_code` + plotter, or the legacy function. Never hand-build `Property` arrays or call parser internals to skip a broken step. If the user path crashes, that crash is the result.
- Capture before/after when verifying a change. Run the same driver on the base commit (a detached worktree after `$H worktree-setup`) and on the change, and compare the summaries and images.
- `use_cache=True` loads the fixture's shipped `ebs.pkl`, which may be stale. Verify parser changes with `use_cache=False`, the default.
- No mocks. The fixtures are real DFT output.

## Cleanup

```bash
$H clean data/verify-runs/<run>
```

This removes only that run's `work/` scratch copy (including its `TMPDIR`) and keeps `evidence/`. It refuses paths outside `data/verify-runs/`. Clean every run you made before hand-back, and leave other agents' runs alone. There are no processes to kill.

```bash
$H gc [hours]     # default 24
```

This removes the `work/` copy of every run that started more than `hours` ago, from any agent, and keeps each `evidence/`. It skips a run whose harness process is still alive, which the harness records in the run's `.pid` file. Run it when `$H run` refuses for lack of space. Fetched fixtures in `data/examples/` are shared cache; leave them.

## Helpers

- `scripts/verify.sh`: `doctor | fetch <relpath>... | run <name> <fixture> <driver.py> | clean <run-dir> | gc [hours] | worktree-setup | exec <cmd>...`
- `scripts/examples/bands_plain.py`: the minimal single-call template, which exits 0. Copy it for a one-off driver.
- `scripts/examples/{bands,dos,fermi3d,fermi2d,bs2d,parsers,utilities}.py`: full per-feature drivers. Run them as shown under Drive.
- `scripts/lib/verify_steps.py`: `step`, `png`, `distinct_colors`, `writable_copy` and `finish` for multi-step drivers. It is importable because the harness puts `scripts/lib` on `PYTHONPATH`.
- `scripts/lib/references/`: independent references (unfolding weights, reduced spin mesh, tiled cut orbits), each with an analytic `validate()` and a `ref_*.py` driver. Compare against one of these before you write your own; `features/README.md` lists them.

Known repo issues that affect verification (as of dev @ d6d4aaa7):
- `pyprocar.download_from_hf(relpath, output_path=".")` crashes when given a str; it needs a `Path`. The harness passes a `Path`.

A PR that changes a documented side effect updates this skill in the same PR. basedpyright type-checks `scripts/`, so a PR that breaks a driver's import or call fails CI's typecheck job until it fixes the driver.
