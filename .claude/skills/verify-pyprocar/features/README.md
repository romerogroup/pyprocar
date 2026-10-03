# pyprocar feature map

One file per user-facing feature. Each lists every entry point a user can reach it by; a proof that covers one entry point is incomplete when the file lists others.

Each feature has a reusable driver in `scripts/examples/<feature>.py`. It runs every entry point as an isolated step, records each step's result or crash in `summary.json`, and exits 1 if any step failed. Read `summary.json` per step instead of trusting the exit code alone.

| Feature | File | Driver / fixture | Status (dos-rewrite @ 1f36aae1, re-driven 2026-10-02) |
|---|---|---|---|
| Band structure along a k-path | [bands.md](bands.md) | `bands.py` / `data/examples/bands/non-spin-polarized` | Object API plain/parametric/scatter/`plot_quiver`/flip work; `plot(vectors_mode=)` and `plot_overlay` crash; every legacy `bandsplot` mode and `bandsdosplot` crash |
| Density of states | [dos.md](dos.md) | `dos.py` / `data/examples/dos/non-spin-polarized` | Object API works; legacy `dosplot` crashes in every mode |
| 3D Fermi surface | [fermi3d.md](fermi3d.md) | `fermi3d.py` / `data/examples/fermi3d/non-spin-polarized` | Object API works; legacy `FermiHandler` parametric/fermi_speed work, plain crashes |
| 2D Fermi slice | [fermi2d.md](fermi2d.md) | `fermi2d.py` / `data/examples/fermi2d/non-spin-polarized` | `FermiSlicePlotter` with scalars works, without scalars draws nothing; legacy `fermi2D` unreachable in every mode |
| 2D band structure surface | [bs2d.md](bs2d.md) | `bs2d.py` / `data/examples/bands/2d-bands` | Object API works; legacy `BandStructure2DHandler` crashes |
| Code parsers (VASP, QE, Elk, Abinit, Siesta, Lobster, BXSF, FRMSF, DFTB+) | [parsers.md](parsers.md) | `parsers.py` / `data/examples/bands/non-spin-polarized` + `data/codes/qe` | VASP, QE and Elk bands proven; others not driven |
| File utilities (`bandgap`, `kpath`, `filter`, `repair`, `cat`, `generate2dkmesh`) | [utilities.md](utilities.md) | `utilities.py` / `data/examples/bands/non-spin-polarized` | `kpath`/`filter`/`repair`/`cat`/`generate2dkmesh` work; `bandgap` crashes |

Update the Status column whenever a run changes what is known.
