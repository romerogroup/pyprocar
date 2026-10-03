# Code parsers

The layer that turns raw DFT output directories into canonical objects. Every plot depends on it.

## Sub-features

- Codes: the dispatcher accepts exactly `vasp`, `qe`, `elk`, `abinit`, `siesta`, `lobster`, `bxsf`, `frmsf` and `dftbplus`. The DFT+ string is `dftbplus`, not `dftb+`; that raises `ValueError: Invalid code`. Adapters live in `pyprocar/io/<code>/`; DFTB+ is the single file `pyprocar/io/dftbplus.py`.
- Products: `Parser(code, dirpath)` has `.ebs`, `.dos`, `.structure`, `.kpath` and `.reciprocal_lattice`. There is no `Parser.fermi`; use `p.ebs.fermi`.
- Elk: bands from `BAND.OUT` (task 20) or `BAND_S01_A0001.OUT` (tasks 21 and 22) plus `BANDLINES.OUT`, the Fermi level from `FERMI.OUT` or `EFERMI.OUT`, `elk.in` plot1d paths, and `GEOMETRY.OUT`. Elk never writes `BANDS.OUT`.

## How to get to it (user POV)

1. **Directly:** `from pyprocar.io import Parser; p = Parser(code="vasp", dirpath=...); p.ebs; p.dos`.
2. **Indirectly:** every `*.from_code(code=..., dirpath=...)` and every legacy plotting function goes through `Parser`.

## Driving it with verify.sh

```bash
$H run parsers data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/parsers.py
```

The proven end state (1f36aae1):
- `vasp` (SrVO3 bands): `bands_shape` is `[200, 20, 1]`, ticks are `Γ M Γ R X`, species are `O Sr V`, and `ebs_fermi` is 4.9992.
  - That 4.9992 is the nscf vasprun value. It is not the SCF value 5.3017 that the examples pass as `fermi=`, so object-API plots are shifted by 4.9992.
- `qe` (`data/codes/qe/7.2/SrVO3/non-spin-polarized/bands`): `bands_shape` is `[155, 25, 1]`, ticks are `Γ M Γ R X`, and `ebs_fermi` is 12.5491.
  - The cell is cubic, so a private copy with `K_POINTS crystal_b` changed to `tpiba_b` must give the same 155 k-points and ticks (proven at #262). `tpiba_c` and `crystal_c` give `kpath` `None`; only the synthetic tests in `tests/pyprocar/io/qe/test_qe_parser.py` cover them.
- `elk` structure: all four `data/codes/elk/6.3/SrVO3/*` dirs give a 3.841244 Angstrom cubic lattice, through `GEOMETRY.OUT` or, in `non-spin-polarized/bands`, the `elk.in` fallback. Their `structure.pkl` files hold the pre-#242 Bohr lattice (7.2589), so do not compare against them.
- `Parser` writes nothing into the dir.
- `abinit` (`data/codes/abinit/9.6/Fe/*/bands`, #255): `kpath` ticks are `Γ(0) H(50) N(100) Γ(150) P(200) H|P(250) N(300)`, and `ebs` is an `ElectronicBandStructurePath`. Abinit writes each segment boundary once, so a wrong segmentation shows as 3 ticks `Γ(0) H|H(250) N(300)`. With Cartesian x-distances, Γ–H is `1/a` = 0.3521 Å⁻¹ (a = 2.84 Å).
- Band x-distances are Cartesian (Å⁻¹, no 2π) since #255. On the hexagonal `data/examples/bands/unfolding/primitive` set the tick x values are `0, 0.1878, 0.2963, 0.5132`; fractional distances give `0, 0.5, 0.8727, 1.3441`.

## Gotchas

- The HF `data/examples/*` sets are VASP only. Non-VASP fixtures are already extracted locally at `data/codes/{abinit,elk,qe,vasp}` and `data/io/vasp`, and the unit tests read them from `ROOT/data`. If they are missing, `$H fetch data/codes` (or `data/io`) should fetch and extract the dataset zip, according to `utils/download_examples.py`. This hasn't been driven, because the dirs already exist and the fetch returns early. There are no siesta, lobster, bxsf, frmsf or dftb fixtures.
- **BXSF has no fixture in `data/`, so make one.** ABINIT: `pixi exec -s abinit abinit run.abi` with `prtfsurf 1`, `shiftk 0 0 0` and a metal (the conda package ships only the H GTH pseudopotential `info/test/01h.pspgth`; simple-cubic H with `acell 3*3.0` works, and `acell 3*5.0` plus `nsppol 2` gives distinct spin channels). It writes `runo_BXSF` in Hartree. QE: `pw.x` scf and nscf, then `fs.x`, which writes `<prefix>_fs.bxsf`. `ElectronicBandStructureMesh.from_code("bxsf", dir)` finds either file. Check Gamma energies against `runo_EIG` or `nscf.out`.
- Elk bands (proven at #268): `data/codes/elk/6.3/SrVO3/non-spin-polarized/bands` gives bands `(54, 41, 1)` and `spin-polarized-colinear/bands` gives `(44, 71, 2)`, with ticks `Γ X M Γ R X` at the plot1d vertices. Elk lists each vertex once, and the parser repeats the 4 inner ones, so 50 and 40 points become 54 and 44.
