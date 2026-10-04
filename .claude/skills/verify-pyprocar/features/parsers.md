# Code parsers

The layer that turns raw DFT output directories into canonical objects. Every plot depends on it.

## Sub-features

- Codes: `get_parser` accepts exactly `vasp`, `qe`, `elk`, `abinit`, `siesta`, `lobster`, `bxsf` and `frmsf`; any other string raises `ValueError: Invalid code`. Adapters live in `pyprocar/io/<code>/`.
- Products: `get_parser(code, dirpath)` returns a `BaseParser` with `.ebs`, `.dos`, `.structure`, `.kpath` and `.reciprocal_lattice`. The driver reads the Fermi energy from `p.ebs.fermi`.
- Elk: bands from `BAND.OUT` (task 20) or `BAND_S01_A0001.OUT` (tasks 21 and 22) plus `BANDLINES.OUT`, the Fermi level from `FERMI.OUT` or `EFERMI.OUT`, `elk.in` plot1d paths, and `GEOMETRY.OUT`. Elk never writes `BANDS.OUT`.
- Elk task-22 orbital names (#285): Elk 10.7.8 and later write task 22's characters in each site's irreducible-representation basis unless `lmirep` is `.false.` (release notes, `bandstr.f90`). The parser reads the version from `INFO.OUT` (`Elk code version x.y.z`) and names those columns `Y{l}_ir{i}`, the i-th basis function of l in `ELMIREP.OUT` order; Ylm runs keep `Y{l}{m}`. Without `INFO.OUT`, an `ELMIREP.OUT` from a run with no task 10 marks the irrep basis, and one that also lists task 10 warns. Elk DOS (task 10) columns are also in that basis whenever `lmirep` is on, but they keep the `Y{l}{m}` slot names #298 gave them.

## How to get to it (user POV)

1. **Directly:** `from pyprocar.io import get_parser; p = get_parser(code="vasp", dirpath=...); p.ebs; p.dos`.
2. **Indirectly:** every `*.from_code(code=..., dirpath=...)` and every legacy plotting function goes through `get_parser`.

## Driving it with verify.sh

```bash
$H run parsers data/examples/bands/non-spin-polarized .claude/skills/verify-pyprocar/scripts/examples/parsers.py
```

The proven end state (1f36aae1; `vasp` and `qe` re-driven at d6d4aaa7):
- `vasp` (SrVO3 bands): `bands_shape` is `[200, 20, 1]`, ticks are `Γ X M Γ R X`, species are `O Sr V`, and `ebs_fermi` is 4.9992.
  - That 4.9992 is the nscf vasprun value. It is not the SCF value 5.3017 that the examples pass as `fermi=`, so object-API plots are shifted by 4.9992.
- `qe` (`data/codes/qe/7.2/SrVO3/non-spin-polarized/bands`): `bands_shape` is `[155, 25, 1]`, ticks are `Γ X M Γ R X`, and `ebs_fermi` is 12.5491.
  - The cell is cubic, so a private copy with `K_POINTS crystal_b` changed to `tpiba_b` must give the same 155 k-points and ticks (proven at #262). `tpiba_c` and `crystal_c` give `kpath` `None`; only the synthetic tests in `tests/pyprocar/io/qe/test_qe_parser.py` cover them.
- `qe` non-collinear (`data/codes/qe/7.2/SrVO3/non-colinear/bands`, #293): `projected` is `[155, 50, 4, 5, 16]` (total, Sx, Sy, Sz on projwfc.x's real (l, m) orbitals `s pz px py dz2 dxz dyz dx2-y2 dxy ...`), and `bands` keeps 1 channel. At Gamma, band 41 has V t2g 0.984 and eg 0.000, and band 47 has t2g 0.000 and eg 0.833, matching the j-state weights in `kpdos.out`. A wrong spinor transform moves weight between dxy and dx2-y2. Every parser passes orbital names; a PROCAR that `pyprocar.filter` rewrote keeps its `o0` header name and label (`Bi-(o0)` on `data/examples/bands/auto`).
- `elk` structure: all four `data/codes/elk/6.3/SrVO3/*` dirs give a 3.841244 Angstrom cubic lattice, through `GEOMETRY.OUT` or, in `non-spin-polarized/bands`, the `elk.in` fallback. Their `structure.pkl` files hold the pre-#242 Bohr lattice (7.2589), so do not compare against them.
- `get_parser` writes nothing into the dir.
- `abinit` (`data/codes/abinit/9.6/Fe/*/bands`, #255): `kpath` ticks are `Γ(0) H(50) N(100) Γ(150) P(200) H|P(250) N(300)`, and `ebs` is an `ElectronicBandStructurePath`. Abinit writes each segment boundary once, so a wrong segmentation shows as 3 ticks `Γ(0) H|H(250) N(300)`. With Cartesian x-distances, Γ–H is `1/a` = 0.3521 Å⁻¹ (a = 2.84 Å).
- Band x-distances are Cartesian (Å⁻¹, no 2π) since #255. On the hexagonal `data/examples/bands/unfolding/primitive` set the tick x values are `0, 0.1878, 0.2963, 0.5132`; fractional distances give `0, 0.5, 0.8727, 1.3441`. The `bands/unfolding/supercell` set gives the same tick x values since #277; its VASP 6 OUTCAR prints a "Primitive cell" lattice block first, and `Outcar.reciprocal_lattice` takes the last block.

## Gotchas

- The HF `data/examples/*` sets are VASP only. Non-VASP fixtures are already extracted locally at `data/codes/{abinit,elk,qe,vasp}` and `data/io/vasp`, and the unit tests read them from `ROOT/data`. If they are missing, `$H fetch data/codes` (or `data/io`) should fetch and extract the dataset zip, according to `utils/download_examples.py`. This hasn't been driven, because the dirs already exist and the fetch returns early. There are no siesta, lobster, bxsf or frmsf fixtures.
- **BXSF has no fixture in `data/`, so make one.** ABINIT: `pixi exec -s abinit abinit run.abi` with `prtfsurf 1`, `shiftk 0 0 0` and a metal (the conda package ships only the H GTH pseudopotential `info/test/01h.pspgth`; simple-cubic H with `acell 3*3.0` works, and `acell 3*5.0` plus `nsppol 2` gives distinct spin channels). It writes `runo_BXSF` in Hartree. QE: `pw.x` scf and nscf, then `fs.x`, which writes `<prefix>_fs.bxsf`. `ElectronicBandStructureMesh.from_code("bxsf", dir)` finds either file. Check Gamma energies against `runo_EIG` or `nscf.out`.
- Elk bands (proven at #268): `data/codes/elk/6.3/SrVO3/non-spin-polarized/bands` gives bands `(54, 41, 1)` and `spin-polarized-colinear/bands` gives `(44, 71, 2)`, with ticks `Γ X M Γ R X` at the plot1d vertices. Elk lists each vertex once, and the parser repeats the 4 inner ones, so 50 and 40 points become 54 and 44.
