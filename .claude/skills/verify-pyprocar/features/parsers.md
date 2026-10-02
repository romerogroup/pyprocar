# Code parsers

The layer that turns raw DFT output directories into canonical objects. Every plot depends on it.

## Sub-features

- Codes: the dispatcher accepts exactly `vasp`, `qe`, `elk`, `abinit`, `siesta`, `lobster`, `bxsf`, `frmsf` and `dftbplus`. The DFT+ string is `dftbplus`, not `dftb+`; that raises `ValueError: Invalid code`. Adapters live in `pyprocar/io/<code>/`; DFTB+ is the single file `pyprocar/io/dftbplus.py`.
- Products: `Parser(code, dirpath)` has `.ebs`, `.dos`, `.structure`, `.kpath` and `.reciprocal_lattice`. There is no `Parser.fermi`; use `p.ebs.fermi`.
- Elk: bands from `BANDS.OUT` plus `BANDLINES.OUT`, the Fermi level from `FERMI.OUT`, `elk.in` plot1d paths, and `GEOMETRY.OUT`. These are recent work.

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
- `Parser` writes nothing into the dir.

## Gotchas

- The HF `data/examples/*` sets are VASP only. Non-VASP fixtures are already extracted locally at `data/codes/{abinit,elk,qe,vasp}` and `data/io/vasp`, and the unit tests read them from `ROOT/data`. If they are missing, `$H fetch data/codes` (or `data/io`) should fetch and extract the dataset zip, according to `utils/download_examples.py`. This hasn't been driven, because the dirs already exist and the fetch returns early. There are no siesta, lobster, bxsf, frmsf or dftb fixtures.
- **Elk is verified-unreachable with the local fixtures.** The `data/codes/elk/*` dirs use the older layout (`BAND_S0x_A000y.OUT`, `EFERMI.OUT`, no `BANDS.OUT`/`FERMI.OUT`), so `Parser("elk", ...).ebs` is `None`. Proving Elk needs a calc made by a current Elk version. The Elk unit tests build inline `elk.in` text instead.
