# CLAUDE.md

Before you set up a git worktree or run tests, type checks, lint or the library, read the Launch section of `.claude/skills/verify-pyprocar/SKILL.md`: its gates paragraph and "Agents in worktrees".

## Rules and what enforces them

When a reviewer corrects a mistake, fix it and add its rule here. If the rule is already here with nothing enforcing it, the mistake is a repeat: enforce it in the same change, with a check whose error names what to use instead.

| Rule | Enforced by |
|---|---|
| No undefined names or NumPy 2 removed API anywhere in the repo. | CI step `ruff check --select F821,F822,F823,NPY201 .` |
| Code runs on the declared floor, Python 3.12. | ruff `target-version` and basedpyright `pythonVersion`; `tests/test_python_floor.py` keeps those, `pyproject.toml` and `pixi.toml` equal |
| Physical constants come from `pyprocar/utils/units.py`. | `tests/test_unit_constants.py` |
| `pixi.toml` tasks name only defined tasks, environments and files. | `tests/test_pixi_manifest.py` |
| Parsers hand the core projections as `(k, band, spin, atom, orbital)` for bands and `(energy, spin, atom, orbital)` for DOS, with one orbital name per orbital. | `check_projected_layout` in the `ElectronicBandStructure` and `DensityOfStates` constructors |
| Scripts, notebooks and docs call `pyprocar.*` with its current signature. | `tests/test_documented_calls.py` |
| Tests never write `data/`; a test that reads it is marked `data`. | audit hook in `tests/conftest.py` |
| Baseline entries in `.basedpyright/baseline.json` go only with the code they cover. | basedpyright lock mode in CI |
| A new test fails on the base library for the reason it names, and passes with the fix. | nothing; the verifier's red/green lane |
| Expected values come from an independent source (formula, manual, other code path), never from the code under test. | nothing; review |
| A fix covers every sibling with the same defect: grep for the pattern across handlers, classes and parsers. | nothing; review |
| Units convert once, at the parser boundary. The core holds eV, Angstrom, and 1/Angstrom reciprocal lattices without 2 pi in the row convention (`frac @ B`). | nothing; review |
| A fix never turns a loud failure into silently wrong output. If a newly reachable path cannot be correct, raise. | nothing; review |
