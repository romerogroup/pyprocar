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
| Parsers hand the core projections as `(k, band, spin, atom, orbital)` for bands and `(energy, spin, atom, orbital)` for DOS. The leading axes match the bands or total DOS, the atom axis matches the structure (or the `atom_groups` count of a PROCAR filtered by atoms), and there are 1, 2 or 4 spin channels (non-collinear is total, Sx, Sy, Sz, with 1 or 4 band channels). Every parser passes one name per orbital it projects. | `check_projected_layout` in the `ElectronicBandStructure` and `DensityOfStates` constructors; `tests/pyprocar/io/test_orbital_names.py` |
| Scripts, notebooks and docs call `pyprocar.*` with its current signature, and every `ebs.<member>` they name exists. | `tests/test_documented_calls.py` |
| Tests never write `data/`; a test that reads it is marked `data`. | audit hook in `tests/conftest.py`; fixtures are read-only after `verify.sh fetch`, and `verify.sh doctor` counts writable ones |
| Code that writes next to a fixture works on a writable copy of it. | `PermissionError` from the read-only fixture; `writable_copy` in `tests.utils` and `verify_steps` makes the copy |
| `verify.sh` writes only below the data root (the main checkout's `data/`, resolved), except that `worktree-setup` and `exec` write their own worktree's `pyprocar/_version.py`, `.tmp` and `data` link. It chmods and deletes only strictly below the data root, never through a symlink or a second hard link, and chmods only in `chmod_below_data`. | `tests/test_verify_fixtures.py`, which runs every command against a sacrificial tree outside `data/` |
| A message the user must see or act on is a `UserWarning` from `warn_user` (`pyprocar/utils/log_utils.py`), which names the user's line; the `user` logger carries only info and debug progress. Tests assert a warning with `tests/utils/user_warning.py`, which also checks the line it names. | `tests/test_user_warnings.py`. In `pyprocar/` it fails on `.warning`, `.error`, `.critical`, `.exception`, `.fatal`, `.warn`, or `.log` with a constant or named level of WARNING or above, called on `getLogger("user")`, on a name bound to it in the same function or module, on a `self.` attribute bound to it in the same class, or on a name imported from another module that binds it; and on `warn` or `warn_explicit` from the `warnings` module outside `log_utils.py`. In `tests/` it fails on `caplog.at_level` or `set_level` with logger `"user"`, on `setLevel` on the user logger, and on `pytest.warns(UserWarning, ...)` outside the helper. |
| Baseline entries in `.basedpyright/baseline.json` go only with the code they cover. | basedpyright lock mode in CI |
| A new or changed test fails on the base code for the reason it names, and passes with the fix. A test that guards existing behaviour says so with `@pytest.mark.guards_existing_behaviour(reason="...")`. | CI job `red-green` (`.github/scripts/red_green.py`), non-blocking for now, so read its log; the verifier still judges the reason a test fails |
| Expected values come from an independent source (formula, manual, other code path), never from the code under test. | nothing; review |
| A fix covers every sibling with the same defect: grep for the pattern across handlers, classes and parsers. | nothing; review |
| Units convert once, at the parser boundary. The core holds eV, Angstrom, and 1/Angstrom reciprocal lattices without 2 pi in the row convention (`frac @ B`). | nothing; review |
| A fix never turns a loud failure into silently wrong output. If a newly reachable path cannot be correct, raise. | nothing; review |
