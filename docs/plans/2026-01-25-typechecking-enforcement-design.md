# Enforcing Strict Typechecking for PyProcar

**Date:** 2026-01-25
**Goal:** Gradual improvement toward full strict compliance
**Current state:** 17,683 type errors under `typeCheckingMode: strict`

## Scope and Phasing

### Phase 1: Core Module + Supporting Utils (this effort)

Classes in dependency order:

| Order | Class | Depends On | Est. Errors |
|-------|-------|------------|-------------|
| 1 | Property | - | TBD |
| 2 | KPath | - | ~432 |
| 3 | Structure | - | TBD |
| 4 | EBS | Property, KPath, Structure | ~842 |
| 5 | DOS | Property, Structure | ~383 |
| 6 | FermiSurface | EBS | ~281 |

Each commit may include related `utils/` changes if the class depends on them.

### Phase 2: IO Module (future)

Parsers that create core objects.

### Phase 3: Remaining Modules (future)

plotter/, scripts/, remaining utils/, cfg/, pyposcar/

## Commit Strategy

- One commit per core class
- Each commit may include related `utils/` changes
- Each commit should pass typecheck for modified files
- Tests must pass after each commit

## pyrightconfig.json Updates

Add to `allowUntypedLibraries`:
- `numpy`
- `scipy`
- `pyvista`

(matplotlib already present)

## Typing Conventions

### Handling External Libraries

```python
# Libraries in allowUntypedLibraries return Any
# Annotate on assignment to establish types in your code
self.bands: npt.NDArray[np.float64] = np.loadtxt(file)
self.kpoints: npt.NDArray[np.float64] = data[:, :3]
```

### Imports

```python
from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Sequence, Mapping, Iterable
import numpy.typing as npt

if TYPE_CHECKING:
    from pyprocar.core import Structure, KPath  # Avoid circular imports
```

### Prefer Generic/Covariant Types for Parameters

```python
# Use Sequence, Mapping, Iterable for function parameters (read-only)
def process(atoms: Sequence[str]) -> None: ...
def lookup(mapping: Mapping[str, int]) -> None: ...

# Use concrete types for return values and attributes
def get_atoms(self) -> list[str]: ...
self._atoms: list[str] = list(atoms)
```

### Type Narrowing for Optional

```python
# Guard narrows type
def get_energy(self) -> float:
    if self.energy is None:
        raise ValueError("energy not set")
    return self.energy.real
```

### Common Patterns

```python
# Arrays
npt.NDArray[np.float64]      # Float arrays
npt.NDArray[np.int64]        # Integer arrays

# Optional values
energy: float | None = None

# Sequences for inputs, concrete for storage/returns
def __init__(self, atoms: Sequence[str]) -> None:
    self._atoms: list[str] = list(atoms)
```

## Workflow Per Commit

For each core class:

1. **Analyze errors:**
   ```bash
   pixi run -e dev basedpyright --project pyrightconfig.json pyprocar/core/<file>.py
   ```

2. **Add type annotations:**
   - Start with `__init__` parameters and instance attributes
   - Then public methods (parameters and return types)
   - Then private methods
   - Fix any `utils/` functions this class depends on

3. **Handle common fixes:**
   - Add null guards for `Optional` access
   - Annotate assignments from numpy/scipy calls
   - Add `from __future__ import annotations` if needed

4. **Verify:**
   ```bash
   pixi run typecheck  # Quick check
   pixi run test       # Ensure behavior unchanged
   ```

5. **Commit:**
   ```
   fix(core): Add type annotations to <ClassName>

   - Add parameter and return type annotations
   - Add instance attribute annotations
   - Fix related utils/ functions as needed
   ```

## Success Criteria

Phase 1 complete when:
- All 6 core classes pass strict typecheck
- Tests pass
- No regressions in functionality
