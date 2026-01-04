## Summary

PyProcar is a Python library for electronic structure pre/post-processing of DFT calculations (VASP, Quantum ESPRESSO, Abinit, Elk, Lobster). The codebase follows a **layered dataflow architecture** with four distinct layers: Input/Extraction, Parser/Adapter, Data, and Visualization. Core domain objects share a common `PointSet`/`Property` interface for consistent data handling.

## Project Details

### 1. Project Overview

**Purpose**: PyProcar analyzes and visualizes first-principles electronic-structure calculations, providing tools for:
- Band structure plotting (plain, parametric, spin-textured)
- Density of states analysis
- Fermi surface visualization (2D and 3D)
- Band unfolding for supercells
- K-path generation

**Supported DFT Codes**:
- VASP (full support)
- Quantum ESPRESSO (full support)
- Abinit (DOS in development)
- Elk (in development)
- Lobster (in development)
- SIESTA
- DFTB+

### 2. Architecture Layers

#### Layer 1: Input/Extraction (`pyprocar/io/`)
- **Purpose**: Read raw simulator files with minimal interpretation
- **Pattern**: Stateless extractors returning dicts/arrays
- **Files**: Code-specific extractors (e.g., `vasp/procar.py`, `qe/pw/pwout.py`)
- **Guidance**: No unit conversion, no canonical objects, no plotting

#### Layer 2: Parser/Adapter (`pyprocar/io/parser.py`)
- **Purpose**: Combine extractor outputs, perform conversions, produce canonical objects
- **Entry Point**: `Parser(code="vasp", dirpath=...)`
- **Pattern**: Factory pattern with `get_parser()` function
- **Guidance**: Deterministic processing, never touch visualization

#### Layer 3: Data Layer (`pyprocar/core/`)
- **Purpose**: Canonical domain objects representing physics concepts
- **Key Classes**:
  - `PointSet` / `Property`: Foundation pattern for aligned data
  - `ElectronicBandStructure`: K-path band energies
  - `ElectronicBandStructureMesh`: Uniform k-grid data
  - `DensityOfStates`: Energy-aligned DOS
  - `FermiSurface`: Isosurface representations
  - `Structure`: Atomic positions and lattice
- **Guidance**: Pure data + analysis; keep I/O and plotting out

#### Layer 4: Visualization (`pyprocar/plotter/`)
- **Purpose**: Render plots from canonical objects
- **Key Classes**: `DOSPlotter`, `BandStructurePlotter`, `FermiPlotter`
- **Pattern**: Accept fully-prepared data objects, use matplotlib/PyVista
- **Guidance**: Never read files or recompute physics

### 3. Core Design Patterns

#### PointSet/Property Pattern (`pyprocar/core/property_store.py`)
```python
# PointSet: Container for aligned sample points + properties
# Property: Metadata-rich wrapper around NumPy arrays
```
- Weak references prevent circular dependencies
- Gradient computation via pluggable `gradient_func`
- Properties can have sub-keys for gradients

#### Factory Pattern
- `Parser.from_code()`, `DensityOfStates.from_code()`, `FermiSurface.from_ebs()`
- Encapsulates complex initialization

#### Lazy Loading via `@cached_property`
- Extractors delay file I/O until property access
- Caches parsed results for reuse

### 4. Code Style Conventions

- **Formatting**: Black with line length 88, isort for imports
- **Naming**:
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Paths: `filepath` / `dirpath`
  - Lists: pluralized
- **Type hints**: Required
- **Guard clauses**: Prefer early returns
- **No nested functions**

### 5. Logging Conventions

```python
# Logger access by name
logging.getLogger(__file__)

# User-facing messages
logging.getLogger("user")

# Levels
# INFO: initialization/completion
# DEBUG: detailed info (shapes, values)
# WARNING/ERROR/CRITICAL: as appropriate

# Never log array values, log shapes instead
# Add logs at function start/end
```

### 6. Testing Patterns

- Whenever creating test reference the `testing-python` skill

### 7. Development Environment

**Package Manager**: pixi (preferred)

```bash
# Install deps
pixi install -e dev

# Run in environment shell
pixi shell -e dev

# Run commands
pixi run -e dev <command>

# Run tests
pixi run -e dev pytest tests/pyprocar/core/test_dos.py
```

**Environments**:
- `default`: build, dev, pytest
- `dev`: Full development (lint, rust, pytest, build, llm)
- `docs`: Documentation building
- `lint`: Linting/formatting

**Key Tasks**:
- `pixi run test`: Run all tests
- `pixi run lint`: Run linters
- `pixi run typecheck-python`: Run basedpyright

### 8. Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, build |
| `pixi.toml` | Environment management, tasks |
| `pyrightconfig.json` | Type checking settings |
| `.config/.pytest.toml` | Test configuration |
| `.config/.ruff.toml` | Python linting (line-length: 100) |
| `.config/lefthook.yaml` | Git hooks |

### 9. Git/PR Conventions

- **Branch names**: `feature/*`, `bugfix/*`, `hotfix/*`
- **Commit style**: Imperative mood, ≤72 chars first line
- **PR titles**: Describe intent clearly
- **Never commit**: `.env` files

### 10. Layer Boundaries (Critical)

From `AGENTS.md`:

- **Extraction**: No conversions
- **Parser**: No plotting
- **Data**: No I/O or plotting
- **Visualization**: No parsing

**Rule**: Inject dependencies downstream, don't make upstream calls.

### 11. Public API Entry Points

**Main Functions** (`pyprocar/__init__.py` → `pyprocar/scripts/`):
- `bandsplot()`: Band structure visualization
- `dosplot()`: Density of states
- `bandsdosplot()`: Combined plot
- `fermi2D()`: 2D Fermi surface
- `FermiHandler`: 3D Fermi surface
- `unfold()`: Band unfolding
- `bandgap()`: Calculate band gap
- `kpath()`: Generate k-paths

**Parser** (`pyprocar.io.Parser`):
```python
parser = Parser(code='vasp', dirpath='path/to/calc')
ebs = parser.ebs      # ElectronicBandStructure
dos = parser.dos      # DensityOfStates
structure = parser.structure  # Structure
```

### 12. Key File Locations

| Component | Location |
|-----------|----------|
| Core data objects | `pyprocar/core/` |
| IO parsers | `pyprocar/io/` |
| Plotters | `pyprocar/plotter/` |
| Utilities | `pyprocar/utils/` |
| User scripts | `pyprocar/scripts/` |
| Tests | `tests/pyprocar/` |
| Config files | `.config/` |

## Code References

- `pyprocar/core/property_store.py:488-745` - PointSet class
- `pyprocar/core/property_store.py:79-486` - Property class
- `pyprocar/core/ebs.py:168-1004` - ElectronicBandStructure
- `pyprocar/core/dos.py:400-2427` - DensityOfStates
- `pyprocar/io/__init__.py:54-93` - Parser class
- `pyprocar/plotter/bs_plot.py:46-1137` - BandStructurePlotter
- `pyprocar/plotter/dos_plot.py:150-966` - DOSPlotter

## Architecture Documentation

The codebase implements a unidirectional dataflow:

```
[Simulation Outputs] → [Extractors] → [Parser] → [Data Objects] → [Plotters]
```

Key architectural decisions:
1. **PointSet/Property foundation**: All physics objects share this pattern
2. **Factory-based creation**: Minimal constructors, rich factory methods
3. **Immutable transformations**: Methods return new instances
4. **Weak references**: Prevent circular dependencies in Property→PointSet

## Historical Context (from AGENTS.md/ARCHITECTURE.md)

- Project follows layered architecture with strict boundary enforcement
- PointSet/Property pattern is the "contract" between layers
- Ongoing work to harmonize all data classes around this pattern
- Currently expanding parser coverage for additional DFT codes

## Public API

Main entry points via `pyprocar.scripts/`:
- `bandsplot()`, `dosplot()`, `bandsdosplot()`
- `fermi2D()`, `FermiHandler` (3D)
- `unfold()`, `bandgap()`, `kpath()`

Parser usage:
```python
from pyprocar.io import Parser
parser = Parser(code='vasp', dirpath='path/to/calc')
ebs = parser.ebs      # ElectronicBandStructure
dos = parser.dos      # DensityOfStates
```
