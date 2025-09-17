# AGENTS.md
 
## Dev Environment Setup
- Install deps: `pixi install -e tests`
- To run in environment shell: `pixi shell -e tests`
- To run in terminal: `pixi run -e tests`

## Testing Instructions
- Run tests on single testing files and not the full test suite. example: `pixi run -e tests pytest tests/pyprocar/core/test_dos.py`
- Try to keep tests. Use generated test data for testing.
- These should be testing different execution paths of a layer. They should be contained within a single function with an appropiate name. 
- Prefer a single assert per test.

## Code Style Conventions

- Naming conventions:
  - Paths → `filepath` / `dirpath`
  - Lists → pluralized
  - Parameter objects → use `@dataclass` if parameter count > 3  
- Use type hints.
- Prefer to use guard clauses for early returns and error handling.

## Logging Conventions
- Loggers should be accesed by their name. example: `logging.getLogger(__file__)`
- Use user logger `logging.getLogger("user")` for user facing messages.
- Use info level for initialization and completion messages.
- Use debug level to get more detailed information such as array shapes, float, string, int values.
- Never return array values in logs
- Use warning level for warnings.
- Use error level for errors.
- Use critical level for critical errors.
- When adding logs, prefer to add them at the start and end of a functions.

## Commit & PR Guidelines

- Use **imperative mood** (`Add feature`, `Fix bug`, `Refactor parser`)  
- Keep first line ≤ 72 chars
- PR titles should describe intent clearly, not just “fix” or “update.”


Perfect — the diagram you shared is exactly the kind of **dataflow / layering concept** that an AGENT needs to understand so it doesn’t “break abstraction boundaries” when making changes.  

In `AGENTS.md`, this should become a **dedicated section** (let’s call it **DataFlow**) that sits alongside build/tests/style rules. This section isn’t for human readers to learn the science — it’s for agents to know *where to put code, what each layer is responsible for, and what not to mix together*.  

## DataFlow and Layer Responsibilities

This project follows a **layered dataflow architecture**.  
Agents should respect these layers and place new functionality in the correct layer.  
⚠️ **Never mix responsibilities across layers** (e.g., don’t put plotting logic inside a parser).


### Input Layer (Extraction)
- **Responsibility**: Raw extraction from simulation output files with minimal interpretation.  
- **Examples**: `PwSCF`, `Procar`, `VaspXML`, `AtomicProjXML` extractors.  
- **Agent guidance**:  
  - Only read files → produce raw Python objects or dict-like structures.  
  - Do not apply unit conversions or build canonical objects here.  

✅ Good: Extract numerical arrays directly from `vasprun.xml`.  
❌ Bad: Convert units to eV here (belongs in Parser layer).  

---

### Parser Layer (Adapter)
- **Responsibility**: Coordinates low-level input readers, applies **unit conversions**, and builds **canonical Data Objects**.  
- **Examples**: `QEParser`, `VaspParser`, common `Parser` superclass.  
- **Agent guidance**:  
  - Translate raw extracted data into well-structured domain objects (`ElectronicBandStructure`, `DensityOfStates`, etc.).  
  - Apply consistent units (eV, reciprocal space, etc.) here.  
  - Keep logic **deterministic and stateless**.  

✅ Good: Convert raw energy levels into an `ElectronicBandStructure` instance.  
❌ Bad: Generate or save plots here.  

---

### Data Layer
- **Responsibility**: Stores **canonical domain objects** (standardized representation of band structures, DOS, Fermi surfaces).  
- **Examples**: `ElectronicBandStructure`, `DensityOfStates`, `FermiSurface`.  
- **Agent guidance**:  
  - These are pure data containers.  
  - Do not mix them with visualization or I/O.  
  - If adding a new property: ensure it respects canonical units and domain consistency.  

✅ Good: Add a `.fermi_energy` attribute.  
❌ Bad: Add a `.plot()` method (this belongs in Visualization).  


### Visualization Layer
- **Responsibility**: Consumes **Data Layer objects** and produces plots.  
- **Examples**: `BandStructurePlotter`, `DOSPlotter`, `Fermi3DPlotter`.  
- **Agent guidance**:  
  - Input = standardized Data Objects only.  
  - Output = visualizations (matplotlib, plotly, etc.).  
  - Never re-implement parsing or unit conversion here.  

✅ Good: A plotter that accepts a `DensityOfStates` object and produces a Matplotlib figure.  
❌ Bad: A plotter that opens `vasprun.xml` directly.  

### DataFlow Summary

| Layer              | Responsibility                              | Output Type                   |
|--------------------|----------------------------------------------|-------------------------------|
| Input (Extraction) | Minimal file reading, raw structures         | Raw dicts/arrays              |
| Parser (Adapter)   | Builds canonical objects, unit conversions   | Domain Data Objects           |
| Data               | Canonical domain representation              | `ElectronicBandStructure`, etc. |
| Visualization      | Uses Data objects → produces visual output   | Plots / figures               |

### Agent Rules of Thumb
- **Keep boundaries clean**:  
  - Extraction → No conversions  
  - Parser → No plotting  
  - Data → No I/O or plotting  
  - Visualization → No parsing  
- **Inject dependencies**: Pass data objects **downstream**, don’t make upstream calls.  
- **Canonical objects are the "contract"** between layers: all layers must respect them.  

- When adding or editing code, first ask **“Which layer am I working in?”** and only add responsibilities appropriate for that layer. If unsure, default to **Data Layer first, Parser second** — never sneak application logic into Input or Visualization.  
