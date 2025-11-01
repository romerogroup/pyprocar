# General Taks Guidelines
- Always make a todo list on how to complete the task before starting.
- The todo list should start with creating tests for a new behavior, bug fix, or feature.
- Keep iterating until all the tests are passing. This should be module specific, do not worry about tests in other unrelated modules.
# Architecture Overview
- Make sure to always read and understand the architecture overview of the repository in `ARCHITECTURE.md`

# Deveolpment and Testing Guidelines
- Install deps: `pixi install -e dev`
- To run in environment shell: `pixi shell -e dev`
- To run in terminal: `pixi run -e dev`
- Run tests on single testing files and not the full test suite. example: `pixi run -e tests pytest tests/pyprocar/core/test_dos.py`
- Use generated test data for testing.
- These should be testing different execution paths of a layer. They should be contained within a single function with an appropiate name. 
- Prefer a single assert per test.
- create new test files in the `tests/` directory. This will likely mirror the file structure of the `pyprocar/` directory.

# Code Style Conventions
- Use `black` with line length 88
- Import order must follow `isort`
- Variable and function names must be snake_case
- Class names must be PascalCase
- Naming conventions:
  - Paths → `filepath` / `dirpath`
  - Lists → pluralized
- Use type hints.
- Use assert statements for validation and error handling.
- Prefer to use guard clauses for early returns and error handling.
- Do not use nested functions.

# Logging Conventions
- Loggers should be accesed by their name. example: `logging.getLogger(__file__)`
- Use user logger `logging.getLogger("user")` for user facing messages.
- Use info level for initialization and completion messages.
- Use debug level to get more detailed information such as array shapes, float, string, int values.
- Never return array values in logs, return the array shape instead.
- Use warning level for warnings.
- Use error level for errors.
- Use critical level for critical errors.
- When adding logs, prefer to add them at the start and end of a functions.

# Commit & PR Guidelines
- Branch names: `feature/*`, `bugfix/*`, or `hotfix/*`
- Use **imperative mood** (`Add feature`, `Fix bug`, `Refactor parser`)  
- Keep first line ≤ 72 chars
- PR titles should describe intent clearly, not just “fix” or “update.”

# Security
- Do NOT commit `.env` file

# Agent Rules of Thumb
- **Keep boundaries clean**:  
  - Extraction → No conversions  
  - Parser → No plotting  
  - Data → No I/O or plotting  
  - Visualization → No parsing  
- **Inject dependencies**: Pass data objects **downstream**, don’t make upstream calls.  
- **Canonical objects are the "contract"** between layers: all layers must respect them.  

- When adding or editing code, first ask **“Which layer am I working in?”** and only add responsibilities appropriate for that layer. If unsure, default to **Data Layer first, Parser second** — never sneak application logic into Input or Visualization.  
