# Typing Standards for YKK-Point-Cloud

This document is the source of truth for Python typing standards in this repository. All code must conform to these rules. This codebase uses basedpyright with `typeCheckingMode: "all"`.

## Core Principles

### Philosophy

1. **Types are documentation that the compiler verifies.** Every function signature is a contract. Violations are bugs.

2. **No escape hatches.** Never use `Any`, `# type: ignore`, or `cast()` to silence errors. Fix the underlying type issue or write a proper stub.

3. **Explicit over implicit.** Annotate all function parameters and return types. Never rely on type inference for public APIs.

4. **Stubs for the untyped.** When a third-party library lacks types, write stubs in `typings/` rather than suppressing errors.

### When to Write Stubs vs Inline Types

**Write stubs (`typings/<package>.pyi`) when:**
- Third-party library has no `py.typed` marker
- Library's bundled types are incomplete or incorrect
- You need to type a C extension module

**Use inline annotations when:**
- Writing your own code (always)
- The types are part of your public API

### Type Checking Command

Always verify types pass before committing:
```bash
pixi run -q -e dev typecheck
```

---

## Function Signatures

### Rules

1. **MUST** annotate all parameters and return types for every function and method.
2. **MUST** use `None` return annotation for functions that don't return a value.
3. **MUST** use `| None` (not `Optional`) for nullable types.
4. **MUST** use `@overload` when a function's return type depends on input types.
5. **MUST NOT** use `*args: Any` or `**kwargs: Any`. Type them precisely or use `ParamSpec`.

### Examples

```python
# BAD: Missing annotations
def process(data, validate):
    if validate:
        return data.strip()
    return data

# GOOD: Fully annotated
def process(data: str, validate: bool) -> str:
    if validate:
        return data.strip()
    return data

# BAD: Returns None implicitly
def log_message(msg: str):
    print(msg)

# GOOD: Explicit None return
def log_message(msg: str) -> None:
    print(msg)

# BAD: Union return obscures actual behavior
def get_user(id: int) -> User | None:
    ...

# GOOD: Use overloads when behavior differs by input
@overload
def get_user(id: int, required: Literal[True]) -> User: ...
@overload
def get_user(id: int, required: Literal[False] = ...) -> User | None: ...
def get_user(id: int, required: bool = False) -> User | None:
    ...
```

### Relevant Diagnostics
- `reportMissingParameterType`
- `reportUnknownParameterType`
- `reportMissingReturnType`

---

## Generics

### Rules

1. **MUST** use `TypeVar` for functions that preserve input/output type relationships.
2. **MUST** use `bound` or `constraints` on TypeVars when the type must have specific capabilities.
3. **MUST** use `ParamSpec` to preserve callable signatures when wrapping functions.
4. **MUST** use `TypeVarTuple` for variadic generics (tuple unpacking, *args typing).
5. **MUST NOT** leave generic types unparameterized (e.g., `list` instead of `list[str]`).

### Examples

```python
from typing import TypeVar, ParamSpec, Callable, TypeVarTuple

# BAD: Loses type information
def first(items: list) -> object:
    return items[0]

# GOOD: Preserves type
_T = TypeVar("_T")

def first(items: list[_T]) -> _T:
    return items[0]

# BAD: Wrapper loses signature information
def logged(fn: Callable[..., Any]) -> Callable[..., Any]:
    ...

# GOOD: ParamSpec preserves signature
_P = ParamSpec("_P")
_R = TypeVar("_R")

def logged(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        print(f"Calling {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper

# GOOD: Bounded TypeVar for specific capabilities
_Comparable = TypeVar("_Comparable", bound="SupportsLessThan")

def minimum(a: _Comparable, b: _Comparable) -> _Comparable:
    return a if a < b else b
```

### Relevant Diagnostics
- `reportMissingTypeArgument`
- `reportInvalidTypeVarUse`
- `reportUnknownMemberType`

---

## Protocols & Structural Typing

### Rules

1. **MUST** use `Protocol` for structural typing instead of `ABC` when you only need method signatures.
2. **MUST** prefer protocols over concrete types for function parameters (accept interfaces, return concrete).
3. **MUST** mark protocol classes with `@runtime_checkable` only if you need `isinstance()` checks.
4. **MUST NOT** inherit from both `Protocol` and a concrete class.
5. **SHOULD** use `collections.abc` types (`Mapping`, `Sequence`, `Iterable`) over concrete types (`dict`, `list`) for parameters.

### Examples

```python
from typing import Protocol, runtime_checkable
from collections.abc import Iterable, Mapping

# BAD: Requires specific concrete type
def process_items(items: list[str]) -> None:
    for item in items:
        print(item)

# GOOD: Accepts any iterable
def process_items(items: Iterable[str]) -> None:
    for item in items:
        print(item)

# BAD: ABC when you only need structure
from abc import ABC, abstractmethod

class Readable(ABC):
    @abstractmethod
    def read(self) -> str: ...

# GOOD: Protocol for structural typing
class Readable(Protocol):
    def read(self) -> str: ...

def load_content(source: Readable) -> str:
    return source.read()

# GOOD: runtime_checkable only when needed
@runtime_checkable
class SupportsClose(Protocol):
    def close(self) -> None: ...

def maybe_close(obj: object) -> None:
    if isinstance(obj, SupportsClose):
        obj.close()
```

### Relevant Diagnostics
- `reportArgumentType`
- `reportAttributeAccessIssue`

---

## TypedDict & Data Structures

### Rules

1. **MUST** use `TypedDict` for dictionary structures with known keys, not `dict[str, Any]`.
2. **MUST** use class-based syntax for TypedDict, not functional syntax.
3. **MUST** mark optional keys with `NotRequired[]` or use `total=False`.
4. **MUST** use `dataclass` or `NamedTuple` for structured data with behavior or attribute access.
5. **MUST NOT** use `dict[str, Any]` as a catch-all for JSON or config data.

### Examples

```python
from typing import TypedDict, NotRequired, NamedTuple
from dataclasses import dataclass

# BAD: Untyped dictionary structure
def process_config(config: dict[str, Any]) -> None:
    host = config["host"]  # Unknown type
    port = config["port"]

# GOOD: TypedDict with known structure
class ServerConfig(TypedDict):
    host: str
    port: int
    timeout: NotRequired[float]

def process_config(config: ServerConfig) -> None:
    host = config["host"]  # str
    port = config["port"]  # int

# BAD: Functional TypedDict syntax
UserData = TypedDict("UserData", {"name": str, "age": int})

# GOOD: Class-based syntax
class UserData(TypedDict):
    name: str
    age: int

# GOOD: Use dataclass for objects with methods
@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

# GOOD: NamedTuple for immutable record types
class Coordinate(NamedTuple):
    lat: float
    lon: float
```

### Relevant Diagnostics
- `reportTypedDictNotRequiredAccess`
- `reportAny`
- `reportUnknownMemberType`

---

## Type Narrowing & Guards

### Rules

1. **MUST** use `isinstance()` or `type()` checks to narrow union types before access.
2. **MUST** use `TypeGuard` or `TypeIs` for custom narrowing functions.
3. **MUST** use `assert` statements or early returns to narrow types, not `# type: ignore`.
4. **MUST** use `Literal` types with equality checks for narrowing tagged unions.
5. **MUST NOT** use `cast()` to "narrow" types—use proper narrowing instead.

### Examples

```python
from typing import TypeGuard, TypeIs, Literal, assert_never

# BAD: Using cast to fake narrowing
def get_name(value: str | None) -> str:
    return cast(str, value)  # Unsafe!

# GOOD: Proper narrowing with isinstance
def get_name(value: str | None) -> str:
    if value is None:
        raise ValueError("Value required")
    return value  # Narrowed to str

# GOOD: TypeGuard for custom narrowing (narrows in True branch only)
def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

# GOOD: TypeIs for bidirectional narrowing (Python 3.13+ / typing_extensions)
def is_str(val: object) -> TypeIs[str]:
    return isinstance(val, str)

# GOOD: Exhaustive matching with assert_never
class Status:
    kind: Literal["pending", "done", "failed"]

def handle_status(status: Literal["pending", "done", "failed"]) -> str:
    match status:
        case "pending":
            return "Waiting..."
        case "done":
            return "Complete!"
        case "failed":
            return "Error occurred"
        case _ as unreachable:
            assert_never(unreachable)
```

### Relevant Diagnostics
- `reportUnnecessaryCast`
- `reportInvalidCast`
- `reportMatchNotExhaustive`
- `reportUnreachable`

---

## Handling Third-Party Libraries

### Rules

1. **MUST** write stubs in `typings/<package_name>/` for untyped third-party libraries.
2. **MUST** create `__init__.pyi` for package stubs, or `<module>.pyi` for single modules.
3. **MUST** use `_typeshed.Incomplete` for partially typed stubs, never `Any`.
4. **MUST** include `def __getattr__(name: str) -> Incomplete: ...` in partial module stubs.
5. **MUST NOT** use `allowedUntypedLibraries` as a first resort—write stubs instead.
6. **MUST NOT** suppress `reportMissingTypeStubs` globally.

### Stub File Structure

```
typings/
├── some_untyped_lib/
│   ├── __init__.pyi
│   └── submodule.pyi
└── another_lib.pyi
```

### Examples

```python
# typings/untyped_lib/__init__.pyi

from _typeshed import Incomplete

# Fully typed exports
def connect(host: str, port: int) -> Connection: ...

class Connection:
    def execute(self, query: str) -> list[dict[str, str]]: ...
    def close(self) -> None: ...

# Partially typed - use Incomplete, not Any
def advanced_query(params: Incomplete) -> Incomplete: ...

# Mark module as partial
def __getattr__(name: str) -> Incomplete: ...


# typings/simple_module.pyi

from _typeshed import Incomplete
from collections.abc import Callable

# Type what you use, mark rest as incomplete
def parse(data: str) -> dict[str, str]: ...
def transform(fn: Callable[[str], str], data: str) -> str: ...

def __getattr__(name: str) -> Incomplete: ...
```

### Relevant Diagnostics
- `reportMissingTypeStubs`
- `reportUnknownMemberType`
- `reportUnknownVariableType`
- `reportAny`

---

## Prohibited Patterns

### Absolute Prohibitions

These patterns are **never acceptable** in this codebase:

| Pattern | Why It's Banned | What To Do Instead |
|---------|-----------------|-------------------|
| `Any` type | Defeats type safety entirely | Use proper types, generics, or `object` |
| `# type: ignore` | Unsafe, no rule specification | Fix the type error properly |
| `# pyright: ignore` (without rule) | Hides multiple potential errors | Never use; fix the underlying issue |
| `cast()` for narrowing | Lies to the type checker | Use `isinstance()`, `TypeGuard`, or assertions |
| `dict[str, Any]` | Untyped structure | Use `TypedDict` or proper value types |
| `*args: Any, **kwargs: Any` | Untyped variadic | Use `ParamSpec` or concrete types |
| Unparameterized generics | `list`, `dict`, `set` without type args | Always specify: `list[str]`, `dict[str, int]` |

### Conditional Prohibitions

These require explicit justification in rare cases:

| Pattern | When Acceptable | Required Action |
|---------|-----------------|-----------------|
| `cast()` | Widening types or post-narrowing in comprehensions | Add comment explaining why narrowing won't work |
| `object` type | Truly accepting any type | Confirm you don't need type info from the value |
| `@no_type_check` | Never | Remove and fix types |

### Examples of Fixes

```python
# BANNED: Any in signature
def process(data: Any) -> Any: ...

# FIXED: Proper generic
_T = TypeVar("_T")
def process(data: _T) -> _T: ...

# BANNED: Ignoring errors
x: str = some_untyped_func()  # type: ignore

# FIXED: Write a stub for some_untyped_func in typings/

# BANNED: Cast for narrowing
items: list[str | int] = [...]
strings = cast(list[str], [x for x in items if isinstance(x, str)])

# FIXED: Let type checker infer
strings = [x for x in items if isinstance(x, str)]  # Inferred as list[str]
```

### Relevant Diagnostics
- `reportAny`
- `reportExplicitAny`
- `reportIgnoreCommentWithoutRule`
- `reportInvalidCast`
- `reportUnnecessaryCast`

---

## Basedpyright Diagnostics Reference

### Critical Diagnostics (Always Fix Immediately)

| Diagnostic | Meaning |
|------------|---------|
| `reportGeneralTypeIssues` | Core type mismatches—assignment, argument, return |
| `reportArgumentType` | Wrong argument type passed to function |
| `reportReturnType` | Return value doesn't match declared type |
| `reportAttributeAccessIssue` | Accessing attribute that doesn't exist on type |
| `reportAny` | Expression has type `Any` |
| `reportExplicitAny` | Direct usage of `Any` in annotations |
| `reportUnknownParameterType` | Parameter type cannot be determined |
| `reportUnknownMemberType` | Attribute/method type is unknown |
| `reportMissingTypeStubs` | Third-party library needs stubs |

### Safety Diagnostics (Prevent Runtime Errors)

| Diagnostic | Meaning |
|------------|---------|
| `reportOptionalMemberAccess` | Accessing member on potentially `None` value |
| `reportOptionalSubscript` | Indexing potentially `None` value |
| `reportUnboundVariable` | Variable used before assignment |
| `reportPossiblyUnboundVariable` | Variable might not be assigned on all paths |
| `reportInvalidCast` | Cast to non-overlapping type |

### Code Quality Diagnostics

| Diagnostic | Meaning |
|------------|---------|
| `reportUnusedImport` | Import not used in file |
| `reportUnusedVariable` | Variable assigned but never read |
| `reportUnusedParameter` | Function parameter never used |
| `reportIgnoreCommentWithoutRule` | Ignore comment missing rule code |
| `reportImplicitOverride` | Override method missing `@override` decorator |
| `reportPrivateLocalImportUsage` | Using non-exported symbol from module |

### Running Diagnostics

```bash
# Check all files
pixi run -q -e dev typecheck

# Check specific file with full output
pixi run -q -e dev typecheck src/mymodule.py --verbose
```
