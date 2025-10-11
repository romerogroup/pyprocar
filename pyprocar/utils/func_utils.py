import inspect
from collections.abc import Sequence, Iterable
from functools import wraps
from itertools import product
from typing import Any, Callable, TypeVar, Union, Literal


def example_func(a, b, c=10, d=20, *, e=30, f=40, **kwargs):
    pass

        
def get_kwargs(func, defaults=True):
    sig = inspect.signature(func)
    return {
        name: param.default if param.default is not param.empty else None
        for name, param in sig.parameters.items()
        if param.default is not param.empty or param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        )
    }
    
    
def get_args(func, defaults=True):
    sig = inspect.signature(func)
    return [name for name, param in sig.parameters.items() if param.default is not param.empty]
    
    
def keep_func_kwargs(kwargs, func, ):
    func_kwargs = get_kwargs(func)
    return {k: v for k, v in kwargs.items() if k in func_kwargs}
    
def keep_func_args(args, func, defaults=True):
    func_args = get_args(func, defaults)
    return [v for v in args if v in func_args]
    
def keep_func_kwargs_and_args(kwargs, args, func, defaults=True):
    func_kwargs = get_kwargs(func, defaults)
    func_args = get_args(func, defaults)
    return {k: v for k, v in kwargs.items() if k in func_kwargs}, [v for v in args if v in func_args]
    
T = TypeVar("T")
Mode = Literal["grouped", "explode"]

# --- helpers ---------------------------------------------------------------

_SPECIAL_NESTED_KEYS = ("kwargs", "options", "config", "params")

def _flatten_special_kwargs(d: dict[str, Any],
                            keys: tuple[str, ...] = _SPECIAL_NESTED_KEYS,
                            deep: bool = True) -> dict[str, Any]:
    out = dict(d)
    while True:
        expanded = False
        for k in keys:
            v = out.pop(k, None)
            if isinstance(v, dict):
                out.update(v)
                expanded = True
        if not (deep and expanded):
            break
    return out

def _is_seq(x: Any) -> bool:
    # treat numpy arrays / lists / tuples as sequences; exclude str/bytes
    try:
        from collections.abc import Iterable
    except Exception:
        Iterable = tuple  # fallback, shouldn't happen
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

def _check_for_groups(x: Any) -> bool:
    """Sequence of sequences? e.g., [[...],[...]]"""
    if not _is_seq(x):
        return False
    try:
        it = iter(x)
        first = next(it)
    except StopIteration:
        return False
    return _is_seq(first)

def _as_selection_default(x: Any) -> Any:
    """
    Normalize a single selection. If you need int-> [int] behavior for
    certain parameters, plug that in here or per-parameter.
    """
    return x

def _to_groups_grouped(x: Any, as_selection: Callable[[Any], Any]) -> list[Any]:
    """
    Treat list as one group unless it's list-of-lists (already grouped).
    """
    if _check_for_groups(x):
        return [as_selection(g) for g in x]
    return [as_selection(x)]

def _to_groups_explode(x: Any, as_selection: Callable[[Any], Any]) -> list[Any]:
    """
    Treat list as many groups; list-of-lists keeps each inner as its own group.
    None -> [None].
    """
    if x is None:
        return [None]
    if _check_for_groups(x):
        # already groups: each inner selection remains a group
        return [as_selection(g) for g in x]
    if _is_seq(x):
        return [as_selection(v) for v in x]
    return [as_selection(x)]

# --- decorator -------------------------------------------------------------

ParamSpec = Union[str, tuple[str, Mode]]

def expand_grouped_params(
    *params: ParamSpec,
    default_mode: Mode = "grouped",
    selection_normalizers: dict[str, Callable[[Any], Any]] | None = None,
    use_all: bool = False,
    exclude: Iterable[str] = (),
) -> Callable[[Callable[..., T]], Callable[..., Union[T, list[T]]]]:
    """
    Expand specified parameters into a Cartesian product of 'groups'.

    Parameters
    ----------
    params:
      - "name"                  -> uses default_mode
      - ("name","grouped"/"explode") -> per-param override
      With use_all=True, these act as *overrides* on top of the auto-detected set.

    default_mode:
      - "grouped": list treated as one selection (list-of-lists = many)
      - "explode": list treated as many groups

    selection_normalizers:
      Optional per-parameter function(selection) -> selection, applied to each
      single selection (e.g., int -> [int] for atoms, or list(...) coercions).

    use_all:
      If True, automatically include all function parameters (except excluded),
      so you don't need to list them in *params.

    exclude:
      Iterable of parameter names to ignore when use_all=True.
    """
    explicit_specs: dict[str, Mode] = {}
    for p in params:
        if isinstance(p, tuple):
            name, mode = p
        else:
            name, mode = p, default_mode
        explicit_specs[name] = mode

    selection_normalizers = selection_normalizers or {}
    exclude = set(exclude) | {"self", "cls"}

    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, list[T]]]:
        try:
            sig = inspect.signature(func)
        except Exception:
            sig = None

        # Build the final spec list now that we have the signature.
        specs: list[tuple[str, Mode]] = []
        if use_all and sig is not None:
            for name, par in sig.parameters.items():
                if name in exclude:
                    continue
                if par.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                mode = explicit_specs.get(name, default_mode)
                specs.append((name, mode))
        else:
            # Use only explicitly provided params
            for name, mode in explicit_specs.items():
                specs.append((name, mode))

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[T, list[T]]:
            # 1) flatten nested kwargs at entry
            kwargs = _flatten_special_kwargs(kwargs)

            # 2) bind for named access
            if sig is not None:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                argmap = dict(bound.arguments)
            else:
                argmap = kwargs.copy()

            # 3) build groups
            group_lists: list[list[Any]] = []
            for pname, mode in specs:
                as_sel = selection_normalizers.get(pname, _as_selection_default)
                groups = (_to_groups_grouped if mode == "grouped" else _to_groups_explode)(
                    argmap.get(pname, None), as_sel
                )
                group_lists.append(groups)

            # 4) product iterate
            results: list[T] = []
            for combo in product(*group_lists):
                callmap = dict(argmap)
                for (pname, _mode), value in zip(specs, combo):
                    callmap[pname] = value

                callmap = _flatten_special_kwargs(callmap)

                if sig is not None:
                    ba = sig.bind_partial(**callmap)
                    ba.apply_defaults()
                    out = func(*ba.args, **ba.kwargs)
                else:
                    out = func(**callmap)
                results.append(out)

            return results[0] if len(results) == 1 else results

        return wrapper
    return decorator

if __name__ == "__main__":
    print(get_kwargs(example_func))