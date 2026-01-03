import inspect
from collections.abc import Callable, Iterable
from functools import wraps
from itertools import product
from typing import Any, Literal, TypeVar, Union


def example_func(a, b, c=10, d=20, *, e=30, f=40, **kwargs):
    pass


def get_kwargs(func, defaults=True):
    sig = inspect.signature(func)
    return {
        name: param.default if param.default is not param.empty else None
        for name, param in sig.parameters.items()
        if param.default is not param.empty
        or param.kind
        in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        )
    }


def get_args(func, defaults=True):
    sig = inspect.signature(func)
    return [
        name
        for name, param in sig.parameters.items()
        if param.default is param.empty and name != "kwargs"
    ]


def get_params(func, defaults=True):
    sig = inspect.signature(func)
    return {name: param for name, param in sig.parameters.items()}


def keep_func_kwargs(
    kwargs,
    func,
):
    func_kwargs = get_kwargs(func)
    return {k: v for k, v in kwargs.items() if k in func_kwargs}


def keep_func_args(args, func, defaults=True):
    func_args = get_args(func, defaults)
    return [v for v in args if v in func_args]


def keep_func_kwargs_and_args(kwargs, func, defaults=True):
    func_kwargs = get_kwargs(func, defaults)
    func_args = get_args(func, defaults)
    return (
        [v for k, v in kwargs.items() if k in func_args],
        {k: v for k, v in kwargs.items() if k in func_kwargs},
    )


def keep_func_params(params, func, defaults=True):
    func_params = get_params(func, defaults)
    func_param_keys = set(func_params.keys())
    return {k: v for k, v in params.items() if k in func_param_keys}


T = TypeVar("T")
Mode = Literal["grouped", "explode"]

# --- helpers ---------------------------------------------------------------

_SPECIAL_NESTED_KEYS = ("kwargs", "options", "config", "params")


def _flatten_special_kwargs(
    d: dict[str, Any], keys: tuple[str, ...] = _SPECIAL_NESTED_KEYS, deep: bool = True
) -> dict[str, Any]:
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


def expand_grouped_params_to_dicts(params: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand grouped parameters into a list of parameter dictionaries.

    Groups are aligned by index, not Cartesian product. If one parameter
    is grouped (list of lists/dicts), others are broadcast to match.
    If multiple parameters are grouped, they must have the same length.

    Parameters
    ----------
    params
        Dictionary where keys are parameter names and values are their values.
        Values can be:
        - Single values (int, list, dict, None)
        - Grouped values (list of lists or list of dicts)

    Returns
    -------
    list[dict[str, Any]]
        List of parameter dictionaries, one per group index.
        If no grouped params, returns list with single dict.

    Examples
    --------
    >>> params = {"atoms": [[0,2], [1]], "orbitals": [0,1,2]}
    >>> expand_grouped_params_to_dicts(params)
    [{"atoms": [0,2], "orbitals": [0,1,2]}, {"atoms": [1], "orbitals": [0,1,2]}]

    >>> params = {"atoms": [[0,2], [1]], "orbitals": [[0,1,2], [4,5,6,7,8]]}
    >>> expand_grouped_params_to_dicts(params)
    [{"atoms": [0,2], "orbitals": [0,1,2]}, {"atoms": [1], "orbitals": [4,5,6,7,8]}]
    """
    # Detect grouped parameters
    grouped_params: dict[str, list[Any]] = {}
    for key, value in params.items():
        if _check_for_groups(value):
            grouped_params[key] = value

    # Determine number of groups
    if not grouped_params:
        # No grouped params, return single dict
        return [params.copy()]

    # Validate: all grouped params must have same length
    group_lengths = {key: len(value) for key, value in grouped_params.items()}
    if len(set(group_lengths.values())) > 1:
        lengths_str = ", ".join(f"{k}={v}" for k, v in group_lengths.items())
        raise ValueError(f"Grouped parameters must have the same length. Found: {lengths_str}")

    n_groups = list(group_lengths.values())[0]

    # Build list of parameter dicts
    result = []
    for i in range(n_groups):
        param_dict = {}
        for key, value in params.items():
            if key in grouped_params:
                # Use i-th element from grouped param
                param_dict[key] = grouped_params[key][i]
            else:
                # Broadcast single value to all groups
                param_dict[key] = value
        result.append(param_dict)

    return result


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
) -> Callable[[Callable[..., T]], Callable[..., T | list[T]]]:
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

    def decorator(func: Callable[..., T]) -> Callable[..., T | list[T]]:
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
        def wrapper(*args: Any, **kwargs: Any) -> T | list[T]:
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
