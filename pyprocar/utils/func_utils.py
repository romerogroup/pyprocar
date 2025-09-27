import inspect
from collections.abc import Sequence, Iterable
from functools import wraps
from itertools import product
from typing import Any, Callable, TypeVar, Union


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

def _is_seq(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

def check_for_groups(x: Any) -> bool:
    """True if x is a sequence of sequences, e.g. [[...], [...]]."""
    if not _is_seq(x):
        return False
    try:
        # Empty sequence -> NOT grouped (treated as a single empty selection)
        return len(x) > 0 and _is_seq(next(iter(x)))
    except StopIteration:
        return False

def _as_selection(x: Any) -> Any:
    """
    Normalize ONE selection (the inner object passed to the core function):
      - None stays None (means "use default/all" semantics if your core supports it).
      - int -> [int]
      - sequence -> list(sequence)
      - anything else -> as-is (you can tighten if needed)
    """
    if x is None:
        return None
    
    if isinstance(x, dict):
        return [x]
    elif _is_seq(x):
        return list(x)
    else:
        return [x]
    return x  # fallback; customize if you want to be stricter

def _to_groups(x: Any) -> list[Any]:
    """
    Normalize an argument into a list of selections ("groups"):
      - If grouped (list of lists), return [ _as_selection(g) for g in x ]
      - Else return [ _as_selection(x) ]
    """
    if check_for_groups(x):
        return [_as_selection(g) for g in x]
    return [_as_selection(x)]

def expand_grouped_params(*param_names: str) -> Callable[[Callable[..., T]], Callable[..., Union[T, list[T]]]]:
    """
    Decorator to expand specified parameters that may be given as lists of lists.
    Scalar ints are promoted to single-item lists (e.g., 3 -> [3]).
    Returns a single value if there’s exactly one combination; else a list.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, list[T]]]:
        sig = None
        try:
            import inspect
            sig = inspect.signature(func)
        except Exception:
            pass

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Union[T, list[T]]:
            # Bind args/kwargs to names so we can read/overwrite parameters easily
            if sig is not None:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                argmap = dict(bound.arguments)
            else:
                argmap = kwargs.copy()

            # Normalize each target parameter into groups (outer list = groups;
            # inner value = ONE selection passed to the core)
            group_lists: list[list[Any]] = []
            for pname in param_names:
                group_lists.append(_to_groups(argmap.get(pname, None)))

            results: list[T] = []

            # Iterate over cartesian product of groups
            for combo in product(*group_lists):
                callmap = dict(argmap)
                for pname, value in zip(param_names, combo):
                    # value here is ONE normalized selection (e.g., None, [3], [1,2], etc.)
                    callmap[pname] = value

                if sig is not None:
                    ba = sig.bind_partial(**callmap)
                    ba.apply_defaults()
                    result = func(*ba.args, **ba.kwargs)
                else:
                    result = func(**callmap)
                results.append(result)

            return results[0] if len(results) == 1 else results

        return wrapper
    return decorator


if __name__ == "__main__":
    print(get_kwargs(example_func))