"""Stub for _pytest.python_api to type the approx function."""

class ApproxBase:
    """Base class for approximate comparisons."""
    ...

def approx(
    expected: object,
    rel: float | None = None,
    abs: float | None = None,
    nan_ok: bool = False,
) -> ApproxBase: ...
