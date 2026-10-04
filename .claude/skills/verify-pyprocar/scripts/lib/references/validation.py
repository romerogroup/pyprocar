"""``require`` still checks under ``python -O``, which strips ``assert``."""


class ValidationError(Exception):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValidationError(message)
