"""Error catalog with duplicate-code AND duplicate-name detection (forge contract).

Design rule (pitfalls.md G): two error classes with the SAME code silently shadow
each other with zero import-time signal. We detect both duplicate *codes* and
duplicate *names* at registration time and hard-fail the import if either occurs.
"""
from __future__ import annotations


class ForgeError(Exception):
    """Base class for all EffAttnForge errors."""

    code = "E000"

    def __init__(self, message: str = "", code: str | None = None) -> None:
        self.message = message
        self.error_code = code or self.code
        super().__init__(f"[{self.error_code}] {message}")


_REGISTRY: dict[str, type[ForgeError]] = {}
_Names: dict[str, str] = {}


def _register(cls: type[ForgeError]) -> type[ForgeError]:
    if cls is ForgeError:
        return cls
    if cls.code in _REGISTRY and _REGISTRY[cls.code] is not cls:
        clash = _REGISTRY[cls.code].__name__
        raise RuntimeError(
            f"duplicate error code {cls.code!r}: {cls.__name__} vs {clash}"
        )
    if cls.__name__ in _Names and _Names[cls.__name__] != cls.code:
        raise RuntimeError(
            f"duplicate error name {cls.__name__!r} bound to code "
            f"{_Names[cls.__name__]!r} (now {cls.code!r})"
        )
    _REGISTRY[cls.code] = cls
    _Names[cls.__name__] = cls.code
    return cls


@_register
class ConfigError(ForgeError):
    code = "E100"


@_register
class InvalidVariantError(ForgeError):
    code = "E101"


@_register
class ShapeError(ForgeError):
    code = "E200"


@_register
class NumericalError(ForgeError):
    code = "E300"


@_register
class RoutingError(ForgeError):
    code = "E400"


@_register
class BenchmarkError(ForgeError):
    code = "E500"
