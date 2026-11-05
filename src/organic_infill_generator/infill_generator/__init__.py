"""Public infill generator API."""
from __future__ import annotations

__all__ = [
    "InfeasibleRequestError",
    "PreflightIssue",
    "PreflightReport",
    "generate_pattern",
    "preflight_pattern_request",
]

_PREFLIGHT_EXPORTS = frozenset({
    "InfeasibleRequestError",
    "PreflightIssue",
    "PreflightReport",
    "preflight_pattern_request",
})


def __getattr__(name: str):
    if name == "generate_pattern":
        from .generator import generate_pattern

        return generate_pattern
    if name in _PREFLIGHT_EXPORTS:
        from . import preflight

        return getattr(preflight, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
