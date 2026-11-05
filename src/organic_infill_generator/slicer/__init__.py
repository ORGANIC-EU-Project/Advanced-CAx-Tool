"""Public slicer API."""
from __future__ import annotations

from .closure import ClosureResult, ContourCloser, SimpleContourCloser
from .pipeline import save_contours, slice_contours

__all__ = [
    "ClosureResult",
    "ContourCloser",
    "SimpleContourCloser",
    "save_contours",
    "slice_contours",
]
