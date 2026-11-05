"""
organic_infill_generator
=============
Basic module of the ORGANIC CAx tool: what a basic slicer does, from a CAD file
to a continuous toolpath for extrusion additive manufacturing.

    cadquery_service   STEP <-> STL conversion (optional CAD backend)
    slicer             planar slicing and closed contours
    infill_generator   walls, global pattern plan, clipping and exports
    pipeline           the complete process
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["PipelineConfig", "PipelineResult", "__version__", "run_pipeline"]


def __getattr__(name: str):
    if name in {"PipelineConfig", "PipelineResult", "run_pipeline"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
