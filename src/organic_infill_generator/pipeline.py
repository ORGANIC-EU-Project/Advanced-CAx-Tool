"""
pipeline.py
===========
The complete process of the tool, from a CAD file to a printable toolpath.

    STEP/STL -> planar slices -> closed contours -> walls and infill domain
             -> one global pattern, clipped per layer -> continuous toolpaths
             -> robot CSV, STL, STEP and a validation summary

Every stage lives in its own package; this module only orders them. Use
:func:`run_pipeline`, or call the stages yourself to stop anywhere in between.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .infill_generator.generator import (
    GlobalPatternPlan,
    _build_layer_outputs,
    _PatternPreviewResult,
    build_global_pattern_plan,
    save_pattern_stack_stl,
    save_pattern_stack_stp,
    save_toolpath_csv,
)
from .infill_generator.geometry.fillet import resolve_fillet
from .infill_generator.settings import ApiDefaults
from .infill_generator.validation import (
    coverage_metrics,
    ordered_path_continuity_metrics,
    path_quality_metrics,
)
from .slicer import ContourCloser, save_contours, slice_contours
from .slicer.models import SlicerLayer, SlicerOutput, SlicerResult

logger = logging.getLogger(__name__)

STEP_SUFFIXES = frozenset({".step", ".stp"})
STL_SUFFIXES = frozenset({".stl"})


@dataclass
class PipelineConfig:
    """Process parameters of one run. Lengths in mm, density in percent."""

    layer_height: float = 1.5
    line_width: float = 6.0
    infill_density: float = 50.0
    infill_pattern: str = ApiDefaults.INFILL_PATTERN
    num_walls: int = 1
    wall_spacing: float = 0.0
    # None rounds the corners with a radius of one line width; 0 keeps them sharp.
    fillet_radius: float | None = ApiDefaults.FILLET_RADIUS
    bottom_layers: int = 0
    top_layers: int = 0

    title: str = "organic_part"
    output_dir: str = "results"
    export_csv: bool = True
    export_stl: bool = True
    # Needs the optional CAD backend: pip install organic-infill-generator[cad]
    export_step: bool = False

    parallel: bool = False
    num_workers: int | None = None
    # None uses the simple cycle closure of the slicer.
    contour_closer: ContourCloser | None = field(default=None, repr=False)


@dataclass
class PipelineResult:
    """Everything one run produced, stage by stage."""

    stl_path: Path
    slicer_result: SlicerResult
    plan: GlobalPatternPlan
    previews: list[_PatternPreviewResult]
    contours_path: Path
    toolpath_csv_path: str | None
    output_stl_path: str | None
    output_step_path: str | None
    summary_path: Path
    summary: dict[str, Any]


class _StageTimer:
    """Collect how long each stage took, for the run summary."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def __call__(self, name: str) -> "_StageTimer":
        self._name = name
        return self

    def __enter__(self) -> None:
        logger.info("[stage] %s", self._name)
        self._started = time.perf_counter()

    def __exit__(self, *_: object) -> None:
        self.seconds[self._name] = round(time.perf_counter() - self._started, 3)


def prepare_stl(input_file: str | Path, output_dir: Path) -> Path:
    """Return an STL of the input part, tessellating it first if it is a STEP."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in STL_SUFFIXES:
        return input_path
    if suffix not in STEP_SUFFIXES:
        raise ValueError(f"Unsupported input '{suffix}'. Use a .step, .stp or .stl file.")

    # The CAD backend is optional, so it is only imported when a STEP arrives.
    from .cadquery_service import step_to_stl

    stl_path = output_dir / f"{input_path.stem}_tessellated.stl"
    step_to_stl(input_path, stl_path)
    return stl_path


def build_pattern_request(config: PipelineConfig, slicer_output: SlicerOutput) -> dict[str, Any]:
    """Translate the run configuration into the pattern request of a sliced part."""
    return {
        "num_walls": config.num_walls,
        "line_width": config.line_width,
        "wall_spacing": config.wall_spacing,
        "fillet_radius": config.fillet_radius,
        "infill_density": config.infill_density,
        "infill_pattern": config.infill_pattern,
        "extrude_height": float(slicer_output.model_bounds.size[2]),
        "layer_height": config.layer_height,
        "bottom_layers": config.bottom_layers,
        "top_layers": config.top_layers,
    }


def layer_area(layer: SlicerLayer) -> float:
    return float(sum(contour.area for contour in layer.contours))


def _largest_preview(previews: list[_PatternPreviewResult]) -> _PatternPreviewResult:
    return max(previews, key=lambda preview: preview.final_contour.area)


def validate_toolpaths(previews: list[_PatternPreviewResult]) -> dict[str, Any]:
    """
    Measure the generated toolpaths against what the process needs.

    Continuity is checked on every layer, because a jump on any of them stops
    the extruder. Coverage and path quality are measured on the largest layer:
    they are the expensive metrics and the largest layer is their worst case.
    """
    jumps_per_layer = [
        ordered_path_continuity_metrics(preview.path_points, tolerance=preview.line_width * 0.01)["jump_count"]
        for preview in previews
    ]
    largest = _largest_preview(previews)
    infill_points = largest.infill_path_points if len(largest.infill_path_points) >= 2 else largest.path_points
    coverage = coverage_metrics(
        path_points=infill_points,
        line_width=largest.line_width,
        printable_geometry=largest.infill_boundary,
    )
    quality = path_quality_metrics(
        path_points=largest.path_points,
        printable_geometry=largest.final_contour,
        line_width=largest.line_width,
    )
    return {
        "layers_with_toolpath": len(previews),
        "total_jumps": int(sum(jumps_per_layer)),
        "total_path_length_mm": float(sum(
            path_quality_metrics(
                path_points=preview.path_points,
                printable_geometry=None,
                line_width=preview.line_width,
            )["path_length"]
            for preview in previews
        )),
        "largest_layer": {
            "z_mm": float(largest.z),
            "area_mm2": float(largest.final_contour.area),
            **{key: value for key, value in coverage.items() if isinstance(value, (bool, int, float))},
            **{key: value for key, value in quality.items() if isinstance(value, (bool, int, float))},
        },
    }


def run_pipeline(
        input_file: str | Path,
        config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the whole tool on a STEP or STL part and write its outputs."""
    config = config or PipelineConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timer = _StageTimer()

    with timer("cad_to_stl"):
        stl_path = prepare_stl(input_file, output_dir)

    with timer("slice_and_close_contours"):
        slicer_result = slice_contours(
            str(stl_path),
            layer_height=config.layer_height,
            contour_closer=config.contour_closer,
            parallel=config.parallel,
            num_workers=config.num_workers,
        )
    slicer_output = slicer_result.output
    if not any(layer.contours for layer in slicer_output.layers):
        raise RuntimeError(f"The slicer found no closed contours in {stl_path}.")
    contours_path = output_dir / f"{config.title}_contours.json"
    save_contours(slicer_result, str(contours_path))

    with timer("global_pattern_plan"):
        plan = build_global_pattern_plan(build_pattern_request(config, slicer_output), slicer_output)

    with timer("layer_toolpaths"):
        previews = _build_layer_outputs(plan, slicer_output, title=config.title)
    if not previews:
        raise RuntimeError("No layer produced a printable toolpath.")

    with timer("validation"):
        validation = validate_toolpaths(previews)

    toolpath_csv_path = output_stl_path = output_step_path = None
    with timer("exports"):
        if config.export_csv:
            toolpath_csv_path = save_toolpath_csv(previews, output_dir=str(output_dir))
        if config.export_stl or config.export_step:
            stl_written = save_pattern_stack_stl(previews, output_dir=str(output_dir))
            output_stl_path = stl_written if config.export_stl else None
            if config.export_step:
                output_step_path = save_pattern_stack_stp(
                    previews, stl_path=stl_written, output_dir=str(output_dir),
                )

    largest = _largest_preview(previews)
    summary = {
        "input_file": Path(input_file).name,
        "process": {
            "layer_height_mm": config.layer_height,
            "line_width_mm": config.line_width,
            "infill_density_percent": config.infill_density,
            "infill_pattern": config.infill_pattern,
            "num_walls": config.num_walls,
            "fillet_radius_mm": largest.fillet_radius,
            "fillet_applied": resolve_fillet(largest.fillet_radius, largest.line_width).applied,
        },
        "slicer": {
            "total_layers": slicer_output.total_layers,
            "closed_layers": slicer_output.closed_layers,
            "failed_layers": slicer_output.failed_layers,
            "noise_layers": slicer_output.noise_layers,
            "model_bounds": asdict(slicer_output.model_bounds),
        },
        "validation": validation,
        "timings_s": timer.seconds,
    }
    summary_path = output_dir / f"{config.title}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return PipelineResult(
        stl_path=stl_path,
        slicer_result=slicer_result,
        plan=plan,
        previews=previews,
        contours_path=contours_path,
        toolpath_csv_path=toolpath_csv_path,
        output_stl_path=output_stl_path,
        output_step_path=output_step_path,
        summary_path=summary_path,
        summary=summary,
    )
