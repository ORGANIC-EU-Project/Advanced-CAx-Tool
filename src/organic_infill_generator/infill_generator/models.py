from __future__ import annotations

from dataclasses import dataclass

from .settings import ApiDefaults, VariationSettings


@dataclass
class PatternRequest:
    """
    Pattern and extrusion recipe.

    ``wall_spacing`` controls the gap between adjacent wall centerlines.
    If it is negative, walls intentionally overlap.

    Prefer ``wall_spacing_ratio``: the gap that matters is relative to the bead,
    so the same absolute value means something different at every line width.
    When set it wins over ``wall_spacing``; use ``wall_spacing_mm`` to read
    whichever applies.

    Geometry inputs such as contours, slicer layers, model bounds, export flags,
    titles, and units intentionally live outside this model.
    """
    num_walls: int
    line_width: float
    wall_spacing: float
    fillet_radius: float
    infill_density: float
    infill_pattern: str
    extrude_height: float
    layer_height: float = 0.2
    bottom_layers: int = 0
    top_layers: int = 0
    # Preferred way to give the gap: a fraction of line_width. When set it
    # overrides ``wall_spacing``, which stays for callers that already pass mm.
    wall_spacing_ratio: float | None = None

    @property
    def wall_spacing_mm(self) -> float:
        """The gap in model units, whichever way it was specified."""
        if self.wall_spacing_ratio is None:
            return float(self.wall_spacing)
        return float(self.wall_spacing_ratio) * float(self.line_width)

    @property
    def global_density(self) -> float:
        """
        Alias kept for utilities that conceptually talk about a global density.
        """
        return float(self.infill_density)


@dataclass
class PatternOutputOptions:
    """Which files a generation call writes, and where."""

    toolpath_csv_file: bool = False
    stl_file: bool = False
    step_file: bool = False
    title: str = ApiDefaults.TITLE
    # None keeps PathSettings.RESULTS_DIR.
    output_dir: str | None = None


@dataclass
class PatternVariationRequest:
    """
    Request model accepted by the density-variation helper.

    It keeps the variation-specific knobs separate from the main pattern
    request so callers do not have to pass loose arguments around.
    """

    original_pattern: PatternRequest
    n_variations: int
    variation_range: float = VariationSettings.DENSITY_VARIATION_RANGE
    step: float = VariationSettings.DENSITY_VARIATION_STEP
