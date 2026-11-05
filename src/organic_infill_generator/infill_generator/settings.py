"""
settings.py
===========
Centralized tunable settings for the infill generator.

This file is intentionally verbose. The goal is to keep the generic knobs in a
single place and explain what each one controls, so geometric or visual tweaks
can be made without searching through multiple modules first.
"""
from __future__ import annotations


class ApiDefaults:
    """
    Default values used by the public JSON-like API when a payload omits them.

    These are the "product-level" defaults: wall count, line width, density,
    export settings, and other values the user can conceptually think of as
    the default slicer profile for the generator.
    """

    # Total number of wall loops to generate, counting the outer wall.
    NUM_WALLS = 1
    # Physical width of a deposited line in model units, usually millimeters.
    LINE_WIDTH = 0.4
    # Global infill density percentage used to derive spacing automatically.
    INFILL_DENSITY = 50.0
    # Public-facing default pattern name when none is specified.
    INFILL_PATTERN = "honeycomb"

    # Fillet radius applied to the final toolpath to smooth sharp corners.
    # None means "use line_width". Set to 0 to disable.
    FILLET_RADIUS = None

    # Default Z height used when extruding the 2D path into a mesh.
    # A value of 0 disables implicit 3D extrusion.
    EXTRUDE_HEIGHT = 0.0
    # Default layer height used to turn top/bottom layer counts into real Z.
    LAYER_HEIGHT = 0.2
    # Default number of fully solid bottom layers.
    BOTTOM_LAYERS = 0
    # Default number of fully solid top layers.
    TOP_LAYERS = 0
    # Default human-readable title for generated outputs.
    TITLE = "Pattern Generator"
    # Default display unit for metadata and plot titles.
    UNIT = "mm"


class DensitySettings:
    """
    Shared bounds for the density math that turns a percentage into spacing.
    """

    # Minimum density fraction used in spacing formulas to avoid division by zero.
    MIN_DENSITY_RATIO = 0.01
    # Maximum density fraction; 1.0 means 100% density.
    MAX_DENSITY_RATIO = 1.0
    # Lower bound for exposed density percentages.
    MIN_DENSITY_PERCENT = 0.0
    # Upper bound for exposed density percentages.
    MAX_DENSITY_PERCENT = 100.0


class VariationSettings:
    """
    Defaults used when generating density-based request variations.

    These settings only affect helper utilities that synthesize alternative
    `PatternRequest` objects around the current global density.
    """

    # Maximum absolute density deviation applied around the original request.
    DENSITY_VARIATION_RANGE = 20.0
    # Smallest allowed increment between generated density alternatives.
    DENSITY_VARIATION_STEP = 5.0


class PatternCatalog:
    """
    Central pattern registry and pattern-specific geometric tuning.

    This class keeps the public pattern names, the internal primitive sequences
    they expand to, and the geometry correction factors that distinguish one
    family of pattern from another.
    """

    # Internal primitive codes used by the honeycomb family.
    HEXAGON_CODES = frozenset({"HH", "HHI"})
    # Internal primitive codes that form a diamond pattern.
    DIAMOND_CODES = frozenset({"S", "I"})
    # Vertical-family codes. ``MV`` is an internal narrow-spacing vertical used in mixed sequences.
    VERTICAL_CODES = frozenset({"V", "MV"})
    # Primitive codes used by the triangular pattern family.
    TRIANGULAR_MODIFIED_CODES = frozenset({"S", "I", "V", "MV"})

    # Public pattern name -> exact primitive sequence used by the generator.
    PUBLIC_PATTERN_MAP = {
        "vertical": ("V",),                # Single straight vertical column.
        "serrated": ("S",),                # Single serrated column leaning one way.
        "serrated_inverted": ("I",),       # Single serrated column leaning the opposite way.
        "sinusoidal": ("SIN",),            # Continuous sinusoidal column.
        "inclined": ("INCL",),             # Straight diagonal column.
        "honeycomb": ("HH", "HHI"),        # Alternating half-hexagons create honeycomb cells.
        "frankenstein": ("S",  "V","I", "SIN", "V", "SIN", "HH", "HHI"),
        "hilbert": ("HILB",),              # Hilbert curve packed into each column.
        "diamond": ("S", "I"),             # Alternating zigzags create diamond cells.
        "triangular": ("S", "V", "I", "V"),  # Triangular pattern with extra vertical stabilizing columns.
    }

    # Pattern-specific correction factors applied to the theoretical line distance.
    DISTANCE_FACTORS = {
        "honeycomb": 1.5,              # Honeycomb uses slicer-like sparse spacing, calibrated with cell ratio below.
        "diamond": 2.0,                # Diamond family uses a larger effective feature spacing than vertical lines.
        "triangular": 2.0,             # Triangular family keeps the same density-distance calibration as diamond.
        "sinusoidal": 2.0,             # Sinusoidal is calibrated like triangular for the current generator.
    }
    # Neutral factor for patterns without an explicit correction.
    DEFAULT_DISTANCE_FACTOR = 1.0

    # Height/width ratio used when deriving triangular pattern step from amplitude.
    TRIANGULAR_STEP_RATIO = 1.0
    # Height/width ratio used when deriving honeycomb pattern step from amplitude.
    HEXAGON_STEP_RATIO = 4.0
    # Neutral step ratio used by patterns without special geometry.
    DEFAULT_STEP_RATIO = 1.0


class ColumnSettings:
    """
    Generic settings that control how infill columns are laid out and sampled.

    These values describe the default dimensions of a column-based infill
    pattern and how much extra coverage we generate around the clip boundary.
    """

    # Default centerline-to-centerline spacing between infill columns.
    DEFAULT_INFILL_SPACING = 0.6
    # Default lateral opening of a pattern inside one column.
    DEFAULT_PATTERN_AMPLITUDE = 0.25
    # Default vertical distance between repeated features in a pattern.
    DEFAULT_PATTERN_STEP = 0.3

    # First column center is placed at half a spacing from the left bound.
    FIRST_CENTER_FACTOR = 0.5
    # Half-width of a nominal column expressed as a fraction of infill spacing.
    MAX_HALF_WIDTH_FACTOR = 0.5
    # Extra vertical padding around the infill boundary, expressed in amplitudes.
    VERTICAL_PADDING_FACTOR = 1.0
    # Minimum vertical padding as a fraction of boundary height (ensures columns
    # extend well beyond the contour).
    MIN_VERTICAL_PADDING_RATIO = 0.20


class VerticalSequenceSettings:
    """
    Special handling for vertical columns when they appear in mixed sequences.
    """

    # Enable a narrow line-width-based slot for vertical columns when they are
    # mixed with other patterns.
    APPLY_NARROW_VERTICAL_SPAN_IN_MIXED_SEQUENCES = True
    # Clearance multiplier used to convert a vertical line into its effective
    # slot width.
    MODIFIED_VERTICAL_CLEARANCE_FACTOR = 2.0


class SinusoidalSettings:
    """
    Sampling controls for sinusoidal columns.
    """

    # Number of vertical samples used per pattern step in sinusoidal paths.
    STEPS_PER_PATTERN_STEP = 10.0
    # Minimum vertical sampling increment for sinusoidal patterns.
    MIN_INCREMENT = 0.005
    # Lower bound on the pattern step the phase integration divides by.
    MIN_PATTERN_STEP = 0.01


class HoneycombSettings:
    """
    Geometric discretization for honeycomb columns.
    """

    # Number of vertical slices used to build one honeycomb cycle.
    VERTICAL_DIVISIONS = 6.0


class HilbertSettings:
    """
    Detail thresholds for Hilbert columns.
    """

    # Mapping from pattern step threshold to Hilbert recursion level.
    STEP_THRESHOLDS = (
        (1.0, 1),   # Coarse pattern step -> low Hilbert detail.
        (0.5, 2),   # Medium-coarse pattern step.
        (0.25, 3),  # Medium detail.
        (0.12, 4),  # Fine detail.
    )
    # Detail level used when pattern_step falls below all thresholds above.
    DEFAULT_LEVEL = 5
    # Chaikin smoothing weight for the first generated point on a segment.
    CHAIKIN_Q_WEIGHT = 0.75
    # Chaikin smoothing weight for the second generated point on a segment.
    CHAIKIN_R_WEIGHT = 0.25


class RingSettings:
    """
    Tuning for contour-ring reordering and seam handling.
    """

    # Fraction of a ring length used to decide whether a reordered start point
    # is already "near enough" to the existing seam.
    REORDER_TOLERANCE_RATIO = 0.001


class WallSettings:
    """
    Shared spacing and connection rules for wall generation and wall-toolpath joins.
    """

    # Multiplier used to shoot a long enough ray when building contour
    # connection lines.
    CONNECTION_RAY_DIAGONAL_MULTIPLIER = 2.0
    # Internal gap reused both for wall-to-wall separation and as the lateral
    # safety margin between neighboring columns, in model units (mm).
    SPACING = 0.0
    # Same quantity expressed as a FRACTION of line_width, which is how it
    # should be specified: the gap that matters is relative to the bead, so an
    # absolute value means something different at every line width. Negative is
    # legitimate and makes adjacent beads overlap on purpose; it is bounded at
    # -1.0, where the centrelines would coincide.
    # None keeps the absolute SPACING above; anything else overrides it.
    SPACING_RATIO: float | None = None
    # Extra additive gap used at the wall loop connection point.
    CONNECTION_GAP_EXTRA = SPACING + 0.2
    # Whether the first infill segment is stretched toward the innermost wall.
    # The probe that does it is VERTICAL, so on a wall at angle theta the real
    # perpendicular clearance is wall_gap_length * cos(theta): measured 3.70 mm
    # on an axis-aligned wall, 2.62 on a 45 degree chamfer, 1.39 on an ellipse.
    # The lead-in is not needed -- the infill reaches the wall through the
    # ordinary connector -- and it was the only wall contact left in the layer.
    EXTEND_INFILL_START_TO_WALL = False
    # Preferred angle for the wall-to-wall seam connector, measured in degrees
    # relative to the local inward normal.
    CONNECTION_ANGLE_DEGREES = 0
    # When enabled, wall loops are traversed in the opposite direction after
    # the connector so the seam gap stays on the far side relative to the
    # incoming infill direction.
    SWAP_GAP_SIDE_WITH_INFILL_DIRECTION = True


class FilletSettings:
    """
    Discretization and thresholds for rounded-corner path styling.
    """

    # Arc discretization used by Shapely when approximating rounded fillets.
    QUAD_SEGS = 8
    # Minimum turn angle before a vertex is considered a candidate for filleting.
    MIN_TURN_DEGREES = 8.0
    # Minimum viable fillet radius as a fraction of the line width. A radius
    # below ``MIN_RADIUS_FACTOR * line_width`` (or exactly 0) disables the fillet
    # entirely; it is never clamped upward.
    MIN_RADIUS_FACTOR = 0.5


class MeshSettings:
    """
    Numeric cleanup settings used before and during 3D extrusion.
    """

    # Small additive width applied only when building 3D mesh geometry from a
    # toolpath. This leaves the XY centerline/path planning untouched while
    # making exported ribbons slightly easier for slicers such as Cura to
    # detect as printable walls.
    LINE_WIDTH_EXPORT_EXTRA = 0.005
    # Grid size used to snap the fused swept geometry before 3D extrusion.
    # This helps remove almost-coincident edges and near-tangent contacts that
    # some slicers interpret poorly even when the mesh is technically watertight.
    SWEPT_GEOMETRY_PRECISION_GRID = 1e-5


class GeometrySettings:
    """
    Shared geometric tolerances and low-level numeric constants.

    Almost every geometry-heavy module needs small epsilons for clipping,
    deduplication, offsetting, or line intersection logic. Keeping them here
    makes the behaviour easier to tune consistently.
    """

    # Minimum non-trivial length used to keep or discard lines.
    LENGTH_EPSILON = 1e-6
    # Tiny area-like threshold used for generic near-zero comparisons.
    AREA_EPSILON = 1e-9
    # Tolerance used when comparing XY coordinates for equality.
    COORD_EPSILON = 1e-8
    # Threshold used when detecting nearly-vertical intersections in clipping code.
    VERTICAL_INTERSECTION_EPSILON = 1e-12


class PlotSettings:
    """
    Plotting defaults for previews, debug images, and smoke tests.

    These are visual-only settings: colors, line widths, legend style, figure
    size, and title formatting. They should not affect the actual toolpath
    geometry.
    """

    # Fallback matplotlib linewidth when a helper is called without an explicit value.
    DEFAULT_LINEWIDTH = 1.5
    # Alpha used when filling polygons for debug display.
    FILL_ALPHA = 0.4

    # Default figure size used by plot_pattern_result and generated previews.
    RESULT_FIGSIZE = (10, 8)
    # Default image resolution for saved PNG previews.
    SAVE_DPI = 150
    # Number of typographic points per inch, used to convert real line width to
    # plot linewidth.
    POINTS_PER_INCH = 72.0
    # Font size used by compact legends in debug plots.
    LEGEND_FONT_SIZE = 6
    # Default legend anchor position.
    LEGEND_LOCATION = "lower left"

    # Style for the final target contour outline.
    CONTOUR_COLOR = "gray"
    CONTOUR_LINEWIDTH = 1.0

    # Style for the inner boundary that clips the infill.
    INFILL_BOUNDARY_COLOR = "red"
    INFILL_BOUNDARY_LINEWIDTH = 0.6
    INFILL_BOUNDARY_LINESTYLE = "--"

    # Style for wall centerlines.
    WALL_COLOR = "dimgray"
    WALL_LINEWIDTH = 0.9
    WALL_ALPHA = 0.8

    # Style for the main toolpath centerline.
    CENTERLINE_COLOR = "blue"
    CENTERLINE_LINEWIDTH = 0.5

    # Style for the 3D mesh preview path with minimum corner radius applied.
    MESH_PREVIEW_COLOR = "darkorange"
    MESH_PREVIEW_LINEWIDTH = 1.1
    MESH_PREVIEW_LINESTYLE = "--"
    # Style used to overlay the physical line width on top of the centerline,
    # drawn as one translucent ribbon per polyline segment.
    LINE_WIDTH_COLOR = "cornflowerblue"
    LINE_WIDTH_ALPHA = 0.2

    # Start marker appearance.
    START_COLOR = "green"
    START_MARKER_SIZE = 60

    # Infill-start marker appearance.
    INFILL_START_COLOR = "lime"
    INFILL_START_MARKER_SIZE = 40
    INFILL_START_EDGE_COLOR = "black"
    INFILL_START_EDGE_LINEWIDTH = 0.5

    # End marker appearance.
    END_COLOR = "red"
    END_MARKER_SIZE = 60
    END_MARKER = "s"

    # Human-readable labels used in legends.
    LABEL_CENTERLINE = "Centerline"
    LABEL_MESH_PREVIEW = "3D Fillet Preview"
    LABEL_LINE_WIDTH = "Line width"
    LABEL_START = "Start"
    LABEL_INFILL_START = "Infill Start"
    LABEL_END = "End"

    # Number of decimals shown for density in plot titles.
    TITLE_DENSITY_DECIMALS = 1
    # Number of decimals shown for line distance in plot titles.
    TITLE_DISTANCE_DECIMALS = 3


class PathSettings:
    """
    Output-path and messaging settings.

    This class keeps the few generic strings that describe where artifacts are
    written and what message should be shown for known path-generation
    limitations.
    """

    # Directory where generated CSV, STL, and PNG artifacts are stored.
    RESULTS_DIR = "results"
    # Explicit error shown when someone requests a continuous Eulerian path on a shape with holes.
    HOLES_NOT_SUPPORTED_MESSAGE = (
        "assemble_eulerian_path currently supports infill boundaries without holes. "
        "Use the clipped-column preview for donut-style geometries."
    )
    # Amount trimmed from each clipped column endpoint before connecting columns.
    # This reduces harsh turns right on the contour and leaves room for a cleaner connector.
    COLUMN_END_TRIM_FACTOR = 0.5
    # Small tolerance used when checking whether a direct connector stays inside the boundary.
    CONNECTOR_COVER_TOLERANCE = 1e-7
    # Maximum center-to-center compression allowed when a fully dense solid
    # CSV raster must touch both infill boundaries.  The unavoidable width
    # remainder is confined to symmetric edge bands while the interior keeps
    # the exact requested linewidth pitch.
    SOLID_RASTER_MAX_OVERLAP_RATIO = 0.05


class PreflightSettings:
    """
    Thresholds for the manufacturability check run before any generation.

    The check is pure arithmetic on the request and the slicer bounds, so it
    costs nothing next to a generation pass. Its job is to reject what the
    process cannot deposit before the heavy work starts, and to flag what it
    can deposit but not the way it was asked for.
    """

    # How far ``extrude_height`` may sit from a whole number of layers before
    # it is reported, as a fraction of one layer. Layer counts are integers, so
    # a requested height almost never lands exactly on one; this absorbs both
    # float noise and the small process allowance, and only a real mismatch
    # (a height that will visibly round up or down) gets a warning.
    # A height equal to one layer height is a printable single layer.
    LAYER_HEIGHT_TOLERANCE_RATIO = 0.05

    # Cell-forming patterns (honeycomb, diamond, triangular) need at least this
    # many infill columns across the part to close a single cell. Below it the
    # pattern still generates, but it degenerates into disconnected lines.
    MIN_COLUMNS_FOR_CELL_PATTERNS = 2

    # Cell-forming pattern families, by public pattern name.
    CELL_PATTERNS = frozenset({"honeycomb", "diamond", "triangular"})

    # How far the density the generator can actually reach may sit from the
    # requested one, in percentage points, before it is reported. Spacing is
    # quantized by the bead, so small deviations are normal.
    DENSITY_TOLERANCE_PERCENT = 5.0
