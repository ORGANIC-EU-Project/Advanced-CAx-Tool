"""Configuration and logging for the slicer package."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_AREA_THRESHOLD = 0.01
MIN_SEGMENTS_FOR_SLICE = 3

# Distance under which two segment endpoints count as the same point, in mm.
ENDPOINT_TOLERANCE = 1e-4

LARGE_SCALE_THRESHOLD = 1000.0
SMALL_SCALE_THRESHOLD = 0.01
SUB_MM_AUTOSCALE_FACTOR = 1000.0

OPEN_LINE_COLOR = "#CC79A7"
NOISE_SEGMENT_THRESHOLD = MIN_SEGMENTS_FOR_SLICE
