"""
Rewrite tests/golden/patterns.json from the current code.

    python tests/regenerate_golden.py

Run it only after a change that is meant to alter the toolpaths, and review the
diff of the file: it is the record of what changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from _fixtures import layer, request, slicer_output

from organic_infill_generator.infill_generator.generator import generate_pattern
from organic_infill_generator.infill_generator.settings import PatternCatalog

GOLDEN = Path(__file__).resolve().parent / "golden" / "patterns.json"

OCTAGON = [
    [70.0, 0.0], [170.0, 0.0], [240.0, 70.0], [240.0, 170.0],
    [170.0, 240.0], [70.0, 240.0], [0.0, 170.0], [0.0, 70.0],
]


def measure(pattern: str) -> dict:
    """What identifies one toolpath, rounded so a libm difference does not matter."""
    result = generate_pattern(request(infill_pattern=pattern), slicer_output([layer(0.75, OCTAGON)]), {})
    points = np.asarray(result["path_points"])
    return {
        "points": len(points),
        "length_mm": round(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), 3),
        "start": [round(float(value), 3) for value in points[0]],
        "end": [round(float(value), 3) for value in points[-1]],
    }


def main() -> None:
    GOLDEN.parent.mkdir(exist_ok=True)
    data = {pattern: measure(pattern) for pattern in sorted(PatternCatalog.PUBLIC_PATTERN_MAP)}
    GOLDEN.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {GOLDEN}")


if __name__ == "__main__":
    main()
