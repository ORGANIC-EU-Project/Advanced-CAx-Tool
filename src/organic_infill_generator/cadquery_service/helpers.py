from __future__ import annotations

import io
import sys
import time
from pathlib import Path


def configure_stdout_utf8() -> None:
    """Keep console output readable on Windows terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def now() -> float:
    """Return a monotonic timestamp suitable for elapsed-time measurements."""
    return time.perf_counter()


def file_size_bytes(path: Path) -> int | None:
    """Return the file size in bytes, or ``None`` when the file does not exist."""
    return path.stat().st_size if path.exists() else None


def ensure_parent(path: Path) -> None:
    """Create the parent directory for an output path before writing into it."""
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_nonempty(path: Path, label: str) -> None:
    """Fail fast when an expected conversion artifact was not written."""
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} was not created or is empty: {path}")


def print_report(result: dict) -> None:
    """Print the common conversion report returned by CAD service functions."""
    print()
    print("=" * 72)
    print(f"OPERATION : {result['operation']}")
    print(f"BACKEND   : {result['backend']}")
    print("=" * 72)
    print(f"Input : {result['input']}")
    print(f"Output: {result['output']}")

    print("\nSizes (bytes):")
    for key, value in result["sizes_bytes"].items():
        mb = f"  ({value / 1e6:.2f} MB)" if value else ""
        print(f"  - {key}: {value}{mb}")

    print("\nTimings (s):")
    for key, value in result["timings"].items():
        print(f"  - {key}: {value:.4f}")
    print()
