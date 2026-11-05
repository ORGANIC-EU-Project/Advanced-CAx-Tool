"""
settings.py
===========
Centralized tunable settings for the CadQuery/OCP conversion service.

This mirrors the settings style used by the infill generator: plain classes
with named constants, grouped by the part of the conversion they affect.
"""
from __future__ import annotations


class StepToStlSettings:
    """
    Meshing settings used when exporting a STEP/B-Rep model to STL.

    When auto tolerance is enabled, CadQuery derives the linear deflection from
    the imported part bounding-box diagonal. Manual values are used only when
    `AUTO_TOLERANCE` is disabled.
    """

    AUTO_TOLERANCE = True
    RELATIVE_LINEAR_DEFLECTION = 0.0005
    AUTO_ANGULAR_DEFLECTION = 0.05

    LINEAR_DEFLECTION = 0.01
    ANGULAR_DEFLECTION = 0.05


class StlToStepSettings:
    """
    Sewing settings used when turning STL triangle faces into a STEP shell.
    """

    SEW_TOLERANCE = 1e-6
