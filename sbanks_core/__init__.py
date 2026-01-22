# -*- coding: utf-8 -*-
"""
Sbanks Core Library - Pure Python geometry smoothing algorithms.

This library provides GIS-agnostic implementations of:
- Whittaker-Eilers smoother
- Savitzky-Golay filter
- Geometry utilities for coordinate processing
"""

__author__ = "Anatoly Tsyplenkov"
__copyright__ = "(C) 2026 by Anatoly Tsyplenkov"
__version__ = "0.1.0"

from .whittaker import WhittakerSmoother
from .savgol import smooth_open_geometry, smooth_closed_geometry
from .geometry import (
    haversine_distance,
    calculate_cumulative_distances,
    apply_antihook_padding,
    apply_ring_padding,
    resample_and_smooth,
    snap_endpoints,
    densify_geometry,
)

__all__ = [
    "WhittakerSmoother",
    "smooth_open_geometry",
    "smooth_closed_geometry",
    "haversine_distance",
    "calculate_cumulative_distances",
    "apply_antihook_padding",
    "apply_ring_padding",
    "resample_and_smooth",
    "snap_endpoints",
    "densify_geometry",
]
