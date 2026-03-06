# -*- coding: utf-8 -*-
"""
Savitzky-Golay filter wrappers for geometry smoothing.

This module provides convenience functions for applying Savitzky-Golay
smoothing to open (LineString) and closed (Polygon ring) geometries.
"""

import numpy as np
from scipy.signal import savgol_filter

from .geometry import (
    apply_antihook_padding,
    calculate_cumulative_distances,
    densify_geometry,
)


def _validate_savgol_params(window_length, polyorder):
    """
    Validate Savitzky-Golay core parameters.

    Parameters
    ----------
    window_length : int
        Length of the filter window
    polyorder : int
        Order of polynomial fit
    """
    if isinstance(window_length, bool):
        raise ValueError("window_length must be an odd positive integer")
    if not isinstance(window_length, (int, np.integer)):
        raise ValueError("window_length must be an odd positive integer")
    if window_length <= 0 or window_length % 2 == 0:
        raise ValueError("window_length must be an odd positive integer")

    if isinstance(polyorder, bool):
        raise ValueError("polyorder must be a non-negative integer")
    if not isinstance(polyorder, (int, np.integer)):
        raise ValueError("polyorder must be a non-negative integer")
    if polyorder < 0 or polyorder >= window_length:
        raise ValueError("polyorder must be non-negative and less than window_length")


def _validate_pad_count(pad_count):
    """
    Validate anti-hook padding count.

    Parameters
    ----------
    pad_count : int
        Number of points to pad at each endpoint
    """
    if isinstance(pad_count, bool):
        raise ValueError("pad_count must be a positive integer")
    if not isinstance(pad_count, (int, np.integer)) or pad_count <= 0:
        raise ValueError("pad_count must be a positive integer")


def smooth_open_geometry(
    x, y, window_length, polyorder, pad_count=None, max_segment_length=None
):
    """
    Apply Savitzky-Golay filter to open geometries with anti-hook extrapolation.

    Extrapolates points at both ends of the line before filtering, then trims
    back to the original length. This prevents endpoint artifacts.

    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    window_length : int
        Length of the filter window (must be odd and > polyorder)
    polyorder : int
        Order of the polynomial used to fit samples
    pad_count : int, optional
        Number of points to pad at each end. Defaults to window_length.
    max_segment_length : float, optional
        If provided, densify sparse segments before smoothing.
        Segments longer than this value will have points inserted
        via linear interpolation. This prevents spike artifacts on
        geometries with uneven vertex density.

    Returns
    -------
    tuple
        (x_smooth, y_smooth) smoothed coordinates with endpoints snapped
        to original positions
    """
    x = np.asarray(x)
    y = np.asarray(y)

    _validate_savgol_params(window_length, polyorder)
    pad_count = window_length if pad_count is None else pad_count
    _validate_pad_count(pad_count)

    # Optional densification
    if max_segment_length is not None:
        x, y = densify_geometry(x, y, max_segment_length)

    if len(x) < window_length:
        return x.copy(), y.copy()

    # Store original endpoints
    x_s, y_s = x[0], y[0]
    x_e, y_e = x[-1], y[-1]

    distances = calculate_cumulative_distances(x, y, is_geographic=False)
    x_ext, y_ext, _ = apply_antihook_padding(x, y, distances, pad_count)

    # Apply filter
    x_sm = savgol_filter(x_ext, window_length, polyorder)[pad_count:-pad_count]
    y_sm = savgol_filter(y_ext, window_length, polyorder)[pad_count:-pad_count]

    # Snap endpoints
    x_sm[0], y_sm[0] = x_s, y_s
    x_sm[-1], y_sm[-1] = x_e, y_e

    return x_sm, y_sm


def smooth_closed_geometry(x, y, window_length, polyorder, max_segment_length=None):
    """
    Apply Savitzky-Golay filter to closed geometries with wrap mode.

    Uses the wrap mode of savgol_filter to ensure smooth continuity
    across the closure point of rings.

    Parameters
    ----------
    x : array-like
        X coordinates (excluding the closing duplicate point)
    y : array-like
        Y coordinates (excluding the closing duplicate point)
    window_length : int
        Length of the filter window (must be odd and > polyorder)
    polyorder : int
        Order of the polynomial used to fit samples
    max_segment_length : float, optional
        If provided, densify sparse segments before smoothing.
        Segments longer than this value will have points inserted
        via linear interpolation. This prevents spike artifacts on
        geometries with uneven vertex density.

    Returns
    -------
    tuple
        (x_smooth, y_smooth) smoothed coordinates
    """
    x = np.asarray(x)
    y = np.asarray(y)

    _validate_savgol_params(window_length, polyorder)

    # Optional densification
    if max_segment_length is not None:
        x, y = densify_geometry(x, y, max_segment_length)

    if len(x) < window_length:
        return x.copy(), y.copy()

    return (
        savgol_filter(x, window_length, polyorder, mode="wrap"),
        savgol_filter(y, window_length, polyorder, mode="wrap"),
    )
