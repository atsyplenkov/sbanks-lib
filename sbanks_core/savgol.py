# -*- coding: utf-8 -*-
"""
Savitzky-Golay filter wrappers for geometry smoothing.

This module provides convenience functions for applying Savitzky-Golay
smoothing to open (LineString) and closed (Polygon ring) geometries.
"""

import numpy as np
from scipy.signal import savgol_filter


def smooth_open_geometry(x, y, window_length, polyorder, pad_count=None):
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

    Returns
    -------
    tuple
        (x_smooth, y_smooth) smoothed coordinates with endpoints snapped
        to original positions
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < window_length:
        return x.copy(), y.copy()

    pad_count = pad_count or window_length

    # Store original endpoints
    x_s, y_s = x[0], y[0]
    x_e, y_e = x[-1], y[-1]

    # Calculate direction at start
    dx_s = x[1] - x[0]
    dy_s = y[1] - y[0]

    # Calculate direction at end
    dx_e = x[-1] - x[-2]
    dy_e = y[-1] - y[-2]

    # Extrapolation ranges
    rng_b = np.arange(pad_count, 0, -1)
    rng_f = np.arange(1, pad_count + 1)

    # Extend arrays
    x_ext = np.concatenate([x[0] - dx_s * rng_b, x, x[-1] + dx_e * rng_f])
    y_ext = np.concatenate([y[0] - dy_s * rng_b, y, y[-1] + dy_e * rng_f])

    # Apply filter
    x_sm = savgol_filter(x_ext, window_length, polyorder)[pad_count:-pad_count]
    y_sm = savgol_filter(y_ext, window_length, polyorder)[pad_count:-pad_count]

    # Snap endpoints
    x_sm[0], y_sm[0] = x_s, y_s
    x_sm[-1], y_sm[-1] = x_e, y_e

    return x_sm, y_sm


def smooth_closed_geometry(x, y, window_length, polyorder):
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

    Returns
    -------
    tuple
        (x_smooth, y_smooth) smoothed coordinates
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < window_length:
        return x.copy(), y.copy()

    return (
        savgol_filter(x, window_length, polyorder, mode='wrap'),
        savgol_filter(y, window_length, polyorder, mode='wrap')
    )
