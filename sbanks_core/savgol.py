# -*- coding: utf-8 -*-
"""
Savitzky-Golay filter wrappers for geometry smoothing.

This module provides convenience functions for applying Savitzky-Golay
smoothing to open (LineString) and closed (Polygon ring) geometries.

The arc-length aware functions (smooth_open_geometry_arclength and
smooth_closed_geometry_arclength) resample to uniform arc-length spacing
before filtering, which prevents spatial artifacts when coordinates have
non-uniform spacing.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


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
        savgol_filter(x, window_length, polyorder, mode="wrap"),
        savgol_filter(y, window_length, polyorder, mode="wrap"),
    )


def smooth_open_geometry_arclength(x, y, window_length, polyorder,
                                    is_geographic=False, pad_count=None):
    """
    Apply Savitzky-Golay filter with arc-length parameterization for open geometries.

    This function resamples the input coordinates to uniform arc-length spacing
    before applying the S-G filter. This ensures spatial coherence is maintained
    when smoothing X and Y independently, preventing spike artifacts that can
    occur with non-uniformly spaced data.

    Parameters
    ----------
    x : array-like
        X coordinates (longitude if geographic)
    y : array-like
        Y coordinates (latitude if geographic)
    window_length : int
        Length of the filter window (must be odd and > polyorder)
    polyorder : int
        Order of the polynomial used to fit samples
    is_geographic : bool, optional
        If True, use Haversine formula for distance calculation.
        Default is False.
    pad_count : int, optional
        Number of points to pad at each end. Defaults to window_length.

    Returns
    -------
    tuple
        (x_smooth, y_smooth) smoothed coordinates at uniform arc-length positions,
        with endpoints snapped to original positions
    """
    from sbanks_core.geometry import calculate_cumulative_distances

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < window_length:
        return x.copy(), y.copy()

    # Store original endpoints
    x_start, y_start = x[0], y[0]
    x_end, y_end = x[-1], y[-1]

    # Calculate cumulative arc-length
    distances = calculate_cumulative_distances(x, y, is_geographic)
    total_length = distances[-1]

    if total_length <= 0:
        return x.copy(), y.copy()

    # Determine uniform spacing based on average spacing from original data
    avg_spacing = total_length / (len(x) - 1)

    # Number of uniformly spaced points (at least window_length + 1)
    n_uniform = max(window_length + 1, int(total_length / avg_spacing) + 1)
    uniform_distances = np.linspace(0, total_length, n_uniform)

    # Interpolate to uniform arc-length spacing
    fx = interp1d(distances, x, kind='linear', fill_value='extrapolate')
    fy = interp1d(distances, y, kind='linear', fill_value='extrapolate')
    x_uniform = fx(uniform_distances)
    y_uniform = fy(uniform_distances)

    # Apply S-G filter to uniformly-spaced data
    x_smooth, y_smooth = smooth_open_geometry(
        x_uniform, y_uniform, window_length, polyorder, pad_count
    )

    # Snap endpoints to original positions
    x_smooth[0], y_smooth[0] = x_start, y_start
    x_smooth[-1], y_smooth[-1] = x_end, y_end

    return x_smooth, y_smooth


def smooth_closed_geometry_arclength(x, y, window_length, polyorder,
                                      is_geographic=False):
    """
    Apply Savitzky-Golay filter with arc-length parameterization for closed geometries.

    This function resamples the input coordinates to uniform arc-length spacing
    before applying the S-G filter with wrap mode. This ensures spatial coherence
    is maintained for polygon rings.

    Parameters
    ----------
    x : array-like
        X coordinates (longitude if geographic), excluding the closing duplicate point
    y : array-like
        Y coordinates (latitude if geographic), excluding the closing duplicate point
    window_length : int
        Length of the filter window (must be odd and > polyorder)
    polyorder : int
        Order of the polynomial used to fit samples
    is_geographic : bool, optional
        If True, use Haversine formula for distance calculation.
        Default is False.

    Returns
    -------
    tuple
        (x_smooth, y_smooth) smoothed coordinates at uniform arc-length positions
    """
    from sbanks_core.geometry import calculate_cumulative_distances

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < window_length:
        return x.copy(), y.copy()

    # Calculate distances including the closing segment back to the start
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    distances = calculate_cumulative_distances(x_closed, y_closed, is_geographic)
    ring_distances = distances[:-1]  # Distances for the open vertices
    total_perimeter = distances[-1]

    if total_perimeter <= 0:
        return x.copy(), y.copy()

    # Determine uniform spacing based on average spacing from original data
    avg_spacing = total_perimeter / len(x)

    # Number of uniformly spaced points (at least window_length + 1)
    n_uniform = max(window_length + 1, int(total_perimeter / avg_spacing) + 1)

    # Create uniform distances that don't include the closing point
    # (since we'll use wrap mode)
    uniform_distances = np.linspace(0, total_perimeter, n_uniform, endpoint=False)

    # For interpolation, we need to handle the wrap-around.
    # Extend the data with one period before and after.
    x_extended = np.concatenate([x, x, x])
    y_extended = np.concatenate([y, y, y])
    d_extended = np.concatenate([
        ring_distances - total_perimeter,
        ring_distances,
        ring_distances + total_perimeter
    ])

    # Use linear interpolation
    fx = interp1d(d_extended, x_extended, kind='linear')
    fy = interp1d(d_extended, y_extended)
    x_uniform = fx(uniform_distances)
    y_uniform = fy(uniform_distances)

    # Apply S-G filter with wrap mode to uniformly-spaced data
    x_smooth, y_smooth = smooth_closed_geometry(
        x_uniform, y_uniform, window_length, polyorder
    )

    return x_smooth, y_smooth
