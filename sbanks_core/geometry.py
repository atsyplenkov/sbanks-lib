# -*- coding: utf-8 -*-
"""
Geometry utilities for coordinate processing.

This module provides GIS-agnostic geometry operations including:
- Distance calculations (Haversine and Cartesian)
- Coordinate padding for anti-hook and ring processing
- Spline-based resampling
"""

import numpy as np
from scipy.interpolate import splprep, splev


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two points using Haversine formula.

    Parameters
    ----------
    lon1, lat1 : float
        Longitude and latitude of first point in degrees
    lon2, lat2 : float
        Longitude and latitude of second point in degrees

    Returns
    -------
    float
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def calculate_cumulative_distances(x, y, is_geographic=False):
    """
    Calculate cumulative distances along a line.

    Parameters
    ----------
    x : array-like
        X coordinates (longitude if geographic)
    y : array-like
        Y coordinates (latitude if geographic)
    is_geographic : bool, optional
        If True, use Haversine formula; otherwise use Cartesian distance.
        Default is False.

    Returns
    -------
    np.ndarray
        Cumulative distances starting from 0
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2:
        return np.array([0.0])

    if is_geographic:
        # Use Haversine formula for geographic CRS
        distances = np.array(
            [
                haversine_distance(x[i], y[i], x[i + 1], y[i + 1])
                for i in range(len(x) - 1)
            ]
        )
    else:
        # Use Cartesian distance for projected CRS
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

    return np.concatenate([[0], np.cumsum(distances)])


def apply_antihook_padding(x, y, distances, pad_count):
    """
    Apply anti-hook extrapolation for open geometries (LineStrings).

    Extrapolates points at both ends of the line in the direction of
    the tangent at each endpoint. This prevents the "hook" artifact
    that can occur when smoothing open curves.

    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    distances : array-like
        Cumulative distances along the line
    pad_count : int
        Number of points to pad at each end

    Returns
    -------
    tuple
        (x_extended, y_extended, distances_extended) padded arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    distances = np.asarray(distances)
    n = len(x)

    # Direction at start
    dx_s = x[1] - x[0]
    dy_s = y[1] - y[0]
    d_s = np.sqrt(dx_s**2 + dy_s**2)
    if d_s > 0:
        dx_s /= d_s
        dy_s /= d_s
    else:
        dx_s, dy_s = 1.0, 0.0

    # Direction at end
    dx_e = x[-1] - x[-2]
    dy_e = y[-1] - y[-2]
    d_e = np.sqrt(dx_e**2 + dy_e**2)
    if d_e > 0:
        dx_e /= d_e
        dy_e /= d_e
    else:
        dx_e, dy_e = 1.0, 0.0

    # Average segment length for spacing
    avg_seg = distances[-1] / (n - 1) if n > 1 else 1.0

    # Ranges for padding
    rng_back = np.arange(pad_count, 0, -1)
    rng_fwd = np.arange(1, pad_count + 1)

    # Extend arrays
    x_ext = np.concatenate(
        [x[0] - dx_s * avg_seg * rng_back, x, x[-1] + dx_e * avg_seg * rng_fwd]
    )
    y_ext = np.concatenate(
        [y[0] - dy_s * avg_seg * rng_back, y, y[-1] + dy_e * avg_seg * rng_fwd]
    )
    d_ext = np.concatenate(
        [-avg_seg * rng_back, distances, distances[-1] + avg_seg * rng_fwd]
    )

    return x_ext, y_ext, d_ext


def apply_ring_padding(x, y, distances, pad_count, total_perimeter):
    """
    Apply circular padding for closed geometries (Polygon rings).

    Wraps points from the end of the ring to the beginning and vice versa,
    ensuring smooth continuity across the closure point.

    Parameters
    ----------
    x : array-like
        X coordinates (excluding closing point)
    y : array-like
        Y coordinates (excluding closing point)
    distances : array-like
        Cumulative distances along the ring
    pad_count : int
        Number of points to pad at each end
    total_perimeter : float
        Total perimeter of the ring

    Returns
    -------
    tuple
        (x_extended, y_extended, distances_extended) padded arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    distances = np.asarray(distances)

    # Prepend last pad_count points (with shifted distances)
    x_ext = np.concatenate([x[-pad_count:], x, x[:pad_count]])
    y_ext = np.concatenate([y[-pad_count:], y, y[:pad_count]])
    d_ext = np.concatenate(
        [
            distances[-pad_count:] - total_perimeter,
            distances,
            distances[:pad_count] + total_perimeter,
        ]
    )

    return x_ext, y_ext, d_ext


def resample_and_smooth(x, y, delta_s, smoothing_factor=1.0):
    """
    Resample and smooth coordinates using spline interpolation.

    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    delta_s : float
        Target sampling distance
    smoothing_factor : float, optional
        Smoothing factor for spline. Default is 1.0.

    Returns
    -------
    tuple
        (x_new, y_new) resampled and smoothed coordinates as numpy arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate cumulative distance
    dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))])
    total = dist[-1]

    if total < delta_s:
        return x, y

    # Fit spline (k must be less than the number of points)
    k = min(3, len(x) - 1)
    try:
        tck, _ = splprep([x, y], s=smoothing_factor * len(x), k=k)
    except Exception:
        return x, y

    # Generate new parameter values based on desired spacing
    u_new = np.linspace(0, 1, max(int(total / delta_s), 2))

    # Evaluate spline at new parameter values
    x_new, y_new = splev(u_new, tck)

    return np.asarray(x_new), np.asarray(y_new)


def snap_endpoints(x_smooth, y_smooth, x_start, y_start, x_end, y_end):
    """
    Force-snap first/last points to original endpoints.

    Parameters
    ----------
    x_smooth, y_smooth : array-like
        Smoothed coordinates
    x_start, y_start : float
        Original start point coordinates
    x_end, y_end : float
        Original end point coordinates

    Returns
    -------
    tuple
        (x_out, y_out) coordinates with snapped endpoints
    """
    x_out = np.asarray(x_smooth).copy()
    y_out = np.asarray(y_smooth).copy()
    x_out[0], y_out[0] = x_start, y_start
    x_out[-1], y_out[-1] = x_end, y_end
    return x_out, y_out


def densify_geometry(x, y, max_segment_length):
    """
    Densify geometry by adding points to segments exceeding max length.

    Segments longer than max_segment_length will have intermediate points
    inserted via linear interpolation. This is useful as a pre-processing
    step for Savitzky-Golay smoothing to prevent spike artifacts on
    geometries with uneven vertex density.

    Parameters
    ----------
    x : array-like
        X coordinates
    y : array-like
        Y coordinates
    max_segment_length : float
        Maximum allowed segment length. Segments longer than this
        will have intermediate points inserted via linear interpolation.
        Must be positive.

    Returns
    -------
    tuple
        (x_dense, y_dense) densified coordinates as numpy arrays.
        Original vertices are always preserved.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) < 2:
        return x.copy(), y.copy()

    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive")

    # Calculate segment lengths
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)

    # Build result arrays
    x_dense = [x[0]]
    y_dense = [y[0]]

    for i in range(len(segment_lengths)):
        seg_len = segment_lengths[i]

        if seg_len > max_segment_length:
            # Calculate number of subdivisions needed
            n_subdivisions = int(np.ceil(seg_len / max_segment_length))
            # Generate intermediate points via linear interpolation
            t = np.linspace(0, 1, n_subdivisions + 1)[1:]  # Skip first (already added)
            x_interp = x[i] + t * dx[i]
            y_interp = y[i] + t * dy[i]
            x_dense.extend(x_interp)
            y_dense.extend(y_interp)
        else:
            # Just add the endpoint
            x_dense.append(x[i + 1])
            y_dense.append(y[i + 1])

    return np.array(x_dense), np.array(y_dense)
