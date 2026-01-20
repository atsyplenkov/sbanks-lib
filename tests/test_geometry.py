# -*- coding: utf-8 -*-
"""Tests for geometry utility functions."""

import numpy as np
from sbanks_core.geometry import (
    haversine_distance,
    calculate_cumulative_distances,
    apply_antihook_padding,
    apply_ring_padding,
    resample_and_smooth,
    snap_endpoints,
)


class TestHaversineDistance:
    """Test cases for haversine_distance function."""

    def test_zero_distance(self):
        """Test that same point returns zero distance."""
        dist = haversine_distance(0, 0, 0, 0)
        assert dist == 0.0

    def test_known_distance(self):
        """Test with a known distance (approximately)."""
        # London to Paris: ~344 km
        lon1, lat1 = -0.1276, 51.5074  # London
        lon2, lat2 = 2.3522, 48.8566   # Paris
        dist = haversine_distance(lon1, lat1, lon2, lat2)
        # Allow 5% tolerance
        assert 320000 < dist < 360000

    def test_equator_distance(self):
        """Test distance along equator (1 degree ~ 111 km)."""
        dist = haversine_distance(0, 0, 1, 0)
        # 1 degree at equator is approximately 111 km
        assert 110000 < dist < 112000

    def test_symmetry(self):
        """Test that distance is symmetric."""
        dist1 = haversine_distance(10, 20, 30, 40)
        dist2 = haversine_distance(30, 40, 10, 20)
        np.testing.assert_allclose(dist1, dist2, rtol=1e-10)


class TestCalculateCumulativeDistances:
    """Test cases for calculate_cumulative_distances function."""

    def test_single_point(self):
        """Test with single point."""
        x = [0]
        y = [0]
        dist = calculate_cumulative_distances(x, y)
        np.testing.assert_array_equal(dist, [0.0])

    def test_two_points_cartesian(self):
        """Test Cartesian distance between two points."""
        x = [0, 3]
        y = [0, 4]
        dist = calculate_cumulative_distances(x, y, is_geographic=False)
        np.testing.assert_array_almost_equal(dist, [0.0, 5.0])

    def test_three_points_cartesian(self):
        """Test cumulative distances with three points."""
        x = [0, 3, 3]
        y = [0, 0, 4]
        dist = calculate_cumulative_distances(x, y, is_geographic=False)
        np.testing.assert_array_almost_equal(dist, [0.0, 3.0, 7.0])

    def test_geographic_mode(self):
        """Test that geographic mode uses Haversine."""
        x = [0, 1]  # longitude
        y = [0, 0]  # latitude
        dist = calculate_cumulative_distances(x, y, is_geographic=True)
        # 1 degree at equator is ~111 km
        assert dist[0] == 0.0
        assert 110000 < dist[1] < 112000

    def test_returns_numpy_array(self):
        """Test that function returns numpy array."""
        x = [0, 1, 2]
        y = [0, 1, 2]
        dist = calculate_cumulative_distances(x, y)
        assert isinstance(dist, np.ndarray)


class TestApplyAntihookPadding:
    """Test cases for apply_antihook_padding function."""

    def test_output_length(self):
        """Test that output has correct padded length."""
        n = 10
        pad = 5
        x = np.linspace(0, 10, n)
        y = np.linspace(0, 10, n)
        dist = calculate_cumulative_distances(x, y)

        x_ext, y_ext, d_ext = apply_antihook_padding(x, y, dist, pad)

        assert len(x_ext) == n + 2 * pad
        assert len(y_ext) == n + 2 * pad
        assert len(d_ext) == n + 2 * pad

    def test_original_data_preserved(self):
        """Test that original data is in the middle of output."""
        n = 10
        pad = 5
        x = np.linspace(0, 10, n)
        y = np.sin(np.linspace(0, np.pi, n))
        dist = calculate_cumulative_distances(x, y)

        x_ext, y_ext, d_ext = apply_antihook_padding(x, y, dist, pad)

        np.testing.assert_array_almost_equal(x_ext[pad:-pad], x)
        np.testing.assert_array_almost_equal(y_ext[pad:-pad], y)

    def test_extrapolation_direction(self):
        """Test that extrapolation follows tangent direction."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 2, 3, 4])  # 45-degree line
        dist = calculate_cumulative_distances(x, y)

        x_ext, y_ext, _ = apply_antihook_padding(x, y, dist, 2)

        # Start padding should extend in negative direction
        assert x_ext[0] < x[0]
        assert y_ext[0] < y[0]

        # End padding should extend in positive direction
        assert x_ext[-1] > x[-1]
        assert y_ext[-1] > y[-1]


class TestApplyRingPadding:
    """Test cases for apply_ring_padding function."""

    def test_output_length(self):
        """Test that output has correct padded length."""
        n = 10
        pad = 3
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        dist = np.linspace(0, 2 * np.pi, n)
        perimeter = 2 * np.pi

        x_ext, y_ext, d_ext = apply_ring_padding(x, y, dist, pad, perimeter)

        assert len(x_ext) == n + 2 * pad
        assert len(y_ext) == n + 2 * pad
        assert len(d_ext) == n + 2 * pad

    def test_circular_wrap(self):
        """Test that padding wraps from end to start."""
        n = 10
        pad = 2
        x = np.arange(n, dtype=float)
        y = np.arange(n, dtype=float)
        dist = np.arange(n, dtype=float)
        perimeter = n

        x_ext, y_ext, _ = apply_ring_padding(x, y, dist, pad, perimeter)

        # Start should have last 'pad' elements
        np.testing.assert_array_equal(x_ext[:pad], x[-pad:])
        # End should have first 'pad' elements
        np.testing.assert_array_equal(x_ext[-pad:], x[:pad])


class TestResampleAndSmooth:
    """Test cases for resample_and_smooth function."""

    def test_resampling_increases_points(self):
        """Test that resampling with small delta_s increases point count."""
        x = np.array([0, 10, 20])
        y = np.array([0, 5, 0])

        x_new, y_new = resample_and_smooth(x, y, delta_s=2.0)

        assert len(x_new) > len(x)

    def test_short_line_passthrough(self):
        """Test that very short lines are returned unchanged."""
        x = np.array([0, 1])
        y = np.array([0, 0])

        x_new, y_new = resample_and_smooth(x, y, delta_s=100.0)

        np.testing.assert_array_equal(x_new, x)
        np.testing.assert_array_equal(y_new, y)

    def test_returns_numpy_arrays(self):
        """Test that function returns numpy arrays."""
        x = np.linspace(0, 100, 20)
        y = np.sin(np.linspace(0, 2 * np.pi, 20))

        x_new, y_new = resample_and_smooth(x, y, delta_s=5.0)

        assert isinstance(x_new, np.ndarray)
        assert isinstance(y_new, np.ndarray)


class TestSnapEndpoints:
    """Test cases for snap_endpoints function."""

    def test_endpoint_snapping(self):
        """Test that endpoints are correctly snapped."""
        x = np.array([0.1, 1.0, 2.0, 2.9])
        y = np.array([0.1, 1.0, 2.0, 2.9])

        x_out, y_out = snap_endpoints(x, y, 0.0, 0.0, 3.0, 3.0)

        assert x_out[0] == 0.0
        assert y_out[0] == 0.0
        assert x_out[-1] == 3.0
        assert y_out[-1] == 3.0

    def test_interior_unchanged(self):
        """Test that interior points are unchanged."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])

        x_out, y_out = snap_endpoints(x, y, -1.0, -1.0, 4.0, 4.0)

        np.testing.assert_array_equal(x_out[1:-1], x[1:-1])
        np.testing.assert_array_equal(y_out[1:-1], y[1:-1])

    def test_returns_copies(self):
        """Test that function returns copies, not views."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])

        x_out, y_out = snap_endpoints(x, y, 0.0, 0.0, 2.0, 2.0)

        # Modifying output should not affect input
        x_out[1] = 999.0
        assert x[1] == 1.0
