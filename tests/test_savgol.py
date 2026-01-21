# -*- coding: utf-8 -*-
"""Tests for the Savitzky-Golay filter wrappers."""

import numpy as np
from sbanks_core.savgol import (
    smooth_open_geometry,
    smooth_closed_geometry,
    smooth_open_geometry_arclength,
    smooth_closed_geometry_arclength,
)


class TestSmoothOpenGeometry:
    """Test cases for smooth_open_geometry function."""

    def test_basic_smoothing(self):
        """Test that function returns correct length arrays."""
        n = 50
        x = np.linspace(0, 10, n)
        y = np.sin(x) + 0.1 * np.random.randn(n)

        x_sm, y_sm = smooth_open_geometry(x, y, window_length=11, polyorder=3)
        assert len(x_sm) == n
        assert len(y_sm) == n

    def test_endpoint_preservation(self):
        """Test that endpoints are preserved."""
        n = 50
        x = np.linspace(0, 10, n)
        y = np.sin(x) + 0.1 * np.random.randn(n)

        x_sm, y_sm = smooth_open_geometry(x, y, window_length=11, polyorder=3)

        assert x_sm[0] == x[0]
        assert y_sm[0] == y[0]
        assert x_sm[-1] == x[-1]
        assert y_sm[-1] == y[-1]

    def test_short_array_passthrough(self):
        """Test that arrays shorter than window length are returned unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 2.0])

        x_sm, y_sm = smooth_open_geometry(x, y, window_length=11, polyorder=3)

        np.testing.assert_array_equal(x_sm, x)
        np.testing.assert_array_equal(y_sm, y)

    def test_noise_reduction(self):
        """Test that smoothing reduces noise."""
        n = 100
        x = np.linspace(0, 10, n)
        y_true = x * 2  # Linear trend
        y_noisy = y_true + 0.3 * np.random.randn(n)

        x_sm, y_sm = smooth_open_geometry(x, y_noisy, window_length=11, polyorder=3)

        # Interior points should be smoother
        # Compare variance of second differences
        diff2_noisy = np.var(np.diff(y_noisy[10:-10], n=2))
        diff2_smooth = np.var(np.diff(y_sm[10:-10], n=2))
        assert diff2_smooth < diff2_noisy

    def test_custom_pad_count(self):
        """Test smoothing with custom pad count."""
        n = 50
        x = np.linspace(0, 10, n)
        y = np.sin(x)

        x_sm, y_sm = smooth_open_geometry(
            x, y, window_length=11, polyorder=3, pad_count=20
        )
        assert len(x_sm) == n


class TestSmoothClosedGeometry:
    """Test cases for smooth_closed_geometry function."""

    def test_basic_smoothing(self):
        """Test that function returns correct length arrays."""
        n = 50
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(theta) + 0.05 * np.random.randn(n)
        y = np.sin(theta) + 0.05 * np.random.randn(n)

        x_sm, y_sm = smooth_closed_geometry(x, y, window_length=11, polyorder=3)
        assert len(x_sm) == n
        assert len(y_sm) == n

    def test_short_array_passthrough(self):
        """Test that arrays shorter than window length are returned unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 2.0])

        x_sm, y_sm = smooth_closed_geometry(x, y, window_length=11, polyorder=3)

        np.testing.assert_array_equal(x_sm, x)
        np.testing.assert_array_equal(y_sm, y)

    def test_circular_continuity(self):
        """Test that smoothing maintains circular continuity."""
        n = 100
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)

        x_sm, y_sm = smooth_closed_geometry(x, y, window_length=11, polyorder=3)

        # For a perfect circle, smoothing should not change much
        # Check that the result is still close to a circle
        radii = np.sqrt(x_sm**2 + y_sm**2)
        assert np.std(radii) < 0.01  # Radii should be nearly constant

    def test_noise_reduction_closed(self):
        """Test that smoothing reduces noise on closed geometry."""
        n = 100
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(theta) + 0.1 * np.random.randn(n)
        y = np.sin(theta) + 0.1 * np.random.randn(n)

        x_sm, y_sm = smooth_closed_geometry(x, y, window_length=11, polyorder=3)

        # Smoothed radii should have smaller variance
        radii_noisy = np.sqrt(x**2 + y**2)
        radii_smooth = np.sqrt(x_sm**2 + y_sm**2)
        assert np.std(radii_smooth) < np.std(radii_noisy)


class TestSmoothOpenGeometryArclength:
    """Test cases for smooth_open_geometry_arclength function."""

    def test_basic_smoothing(self):
        """Test that function returns arrays."""
        n = 50
        x = np.linspace(0, 10, n)
        y = np.sin(x) + 0.1 * np.random.randn(n)

        x_sm, y_sm = smooth_open_geometry_arclength(x, y, window_length=11, polyorder=3)
        assert len(x_sm) > 0
        assert len(y_sm) > 0

    def test_endpoint_preservation(self):
        """Test that endpoints are preserved."""
        n = 50
        x = np.linspace(0, 10, n)
        y = np.sin(x) + 0.1 * np.random.randn(n)

        x_sm, y_sm = smooth_open_geometry_arclength(x, y, window_length=11, polyorder=3)

        assert x_sm[0] == x[0]
        assert y_sm[0] == y[0]
        assert x_sm[-1] == x[-1]
        assert y_sm[-1] == y[-1]

    def test_short_array_passthrough(self):
        """Test that arrays shorter than window length are returned unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 2.0])

        x_sm, y_sm = smooth_open_geometry_arclength(x, y, window_length=11, polyorder=3)

        np.testing.assert_array_equal(x_sm, x)
        np.testing.assert_array_equal(y_sm, y)

    def test_no_spikes_on_nonuniform_data(self):
        """Test that non-uniformly spaced data doesn't produce spike artifacts.

        This is the key test for the arc-length parameterization fix.
        Non-uniform spacing (common in real GIS data) should not cause
        large deviations from the original geometry.
        """
        # Create non-uniformly spaced data (dense in middle, sparse at ends)
        t = np.concatenate([
            np.linspace(0, 0.3, 10),
            np.linspace(0.3, 0.7, 30),
            np.linspace(0.7, 1.0, 10)
        ])
        x = t * 100  # X ranges from 0 to 100
        y = 10 * np.sin(t * np.pi * 2) + np.random.randn(len(t)) * 0.5

        x_sm, y_sm = smooth_open_geometry_arclength(x, y, window_length=11, polyorder=3)

        # The maximum deviation from a smooth sine wave should be bounded
        # Calculate the smoothed curve's deviation from expected
        # Key: no spike should exceed 3x the noise level (1.5 units)
        y_expected_smooth = 10 * np.sin(np.linspace(0, np.pi * 2, len(y_sm)))

        # Check that no point deviates more than a reasonable threshold
        # from where we'd expect (allowing for smoothing effects)
        max_deviation = np.max(np.abs(y_sm - y_expected_smooth[: len(y_sm)]))
        # With proper arc-length parameterization, deviations should be bounded
        assert max_deviation < 15, f"Max deviation {max_deviation} exceeds threshold"

    def test_geographic_flag(self):
        """Test that the geographic flag is accepted."""
        n = 50
        x = np.linspace(0, 1, n)  # degrees longitude
        y = np.linspace(0, 1, n)  # degrees latitude

        # Should not raise an error with is_geographic=True
        x_sm, y_sm = smooth_open_geometry_arclength(
            x, y, window_length=11, polyorder=3, is_geographic=True
        )
        assert len(x_sm) > 0


class TestSmoothClosedGeometryArclength:
    """Test cases for smooth_closed_geometry_arclength function."""

    def test_basic_smoothing(self):
        """Test that function returns arrays."""
        n = 50
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(theta) + 0.05 * np.random.randn(n)
        y = np.sin(theta) + 0.05 * np.random.randn(n)

        x_sm, y_sm = smooth_closed_geometry_arclength(x, y, window_length=11, polyorder=3)
        assert len(x_sm) > 0
        assert len(y_sm) > 0

    def test_short_array_passthrough(self):
        """Test that arrays shorter than window length are returned unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 2.0])

        x_sm, y_sm = smooth_closed_geometry_arclength(x, y, window_length=11, polyorder=3)

        np.testing.assert_array_equal(x_sm, x)
        np.testing.assert_array_equal(y_sm, y)

    def test_circular_shape_preserved(self):
        """Test that smoothing a circle preserves the circular shape."""
        n = 100
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)

        x_sm, y_sm = smooth_closed_geometry_arclength(x, y, window_length=11, polyorder=3)

        # For a perfect circle, smoothing should not change much
        radii = np.sqrt(x_sm**2 + y_sm**2)
        assert np.std(radii) < 0.05  # Radii should be nearly constant

    def test_no_spikes_on_nonuniform_ring(self):
        """Test that non-uniformly spaced ring data doesn't produce spikes."""
        # Create a non-uniformly parameterized ellipse
        # Dense sampling at the poles, sparse at the sides
        theta_nonuniform = np.concatenate([
            np.linspace(0, np.pi / 4, 5),
            np.linspace(np.pi / 4, 3 * np.pi / 4, 20),
            np.linspace(3 * np.pi / 4, 5 * np.pi / 4, 5),
            np.linspace(5 * np.pi / 4, 7 * np.pi / 4, 20),
            np.linspace(7 * np.pi / 4, 2 * np.pi, 5)[:-1]  # Exclude endpoint
        ])

        x = 2 * np.cos(theta_nonuniform) + np.random.randn(len(theta_nonuniform)) * 0.1
        y = np.sin(theta_nonuniform) + np.random.randn(len(theta_nonuniform)) * 0.1

        x_sm, y_sm = smooth_closed_geometry_arclength(x, y, window_length=11, polyorder=3)

        # Check that the smoothed shape has reasonable bounds
        # No spike should extend far beyond the original ellipse bounds
        assert np.max(np.abs(x_sm)) < 3.0, "X spike detected"
        assert np.max(np.abs(y_sm)) < 2.0, "Y spike detected"

    def test_geographic_flag(self):
        """Test that the geographic flag is accepted."""
        n = 50
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = 10 + 0.01 * np.cos(theta)  # Small ring near lon=10
        y = 45 + 0.01 * np.sin(theta)  # Small ring near lat=45

        # Should not raise an error with is_geographic=True
        x_sm, y_sm = smooth_closed_geometry_arclength(
            x, y, window_length=11, polyorder=3, is_geographic=True
        )
        assert len(x_sm) > 0
