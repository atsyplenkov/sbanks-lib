# -*- coding: utf-8 -*-
"""Tests for the Savitzky-Golay filter wrappers."""

import numpy as np
from sbanks_core.savgol import (
    smooth_open_geometry,
    smooth_closed_geometry,
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


class TestDensificationIntegration:
    """Test cases for max_segment_length parameter integration."""

    def test_open_geometry_with_densification(self):
        """Test that densification increases output point count for open geometry."""
        # Geometry with uneven spacing: short segments then a long one
        x = np.array([0.0, 1.0, 2.0, 3.0, 103.0])
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        x_sm, y_sm = smooth_open_geometry(
            x, y, window_length=5, polyorder=2, max_segment_length=30.0
        )

        # Output should have more points due to densification
        assert len(x_sm) > len(x)

    def test_closed_geometry_with_densification(self):
        """Test that densification increases output point count for closed geometry."""
        # Simple square with long segments
        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        x_sm, y_sm = smooth_closed_geometry(
            x, y, window_length=5, polyorder=2, max_segment_length=30.0
        )

        # Output should have more points due to densification
        assert len(x_sm) > len(x)

    def test_densification_none_preserves_original_count(self):
        """Test that None max_segment_length preserves original behavior."""
        n = 20
        x = np.linspace(0, 10, n)
        y = np.sin(x)

        x_sm, y_sm = smooth_open_geometry(
            x, y, window_length=5, polyorder=2, max_segment_length=None
        )

        assert len(x_sm) == n

    def test_densification_reduces_spikes_on_uneven_geometry(self):
        """Test that densification prevents overshoot artifacts."""
        # Create a staircase-like geometry typical of raster-derived polygons
        # with very uneven spacing: dense in corners, sparse in long edges
        x = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
        y = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0])

        # Smooth without densification
        x_no_dense, y_no_dense = smooth_open_geometry(
            x, y, window_length=5, polyorder=3
        )

        # Smooth with densification
        x_dense, y_dense = smooth_open_geometry(
            x, y, window_length=5, polyorder=3, max_segment_length=20.0
        )

        # The densified version should have a more controlled range
        # (less extreme y values from polynomial overshoot)
        y_range_no_dense = np.max(y_no_dense) - np.min(y_no_dense)
        y_range_dense = np.max(y_dense) - np.min(y_dense)

        # Densification should keep y values more bounded
        assert y_range_dense <= y_range_no_dense + 0.5

    def test_open_geometry_endpoints_preserved_with_densification(self):
        """Test that endpoints are still snapped after densification."""
        x = np.array([0.0, 50.0, 100.0])
        y = np.array([0.0, 10.0, 0.0])

        x_sm, y_sm = smooth_open_geometry(
            x, y, window_length=5, polyorder=2, max_segment_length=15.0
        )

        # Endpoints should match original
        assert x_sm[0] == x[0]
        assert y_sm[0] == y[0]
        assert x_sm[-1] == x[-1]
        assert y_sm[-1] == y[-1]

    def test_closed_geometry_short_with_densification(self):
        """Test that short geometries with densification are handled correctly."""
        # Very short geometry that becomes just enough after densification
        x = np.array([0.0, 100.0])
        y = np.array([0.0, 0.0])

        x_sm, y_sm = smooth_closed_geometry(
            x, y, window_length=5, polyorder=2, max_segment_length=20.0
        )

        # Should densify first, then be long enough to smooth
        assert len(x_sm) >= 5  # At least window_length points
