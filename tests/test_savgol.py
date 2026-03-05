# -*- coding: utf-8 -*-
"""Tests for the Savitzky-Golay filter wrappers."""

import numpy as np
import pytest
from sbanks_core.geometry import (
    densify_geometry,
    resample_and_smooth,
    snap_endpoints,
)
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

    def test_random_input_is_reproducible_with_test_fixture(self):
        n = 20
        x = np.linspace(0, 5, n)
        y1 = np.sin(x) + 0.1 * np.random.randn(n)

        np.random.seed(12345)
        y2 = np.sin(x) + 0.1 * np.random.randn(n)

        np.testing.assert_array_equal(y1, y2)

    @pytest.mark.parametrize("window_length", [0, -3, 10, 11.5, "11"])
    def test_open_raises_for_invalid_window_length_values(self, window_length):
        x = np.linspace(0, 10, 21)
        y = np.sin(x)

        with pytest.raises(ValueError, match="window_length"):
            smooth_open_geometry(x, y, window_length=window_length, polyorder=3)

    @pytest.mark.parametrize("polyorder", [-1, 11, 1.5, "3"])
    def test_open_raises_for_invalid_polyorder_before_filter_invocation(
        self, polyorder, monkeypatch
    ):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError(
                "savgol_filter should not be called for invalid polyorder"
            )

        monkeypatch.setattr("sbanks_core.savgol.savgol_filter", _fail_if_called)
        x = np.linspace(0, 10, 21)
        y = np.sin(x)

        with pytest.raises(ValueError, match="polyorder"):
            smooth_open_geometry(x, y, window_length=11, polyorder=polyorder)

    @pytest.mark.parametrize("pad_count", [-1, 0, 1.5, "2", True])
    def test_open_raises_for_invalid_pad_count_values(self, pad_count):
        x = np.linspace(0, 10, 21)
        y = np.sin(x)

        with pytest.raises(ValueError, match="pad_count"):
            smooth_open_geometry(x, y, window_length=11, polyorder=3, pad_count=pad_count)

    def test_short_open_raises_for_invalid_params(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 0.0])

        with pytest.raises(ValueError, match="polyorder"):
            smooth_open_geometry(x, y, window_length=11, polyorder=11)

    def test_open_invalid_params_raise_before_densification(self, monkeypatch):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("densify_geometry should not run for invalid params")

        monkeypatch.setattr("sbanks_core.savgol.densify_geometry", _fail_if_called)

        x = np.array([0.0, 100.0])
        y = np.array([0.0, 0.0])

        with pytest.raises(ValueError, match="window_length"):
            smooth_open_geometry(x, y, window_length=4, polyorder=2, max_segment_length=10.0)

    def test_open_uneven_spacing_fixture_has_stable_output_bounds(self):
        x = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
        y = np.array([0.0, 0.5, 0.0, 0.0, 0.5, 0.0])

        x_sm, y_sm = smooth_open_geometry(x, y, window_length=5, polyorder=3)

        assert len(x_sm) == len(x)
        assert len(y_sm) == len(y)
        assert np.all(np.isfinite(x_sm))
        assert np.all(np.isfinite(y_sm))
        assert x_sm[0] == x[0]
        assert y_sm[0] == y[0]
        assert x_sm[-1] == x[-1]
        assert y_sm[-1] == y[-1]
        assert np.max(np.abs(y_sm)) < 2.0


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

    @pytest.mark.parametrize("window_length", [0, -3, 10, 11.5, "11"])
    def test_closed_raises_for_invalid_window_length_values(self, window_length):
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)

        with pytest.raises(ValueError, match="window_length"):
            smooth_closed_geometry(x, y, window_length=window_length, polyorder=3)

    @pytest.mark.parametrize("polyorder", [-1, 11, 1.5, "3"])
    def test_closed_raises_for_invalid_polyorder_before_filter_invocation(
        self, polyorder, monkeypatch
    ):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError(
                "savgol_filter should not be called for invalid polyorder"
            )

        monkeypatch.setattr("sbanks_core.savgol.savgol_filter", _fail_if_called)
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)

        with pytest.raises(ValueError, match="polyorder"):
            smooth_closed_geometry(x, y, window_length=11, polyorder=polyorder)

    def test_short_closed_raises_for_invalid_params(self):
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0])

        with pytest.raises(ValueError, match="polyorder"):
            smooth_closed_geometry(x, y, window_length=11, polyorder=11)

    def test_closed_invalid_params_raise_before_densification(self, monkeypatch):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("densify_geometry should not run for invalid params")

        monkeypatch.setattr("sbanks_core.savgol.densify_geometry", _fail_if_called)

        x = np.array([0.0, 100.0, 100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, 100.0])

        with pytest.raises(ValueError, match="window_length"):
            smooth_closed_geometry(
                x, y, window_length=4, polyorder=2, max_segment_length=10.0
            )


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


class TestPipelineIntegration:
    """Test cases for full open- and closed-geometry processing pipelines."""

    def test_open_pipeline_exact_window_boundary(self):
        """Test open pipeline on geometry that densifies to exact window length."""
        x = np.array([0.0, 4.0])
        y = np.array([0.0, 0.0])

        x_dense, y_dense = densify_geometry(x, y, max_segment_length=1.0)
        assert len(x_dense) == 5

        x_sm, y_sm = smooth_open_geometry(
            x_dense, y_dense, window_length=5, polyorder=2, max_segment_length=None
        )
        x_sm_raw, y_sm_raw = smooth_open_geometry(
            x, y, window_length=5, polyorder=2, max_segment_length=1.0
        )
        x_rs, y_rs = resample_and_smooth(x_sm, y_sm, delta_s=0.5)
        x_snap, y_snap = snap_endpoints(x_rs, y_rs, x[0], y[0], x[-1], y[-1])

        assert np.all(np.isfinite(x_dense))
        assert np.all(np.isfinite(y_dense))
        assert np.all(np.isfinite(x_sm))
        assert np.all(np.isfinite(y_sm))
        assert np.all(np.isfinite(x_sm_raw))
        assert np.all(np.isfinite(y_sm_raw))
        assert np.all(np.isfinite(x_rs))
        assert np.all(np.isfinite(y_rs))
        assert np.all(np.isfinite(x_snap))
        assert np.all(np.isfinite(y_snap))
        np.testing.assert_allclose(x_sm_raw, x_sm, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(y_sm_raw, y_sm, rtol=0.0, atol=1e-12)

        assert x_snap[0] == x[0]
        assert y_snap[0] == y[0]
        assert x_snap[-1] == x[-1]
        assert y_snap[-1] == y[-1]
        assert np.all(np.diff(x_snap) >= 0)
        assert np.max(np.abs(y_snap)) < 1e-10

    def test_closed_pipeline_rotation_invariance_at_boundary(self):
        """Test closed smoothing is rotation-invariant at the seam boundary."""
        x = np.array([0.0, 4.0, 4.2])
        y = np.array([0.0, 0.0, 0.2])

        x_dense, y_dense = densify_geometry(x, y, max_segment_length=1.5)
        assert len(x_dense) == 5

        x_sm, y_sm = smooth_closed_geometry(x_dense, y_dense, window_length=5, polyorder=2)

        x_roll = np.roll(x_dense, 1)
        y_roll = np.roll(y_dense, 1)
        x_sm_roll, y_sm_roll = smooth_closed_geometry(
            x_roll, y_roll, window_length=5, polyorder=2
        )

        x_sm_unroll = np.roll(x_sm_roll, -1)
        y_sm_unroll = np.roll(y_sm_roll, -1)
        np.testing.assert_allclose(x_sm_unroll, x_sm, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(y_sm_unroll, y_sm, rtol=1e-7, atol=1e-9)
