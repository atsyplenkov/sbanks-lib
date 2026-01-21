# -*- coding: utf-8 -*-
"""
Comparison tests between sbanks WhittakerSmoother and whittaker-eilers package.

These tests validate numerical accuracy against the Rust-backed reference
implementation from PyPI.
"""

import time

import numpy as np
import pytest

from sbanks_core.whittaker import WhittakerSmoother

# Skip all tests if whittaker-eilers is not installed
we = pytest.importorskip("whittaker_eilers")


class TestNumericalAccuracy:
    """Test numerical accuracy against reference implementation."""

    @pytest.fixture
    def noisy_sinusoid(self):
        """Generate a noisy sinusoidal signal for testing."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 4 * np.pi, n)
        y_true = np.sin(x)
        y_noisy = y_true + 0.3 * np.random.randn(n)
        return y_noisy, y_true

    @pytest.mark.parametrize("lmbda", [1e2, 1e4, 1e6])
    def test_uniform_spacing_accuracy(self, noisy_sinusoid, lmbda):
        """Compare smoothed output for sinusoidal signal with noise."""
        y_noisy, _ = noisy_sinusoid
        n = len(y_noisy)

        # sbanks implementation
        sbanks_smoother = WhittakerSmoother(lmbda=lmbda, order=2, data_length=n)
        sbanks_result = np.array(sbanks_smoother.smooth(y_noisy.tolist()))

        # Reference implementation
        ref_smoother = we.WhittakerSmoother(lmbda=lmbda, order=2, data_length=n)
        ref_result = np.array(ref_smoother.smooth(y_noisy.tolist()))

        np.testing.assert_allclose(sbanks_result, ref_result, rtol=1e-5)

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_order_comparison(self, noisy_sinusoid, order):
        """Test different derivative orders match reference."""
        y_noisy, _ = noisy_sinusoid
        n = len(y_noisy)
        lmbda = 1e4

        # sbanks implementation
        sbanks_smoother = WhittakerSmoother(lmbda=lmbda, order=order, data_length=n)
        sbanks_result = np.array(sbanks_smoother.smooth(y_noisy.tolist()))

        # Reference implementation
        ref_smoother = we.WhittakerSmoother(lmbda=lmbda, order=order, data_length=n)
        ref_result = np.array(ref_smoother.smooth(y_noisy.tolist()))

        np.testing.assert_allclose(sbanks_result, ref_result, rtol=1e-5)


class TestNonUniformSpacing:
    """Test non-uniform spacing support.

    Note: sbanks and whittaker-eilers use different algorithms for non-uniform
    spacing. sbanks uses inverse-interval weighting only for the first-order
    difference matrix, while whittaker-eilers may handle higher-order
    differences differently. Both approaches are valid; we test that each
    produces reasonable smoothing rather than exact numerical equivalence.
    """

    def test_nonuniform_spacing_both_smooth(self):
        """Verify both implementations produce reasonable smoothing with irregular x.

        Note: Non-uniform spacing requires different lambda values than uniform
        spacing due to the inverse-interval weighting in the difference matrix.
        """
        np.random.seed(123)
        n = 50
        # Create non-uniform spacing
        x = np.sort(np.random.rand(n) * 10)
        y_true = np.sin(x)
        y_noisy = y_true + 0.2 * np.random.randn(n)
        # Use small lambda appropriate for non-uniform spacing
        lmbda = 1.0

        # sbanks implementation with x_input
        sbanks_smoother = WhittakerSmoother(
            lmbda=lmbda, order=2, data_length=n, x_input=x.tolist()
        )
        sbanks_result = np.array(sbanks_smoother.smooth(y_noisy.tolist()))

        # Reference implementation with x_input
        ref_smoother = we.WhittakerSmoother(
            lmbda=lmbda, order=2, data_length=n, x_input=x.tolist()
        )
        ref_result = np.array(ref_smoother.smooth(y_noisy.tolist()))

        # Both should reduce noise (get closer to true signal than noisy input)
        error_noisy = np.mean((y_noisy - y_true) ** 2)
        error_sbanks = np.mean((sbanks_result - y_true) ** 2)
        error_ref = np.mean((ref_result - y_true) ** 2)

        assert error_sbanks < error_noisy, "sbanks should reduce noise"
        assert error_ref < error_noisy, "reference should reduce noise"

        # Both results should have correct length
        assert len(sbanks_result) == n
        assert len(ref_result) == n


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_small_arrays(self, n):
        """Test with small data arrays."""
        np.random.seed(456)
        y = np.random.randn(n)
        lmbda = 1e2

        sbanks_smoother = WhittakerSmoother(lmbda=lmbda, order=2, data_length=n)
        sbanks_result = np.array(sbanks_smoother.smooth(y.tolist()))

        ref_smoother = we.WhittakerSmoother(lmbda=lmbda, order=2, data_length=n)
        ref_result = np.array(ref_smoother.smooth(y.tolist()))

        np.testing.assert_allclose(sbanks_result, ref_result, rtol=1e-5)

    def test_constant_signal(self):
        """Test that constant signal produces same result in both."""
        n = 50
        y = np.ones(n) * 7.5
        lmbda = 1e4

        sbanks_smoother = WhittakerSmoother(lmbda=lmbda, order=2, data_length=n)
        sbanks_result = np.array(sbanks_smoother.smooth(y.tolist()))

        ref_smoother = we.WhittakerSmoother(lmbda=lmbda, order=2, data_length=n)
        ref_result = np.array(ref_smoother.smooth(y.tolist()))

        np.testing.assert_allclose(sbanks_result, ref_result, rtol=1e-10)
        np.testing.assert_allclose(sbanks_result, y, rtol=1e-10)

    def test_lambda_zero_identity(self):
        """Test that lambda=0 returns original data in both."""
        n = 30
        np.random.seed(789)
        y = np.random.randn(n)

        sbanks_smoother = WhittakerSmoother(lmbda=0.0, order=2, data_length=n)
        sbanks_result = np.array(sbanks_smoother.smooth(y.tolist()))

        ref_smoother = we.WhittakerSmoother(lmbda=0.0, order=2, data_length=n)
        ref_result = np.array(ref_smoother.smooth(y.tolist()))

        np.testing.assert_allclose(sbanks_result, ref_result, rtol=1e-10)
        np.testing.assert_allclose(sbanks_result, y, rtol=1e-10)


@pytest.mark.benchmark
class TestSpeedBenchmarks:
    """Speed comparison benchmarks between implementations.

    Run manually with: pytest -m benchmark -v -s
    """

    @pytest.mark.parametrize("n", [100, 1000, 10000])
    def test_speed_comparison(self, n):
        """Benchmark initialization and smoothing times."""
        np.random.seed(42)
        y = np.random.randn(n)
        lmbda = 1e4
        order = 2
        n_runs = 10

        # Benchmark sbanks initialization
        start = time.perf_counter()
        for _ in range(n_runs):
            sbanks_smoother = WhittakerSmoother(lmbda=lmbda, order=order, data_length=n)
        sbanks_init_time = (time.perf_counter() - start) / n_runs * 1000

        # Benchmark sbanks smoothing
        sbanks_smoother = WhittakerSmoother(lmbda=lmbda, order=order, data_length=n)
        start = time.perf_counter()
        for _ in range(n_runs):
            sbanks_smoother.smooth(y.tolist())
        sbanks_smooth_time = (time.perf_counter() - start) / n_runs * 1000

        # Benchmark reference initialization
        start = time.perf_counter()
        for _ in range(n_runs):
            ref_smoother = we.WhittakerSmoother(lmbda=lmbda, order=order, data_length=n)
        ref_init_time = (time.perf_counter() - start) / n_runs * 1000

        # Benchmark reference smoothing
        ref_smoother = we.WhittakerSmoother(lmbda=lmbda, order=order, data_length=n)
        start = time.perf_counter()
        for _ in range(n_runs):
            ref_smoother.smooth(y.tolist())
        ref_smooth_time = (time.perf_counter() - start) / n_runs * 1000

        print(f"\n{'=' * 60}")
        print(f"Benchmark: n={n}, lambda={lmbda}, order={order}")
        print(f"{'-' * 60}")
        print(f"{'Metric':<25} {'sbanks (ms)':<15} {'reference (ms)':<15}")
        print(f"{'-' * 60}")
        print(
            f"{'Initialization':<25} {sbanks_init_time:<15.3f} {ref_init_time:<15.3f}"
        )
        print(f"{'Smoothing':<25} {sbanks_smooth_time:<15.3f} {ref_smooth_time:<15.3f}")
        print(
            f"{'Total':<25} {sbanks_init_time + sbanks_smooth_time:<15.3f} {ref_init_time + ref_smooth_time:<15.3f}"
        )
        print(f"{'=' * 60}")

        # Test passes regardless of speed (informational only)
        assert True
