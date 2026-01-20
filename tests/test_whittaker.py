# -*- coding: utf-8 -*-
"""Tests for the Whittaker-Eilers smoother."""

import numpy as np
from sbanks_core.whittaker import WhittakerSmoother


class TestWhittakerSmoother:
    """Test cases for WhittakerSmoother class."""

    def test_basic_smoothing(self):
        """Test that smoother produces output of correct length."""
        n = 100
        y = np.random.randn(n)
        smoother = WhittakerSmoother(lmbda=1e4, order=2, data_length=n)
        y_smooth = smoother.smooth(y)
        assert len(y_smooth) == n

    def test_smoothing_reduces_noise(self):
        """Test that smoothing reduces noise in noisy sinusoid."""
        np.random.seed(123)  # Set seed for reproducibility
        n = 100
        x = np.linspace(0, 4 * np.pi, n)
        y_true = np.sin(x)
        y_noisy = y_true + 0.3 * np.random.randn(n)

        smoother = WhittakerSmoother(lmbda=1e3, order=2, data_length=n)
        y_smooth = np.array(smoother.smooth(y_noisy.tolist()))

        # Smoothed signal should be closer to true signal than noisy signal
        error_noisy = np.mean((y_noisy - y_true) ** 2)
        error_smooth = np.mean((y_smooth - y_true) ** 2)
        assert error_smooth < error_noisy

    def test_higher_lambda_smoother(self):
        """Test that higher lambda produces smoother results."""
        n = 100
        y = np.random.randn(n)

        smoother_low = WhittakerSmoother(lmbda=1e2, order=2, data_length=n)
        smoother_high = WhittakerSmoother(lmbda=1e6, order=2, data_length=n)

        y_low = np.array(smoother_low.smooth(y.tolist()))
        y_high = np.array(smoother_high.smooth(y.tolist()))

        # Higher lambda should produce smaller second differences (smoother)
        diff2_low = np.sum(np.diff(y_low, n=2) ** 2)
        diff2_high = np.sum(np.diff(y_high, n=2) ** 2)
        assert diff2_high < diff2_low

    def test_different_orders(self):
        """Test smoother with different derivative orders."""
        n = 50
        y = np.random.randn(n)

        for order in [1, 2, 3, 4]:
            smoother = WhittakerSmoother(lmbda=1e4, order=order, data_length=n)
            y_smooth = smoother.smooth(y.tolist())
            assert len(y_smooth) == n

    def test_non_uniform_spacing(self):
        """Test smoother with non-uniform x values."""
        n = 50
        x = np.cumsum(np.random.rand(n))  # Non-uniform spacing
        y = np.sin(x) + 0.1 * np.random.randn(n)

        smoother = WhittakerSmoother(
            lmbda=1e3, order=2, data_length=n, x_input=x.tolist()
        )
        y_smooth = smoother.smooth(y.tolist())
        assert len(y_smooth) == n

    def test_small_data(self):
        """Test smoother with small data arrays."""
        for n in [3, 5, 10]:
            y = np.random.randn(n)
            smoother = WhittakerSmoother(lmbda=1e2, order=2, data_length=n)
            y_smooth = smoother.smooth(y.tolist())
            assert len(y_smooth) == n

    def test_zero_lambda_identity(self):
        """Test that zero lambda returns approximately the original data."""
        n = 20
        y = np.random.randn(n)
        smoother = WhittakerSmoother(lmbda=0.0, order=2, data_length=n)
        y_smooth = np.array(smoother.smooth(y.tolist()))
        np.testing.assert_allclose(y_smooth, y, rtol=1e-10)

    def test_constant_signal_unchanged(self):
        """Test that constant signal remains constant after smoothing."""
        n = 50
        y = np.ones(n) * 5.0
        smoother = WhittakerSmoother(lmbda=1e4, order=2, data_length=n)
        y_smooth = np.array(smoother.smooth(y.tolist()))
        np.testing.assert_allclose(y_smooth, y, rtol=1e-10)
