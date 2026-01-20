# -*- coding: utf-8 -*-
"""
Whittaker-Eilers Smoother - Pure Python implementation using scipy.sparse.

This module provides a Whittaker-Eilers smoother that solves:
    (I + lambda * D'D) * y_smooth = y

where D is a finite difference matrix of the specified order.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class WhittakerSmoother:
    """
    Whittaker-Eilers smoother using sparse matrices.

    The Whittaker-Eilers smoother balances fidelity to the original data
    against smoothness of the result. It solves a penalized least squares
    problem where the penalty is based on finite differences.

    Parameters
    ----------
    lmbda : float
        Smoothing strength parameter. Higher values produce smoother results.
        Typical values range from 1e2 to 1e7.
    order : int
        Order of the difference penalty (1-4). Higher orders preserve more
        detail while still smoothing.
    data_length : int
        Length of the data to be smoothed.
    x_input : array-like, optional
        Non-uniform spacing values. If provided, the smoother uses
        inverse-interval weighting for the difference matrix.

    Examples
    --------
    >>> smoother = WhittakerSmoother(lmbda=1e4, order=2, data_length=100)
    >>> y_smooth = smoother.smooth(y)
    """

    def __init__(self, lmbda, order, data_length, x_input=None):
        self.lmbda = lmbda
        self.order = order
        self.data_length = data_length
        self.x_input = np.asarray(x_input) if x_input is not None else None
        self._coef_matrix = self._build_coefficient_matrix()

    def _build_difference_matrix(self):
        """
        Build the difference matrix D of the specified order.

        For uniform spacing, uses standard finite differences.
        For non-uniform spacing, weights by inverse intervals.

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse difference matrix
        """
        n = self.data_length
        d = self.order

        if n <= d:
            return sparse.eye(n, format="csc")

        if self.x_input is None:
            # Uniform spacing: standard finite differences
            e = np.ones(n)
            D = sparse.diags([e[:-1], -e[:-1]], [0, 1], shape=(n - 1, n), format="csc")
            for _ in range(1, d):
                m = D.shape[0]
                if m <= 1:
                    break
                D_next = sparse.diags(
                    [np.ones(m - 1), -np.ones(m - 1)],
                    [0, 1],
                    shape=(m - 1, m),
                    format="csc",
                )
                D = D_next @ D
            return D
        else:
            # Non-uniform spacing: weight by inverse intervals
            h = np.diff(self.x_input)
            h[h == 0] = 1e-10  # Avoid division by zero
            D = sparse.diags(
                [-1.0 / h, 1.0 / h], [0, 1], shape=(n - 1, n), format="csc"
            )
            for _ in range(1, d):
                m = D.shape[0]
                if m <= 1:
                    break
                D_next = sparse.diags(
                    [np.ones(m - 1), -np.ones(m - 1)],
                    [0, 1],
                    shape=(m - 1, m),
                    format="csc",
                )
                D = D_next @ D
            return D

    def _build_coefficient_matrix(self):
        """
        Build the coefficient matrix (I + lambda * D'D).

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse coefficient matrix for the linear system
        """
        n = self.data_length
        I = sparse.eye(n, format="csc")
        D = self._build_difference_matrix()
        return (I + self.lmbda * D.T @ D).tocsc()

    def smooth(self, y):
        """
        Apply the Whittaker-Eilers smoother to data.

        Parameters
        ----------
        y : array-like
            Input data to smooth

        Returns
        -------
        list
            Smoothed data as a list
        """
        y = np.asarray(y, dtype=np.float64)
        return spsolve(self._coef_matrix, y).tolist()
