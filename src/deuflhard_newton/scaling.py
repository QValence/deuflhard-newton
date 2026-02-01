"""
Implementation of global Newton-type methods for nonlinear problems.

This code is an independent implementation based on the algorithmic descriptions found in:

    Deuflhard, P. (2011).
    Newton Methods for Nonlinear Problems.
    Springer Series in Computational Mathematics, Vol. 35.
    https://doi.org/10.1007/978-3-642-23899-4

Important notes:
- This implementation is written from scratch.
- No text, pseudocode, or figures from the book are reproduced.
- The code reflects my interpretation of the underlying mathematical ideas.
- This software is not affiliated with or endorsed by the author or the publisher.

License:
- This code is licensed under the MIT License (see LICENSE file).
"""

from __future__ import annotations

import numpy as np


class Scale:
    """
    Class to handle the numerical stability of the algorithm through scaling.
    """
    _epsilon: float = 1e-30

    def __init__(self,
                 dimension: int,
                 tol: float,
                 init_weight: float | np.ndarray = None):
        """
        Constructor.

        Parameters
        ----------
        dimension:
            Dimension of the space to apply scaling to (x-space, f-space ...).
        tol:
            Tolerance to be achieved in the unscaling paradigm.
        init_weight:
            Initial values of the scaling coefficients.

        """
        self.dimension = dimension

        if init_weight is None:
            init_weight = np.ones(dimension, dtype=float)

        if not isinstance(init_weight, np.ndarray):
            init_weight = np.full(dimension, init_weight, dtype=float)

        self.init_weights = init_weight.copy()
        self.weights = init_weight.copy()

        self.tol_unscaled = tol
        self.tol = tol / np.max(self.weights)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Scales variable according to current scaling coefficients.
        """
        return x / self.weights

    def update(self, x_new: np.ndarray, x_old: np.ndarray) -> None:
        """
        Updates the scaling coefficients and the scaled tolerance based on the new solution.
        """
        self.weights[:] = np.maximum(
            self.init_weights,
            np.maximum(
                0.5 * (np.abs(x_new) + np.abs(x_old)),
                self._epsilon,
            ),
        )

        self.tol = self.tol_unscaled / np.max(self.weights)

    def evaluate_norm(self, x: np.ndarray) -> float:
        """
        Computes the norm of scaled variable "x".
        """
        x_scaled = x / self.weights
        return np.linalg.norm(x_scaled)

    def evaluate_inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the inner product of scaled variable "x" and "y".
        """
        x_scaled = x / self.weights
        y_scaled = y / self.weights
        return np.dot(x_scaled, y_scaled)
