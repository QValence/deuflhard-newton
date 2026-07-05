"""
Affine-invariant scaling for Newton-type solvers.

This code is an independent implementation based on the algorithmic descriptions found in:

    Deuflhard, P. (2011).
    Newton Methods for Nonlinear Problems.
    Springer Series in Computational Mathematics, Vol. 35.
    https://doi.org/10.1007/978-3-642-23899-4

License: MIT (see LICENSE file).
"""

from __future__ import annotations

import numpy as np


class Scale:
    """
    Adaptive affine-invariant scaling for a vector space of given dimension.

    Role in the solver
    ------------------
    The ``Scale`` object defines the *natural norm* used throughout the algorithm:

        ‖v‖_D = ‖D⁻¹ v‖₂,   D = diag(weights)

    Convergence is declared when this scaled norm falls below ``tol``.  Because
    the weights adapt to the magnitude of the iterates, convergence is measured
    in units that are *natural to the problem* — not in absolute Euclidean units.

    This is what makes the algorithm affine-invariant: a change of variable
    x → A x for any non-singular A does not change the convergence behaviour.

    Weight update rule
    ------------------
    After each accepted Newton step (x_old → x_new), the weights are updated:

        w_i = max(w₀_i, ½(|x_new_i| + |x_old_i|), ε)

    Three properties follow:
    - Weights never shrink below the initial value ``w₀`` (set by ``init_weight``
      and the user's ``scaling=`` argument in ``solve()``).
    - Weights track the *average magnitude* of the iterate, so variables that
      grow during the solve are rescaled accordingly.
    - The floor ``ε = 1e-30`` prevents division by zero for variables that are
      genuinely near zero throughout the solve.

    Effect on convergence target
    ----------------------------
    ``tol`` is interpreted in the scaled norm: the solver stops when
    ``‖v‖_D ≤ tol``, which in component form means roughly
    ``|v_i| ≤ tol · w_i`` for each i.

    Consequently:
    - **Poor ``init_weight``** (too small) makes the solver declare convergence
      too early for large-magnitude variables.
    - **Poor ``init_weight``** (too large) makes convergence harder to reach
      for small-magnitude variables and can steer the path to a different root.
    - **Recommended practice**: if the order of magnitude of the solution is
      known, pass ``scaling=np.abs(x0_estimate)`` to ``solve()``.  If unknown,
      the default (unit scaling) is conservative and correct.

    Parameters
    ----------
    dimension:
        Size of the vector space (n for x-space, m for f-space).
    tol:
        Convergence tolerance in the scaled norm ‖·‖_D.  Convergence is
        declared when ``‖v‖_D = ‖v/weights‖₂ ≤ tol``.  With unit weights
        this is the plain Euclidean norm; with adaptive weights it is a
        relative criterion in the natural scale of the problem.
    init_weight:
        Initial scaling weights.  If None, defaults to ones (unit scaling).
        If a scalar, broadcast to all components.
        If an array, must have shape ``(dimension,)``.
    """

    _epsilon: float = 1e-30

    def __init__(
        self,
        dimension: int,
        tol: float,
        init_weight: float | np.ndarray | None = None,
    ) -> None:
        self.dimension = dimension

        if init_weight is None:
            init_weight = np.ones(dimension, dtype=float)
        elif not isinstance(init_weight, np.ndarray):
            init_weight = np.full(dimension, float(init_weight), dtype=float)

        self.init_weights = init_weight.copy()
        self.weights = init_weight.copy()

        self.tol_unscaled = tol
        # Pure Deuflhard criterion: ‖Δx‖_D ≤ tol, no rescaling by max(w).
        self.tol = tol

        # Pre-computed inverse weights and buffer for allocation-free norm.
        self._inv_weights: np.ndarray = 1.0 / self.weights
        self._tmp: np.ndarray = np.empty(dimension, dtype=float)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return x scaled by current weights: x / weights."""
        return x / self.weights

    def update(self, x_new: np.ndarray, x_old: np.ndarray) -> None:
        """
        Update weights after an accepted Newton step (x_old → x_new).

        Weights grow to track the iterate magnitude but never fall below
        ``init_weights`` or ``_epsilon``.
        """
        self.weights[:] = np.maximum(
            self.init_weights,
            np.maximum(
                0.5 * (np.abs(x_new) + np.abs(x_old)),
                self._epsilon,
            ),
        )
        # self.tol is constant (= tol_unscaled); no recomputation needed.
        np.reciprocal(self.weights, out=self._inv_weights)

    def evaluate_norm(self, x: np.ndarray) -> float:
        """Return ‖x‖_D = ‖x / weights‖₂."""
        np.multiply(x, self._inv_weights, out=self._tmp)
        return float(np.sqrt(np.dot(self._tmp, self._tmp)))

    def evaluate_inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return ⟨x, y⟩_D = ⟨x/weights, y/weights⟩."""
        return float(np.dot(x * self._inv_weights, y * self._inv_weights))
