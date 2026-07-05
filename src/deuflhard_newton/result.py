"""
Result type for all solvers in deuflhard_newton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SolveResult:
    """
    Result returned by solve(), nleq_err(), nleq_res(), and lm().

    Attributes
    ----------
    x:
        Solution array (last accepted iterate if not converged).
    fun:
        f(x) evaluated at x.
    success:
        True if the solver converged within the requested tolerance.
    message:
        Human-readable description of the termination condition.
    nit:
        Number of outer iterations performed.
    nfev:
        Total number of function evaluations (f calls).
    njev:
        Total number of Jacobian evaluations (jac calls).
    history:
        Dict of per-iteration arrays recorded during the solve.
        Keys depend on the method used:
          - 'f_norm'      : scaled norm of f(x) at each iteration
          - 'dx_norm'     : scaled norm of Newton correction dx
          - 'dx_bar_norm' : scaled norm of simplified Newton correction (nleq_err only)
          - 'lb'          : damping factor λ
          - 'mu'          : Lipschitz estimate μ
          - 'theta'       : contraction ratio θ = |dx_bar| / |dx|
        Index 0 is the initial evaluation (pre-iteration). All arrays are
        fully valid with no NaN padding.
    method:
        Name of the algorithm used: 'nleq_err', 'nleq_res', or 'lm'.
    """

    x: np.ndarray
    fun: np.ndarray
    success: bool
    message: str
    nit: int
    nfev: int
    njev: int
    history: dict
    method: str

    def __repr__(self) -> str:
        status = "converged" if self.success else "failed"
        return (
            f"SolveResult({status}, nit={self.nit}, nfev={self.nfev}, "
            f"njev={self.njev}, method='{self.method}')"
        )
