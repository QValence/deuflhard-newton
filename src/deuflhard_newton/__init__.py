"""
deuflhard_newton — Newton-type methods for nonlinear equation systems.

Implements Deuflhard's affine-invariant globalised Newton methods with
QNERR/QNRES quasi-Newton acceleration, plus Levenberg-Marquardt for
nonlinear least squares.

Quick start
-----------
    from deuflhard_newton import solve
    import numpy as np

    def f(x):
        return np.array([x[0]**2 + x[1] - 1, x[0] - x[1]**2])

    result = solve(f, np.array([0.5, 0.5]))
    # result.x  — solution
    # result.success — True if converged

Public API
----------
solve         — unified entry point; auto-selects algorithm from problem shape
nleq_err      — NLEQ-ERR: error-oriented Newton (Deuflhard §2.1 + §2.1.4)
nleq_res      — NLEQ-RES: residual-oriented Newton (Deuflhard §2.2 + §2.2.3)
lm            — Levenberg-Marquardt for nonlinear least squares
SolveResult   — result type returned by all solvers
ProblemType   — enum controlling initial damping (LINEAR → EXTREMELY_NONLINEAR)

Utility modules (importable)
-----------------------------
deuflhard_newton.jac      — resolve_jacobian() (auto-diff via csdiff)
deuflhard_newton.scaling  — Scale class (affine-invariant scaling)
deuflhard_newton.result   — SolveResult dataclass
deuflhard_newton.nleq     — nleq_err, nleq_res, IterationPrinter
deuflhard_newton.lm       — lm
deuflhard_newton.solve    — solve
"""

from deuflhard_newton.result import SolveResult
from deuflhard_newton.nleq import nleq_err, nleq_res, ProblemType
from deuflhard_newton.lm import lm
from deuflhard_newton.solve import solve

__all__ = [
    "solve",
    "nleq_err",
    "nleq_res",
    "lm",
    "SolveResult",
    "ProblemType",
]

__version__ = "0.2.0"
