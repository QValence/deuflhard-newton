"""
Unified entry point for all deuflhard_newton solvers.

``solve()`` inspects the problem dimensions and dispatches to the most
appropriate algorithm following Deuflhard's recommendations.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from deuflhard_newton.nleq import nleq_err, nleq_res, ProblemType
from deuflhard_newton.lm import lm
from deuflhard_newton.jac import resolve_jacobian
from deuflhard_newton.result import SolveResult


# ---------------------------------------------------------------------------
# Problem-type string → enum
# ---------------------------------------------------------------------------

_PROBLEM_TYPE_MAP: dict[str, ProblemType] = {
    "linear":               ProblemType.LINEAR,
    "mildly_nonlinear":     ProblemType.MILDLY_NONLINEAR,
    "highly_nonlinear":     ProblemType.HIGHLY_NONLINEAR,
    "extremely_nonlinear":  ProblemType.EXTREMELY_NONLINEAR,
}


def _resolve_problem_type(pt) -> ProblemType:
    if isinstance(pt, ProblemType):
        return pt
    if isinstance(pt, int):
        return ProblemType(pt)
    if isinstance(pt, str):
        key = pt.lower().replace(" ", "_").replace("-", "_")
        if key in _PROBLEM_TYPE_MAP:
            return _PROBLEM_TYPE_MAP[key]
        raise ValueError(
            f"Unknown problem_type {pt!r}. "
            f"Valid strings: {list(_PROBLEM_TYPE_MAP)}"
        )
    raise TypeError(
        f"problem_type must be a ProblemType, int, or str, got {type(pt).__name__!r}."
    )


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------

_VALID_METHODS = ("nleq_err", "nleq_res", "lm", "auto")


def _pick_method(n: int, m: int) -> str:
    """
    Select the best algorithm for an (n, m) problem following Deuflhard's
    recommendations:

    - n = m  → NLEQ-ERR  (square, error criterion)
    - n < m ≤ 4n  → NLEQ-RES  (moderate overdetermination, residual criterion)
    - m > 4n  → LM  (heavy overdetermination / true least-squares regime)
    - m < n  → NLEQ-RES  (underdetermined; lstsq gives minimum-norm solution)

    The 4n threshold is a heuristic: beyond 4× overdetermination the normal
    equations are well-conditioned enough that Gauss-Newton + LM regularisation
    is more reliable than the full NLEQ-RES damping loop.
    """
    if n == m:
        return "nleq_err"
    if m > 4 * n:
        return "lm"
    return "nleq_res"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(
    func: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    jac: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    tol: float = 1e-6,
    method: str = "auto",
    problem_type: ProblemType | str = "mildly_nonlinear",
    max_iter: int = 100,
    scaling: np.ndarray | float | None = None,
    use_qn: bool = True,
    callback: Callable | None = None,
    display: bool = True,
) -> SolveResult:
    """
    Unified Newton-type solver — finds a root (or least-squares minimum) of
    ``func(x) = 0``.

    Dispatches automatically to the best algorithm for the problem shape:

    +------------------+-------------+-----------------------------------+
    | Shape            | Algorithm   | When                              |
    +------------------+-------------+-----------------------------------+
    | n = m (square)   | NLEQ-ERR    | n equations, n unknowns           |
    | n < m ≤ 4n       | NLEQ-RES    | moderately overdetermined          |
    | m > 4n           | LM          | heavily overdetermined / data fit  |
    | m < n            | NLEQ-RES    | underdetermined (min-norm soln)   |
    +------------------+-------------+-----------------------------------+

    All algorithms use Deuflhard's affine-invariant adaptive damping.
    NLEQ-ERR and NLEQ-RES include QNERR/QNRES quasi-Newton acceleration.

    Parameters
    ----------
    func:
        Function F: R^n → R^m.  Must accept a 1-D float numpy array and
        return a 1-D array of length m.
    x0:
        Initial guess in R^n.  Converted to float64 automatically.
    jac:
        Jacobian callable ``jac(x) -> ndarray of shape (m, n)``.
        If None (default), the Jacobian is computed automatically via
        complex-step differentiation (``csdiff`` package, machine-precision
        accuracy).  Supply an explicit ``jac`` when:

        - The function does not support complex arithmetic (external C code,
          functions using ``np.abs`` on intermediates, integer operations).
        - You have an analytical Jacobian and want maximum throughput.
        - You are exploiting Jacobian sparsity.

    tol:
        Convergence tolerance in the affine-invariant norm.

        For NLEQ-ERR: convergence when ``‖Δx‖_D ≤ tol``.
        For NLEQ-RES: convergence when ``‖f(x)‖_D ≤ tol``.
        For LM:       convergence when ``‖Jᵀf‖_D ≤ tol``.

        The scaling D adapts to the natural magnitude of the problem, so
        ``tol`` is effectively a relative tolerance with respect to the
        problem's own scale.  Typical values:

        - 1e-6  for engineering (6 significant digits)
        - 1e-10 for scientific  (10 significant digits)

    method:
        Which algorithm to use.  One of ``'nleq_err'``, ``'nleq_res'``,
        ``'lm'``, or ``'auto'`` (default).  With ``'auto'``, the algorithm
        is chosen based on the problem dimensions after one probe evaluation
        of ``func(x0)``.
    problem_type:
        Nonlinearity classification, controlling the initial damping factor
        and minimum threshold.  Accepts a ``ProblemType`` enum value or one
        of the strings: ``'linear'``, ``'mildly_nonlinear'`` (default),
        ``'highly_nonlinear'``, ``'extremely_nonlinear'``.

        Choose based on your expectation of the problem's curvature:
        - Start with ``'mildly_nonlinear'`` (λ₀ = 1).
        - Use ``'highly_nonlinear'`` (λ₀ = 1e-2) if the problem is known
          to have large curvature near x0.
        - Use ``'extremely_nonlinear'`` (λ₀ = 1e-4) as a last resort for
          very difficult problems.

        Does not apply to ``method='lm'``, which uses its own trust-region
        parameter.

    max_iter:
        Maximum number of outer Newton iterations.
    scaling:
        Lower bound on the scale of each component (x-space for NLEQ-ERR;
        f-space for NLEQ-RES).  Controls what the algorithm treats as
        «near zero».

        **This parameter matters for correctness, not just speed.**
        The adaptive weights satisfy ``w_i ≥ max(|x0_i|, scaling_i)``.
        Two consequences:
        - Components near zero in x0 but large in the solution can cause
          premature convergence declaration if ``scaling`` is not set.
        - Supplying ``scaling=np.abs(x0)`` (or an estimate of the solution
          magnitude) prevents this.

        If None, unit scaling (weights = 1) is used.
    use_qn:
        Enable QNERR/QNRES quasi-Newton Jacobian reuse (default True).
        Passed through to nleq_err/nleq_res only (not used by LM).
        See ``nleq_err`` documentation for the trade-off analysis.
    callback:
        Optional ``callback(k, x, fx, dx)`` called at the end of each
        accepted outer iteration.  Arguments:
        - ``k``:  iteration index (1-based integer)
        - ``x``:  current iterate (copy)
        - ``fx``: ``func(x)`` at current iterate (copy)
        - ``dx``: Newton correction that produced ``x`` (copy)
    display:
        Print a live iteration table to stdout during the solve.  Uses the
        same display infrastructure as calling the underlying algorithm
        directly.

    Returns
    -------
    SolveResult
        ``result.x``       — solution (or last accepted iterate on failure)
        ``result.fun``     — ``func(x)`` at solution
        ``result.success`` — True if converged
        ``result.message`` — human-readable termination reason
        ``result.nit``     — outer iterations performed
        ``result.nfev``    — total function evaluations
        ``result.njev``    — total Jacobian evaluations
        ``result.history`` — dict of per-iteration arrays (see SolveResult)
        ``result.method``  — algorithm used

    Raises
    ------
    ValueError:
        If ``method`` or ``problem_type`` is an unknown string.
    TypeError:
        If ``jac`` is not None or callable, or ``problem_type`` is wrong type.

    Examples
    --------
    >>> import numpy as np
    >>> from deuflhard_newton import solve
    >>> def f(x):
    ...     return np.array([x[0]**2 + x[1] - 1, x[0] - x[1]**2])
    >>> r = solve(f, np.array([0.5, 0.5]), display=False)
    >>> r.success
    True
    >>> np.allclose(r.fun, 0, atol=1e-6)
    True
    """
    # --- Input normalisation ---
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method {method!r}.  Valid options: {list(_VALID_METHODS)}"
        )

    x0 = np.asarray(x0, dtype=float).ravel()
    pt = _resolve_problem_type(problem_type)
    jac_fn = resolve_jacobian(func, jac)

    # --- Method selection and probe ---
    extra_fev = 0   # function evals added by solve() itself (probe etc.)

    if method == "auto":
        f0 = np.asarray(func(x0), dtype=float).ravel()
        extra_fev += 1
        n, m = x0.size, f0.size
        _method = _pick_method(n, m)
    else:
        _method = method

    # --- Tracking callback: capture last iterate for failure reporting ---
    # nleq_err/nleq_res return last accepted x on failure, so this is only
    # needed when a user callback is also present (to avoid wrapping logic).
    # We pass the tracking callback to the algorithm regardless so that
    # the user's callback (if any) is always invoked.
    _user_callback = callback

    def _wrapped_callback(k, x, fx, dx):
        if _user_callback is not None:
            _user_callback(k, x, fx, dx)

    # --- Dispatch ---
    common_kw = dict(
        tol=tol,
        max_iter=max_iter,
        display=display,
        callback=_wrapped_callback,
    )

    if _method == "nleq_err":
        result = nleq_err(
            func, x0, jac_fn,
            user_scaling=scaling,
            problem_type=pt,
            use_qn=use_qn,
            **common_kw,
        )
    elif _method == "nleq_res":
        result = nleq_res(
            func, x0, jac_fn,
            user_scaling=scaling,
            problem_type=pt,
            use_qn=use_qn,
            **common_kw,
        )
    else:  # lm
        result = lm(
            func, x0, jac_fn,
            lam0=1e-3,
            scaling=scaling,
            **common_kw,
        )

    # Adjust nfev to include solve()'s own probe evaluation
    if extra_fev:
        result.nfev += extra_fev

    return result
