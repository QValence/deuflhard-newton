"""
Levenberg-Marquardt solver for nonlinear least squares.

Minimises ½‖F(x)‖² when m > n or when the Jacobian is (near-)rank-deficient.
Unlike NLEQ-ERR/RES which seek a root F(x) = 0, this method converges to a
*local minimum* of the objective — the residual at the solution may be nonzero.

The implementation uses an affine-invariant trust-region formulation consistent
with Deuflhard's scaling philosophy:

  - Scaling matrix D = diag(‖J_i‖₂) (column norms of J), updated each iteration.
  - Step Δx solves the regularised normal equations:
        (JᵀJ + λ DᵀD) Δx = -Jᵀf
  - Trust-region parameter λ is updated based on the ratio of actual to
    predicted decrease (standard Moré 1978 update rule).
  - Convergence criterion: ‖Jᵀf‖_D ≤ tol  (gradient of ½‖f‖² is small).

License: MIT (see LICENSE file).
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

import numpy as np

from deuflhard_newton.result import SolveResult
from deuflhard_newton.nleq import IterationPrinter, ProblemType
from deuflhard_newton.jac import resolve_jacobian

Array = np.ndarray


# ---------------------------------------------------------------------------
# Display columns for LM
# ---------------------------------------------------------------------------

_COLS_LM = [
    ("iter",     "iter",      7),
    ("‖f‖",     "f_norm",   14),
    ("‖Jᵀf‖",  "grad_norm",14),
    ("‖Δx‖",   "dx_norm",  14),
    ("λ",       "lam",      14),
    ("ρ",       "rho",      14),
]


def lm(
    func: Callable[[Array], Array],
    x0: Array,
    jac: Callable[[Array], Array] | None = None,
    tol: float = 1e-6,
    max_iter: int = 1_000,
    lam0: float = 1e-3,
    scaling: Array | None = None,
    display: bool = True,
    *,
    callback: Callable | None = None,
) -> SolveResult:
    """
    Levenberg-Marquardt method for nonlinear least squares.

    Finds a local minimum of ½‖F(x)‖² by solving the regularised normal
    equations at each step.  Suitable when:

    - m >> n (many more equations than unknowns — true overdetermined system)
    - The Jacobian is rank-deficient or ill-conditioned
    - No exact root F(x) = 0 exists (data-fitting problems)

    For root-finding problems where F(x) = 0 has a solution, nleq_err or
    nleq_res will converge faster.

    Convergence criterion
    ---------------------
    Stops when ‖Jᵀf‖_D ≤ tol, where D = diag(column norms of J).
    This is the gradient of ½‖f‖² scaled by D — it is small when the
    current point is (approximately) a stationary point of the objective.

    Note: convergence does NOT guarantee F(x*) = 0.  Check ``result.fun``
    to assess the residual at the solution.

    Parameters
    ----------
    func:
        Function F: R^n → R^m.
    x0:
        Initial guess in R^n.
    jac:
        Callable ``jac(x) -> ndarray of shape (m, n)``.
    tol:
        Tolerance on the scaled gradient norm ‖Jᵀf‖_D.
        Typical values: 1e-6 (engineering), 1e-10 (scientific).
    max_iter:
        Maximum iterations.
    lam0:
        Initial Levenberg parameter λ₀.  Larger values make the first step
        more gradient-descent-like; smaller values trust the Gauss-Newton
        direction more.  Default 1e-3 is appropriate for most problems.
    scaling:
        Optional diagonal scaling vector for x, shape (n,).  Used only for
        the column-norm scaling D when this is provided; otherwise D is
        derived from the Jacobian column norms.
    display:
        Print live iteration table if True.
    callback:
        Optional ``callback(k, x, fx, dx)`` called after each accepted step.

    Returns
    -------
    SolveResult
        ``success=True`` when the gradient criterion is met.
        ``message`` indicates a minimum was found (not necessarily a root).
        ``history`` contains per-iteration arrays: ``f_norm``, ``grad_norm``,
        ``dx_norm``, ``lam`` (Levenberg parameter), ``rho`` (actual/predicted
        decrease ratio).
    """
    max_iter = int(max_iter)
    jac = resolve_jacobian(func, jac)
    n = int(x0.size)
    x = np.asarray(x0, dtype=float).ravel().copy()

    # Initial evaluation
    f = np.asarray(func(x), dtype=float).ravel()
    nfev = 1
    njev = 0
    m = int(f.size)

    lam = float(lam0)
    lam_factor_up   = 11.0   # increase λ on rejected step
    lam_factor_down = 0.1    # decrease λ on well-predicted step

    # Display
    info = (f"LM | n={n} | m={m} | tol={tol:.2e}")
    printer = IterationPrinter(_COLS_LM, info) if display else None

    # History
    J = np.asarray(jac(x), dtype=float); njev += 1
    g = J.T @ f   # gradient of ½‖f‖²

    # Column-norm scaling (D matrix diagonal)
    col_norms = np.linalg.norm(J, axis=0)
    D = np.maximum(col_norms, 1e-8)   # avoid zero scaling

    grad_norm = np.linalg.norm(g / D)

    hist: dict[str, list] = {
        "f_norm":    [float(np.linalg.norm(f))],
        "grad_norm": [float(grad_norm)],
        "dx_norm":   [np.nan],
        "lam":       [float(lam)],
        "rho":       [np.nan],
    }

    if printer:
        printer.print_row(0, f_norm=hist["f_norm"][0],
                          grad_norm=hist["grad_norm"][0], lam=lam)

    # Check initial convergence
    if grad_norm <= tol:
        msg = "Converged in 0 iteration(s) (initial point is already optimal)."
        if printer:
            printer.print_footer(msg, nfev, njev)
        return SolveResult(
            x=x.copy(), fun=f.copy(), success=True, message=msg,
            nit=0, nfev=nfev, njev=njev,
            history={k: np.array(v) for k, v in hist.items()},
            method="lm",
        )

    for k in range(max_iter):
        n_iter = k + 1

        # Solve regularised normal equations:  (JᵀJ + λ DᵀD) Δx = -Jᵀf
        A = J.T @ J + lam * np.diag(D ** 2)
        try:
            dx, _, _, _ = np.linalg.lstsq(A, -g, rcond=None)
        except np.linalg.LinAlgError:
            # Pathological case — increase regularisation and retry
            lam *= lam_factor_up
            continue

        # Predicted decrease:  L(0) - L(Δx) = -gᵀΔx - ½ΔxᵀJᵀJΔx
        # (quadratic model of ‖f‖²)
        pred = -g @ dx - 0.5 * dx @ (J.T @ (J @ dx))

        # Actual decrease
        x_new = x + dx
        f_new = np.asarray(func(x_new), dtype=float).ravel(); nfev += 1
        actual = 0.5 * (f @ f - f_new @ f_new)

        rho = actual / pred if abs(pred) > 0.0 else 0.0

        if rho > 0.0:
            # Accept step
            x[:] = x_new
            f[:] = f_new

            # Recompute Jacobian and gradient at new point
            J = np.asarray(jac(x), dtype=float); njev += 1
            g = J.T @ f

            # Update column-norm scaling (never decreases, as in Moré)
            col_norms_new = np.linalg.norm(J, axis=0)
            D = np.maximum(D, col_norms_new)

            grad_norm = np.linalg.norm(g / D)

            # Decrease λ if prediction was good
            if rho > 0.75:
                lam = max(lam * lam_factor_down, 1e-16)
        else:
            # Rejected step — increase regularisation
            lam *= lam_factor_up

        dx_norm = float(np.linalg.norm(dx))
        f_norm  = float(np.linalg.norm(f))

        hist["f_norm"].append(f_norm)
        hist["grad_norm"].append(float(grad_norm))
        hist["dx_norm"].append(dx_norm)
        hist["lam"].append(float(lam))
        hist["rho"].append(float(rho))

        if printer:
            printer.print_row(n_iter, f_norm=f_norm, grad_norm=grad_norm,
                              dx_norm=dx_norm, lam=lam, rho=rho)

        if callback and rho > 0.0:
            callback(n_iter, x.copy(), f.copy(), dx.copy())

        # Convergence test: gradient of objective is small
        if grad_norm <= tol:
            msg = (f"Converged to local minimum of ‖f‖² in {n_iter} "
                   f"iteration(s). Residual ‖f‖ = {f_norm:.3e}.")
            if printer:
                printer.print_footer(msg, nfev, njev)
            return SolveResult(
                x=x.copy(), fun=f.copy(), success=True, message=msg,
                nit=n_iter, nfev=nfev, njev=njev,
                history={k: np.array(v) for k, v in hist.items()},
                method="lm",
            )

        # λ ceiling: prevent numerical overflow
        if lam > 1e16:
            break

    msg = f"Maximum iterations ({max_iter}) reached without convergence."
    if printer:
        printer.print_footer(msg, nfev, njev)
    return SolveResult(
        x=x.copy(), fun=f.copy(), success=False, message=msg,
        nit=min(k + 1, max_iter), nfev=nfev, njev=njev,
        history={k: np.array(v) for k, v in hist.items()},
        method="lm",
    )
