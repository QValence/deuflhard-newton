"""
Global Newton-type methods for nonlinear problems.

Implements:
  NLEQ-ERR  — error-oriented Newton with affine-invariant damping (Deuflhard §2.1)
               + QNERR quasi-Newton Jacobian reuse when θ < 0.5, λ = 1  (§2.1.4)
  NLEQ-RES  — residual-oriented Newton with affine-invariant damping (Deuflhard §2.2)
               + QNRES quasi-Newton Jacobian reuse when θ < 0.5, λ = 1  (§2.2.3)

The damping strategy is Deuflhard's «natural monotonicity test»: the step length
λ is reduced until ‖Δx̄‖_D / ‖Δx‖_D < 1 (NLEQ-ERR) or ‖f(x+λΔx)‖_D < ‖f(x)‖_D
(NLEQ-RES), forming an affine-invariant path through solution space.

This code is an independent implementation based on the algorithmic descriptions
found in:
    Deuflhard, P. (2011). Newton Methods for Nonlinear Problems.
    Springer Series in Computational Mathematics, Vol. 35.
    https://doi.org/10.1007/978-3-642-23899-4

No text, pseudocode, or figures from the book are reproduced.
License: MIT (see LICENSE file).
"""

from __future__ import annotations

import sys
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

from deuflhard_newton.result import SolveResult
from deuflhard_newton.scaling import Scale
from deuflhard_newton.jac import resolve_jacobian
from deuflhard_newton._numba_backend import (
    attempt_nleq_err as _attempt_nleq_err,
    attempt_nleq_res as _attempt_nleq_res,
    _HAVE_NUMBA as _HAVE_NUMBA_BACKEND,
)

# Optional scipy for LU factorisation reuse in the regularity loop.
# When available, nleq_err/nleq_res perform one LU decomposition of J per
# outer iteration and reuse it for all inner (damping) solves — relevant
# for highly/extremely nonlinear problems where the inner loop runs many times.
try:
    from scipy.linalg import lu_factor as _lu_factor, lu_solve as _lu_solve
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

Array = np.ndarray


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class ProblemType(IntEnum):
    """
    Classification of problem nonlinearity, following Deuflhard (2011) §2.1.

    Controls the initial damping factor λ₀ and minimum threshold λ_min used
    by nleq_err and nleq_res.

    +---------------------+------+-------+------------+
    | Level               |  λ₀  | λ_min | restricted |
    +---------------------+------+-------+------------+
    | LINEAR              |  1   | 1e-4  |   False    |
    | MILDLY_NONLINEAR    |  1   | 1e-4  |   False    |
    | HIGHLY_NONLINEAR    | 1e-2 | 1e-4  |   False    |
    | EXTREMELY_NONLINEAR | 1e-4 | 1e-8  |    True    |
    +---------------------+------+-------+------------+

    «Restricted» mode tightens the monotonicity test: it requires
    θ ≤ 1 - λ/4 instead of θ < 1.

    When in doubt, start with MILDLY_NONLINEAR and increase if the solver
    fails (lb_min reached without convergence).
    """
    LINEAR = 1
    MILDLY_NONLINEAR = 2
    HIGHLY_NONLINEAR = 3
    EXTREMELY_NONLINEAR = 4


# ---------------------------------------------------------------------------
# Internal helpers (public: swappable for v2 Jacobian-free Newton-Krylov)
# ---------------------------------------------------------------------------

def _linear_solve(J: Array, rhs: Array, square: bool) -> Array:
    """
    Solve J @ dx = rhs.

    For square systems, attempts direct solve via LU; falls back to the
    pseudoinverse (minimum-norm step) if J is singular.  The affine-invariant
    damping loop will then find λ s.t. the monotonicity test still holds.

    Architecture note: this is the single point of contact between the outer
    Newton iteration and the linear algebra layer.  For v2 (Jacobian-free
    Newton-Krylov), replace with a matrix-free GMRES call using directional
    derivatives from csdiff.
    """
    if square:
        try:
            return np.linalg.solve(J, rhs)
        except np.linalg.LinAlgError:
            import warnings
            dx, _, rank, _ = np.linalg.lstsq(J, rhs, rcond=None)
            warnings.warn(
                f"Jacobian is rank-deficient (rank {rank}/{J.shape[0]}). "
                "Using minimum-norm pseudoinverse step; the damping loop "
                "will attempt to find a valid λ. Consider a better starting "
                "point or check your function for structural singularity.",
                RuntimeWarning,
                stacklevel=3,
            )
            return dx
    dx, _, _, _ = np.linalg.lstsq(J, rhs, rcond=None)
    return dx


_LU_THRESHOLD = 50  # Use scipy LU only for n > this value.
# Rationale: for n ≤ ~50, scipy's Python wrapper overhead (~50 μs/call) exceeds
# the saving from not re-factorising (numpy.linalg.solve does LU+solve in one
# C dispatch costing ~8-20 μs for small n).  Measured crossover at n≈64.


def _make_reg_solver(J: Array, square: bool) -> Callable[[Array], Array]:
    """
    Return a solver ``f(rhs) -> dx`` for repeated linear solves with fixed J.

    For large n (> _LU_THRESHOLD): LU-factorise once so all inner damping
    loop solves share the factorisation — the LAPACK O(n³) work dominates.
    For small n: re-solve each time with numpy.linalg.solve (one C dispatch,
    LU+solve fused) — cheaper than scipy's wrapper overhead at n ≤ 50.
    Falls back gracefully when J is singular: lstsq per call.
    """
    if square and _HAVE_SCIPY and J.shape[0] > _LU_THRESHOLD:
        try:
            lu_piv = _lu_factor(J)
            return lambda rhs: _lu_solve(lu_piv, rhs)
        except np.linalg.LinAlgError:
            pass  # singular J: fall through to safe per-call solver
    if square:
        def _safe_solve(rhs: Array) -> Array:
            try:
                return np.linalg.solve(J, rhs)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(J, rhs, rcond=None)[0]
        return _safe_solve
    return lambda rhs: np.linalg.lstsq(J, rhs, rcond=None)[0]


def _damping_params(problem_type: ProblemType):
    """Return (lb_init, lb_min, restricted) for the given ProblemType."""
    if problem_type <= ProblemType.MILDLY_NONLINEAR:
        return 1.0, 1e-4, False
    if problem_type == ProblemType.HIGHLY_NONLINEAR:
        return 1e-2, 1e-4, False
    return 1e-4, 1e-8, True   # EXTREMELY_NONLINEAR


def _build_result(
    x: Array,
    f: Array,
    success: bool,
    msg: str,
    n_iter: int,
    nfev: int,
    njev: int,
    hist: dict,
    method: str,
) -> SolveResult:
    return SolveResult(
        x=x,
        fun=f,
        success=success,
        message=msg,
        nit=n_iter,
        nfev=nfev,
        njev=njev,
        history={k: np.array(v) for k, v in hist.items()},
        method=method,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# Column definitions: (display_header, history_key, column_width)
# Floating-point columns are formatted as centered scientific notation.
# The "iter" column is formatted as a centered integer.

_COLS_ERR = [
    ("iter",      "iter",        7),
    ("|f(x)|",   "f_norm",     14),
    ("|dx|",     "dx_norm",    14),
    ("|dx_bar|", "dx_bar_norm",14),
    ("lb",       "lb",         14),
    ("mu",       "mu",         14),
    ("theta",    "theta",      14),
]

_COLS_RES = [
    ("iter",     "iter",       7),
    ("|f(x)|",  "f_norm",    14),
    ("|dx|",    "dx_norm",   14),
    ("lb",      "lb",        14),
    ("mu",      "mu",        14),
    ("theta",   "theta",     14),
]


class IterationPrinter:
    """
    Prints a live-updating iteration table to stdout.

    Column widths and names are declared once; rows and separators are
    derived from that definition — no character counting anywhere.

    Parameters
    ----------
    columns:
        List of ``(display_header, key, width)`` triples.
    info:
        One-line string printed in the table header (method, tol, etc.).
    stride:
        Reprint the column-name row every ``stride`` data rows.
    """

    def __init__(self, columns: list, info: str, stride: int = 20) -> None:
        self._cols = columns
        self._info = info
        self._stride = stride
        self._total = sum(w for _, _, w in columns) + 3 * (len(columns) - 1)
        self._row_idx = 0

    # -- separators ----------------------------------------------------------

    def _eq(self) -> str:
        return "=" * self._total

    def _dash(self) -> str:
        return "-+-".join("-" * w for _, _, w in self._cols)

    # -- row formatting ------------------------------------------------------

    @staticmethod
    def _fmt(key: str, val, width: int) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return f"{'–':^{width}}"
        if key == "iter":
            return f"{int(val):^{width}d}"
        return f"{float(val):^{width}.6e}"

    def _fmt_row(self, values: dict) -> str:
        return " | ".join(
            self._fmt(key, values.get(key), w)
            for _, key, w in self._cols
        )

    def _col_headers(self) -> str:
        return " | ".join(f"{hdr:^{w}}" for hdr, _, w in self._cols)

    # -- public interface ----------------------------------------------------

    def print_header(self) -> None:
        """Print the full opening block (= line, info, = line, column headers, --- line)."""
        print(self._eq())
        print(f" {self._info}")
        print(self._eq())
        print(self._col_headers())
        print(self._dash())

    def print_row(self, iter_k: int, **values) -> None:
        """Print one data row.  Keyword names must match the column keys."""
        if self._row_idx == 0:
            self.print_header()
        elif self._row_idx % self._stride == 0:
            print(self._dash())
            print(self._col_headers())
            print(self._dash())

        print(self._fmt_row({"iter": iter_k, **values}))
        sys.stdout.flush()
        self._row_idx += 1

    def print_footer(self, message: str, nfev: int, njev: int) -> None:
        """Print the closing block after solver exit."""
        print(self._eq())
        print(f" {message} | nfev={nfev} | njev={njev}")
        print(self._eq())
        sys.stdout.flush()


def _make_printer(method: str, n: int, m: int, tol: float,
                  problem_type: ProblemType) -> IterationPrinter:
    cols = _COLS_ERR if method == "nleq_err" else _COLS_RES
    info = (f"{method.upper().replace('_', '-')} | n={n} | m={m} | "
            f"tol={tol:.2e} | {problem_type.name.lower()}")
    return IterationPrinter(cols, info)


def _display_from_history(
    method: str,
    n: int,
    tol: float,
    problem_type: "ProblemType",
    result: SolveResult,
) -> None:
    """Replay the iteration table from a completed solve's pre-collected history."""
    m  = result.fun.size
    pr = _make_printer(method, n, m, tol, problem_type)
    h  = result.history
    hf   = h.get("f_norm",      np.array([]))
    hdx  = h.get("dx_norm",     np.array([]))
    hdxb = h.get("dx_bar_norm", np.array([]))
    hlb  = h.get("lb",          np.array([]))
    hmu  = h.get("mu",          np.array([]))
    hth  = h.get("theta",       np.array([]))
    for k in range(len(hf)):
        kw: dict = {"f_norm": hf[k]}
        if k < len(hdx):  kw["dx_norm"]     = hdx[k]
        if k < len(hdxb): kw["dx_bar_norm"] = hdxb[k]
        if k < len(hlb):  kw["lb"]          = hlb[k]
        if k < len(hmu):  kw["mu"]          = hmu[k]
        if k < len(hth):  kw["theta"]       = hth[k]
        pr.print_row(k, **kw)
    pr.print_footer(result.message, result.nfev, result.njev)


def _print_inner_iter(
    inner_k: int,
    lb_tried: float,
    theta: float,
    threshold: float,
    outcome: str,
    lb_next: float | None = None,
    lb_min: float | None = None,
) -> None:
    """Print one inner-damping-loop status line when display='verbose'."""
    import math
    line = (
        f"  ↳[inner{inner_k:3d}] λ={lb_tried:.3e}  "
        f"θ={theta:.4f} vs thr={threshold:.4f}  {outcome}"
    )
    if outcome in ("REJECT", "JUMP") and lb_next is not None:
        line += f"  next={lb_next:.3e}"
        if outcome == "REJECT" and lb_min is not None and lb_tried > lb_min > 0.0:
            n_halvings = max(0, math.ceil(math.log2(lb_tried / lb_min)))
            line += f"  (≤{n_halvings} more f-evals)"
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Algorithmic bricks shared by NLEQ-ERR and NLEQ-RES
# ---------------------------------------------------------------------------

def _monotonicity_reject(
    theta: float,
    mu: float,
    lb: float,
    lb_min: float,
    restricted: bool,
    has_failed_once: bool,
) -> tuple[bool, float, bool]:
    """
    Natural monotonicity test (Deuflhard §2.1.3 / §2.1.5).

    Unrestricted: θ ≥ 1 means the simplified Newton step does not contract
    toward the root — the current candidate is rejected.
    Restricted (EXTREMELY_NONLINEAR): tighter condition θ > 1 − λ/4.

    When rejected, the next damping factor is the Lipschitz predictor μ
    capped at λ/2 (Deuflhard 2.1.9).  One «grace» step at lb_min is allowed
    before the regularity test declares genuine failure.

    Returns
    -------
    rejected : bool
    next_lb  : float   next damping factor to try (unchanged if not rejected)
    has_failed_once : bool
    """
    if (not restricted and theta >= 1.0) or (restricted and theta > 1.0 - lb / 4.0):
        # §2.1.9: next λ = min(μ_k, λ_k / 2)
        next_lb = min(mu, 0.5 * lb)
        if next_lb < lb_min and not has_failed_once:
            next_lb = lb_min      # one last chance at the minimum threshold
            has_failed_once = True
        return True, next_lb, has_failed_once
    return False, lb, has_failed_once


def _qn_safety_retry(
    jac_fn: Callable[[Array], Array],
    x: Array,
    f_x: Array,
    square: bool,
    scale_dx: "Scale",
    njev: int,
) -> tuple[Array, Array, float, Callable[[Array], Array], int]:
    """
    QNERR/QNRES safety net (Deuflhard §2.1.4).

    When the cached (simplified) Jacobian causes the damping factor to fall
    below lb_min, the Jacobian may simply be stale rather than the problem
    being genuinely intractable.  This function recomputes a fresh Jacobian,
    solves for a new Newton direction, and resets lb = 1 for one more attempt.

    Returns
    -------
    J_new, dx_new, norm_dx_new, reg_solve_new, njev
    """
    J_new = np.asarray(jac_fn(x), dtype=float)
    njev += 1
    dx_new = _linear_solve(J_new, -f_x, square)
    norm_dx_new = scale_dx.evaluate_norm(dx_new)
    reg_solve_new = _make_reg_solver(J_new, square)
    return J_new, dx_new, norm_dx_new, reg_solve_new, njev


# ---------------------------------------------------------------------------
# NLEQ-ERR  (error-oriented, Deuflhard §2.1 + §2.1.4)
# ---------------------------------------------------------------------------

def nleq_err(
    func: Callable[[Array], Array],
    x0: Array,
    jac: Callable[[Array], Array] | None = None,
    tol: float = 1e-3,
    max_iter: int = 1_000,
    user_scaling: float | Array | None = None,
    problem_type: ProblemType = ProblemType.MILDLY_NONLINEAR,
    display: bool | str = True,
    *,
    use_qn: bool = True,
    callback: Callable | None = None,
) -> SolveResult:
    """
    Global Newton method with error-oriented convergence criterion (NLEQ-ERR).

    Converges when the affine-invariant norm of the Newton correction satisfies

        ‖Δx‖_D ≤ tol,   D = diag(scaling weights)

    Includes QNERR quasi-Newton acceleration (Deuflhard §2.1.4): when a full
    Newton step is accepted (λ = 1) and the contraction ratio θ < 0.5, the
    Jacobian is frozen for the next iteration to avoid redundant factorisation.
    The cached Jacobian is discarded and recomputed if the damping factor
    subsequently falls below λ_min.

    Recommended for **square systems** (n = m).  For overdetermined systems
    (m > n) prefer nleq_res.

    Parameters
    ----------
    func:
        Function f: R^n → R^m to find the root of.  Must accept a 1-D
        numpy array and return a 1-D array of length m.
    x0:
        Initial guess in R^n.
    jac:
        Callable ``jac(x) -> ndarray of shape (m, n)``.  Use
        ``deuflhard_newton.jac.resolve_jacobian(func, None)`` to obtain an
        automatic Jacobian via complex-step differentiation.
    tol:
        Convergence tolerance in the affine-invariant norm.  Typical values:
        1e-6 (engineering), 1e-10 (scientific).  The criterion is
        ``‖Δx‖_D ≤ tol`` where D scales each component to its natural
        magnitude (see ``scaling``).
    max_iter:
        Maximum number of outer Newton iterations.
    user_scaling:
        Lower bound on the scale of each component of x.  The adaptive
        weights satisfy ``w_i ≥ max(|x0_i|, user_scaling_i)``.
        Correct scaling is crucial: it determines *what* the algorithm
        converges to, not just *how fast*.  Pass ``np.abs(x0)`` when the
        order of magnitude of the solution is known.  If None, unit scaling
        (weights = 1) is used.
    problem_type:
        Nonlinearity classification; controls λ₀ and λ_min.
        See ProblemType for details.
    display:
        Controls progress output.  ``False`` — silent.  ``True`` — print
        outer-iteration table (default).  ``'verbose'`` — also print one
        line per inner damping attempt showing λ, θ, the monotonicity
        threshold, outcome (REJECT / JUMP / ACCEPT), the next λ to try,
        and a worst-case f-eval budget for the remaining halvings.  Useful
        when diagnosing slow convergence or verifying that the solver is
        making progress rather than looping.
    use_qn:
        Enable QNERR quasi-Newton Jacobian reuse (default True).  When True,
        the Jacobian is frozen once θ < 0.5 and λ = 1, reducing the number
        of expensive Jacobian evaluations near the solution.  This improves
        total work when the Jacobian is costly (e.g. auto-diff: ~n f-evals).
        Pass ``use_qn=False`` for pure quadratic Newton convergence when the
        Jacobian is cheap (small n, analytical formula).
    callback:
        Optional ``callback(k, x, fx, dx)`` called at the end of each
        accepted outer iteration.  ``k`` is the iteration index (1-based),
        ``x`` and ``fx`` are the current iterate and residual, ``dx`` is the
        Newton correction that produced ``x``.

    Returns
    -------
    SolveResult
        See SolveResult for field documentation.  On failure, ``x`` is the
        last accepted iterate, not None.
    """
    max_iter = int(max_iter)

    # --- Numba fast path (requires: numba installed, square system, no callback,
    #     no verbose display — all checked inside attempt_nleq_err) ---
    if _HAVE_NUMBA_BACKEND and callback is None and display != 'verbose':
        _x0   = np.asarray(x0, dtype=float).ravel()
        _n    = int(_x0.size)
        _lb0, _lb_min, _restr = _damping_params(problem_type)
        _us = (np.ones(_n, dtype=float) if user_scaling is None
               else np.full(_n, float(user_scaling), dtype=float)
               if not isinstance(user_scaling, np.ndarray)
               else np.asarray(user_scaling, dtype=float))
        _r, _ok = _attempt_nleq_err(
            func, jac, _x0, _us,
            float(tol), _lb0, _lb_min, _restr, bool(use_qn), int(max_iter),
        )
        if _ok:
            if display:
                _display_from_history("nleq_err", _n, tol, problem_type, _r)
            return _r

    jac = resolve_jacobian(func, jac)
    n = int(x0.size)
    x_sol = np.asarray(x0, dtype=float).ravel().copy()

    # Initial function evaluation
    f_x = np.asarray(func(x_sol), dtype=float).ravel()
    nfev = 1
    njev = 0
    m = int(f_x.size)
    _square = (n == m)

    # Damping parameters
    lb, lb_min, restricted = _damping_params(problem_type)

    # Scaling
    if user_scaling is None:
        user_scaling = np.ones(n, dtype=float)
    elif not isinstance(user_scaling, np.ndarray):
        user_scaling = np.full(n, float(user_scaling), dtype=float)

    scale_x = Scale(n, tol, np.maximum(np.abs(x0), user_scaling))
    unit_f = Scale(m, tol)   # unit-weighted, for history monitoring only
    unit_x = Scale(n, tol)   # unit-weighted, for history monitoring only

    # History (growing lists → arrays at return)
    hist: dict[str, list] = {
        "f_norm":      [unit_f.evaluate_norm(f_x)],
        "dx_norm":     [np.nan],
        "dx_bar_norm": [np.nan],
        "lb":          [np.nan],
        "mu":          [np.nan],
        "theta":       [np.nan],
    }

    # Display
    _verbose = (display == 'verbose')
    printer = _make_printer("nleq_err", n, m, tol, problem_type) if display else None
    if printer:
        printer.print_row(0, f_norm=hist["f_norm"][0])

    # Working arrays (pre-allocated)
    x_prev = x_sol.copy()
    x_cand = np.empty(n)
    dx     = np.zeros(n)    # zeroed: used for Lipschitz at k=1 with no previous
    dx_bar = np.zeros(n)    # zeroed: same
    w      = np.empty(n)

    # QNERR state
    _simplified = False
    _J_cache: Array | None = None

    mu    = np.nan
    theta = np.nan

    for k in range(max_iter):
        n_iter = k + 1

        # Adaptive weight update: w_i = max(w₀_i, ½(|x_new| + |x_old|), ε)
        # (Deuflhard scaling rule; weights only grow, never shrink)
        scale_x.update(x_sol, x_prev)

        # Capture norms of last iteration's dx, dx_bar BEFORE overwriting them.
        # These are the Δx_{k-1} and Δx̄_{k-1} needed by the Lipschitz estimate.
        if k > 0:
            norm_dx_prev     = scale_x.evaluate_norm(dx)
            norm_dx_bar_prev = scale_x.evaluate_norm(dx_bar)

        # QNERR (§2.1.4): reuse cached J in simplified (chord) Newton mode
        if _simplified and _J_cache is not None:
            J_x = _J_cache     # saves one expensive Jacobian evaluation
        else:
            J_x = np.asarray(jac(x_sol), dtype=float)
            njev += 1
            _J_cache = J_x
            _simplified = False   # will be re-set after the damping loop if eligible

        # Newton direction: Δx_k = -J(x_k)⁻¹ f(x_k)  (affine-invariant formulation)
        dx[:] = _linear_solve(J_x, -f_x, _square)

        # --- Convergence test (NLEQ-ERR error criterion, §2.1) ---
        # Stop when ‖Δx_k‖_D ≤ tol  (Newton step small in natural units)
        norm_dx = scale_x.evaluate_norm(dx)
        if norm_dx <= scale_x.tol:
            x_sol += dx
            f_x = np.asarray(func(x_sol), dtype=float).ravel(); nfev += 1
            hist["f_norm"].append(unit_f.evaluate_norm(f_x))
            hist["dx_norm"].append(unit_x.evaluate_norm(dx))
            hist["dx_bar_norm"].append(np.nan)
            hist["lb"].append(np.nan); hist["mu"].append(np.nan); hist["theta"].append(np.nan)
            msg = f"Converged in {n_iter} iteration(s)."
            if printer:
                printer.print_row(n_iter, f_norm=hist["f_norm"][-1],
                                  dx_norm=hist["dx_norm"][-1])
                printer.print_footer(msg, nfev, njev)
            if callback:
                callback(n_iter, x_sol.copy(), f_x.copy(), dx.copy())
            return _build_result(x_sol.copy(), f_x.copy(), True, msg,
                                  n_iter, nfev, njev, hist, "nleq_err")

        # Lipschitz predictor for λ (§2.1.6): predict the next useful λ
        # from Δx̄_{k-1} and Δx_k via the surrogate w = Δx̄_{k-1} − Δx_k.
        # μ_k = λ_{k-1} · ‖Δx_{k-1}‖ · ‖Δx̄_{k-1}‖ / (‖w‖ · ‖Δx_k‖)
        if k > 0:
            w[:] = dx_bar - dx
            s = scale_x.evaluate_norm(w)
            if s > 0.0:
                mu_lip = lb * (norm_dx_prev * norm_dx_bar_prev) / (s * norm_dx)
                lb = min(1.0, mu_lip)  # §2.1.6: λ_k = min(1, μ_k)

        # LU-factor J once; all inner damping loop solves reuse this factorisation.
        _reg_solve = _make_reg_solver(J_x, _square)

        # --- Regularity (inner damping) loop — FINITE TERMINATION PROOF ---
        # _lb_rejected_max is non-decreasing.  The Lipschitz jump fires only
        # when lb_new > _lb_rejected_max.  Since lb_new ≤ 1.0:
        #   (a) After ANY rejection of a jumped-to λ, _lb_rejected_max ≥ that λ,
        #       so the same λ-range cannot be jumped to again.
        #   (b) Each jump multiplies lb by >4× (condition lb_new > 4×lb), so at
        #       most ⌈log₄(1/lb_min)⌉ ≈ 13 jumps before lb reaches 1.0.
        #   (c) Between/after jumps, lb halves monotonically: at most
        #       ⌈log₂(1/lb_min)⌉ ≤ 27 halvings before lb < lb_min → exit.
        #   (d) QNERR retry fires at most once; the same argument applies to the
        #       retry phase.
        # BOUND: ≤ 2 × (13 + 27 + 1) ≈ 82 inner iterations per outer iteration.
        has_failed_once  = False
        _qn_retried      = False
        mu    = np.nan
        theta = np.nan
        _lb_rejected_max = 0.0   # highest λ evaluated and rejected — prevents cycling
        _inner_k         = 0     # counts inner iterations for verbose display

        while True:

            # Regularity test: has λ fallen below the minimum permitted?
            if lb < lb_min:
                if _simplified and not _qn_retried:
                    # QNERR safety net: stale Jacobian may have caused the failure.
                    # Recompute J, reset λ=1, and retry once before giving up.
                    if _verbose:
                        print("  ↳[QNERR retry] stale Jacobian → "
                              "recomputing J, resetting λ=1", flush=True)
                    _simplified = False
                    J_x, dx[:], norm_dx, _reg_solve, njev = _qn_safety_retry(
                        jac, x_sol, f_x, _square, scale_x, njev)
                    _J_cache = J_x
                    lb = 1.0
                    has_failed_once = False
                    _lb_rejected_max = 0.0
                    _qn_retried = True
                    continue

                # Genuine convergence failure
                msg = (f"Convergence failure: damping factor fell below "
                       f"minimum threshold (lb_min={lb_min:.1e}).")
                hist["f_norm"].append(unit_f.evaluate_norm(f_x))
                hist["dx_norm"].append(unit_x.evaluate_norm(dx))
                hist["dx_bar_norm"].append(np.nan)
                hist["lb"].append(lb); hist["mu"].append(mu); hist["theta"].append(theta)
                if printer:
                    printer.print_row(n_iter, f_norm=hist["f_norm"][-1],
                                      dx_norm=hist["dx_norm"][-1],
                                      lb=lb, mu=mu, theta=theta)
                    printer.print_footer(msg, nfev, njev)
                return _build_result(x_sol.copy(), f_x.copy(), False, msg,
                                      n_iter, nfev, njev, hist, "nleq_err")

            # Evaluate candidate: x_cand = x + λ Δx
            x_cand[:] = x_sol + lb * dx
            f_cand = np.asarray(func(x_cand), dtype=float).ravel(); nfev += 1

            # Simplified Newton step from candidate: Δx̄ = -J⁻¹ f(x_cand)
            # (reuses the LU factorisation — no new J-eval)
            dx_bar[:] = _reg_solve(-f_cand)
            norm_dx_bar = scale_x.evaluate_norm(dx_bar)

            # Natural contraction ratio θ = ‖Δx̄‖_D / ‖Δx‖_D  (§2.1.2)
            # θ < 1 certifies contraction toward the root.
            theta = norm_dx_bar / norm_dx

            # Lipschitz predictor μ for next λ  (§2.1.6, based on current candidate)
            # w = Δx̄ − (1 − λ) Δx  plays role of the Lipschitz surrogate
            w[:] = dx_bar - (1.0 - lb) * dx
            denom = scale_x.evaluate_norm(w)
            mu = (0.5 * norm_dx * lb * lb) / denom if denom > 0.0 else np.inf

            # Save lb before it is updated by the rejection test
            lb_evaluated = lb
            _thr = 1.0 - lb_evaluated / 4.0 if restricted else 1.0

            # Natural monotonicity test + damping update
            rejected, lb, has_failed_once = _monotonicity_reject(
                theta, mu, lb, lb_min, restricted, has_failed_once)
            if rejected:
                _lb_rejected_max = max(_lb_rejected_max, lb_evaluated)
                if _verbose:
                    _print_inner_iter(_inner_k + 1, lb_evaluated, theta, _thr,
                                      "REJECT", lb_next=lb, lb_min=lb_min)
                _inner_k += 1
                continue

            # Candidate accepted; damping proposal for next iteration
            lb_new = min(1.0, mu)

            if lb == 1.0 and lb_new == 1.0:
                # Two-step Newton convergence check (§2.1, after a full step)
                # If ‖Δx̄‖_D ≤ tol, the corrected point x_cand + Δx̄ is already converged.
                if norm_dx_bar <= scale_x.tol:
                    if _verbose:
                        _print_inner_iter(_inner_k + 1, lb_evaluated, theta, _thr, "ACCEPT")
                    x_sol[:] = x_cand + dx_bar
                    f_x = np.asarray(func(x_sol), dtype=float).ravel(); nfev += 1
                    hist["f_norm"].append(unit_f.evaluate_norm(f_x))
                    hist["dx_norm"].append(unit_x.evaluate_norm(dx))
                    hist["dx_bar_norm"].append(unit_x.evaluate_norm(dx_bar))
                    hist["lb"].append(lb); hist["mu"].append(mu); hist["theta"].append(theta)
                    msg = f"Converged in {n_iter} iteration(s)."
                    if printer:
                        printer.print_row(n_iter, f_norm=hist["f_norm"][-1],
                                          dx_norm=hist["dx_norm"][-1],
                                          dx_bar_norm=hist["dx_bar_norm"][-1],
                                          lb=lb, mu=mu, theta=theta)
                        printer.print_footer(msg, nfev, njev)
                    if callback:
                        callback(n_iter, x_sol.copy(), f_x.copy(), dx.copy())
                    return _build_result(x_sol.copy(), f_x.copy(), True, msg,
                                          n_iter, nfev, njev, hist, "nleq_err")

            elif lb_new > 4.0 * lb and lb_new > _lb_rejected_max:
                # Lipschitz predictor suggests a much better untried λ; try it
                # first without accepting the current candidate.
                # Guard: jump only if lb_new exceeds all previously rejected λ,
                # preventing cycles (small lb → jump → reject → halve to small lb).
                if _verbose:
                    _print_inner_iter(_inner_k + 1, lb_evaluated, theta, _thr,
                                      "JUMP", lb_next=lb_new, lb_min=lb_min)
                _inner_k += 1
                lb = lb_new
                continue

            # Accept this candidate; advance iteration.
            # lb carries forward unchanged — it is the accepted λ used by the
            # Lipschitz predictor at the start of the next outer iteration.
            if _verbose:
                _print_inner_iter(_inner_k + 1, lb_evaluated, theta, _thr, "ACCEPT")
            x_prev[:] = x_sol
            x_sol[:] = x_cand
            f_x[:] = f_cand
            break

        # --- Post-iteration bookkeeping ---

        # §2.1.4: enter simplified (chord) Newton when θ < 0.5 and λ=1
        _simplified = use_qn and (lb == 1.0 and theta < 0.5)

        hist["f_norm"].append(unit_f.evaluate_norm(f_x))
        hist["dx_norm"].append(unit_x.evaluate_norm(dx))
        hist["dx_bar_norm"].append(unit_x.evaluate_norm(dx_bar))
        hist["lb"].append(lb); hist["mu"].append(mu); hist["theta"].append(theta)

        if printer:
            printer.print_row(n_iter,
                              f_norm=hist["f_norm"][-1],
                              dx_norm=hist["dx_norm"][-1],
                              dx_bar_norm=hist["dx_bar_norm"][-1],
                              lb=lb, mu=mu, theta=theta)

        if callback:
            callback(n_iter, x_sol.copy(), f_x.copy(), dx.copy())

    # Max iterations reached
    msg = f"Maximum iterations ({max_iter}) reached without convergence."
    if printer:
        printer.print_footer(msg, nfev, njev)
    return _build_result(x_sol.copy(), f_x.copy(), False, msg,
                          max_iter, nfev, njev, hist, "nleq_err")


# ---------------------------------------------------------------------------
# NLEQ-RES  (residual-oriented, Deuflhard §2.2 + §2.2.3)
# ---------------------------------------------------------------------------

def nleq_res(
    func: Callable[[Array], Array],
    x0: Array,
    jac: Callable[[Array], Array] | None = None,
    tol: float = 1e-3,
    max_iter: int = 1_000,
    user_scaling: float | Array | None = None,
    problem_type: ProblemType = ProblemType.MILDLY_NONLINEAR,
    display: bool | str = True,
    *,
    use_qn: bool = True,
    callback: Callable | None = None,
) -> SolveResult:
    """
    Global Newton method with residual-oriented convergence criterion (NLEQ-RES).

    Converges when the affine-invariant norm of the residual satisfies

        ‖f(x)‖_D ≤ tol,   D = diag(scaling weights)

    Includes QNRES quasi-Newton acceleration (Deuflhard §2.2.3): when a full
    Newton step is accepted (λ = 1) and the contraction ratio θ < 0.5, the
    Jacobian is frozen for the next iteration.  The cached Jacobian is
    discarded and recomputed if the damping subsequently fails.

    Recommended for **overdetermined systems** (m ≥ n).  For square systems
    (m = n) nleq_err is preferred (stronger convergence criterion).

    Parameters
    ----------
    func:
        Function f: R^n → R^m.
    x0:
        Initial guess in R^n.
    jac:
        Callable ``jac(x) -> ndarray of shape (m, n)``.
    tol:
        Convergence tolerance.  The criterion is ``‖f(x)‖_D ≤ tol``.
        Typical values: 1e-6 (engineering), 1e-10 (scientific).
    max_iter:
        Maximum outer iterations.
    user_scaling:
        Lower bound on the scale of each component of f(x).  Scales the
        residual norm — determines what residual magnitude counts as «small».
        Pass ``np.abs(func(x0))`` when the initial residual is representative
        of the problem's natural scale.
    problem_type:
        Nonlinearity classification.  See ProblemType.
    display:
        Controls progress output.  ``False`` — silent.  ``True`` — print
        outer-iteration table (default).  ``'verbose'`` — also print one
        line per inner damping attempt (same format as nleq_err).
    use_qn:
        Enable QNRES quasi-Newton Jacobian reuse (default True).  Same
        trade-off as in nleq_err: saves Jacobian evaluations near the
        solution at the cost of linear vs quadratic convergence rate.
    callback:
        Optional ``callback(k, x, fx, dx)`` called after each accepted step.

    Returns
    -------
    SolveResult
    """
    max_iter = int(max_iter)

    # --- Numba fast path ---
    if _HAVE_NUMBA_BACKEND and callback is None and display != 'verbose':
        _x0   = np.asarray(x0, dtype=float).ravel()
        _n    = int(_x0.size)
        _lb0, _lb_min, _restr = _damping_params(problem_type)
        _r, _ok = _attempt_nleq_res(
            func, jac, _x0, user_scaling,
            float(tol), _lb0, _lb_min, _restr, bool(use_qn), int(max_iter),
        )
        if _ok:
            if display:
                _display_from_history("nleq_res", _n, tol, problem_type, _r)
            return _r

    jac = resolve_jacobian(func, jac)
    n = int(x0.size)
    x_sol = np.asarray(x0, dtype=float).ravel().copy()

    # Initial evaluation
    f_x = np.asarray(func(x_sol), dtype=float).ravel()
    nfev = 1
    njev = 0
    m = int(f_x.size)
    _square = (n == m)

    # Damping parameters
    lb, lb_min, restricted = _damping_params(problem_type)

    # Scaling (f-space adaptive)
    if user_scaling is None:
        user_scaling = np.ones(m, dtype=float)
    elif not isinstance(user_scaling, np.ndarray):
        user_scaling = np.full(m, float(user_scaling), dtype=float)

    scale_f = Scale(m, tol, np.maximum(np.abs(f_x), user_scaling))
    unit_f  = Scale(m, tol)
    unit_x  = Scale(n, tol)

    # History
    norm_f = scale_f.evaluate_norm(f_x)
    hist: dict[str, list] = {
        "f_norm":  [unit_f.evaluate_norm(f_x)],
        "dx_norm": [np.nan],
        "lb":      [np.nan],
        "mu":      [np.nan],
        "theta":   [np.nan],
    }

    # Convergence check on initial residual
    if norm_f <= scale_f.tol:
        msg = "Converged in 0 iteration(s)."
        if display:
            pr = _make_printer("nleq_res", n, m, tol, problem_type)
            pr.print_row(0, f_norm=hist["f_norm"][0])
            pr.print_footer(msg, nfev, njev)
        return _build_result(x_sol.copy(), f_x.copy(), True, msg,
                              0, nfev, njev, hist, "nleq_res")

    # Display
    _verbose = (display == 'verbose')
    printer = _make_printer("nleq_res", n, m, tol, problem_type) if display else None
    if printer:
        printer.print_row(0, f_norm=hist["f_norm"][0])

    # Working arrays
    f_prev = f_x.copy()
    x_cand = np.empty(n)
    dx     = np.zeros(n)

    # QNRES state
    _simplified = False
    _J_cache: Array | None = None

    norm_f_prev = norm_f
    mu    = np.nan
    theta = np.nan

    for k in range(max_iter):
        n_iter = k + 1

        # QNRES (§2.2.3): reuse cached J in simplified (chord) Newton mode
        if _simplified and _J_cache is not None:
            J_x = _J_cache
        else:
            J_x = np.asarray(jac(x_sol), dtype=float)
            njev += 1
            _J_cache = J_x
            _simplified = False

        # Newton direction: Δx_k = -J(x_k)⁺ f(x_k)  (pseudoinverse for m ≥ n)
        dx[:] = _linear_solve(J_x, -f_x, _square)

        # Lipschitz predictor for NLEQ-RES (§2.2.2):
        # μ_k = μ_{k-1} · ‖f_{k-1}‖_D / ‖f_k‖_D  (residual-norm ratio update)
        if k > 0 and not np.isnan(mu):
            mu_lip = mu * norm_f_prev / norm_f
            lb = min(1.0, mu_lip)   # §2.1.6: λ_k = min(1, μ_k)

        # LU-factor J once for reuse across all inner damping loop solves.
        _reg_solve = _make_reg_solver(J_x, _square)

        # --- Regularity (inner damping) loop — FINITE TERMINATION PROOF ---
        # Same argument as nleq_err: _lb_rejected_max is non-decreasing,
        # lb_new ≤ 1.0, so after any rejection of a jumped-to λ the jump
        # condition lb_new > _lb_rejected_max can never fire for that range
        # again.  Halvings then bring lb below lb_min in at most
        # ⌈log₂(1/lb_min)⌉ ≤ 27 steps.  QNRES retry at most once.
        # BOUND: ≤ 2 × (13 + 27 + 1) ≈ 82 inner iterations per outer iteration.
        has_failed_once  = False
        _qn_retried      = False
        mu    = np.nan
        theta = np.nan
        _lb_rejected_max = 0.0   # highest λ evaluated and rejected — prevents cycling
        _inner_k         = 0     # counts inner iterations for verbose display

        while True:

            # Regularity test: has λ fallen below the minimum permitted?
            if lb < lb_min:
                if _simplified and not _qn_retried:
                    # QNRES safety net: stale Jacobian may have caused the failure.
                    # Recompute J, reset λ=1, and retry once before giving up.
                    if _verbose:
                        print("  ↳[QNRES retry] stale Jacobian → "
                              "recomputing J, resetting λ=1", flush=True)
                    _simplified = False
                    J_x, dx[:], _, _reg_solve, njev = _qn_safety_retry(
                        jac, x_sol, f_x, _square, Scale(n, tol), njev)
                    _J_cache = J_x
                    lb = 1.0
                    has_failed_once = False
                    _lb_rejected_max = 0.0
                    _qn_retried = True
                    continue

                msg = (f"Convergence failure: damping factor fell below "
                       f"minimum threshold (lb_min={lb_min:.1e}).")
                hist["f_norm"].append(unit_f.evaluate_norm(f_x))
                hist["dx_norm"].append(unit_x.evaluate_norm(dx))
                hist["lb"].append(lb); hist["mu"].append(mu); hist["theta"].append(theta)
                if printer:
                    printer.print_row(n_iter, f_norm=hist["f_norm"][-1],
                                      dx_norm=hist["dx_norm"][-1],
                                      lb=lb, mu=mu, theta=theta)
                    printer.print_footer(msg, nfev, njev)
                return _build_result(x_sol.copy(), f_x.copy(), False, msg,
                                      n_iter, nfev, njev, hist, "nleq_res")

            # Evaluate candidate: x_cand = x + λ Δx
            x_cand[:] = x_sol + lb * dx
            f_cand = np.asarray(func(x_cand), dtype=float).ravel(); nfev += 1

            # Residual contraction ratio θ = ‖f(x_cand)‖_D / ‖f(x)‖_D  (§2.2.1)
            # θ < 1 certifies that the residual is shrinking.
            norm_f_cand = scale_f.evaluate_norm(f_cand)
            theta = norm_f_cand / norm_f

            # Lipschitz predictor from candidate residual: §2.2.2
            # w = f_cand − (1 − λ) f_x  plays role of the Lipschitz surrogate
            w_f = f_cand - (1.0 - lb) * f_x
            denom = scale_f.evaluate_norm(w_f)
            mu = (0.5 * norm_f * lb * lb) / denom if denom > 0.0 else np.inf

            lb_evaluated = lb   # save before rejection test updates lb
            _thr = 1.0 - lb_evaluated / 4.0 if restricted else 1.0

            # Natural monotonicity test + damping update
            rejected, lb, has_failed_once = _monotonicity_reject(
                theta, mu, lb, lb_min, restricted, has_failed_once)
            if rejected:
                _lb_rejected_max = max(_lb_rejected_max, lb_evaluated)
                if _verbose:
                    _print_inner_iter(_inner_k + 1, lb_evaluated, theta, _thr,
                                      "REJECT", lb_next=lb, lb_min=lb_min)
                _inner_k += 1
                continue

            # Try a significantly better untried λ before accepting the current step
            lb_new = min(1.0, mu)
            if lb_new > 4.0 * lb and lb_new > _lb_rejected_max:
                if _verbose:
                    _print_inner_iter(_inner_k + 1, lb_evaluated, theta, _thr,
                                      "JUMP", lb_next=lb_new, lb_min=lb_min)
                _inner_k += 1
                lb = lb_new
                continue

            # Accept this candidate and advance
            if _verbose:
                _print_inner_iter(_inner_k + 1, lb_evaluated, theta, _thr, "ACCEPT")
            f_prev[:] = f_x
            f_x = f_cand
            x_sol[:] = x_cand
            break

        # --- Post-iteration bookkeeping ---

        # Adaptive f-space scaling update (weights track ‖f‖ magnitude)
        norm_f_prev = scale_f.evaluate_norm(f_prev)
        scale_f.update(f_x, f_prev)
        norm_f = scale_f.evaluate_norm(f_x)

        # §2.2.3: enter simplified (chord) Newton when θ < 0.5 and λ=1
        _simplified = use_qn and (lb == 1.0 and theta < 0.5)

        hist["f_norm"].append(unit_f.evaluate_norm(f_x))
        hist["dx_norm"].append(unit_x.evaluate_norm(dx))
        hist["lb"].append(lb); hist["mu"].append(mu); hist["theta"].append(theta)

        if printer:
            printer.print_row(n_iter, f_norm=hist["f_norm"][-1],
                              dx_norm=hist["dx_norm"][-1],
                              lb=lb, mu=mu, theta=theta)

        if callback:
            callback(n_iter, x_sol.copy(), f_x.copy(), dx.copy())

        # Convergence test (NLEQ-RES residual criterion, §2.2)
        if norm_f <= scale_f.tol:
            msg = f"Converged in {n_iter} iteration(s)."
            if printer:
                printer.print_footer(msg, nfev, njev)
            return _build_result(x_sol.copy(), f_x.copy(), True, msg,
                                  n_iter, nfev, njev, hist, "nleq_res")

    # Max iterations reached
    msg = f"Maximum iterations ({max_iter}) reached without convergence."
    if printer:
        printer.print_footer(msg, nfev, njev)
    return _build_result(x_sol.copy(), f_x.copy(), False, msg,
                          max_iter, nfev, njev, hist, "nleq_res")
