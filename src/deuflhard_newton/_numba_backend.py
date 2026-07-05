"""
Optional numba-accelerated backend for NLEQ-ERR and NLEQ-RES.

When numba is available and the user's function (and optionally Jacobian)
can be compiled to nopython mode, the entire inner Newton-damping loop runs
as LLVM machine code — no Python dispatch overhead per iteration.

Expected speedup: 5–30× for n < 50 with a numba-compatible function.
First call pays a compilation cost (1–10 s, cached to disk after that).

Restrictions:
  - Only square systems (m == n) — lstsq is not in numba's nopython subset.
  - Callbacks and display='verbose' disable this backend (both require Python
    calls from inside the inner loop).
"""
from __future__ import annotations

import math
import warnings
import numpy as np
from typing import Callable

# ---------------------------------------------------------------------------
# Numba availability
# ---------------------------------------------------------------------------
try:
    import numba
    from numba import njit
    _HAVE_NUMBA: bool = True
except ImportError:
    _HAVE_NUMBA = False

# ---------------------------------------------------------------------------
# Advice shown to the user when JIT compilation fails
# ---------------------------------------------------------------------------
NUMBA_ADVICE = """\
  Common fixes to make your function numba-compatible:

  1. Return a numpy array, not a Python list:
       Bad:  return [x[0]**2 - 1, x[1]**3 - 8]
       Good: return np.array([x[0]**2 - 1.0, x[1]**3 - 8.0])

  2. No Python objects inside the function (dict, set, class instances).

  3. For scalar guards use max(a, b); for array guards use np.maximum(arr, val).

  4. No scipy calls — scipy is not numba-compatible.
     Rewrite the math directly with numpy or math.

  5. No print() or file I/O inside func.

  6. Prefer math.sqrt / math.log for scalar results; np.sqrt for arrays.

  7. Decorate explicitly for early compilation and clear error messages:
       from numba import njit

       @njit(cache=True)
       def my_func(x):
           return np.array([x[0]**2 - 2.0])

  See https://numba.readthedocs.io/en/stable/reference/numpysupported.html
"""


def _warn_numba_fallback(reason: str) -> None:
    warnings.warn(
        f"\nnumba JIT failed: {reason}\n"
        "Falling back to the pure-Python backend (slower).\n"
        + NUMBA_ADVICE,
        RuntimeWarning,
        stacklevel=4,
    )


# ---------------------------------------------------------------------------
# JIT-compiled helpers and inner loops (only defined when numba is present)
# ---------------------------------------------------------------------------

if _HAVE_NUMBA:

    @njit(cache=True)
    def _wnorm(v, inv_w):
        """‖v‖_D = sqrt(Σ(vᵢ · inv_wᵢ)²) — zero Python overhead for small n."""
        s = 0.0
        for i in range(v.shape[0]):
            t = v[i] * inv_w[i]
            s += t * t
        return s ** 0.5

    @njit(cache=True)
    def _unorm(v):
        """‖v‖₂ (unit-weighted), used for history tracking."""
        s = 0.0
        for i in range(v.shape[0]):
            s += v[i] * v[i]
        return s ** 0.5

    @njit(cache=False)
    def _fd_jac_nb(func, x, f0, h=1e-7):
        """
        Forward-difference Jacobian in nopython mode.
        Calls func n times (f0 already known from caller).
        Accuracy: O(h) ≈ 7 decimal digits.
        """
        n = x.shape[0]
        m = f0.shape[0]
        J = np.empty((m, n))
        x_p = x.copy()
        for j in range(n):
            x_p[j] += h
            f_p = func(x_p)
            for i in range(m):
                J[i, j] = (f_p[i] - f0[i]) / h
            x_p[j] = x[j]
        return J

    # -------------------------------------------------------------------------
    # NLEQ-ERR inner loop
    # -------------------------------------------------------------------------

    @njit(cache=False)
    def _nleq_err_nb_core(
        func, jac, x0, w0, tol, lb0, lb_min, restricted, use_qn, max_iter
    ):
        """
        Full NLEQ-ERR algorithm in nopython mode.

        Parameters
        ----------
        func, jac : @njit callables
        w0        : user_scaling (lower bound on x-space weights)

        Returns
        -------
        13-tuple:
          (x_sol, f_x, success, n_iter, nfev, njev,
           h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th)
        Only the first h_len entries of each history array are valid.
        """
        n   = x0.shape[0]
        eps = 1e-30

        x_sol  = x0.copy()
        x_prev = x0.copy()
        x_cand = np.empty(n)
        dx     = np.zeros(n)
        dx_bar = np.zeros(n)
        w_tmp  = np.empty(n)

        # Initial weights: max(w0_i, |x0_i|, eps)
        weights = np.empty(n)
        for i in range(n):
            weights[i] = max(w0[i], abs(x0[i]), eps)
        inv_w = 1.0 / weights

        # Initial function evaluation
        f_x  = func(x_sol)
        nfev = 1
        njev = 0
        m    = f_x.shape[0]

        # Pre-allocated history (unit-weighted norms, length max_iter+2)
        nh    = max_iter + 2
        h_f   = np.empty(nh)
        h_dx  = np.full(nh, np.nan)
        h_dxb = np.full(nh, np.nan)
        h_lb  = np.full(nh, np.nan)
        h_mu  = np.full(nh, np.nan)
        h_th  = np.full(nh, np.nan)
        h_f[0] = _unorm(f_x)
        h_len  = 1

        # Initial Jacobian (k=0 evaluation; weight update at k=0 is a no-op)
        J_x    = jac(x_sol)
        njev  += 1
        J_cache = J_x
        _simplified = False

        lb    = lb0
        mu    = np.nan
        theta = np.nan
        norm_dx_prev     = 0.0
        norm_dx_bar_prev = 0.0

        for k in range(max_iter):
            n_iter = k + 1

            # Adaptive weight update
            for i in range(n):
                w_i = max(w0[i], 0.5 * (abs(x_sol[i]) + abs(x_prev[i])), eps)
                if w_i > weights[i]:
                    weights[i] = w_i
            inv_w = 1.0 / weights

            # Capture norms of previous Δx, Δx̄ before overwriting
            if k > 0:
                norm_dx_prev     = _wnorm(dx,     inv_w)
                norm_dx_bar_prev = _wnorm(dx_bar, inv_w)

            # QNERR: at k=0 we already have J_x from pre-loop call
            if k > 0:
                if _simplified:
                    J_x = J_cache
                else:
                    J_x    = jac(x_sol)
                    njev  += 1
                    J_cache = J_x
                    _simplified = False

            # Newton step Δx = -J⁻¹ f
            dx[:] = np.linalg.solve(J_x, -f_x)

            # Convergence test: ‖Δx‖_D ≤ tol
            norm_dx = _wnorm(dx, inv_w)
            if norm_dx <= tol:
                for i in range(n):
                    x_sol[i] += dx[i]
                f_x   = func(x_sol)
                nfev += 1
                h_f[h_len]  = _unorm(f_x)
                h_dx[h_len] = _unorm(dx)
                h_len += 1
                return (x_sol, f_x, True, n_iter, nfev, njev,
                        h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th)

            # Inter-outer Lipschitz predictor
            if k > 0:
                for i in range(n):
                    w_tmp[i] = dx_bar[i] - dx[i]
                s = _wnorm(w_tmp, inv_w)
                if s > 0.0:
                    mu_lip = lb * norm_dx_prev * norm_dx_bar_prev / (s * norm_dx)
                    lb = mu_lip if mu_lip < 1.0 else 1.0

            # --- Inner damping loop (finite by _lb_rejected_max argument) ---
            has_failed_once  = False
            _qn_retried      = False
            mu                = np.nan
            theta             = np.nan
            _lb_rejected_max = 0.0
            keep_inner       = True

            while keep_inner:

                if lb < lb_min:
                    if _simplified and not _qn_retried:
                        # QNERR safety net: recompute stale Jacobian
                        J_x    = jac(x_sol)
                        njev  += 1
                        J_cache = J_x
                        dx[:] = np.linalg.solve(J_x, -f_x)
                        norm_dx = _wnorm(dx, inv_w)
                        _simplified = False
                        lb = 1.0
                        has_failed_once  = False
                        _lb_rejected_max = 0.0
                        _qn_retried = True
                        continue

                    # Genuine failure
                    h_f[h_len]  = _unorm(f_x)
                    h_dx[h_len] = _unorm(dx)
                    h_lb[h_len] = lb
                    h_mu[h_len] = mu
                    h_th[h_len] = theta
                    h_len += 1
                    return (x_sol, f_x, False, n_iter, nfev, njev,
                            h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th)

                # Candidate: x_cand = x + λ Δx
                for i in range(n):
                    x_cand[i] = x_sol[i] + lb * dx[i]
                f_cand = func(x_cand)
                nfev  += 1

                # Simplified Newton step: Δx̄ = -J⁻¹ f_cand (J reused)
                dx_bar[:] = np.linalg.solve(J_x, -f_cand)
                norm_dx_bar = _wnorm(dx_bar, inv_w)
                theta = norm_dx_bar / norm_dx

                # Lipschitz predictor from inner surrogate
                for i in range(n):
                    w_tmp[i] = dx_bar[i] - (1.0 - lb) * dx[i]
                denom = _wnorm(w_tmp, inv_w)
                mu = (0.5 * norm_dx * lb * lb / denom) if denom > 0.0 else np.inf

                lb_eval = lb
                thr = 1.0 - lb_eval / 4.0 if restricted else 1.0

                # Monotonicity test
                rejected = (not restricted and theta >= 1.0) or (
                    restricted and theta > thr
                )
                if rejected:
                    next_lb = mu if mu < 0.5 * lb else 0.5 * lb
                    if next_lb < lb_min and not has_failed_once:
                        next_lb = lb_min
                        has_failed_once = True
                    lb = next_lb
                    if lb_eval > _lb_rejected_max:
                        _lb_rejected_max = lb_eval
                    continue

                # Candidate accepted: check for two-step convergence or jump
                lb_new = mu if mu < 1.0 else 1.0

                if lb == 1.0 and lb_new == 1.0 and norm_dx_bar <= tol:
                    # Two-step convergence: advance to x_cand + Δx̄
                    for i in range(n):
                        x_sol[i] = x_cand[i] + dx_bar[i]
                    f_x   = func(x_sol)
                    nfev += 1
                    h_f[h_len]   = _unorm(f_x)
                    h_dx[h_len]  = _unorm(dx)
                    h_dxb[h_len] = _unorm(dx_bar)
                    h_lb[h_len]  = lb
                    h_mu[h_len]  = mu
                    h_th[h_len]  = theta
                    h_len += 1
                    return (x_sol, f_x, True, n_iter, nfev, njev,
                            h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th)

                if lb_new > 4.0 * lb and lb_new > _lb_rejected_max:
                    lb = lb_new
                    continue

                # Accept: advance to x_cand
                x_prev[:] = x_sol
                x_sol[:]  = x_cand
                f_x        = f_cand
                keep_inner = False

            # Post-iteration bookkeeping
            _simplified = use_qn and (lb == 1.0 and theta < 0.5)

            h_f[h_len]   = _unorm(f_x)
            h_dx[h_len]  = _unorm(dx)
            h_dxb[h_len] = _unorm(dx_bar)
            h_lb[h_len]  = lb
            h_mu[h_len]  = mu
            h_th[h_len]  = theta
            h_len += 1

        # Max iterations
        h_f[h_len] = _unorm(f_x)
        h_len += 1
        return (x_sol, f_x, False, max_iter, nfev, njev,
                h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th)

    # -------------------------------------------------------------------------
    # NLEQ-RES inner loop
    # -------------------------------------------------------------------------

    @njit(cache=False)
    def _nleq_res_nb_core(
        func, jac, x0, w0, tol, lb0, lb_min, restricted, use_qn, max_iter
    ):
        """
        Full NLEQ-RES algorithm in nopython mode.

        Parameters
        ----------
        w0 : user_scaling (lower bound on f-space weights)

        Returns
        -------
        12-tuple:
          (x_sol, f_x, success, n_iter, nfev, njev,
           h_len, h_f, h_dx, h_lb, h_mu, h_th)
        """
        n   = x0.shape[0]
        eps = 1e-30

        x_sol  = x0.copy()
        x_cand = np.empty(n)
        dx     = np.zeros(n)

        # Initial function evaluation
        f_x  = func(x_sol)
        nfev = 1
        njev = 0
        m    = f_x.shape[0]

        # f-space weights: max(w0_i, |f0_i|, eps)
        weights = np.empty(m)
        for i in range(m):
            weights[i] = max(w0[i], abs(f_x[i]), eps)
        inv_w = 1.0 / weights

        # History
        nh   = max_iter + 2
        h_f  = np.empty(nh)
        h_dx = np.full(nh, np.nan)
        h_lb = np.full(nh, np.nan)
        h_mu = np.full(nh, np.nan)
        h_th = np.full(nh, np.nan)
        h_f[0] = _unorm(f_x)
        h_len  = 1

        norm_f = _wnorm(f_x, inv_w)

        # Convergence before first iteration
        if norm_f <= tol:
            return (x_sol, f_x, True, 0, nfev, njev,
                    h_len, h_f, h_dx, h_lb, h_mu, h_th)

        # Initial Jacobian
        J_x    = jac(x_sol)
        njev  += 1
        J_cache = J_x
        _simplified = False

        lb          = lb0
        mu          = np.nan
        theta       = np.nan
        norm_f_prev = norm_f
        f_prev      = f_x.copy()

        for k in range(max_iter):
            n_iter = k + 1

            # QNERR: k=0 uses pre-loop J
            if k > 0:
                if _simplified:
                    J_x = J_cache
                else:
                    J_x    = jac(x_sol)
                    njev  += 1
                    J_cache = J_x
                    _simplified = False

            # Newton step
            dx[:] = np.linalg.solve(J_x, -f_x)

            # Inter-outer Lipschitz predictor
            if k > 0 and not math.isnan(mu):
                mu_lip = mu * norm_f_prev / norm_f
                lb = mu_lip if mu_lip < 1.0 else 1.0

            # Inner damping loop
            has_failed_once  = False
            _qn_retried      = False
            mu                = np.nan
            theta             = np.nan
            _lb_rejected_max = 0.0
            keep_inner       = True

            while keep_inner:

                if lb < lb_min:
                    if _simplified and not _qn_retried:
                        J_x    = jac(x_sol)
                        njev  += 1
                        J_cache = J_x
                        dx[:] = np.linalg.solve(J_x, -f_x)
                        _simplified = False
                        lb = 1.0
                        has_failed_once  = False
                        _lb_rejected_max = 0.0
                        _qn_retried = True
                        continue

                    h_f[h_len]  = _unorm(f_x)
                    h_dx[h_len] = _unorm(dx)
                    h_lb[h_len] = lb
                    h_mu[h_len] = mu
                    h_th[h_len] = theta
                    h_len += 1
                    return (x_sol, f_x, False, n_iter, nfev, njev,
                            h_len, h_f, h_dx, h_lb, h_mu, h_th)

                for i in range(n):
                    x_cand[i] = x_sol[i] + lb * dx[i]
                f_cand = func(x_cand)
                nfev  += 1

                norm_f_cand = _wnorm(f_cand, inv_w)
                theta = norm_f_cand / norm_f

                w_f   = f_cand - (1.0 - lb) * f_x
                denom = _wnorm(w_f, inv_w)
                mu = (0.5 * norm_f * lb * lb / denom) if denom > 0.0 else np.inf

                lb_eval = lb
                thr = 1.0 - lb_eval / 4.0 if restricted else 1.0

                rejected = (not restricted and theta >= 1.0) or (
                    restricted and theta > thr
                )
                if rejected:
                    next_lb = mu if mu < 0.5 * lb else 0.5 * lb
                    if next_lb < lb_min and not has_failed_once:
                        next_lb = lb_min
                        has_failed_once = True
                    lb = next_lb
                    if lb_eval > _lb_rejected_max:
                        _lb_rejected_max = lb_eval
                    continue

                lb_new = mu if mu < 1.0 else 1.0
                if lb_new > 4.0 * lb and lb_new > _lb_rejected_max:
                    lb = lb_new
                    continue

                f_prev[:] = f_x
                f_x        = f_cand
                x_sol[:]   = x_cand
                keep_inner = False

            # Post-iteration: adaptive f-space weight update
            _simplified = use_qn and (lb == 1.0 and theta < 0.5)

            norm_f_prev = _wnorm(f_prev, inv_w)   # with OLD weights
            for i in range(m):
                w_new = max(w0[i], 0.5 * (abs(f_x[i]) + abs(f_prev[i])), eps)
                weights[i] = max(weights[i], w_new)
            inv_w  = 1.0 / weights
            norm_f = _wnorm(f_x, inv_w)            # with NEW weights

            h_f[h_len]  = _unorm(f_x)
            h_dx[h_len] = _unorm(dx)
            h_lb[h_len] = lb
            h_mu[h_len] = mu
            h_th[h_len] = theta
            h_len += 1

            if norm_f <= tol:
                return (x_sol, f_x, True, n_iter, nfev, njev,
                        h_len, h_f, h_dx, h_lb, h_mu, h_th)

        h_f[h_len] = _unorm(f_x)
        h_len += 1
        return (x_sol, f_x, False, max_iter, nfev, njev,
                h_len, h_f, h_dx, h_lb, h_mu, h_th)


# ---------------------------------------------------------------------------
# Python-level probe and dispatch helpers
# ---------------------------------------------------------------------------

# Module-level cache: Python callable → (jit_fn, error_str)
# Reusing the same jit_fn object across calls is essential — each new
# numba.njit(fn) call creates a distinct dispatcher type, forcing
# _nleq_err_nb_core to recompile for every solver invocation.
_JIT_FUNC_CACHE: dict = {}


def _try_jit(fn: Callable, x0: np.ndarray, expect_2d: bool = False):
    """
    Try to JIT-compile *fn* and invoke it once to force compilation.

    Uses ``cache=False`` so that functions defined interactively (or in -c
    strings) are supported.  The compiled bytecode lives in-process; for
    repeated calls within the same session numba reuses the compiled version.

    Returns ``(jit_fn, "")`` on success or ``(None, error_str)`` on failure.
    """
    if not _HAVE_NUMBA:
        return None, "numba not installed"
    x0 = np.asarray(x0, dtype=np.float64)

    # Return the cached dispatcher if we already JIT-compiled this function.
    # Same dispatcher object → same numba type → no recompilation of the core loop.
    if fn in _JIT_FUNC_CACHE:
        return _JIT_FUNC_CACHE[fn]

    try:
        jit_fn = numba.njit(cache=False)(fn)
        result = jit_fn(x0)
        if not isinstance(result, np.ndarray):
            entry = (None, f"must return ndarray, got {type(result).__name__}")
            _JIT_FUNC_CACHE[fn] = entry
            return entry
        if expect_2d and result.ndim != 2:
            entry = (None, f"Jacobian must return 2-D array, got shape {result.shape}")
            _JIT_FUNC_CACHE[fn] = entry
            return entry
        entry = (jit_fn, "")
        _JIT_FUNC_CACHE[fn] = entry
        return entry
    except numba.core.errors.TypingError as e:
        entry = (None, "TypingError — " + str(e).split("\n")[0][:200])
        _JIT_FUNC_CACHE[fn] = entry
        return entry
    except numba.core.errors.NumbaError as e:
        entry = (None, str(e).split("\n")[0][:200])
        _JIT_FUNC_CACHE[fn] = entry
        return entry
    except Exception as e:
        entry = (None, str(e)[:200])
        _JIT_FUNC_CACHE[fn] = entry
        return entry


_FD_JAC_CACHE: dict = {}   # maps func_jit (by identity) → fd_jac_jit


def _build_fd_jac_jit(func_jit):
    """
    Return a numba JIT-compiled FD-Jacobian that wraps *func_jit*.

    Caches by *func_jit* identity so the same dispatcher is reused across
    calls, avoiding recompilation of the core loop.
    """
    cached = _FD_JAC_CACHE.get(id(func_jit))
    if cached is not None:
        return cached

    @numba.njit(cache=False)
    def _jac(x):
        f0 = func_jit(x)
        return _fd_jac_nb(func_jit, x, f0)

    _FD_JAC_CACHE[id(func_jit)] = _jac
    return _jac


def _pack_err_history(h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th):
    hl = int(h_len)
    return {
        "f_norm":      np.array(h_f[:hl]),
        "dx_norm":     np.array(h_dx[:hl]),
        "dx_bar_norm": np.array(h_dxb[:hl]),
        "lb":          np.array(h_lb[:hl]),
        "mu":          np.array(h_mu[:hl]),
        "theta":       np.array(h_th[:hl]),
    }


def _pack_res_history(h_len, h_f, h_dx, h_lb, h_mu, h_th):
    hl = int(h_len)
    return {
        "f_norm":  np.array(h_f[:hl]),
        "dx_norm": np.array(h_dx[:hl]),
        "lb":      np.array(h_lb[:hl]),
        "mu":      np.array(h_mu[:hl]),
        "theta":   np.array(h_th[:hl]),
    }


def attempt_nleq_err(
    func: Callable,
    orig_jac,          # user's original jac (before resolve_jacobian), or None
    x0: np.ndarray,
    user_scaling: np.ndarray,
    tol: float,
    lb0: float,
    lb_min: float,
    restricted: bool,
    use_qn: bool,
    max_iter: int,
):
    """
    Try to run NLEQ-ERR via the numba backend.

    Returns ``(SolveResult, True)`` on success, ``(None, False)`` otherwise.
    Emits a ``RuntimeWarning`` describing the failure before returning False.
    """
    from deuflhard_newton.result import SolveResult

    if not _HAVE_NUMBA:
        return None, False

    x0_f = np.asarray(x0, dtype=np.float64)
    n    = x0_f.size

    # Only square systems: lstsq not in numba's nopython subset
    # (m unknown until we call func — checked after JIT compilation succeeds)

    # --- JIT-compile the function ---
    func_jit, err = _try_jit(func, x0_f, expect_2d=False)
    if func_jit is None:
        _warn_numba_fallback(f"could not compile func: {err}")
        return None, False

    # Check that func returns a 1-D array of the same length as x (square)
    test_f = func_jit(x0_f)
    if test_f.ndim != 1 or test_f.size != n:
        # Non-square: fall back silently (no warning — this is expected)
        return None, False

    # --- JIT-compile (or synthesise) the Jacobian ---
    using_fd = False
    if orig_jac is None:
        jac_jit = _build_fd_jac_jit(func_jit)
        using_fd = True
    else:
        jac_jit, err_j = _try_jit(orig_jac, x0_f, expect_2d=True)
        if jac_jit is None:
            warnings.warn(
                f"\nnumba: Jacobian could not be JIT-compiled ({err_j}).\n"
                "Using finite-difference Jacobian inside numba instead "
                "(less accurate: O(1e-7) vs machine precision for csdiff).\n"
                + NUMBA_ADVICE,
                RuntimeWarning,
                stacklevel=4,
            )
            jac_jit = _build_fd_jac_jit(func_jit)
            using_fd = True

    # --- Run the compiled inner loop ---
    try:
        out = _nleq_err_nb_core(
            func_jit, jac_jit, x0_f, user_scaling,
            tol, lb0, lb_min, restricted, use_qn, max_iter,
        )
    except Exception as e:
        _warn_numba_fallback(f"runtime error in numba inner loop: {e}")
        return None, False

    (x_sol, f_x, success, n_iter, nfev, njev,
     h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th) = out

    if success:
        msg = f"Converged in {n_iter} iteration(s)."
    elif n_iter >= max_iter:
        msg = f"Maximum iterations ({max_iter}) reached without convergence."
    else:
        msg = (f"Convergence failure: damping factor fell below "
               f"minimum threshold (lb_min={lb_min:.1e}).")

    method = "nleq_err[numba+fd]" if using_fd else "nleq_err[numba]"
    result = SolveResult(
        x=x_sol.copy(), fun=f_x.copy(),
        success=success, message=msg,
        nit=int(n_iter), nfev=int(nfev), njev=int(njev),
        history=_pack_err_history(h_len, h_f, h_dx, h_dxb, h_lb, h_mu, h_th),
        method=method,
    )
    return result, True


def attempt_nleq_res(
    func: Callable,
    orig_jac,
    x0: np.ndarray,
    raw_user_scaling,            # None | float | ndarray — f-space lower bound
    tol: float,
    lb0: float,
    lb_min: float,
    restricted: bool,
    use_qn: bool,
    max_iter: int,
):
    """
    Try to run NLEQ-RES via the numba backend.

    Returns ``(SolveResult, True)`` on success, ``(None, False)`` otherwise.
    """
    from deuflhard_newton.result import SolveResult

    if not _HAVE_NUMBA:
        return None, False

    x0_f = np.asarray(x0, dtype=np.float64)
    n    = x0_f.size

    func_jit, err = _try_jit(func, x0_f, expect_2d=False)
    if func_jit is None:
        _warn_numba_fallback(f"could not compile func: {err}")
        return None, False

    # Call func once to determine m and build f-space user_scaling
    test_f = func_jit(x0_f)
    if test_f.ndim != 1 or test_f.size != n:
        return None, False   # non-square: fall back silently
    m = test_f.size

    if raw_user_scaling is None:
        user_scaling = np.ones(m, dtype=np.float64)
    elif not isinstance(raw_user_scaling, np.ndarray):
        user_scaling = np.full(m, float(raw_user_scaling), dtype=np.float64)
    else:
        user_scaling = np.asarray(raw_user_scaling, dtype=np.float64)

    using_fd = False
    if orig_jac is None:
        jac_jit  = _build_fd_jac_jit(func_jit)
        using_fd = True
    else:
        jac_jit, err_j = _try_jit(orig_jac, x0_f, expect_2d=True)
        if jac_jit is None:
            warnings.warn(
                f"\nnumba: Jacobian could not be JIT-compiled ({err_j}).\n"
                "Using finite-difference Jacobian inside numba instead.\n"
                + NUMBA_ADVICE,
                RuntimeWarning,
                stacklevel=4,
            )
            jac_jit  = _build_fd_jac_jit(func_jit)
            using_fd = True

    try:
        out = _nleq_res_nb_core(
            func_jit, jac_jit, x0_f, user_scaling,
            tol, lb0, lb_min, restricted, use_qn, max_iter,
        )
    except Exception as e:
        _warn_numba_fallback(f"runtime error in numba inner loop: {e}")
        return None, False

    (x_sol, f_x, success, n_iter, nfev, njev,
     h_len, h_f, h_dx, h_lb, h_mu, h_th) = out

    if success:
        msg = f"Converged in {n_iter} iteration(s)."
    elif n_iter >= max_iter:
        msg = f"Maximum iterations ({max_iter}) reached without convergence."
    else:
        msg = (f"Convergence failure: damping factor fell below "
               f"minimum threshold (lb_min={lb_min:.1e}).")

    method = "nleq_res[numba+fd]" if using_fd else "nleq_res[numba]"
    result = SolveResult(
        x=x_sol.copy(), fun=f_x.copy(),
        success=success, message=msg,
        nit=int(n_iter), nfev=int(nfev), njev=int(njev),
        history=_pack_res_history(h_len, h_f, h_dx, h_lb, h_mu, h_th),
        method=method,
    )
    return result, True
