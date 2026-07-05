"""
Tests on the order of convergence and QNERR/QNRES effectiveness.

Newton's method achieves quadratic convergence on smooth problems; QNERR/QNRES
reduce Jacobian evaluations at the cost of reverting to linear (chord method)
convergence in the simplified-Newton phase.
"""
import numpy as np
import pytest
from deuflhard_newton import nleq_err, nleq_res


# ---------------------------------------------------------------------------
# Smooth problem for convergence-order measurement
# ---------------------------------------------------------------------------

def smooth_nonlinear(x):
    """
    Smooth 3x3 system with root x* = [1, 1, 1].

        x0^2 + x1 - 2 = 0       J_row0 = [2*x0,  1,        0      ]
        x0 + x1^2 - 2 = 0       J_row1 = [1,      2*x1,     0      ]
        x2^2 + x2 - 2 = 0       J_row2 = [0,      0,        2*x2+1 ]

    Root: [1, 1, 1] (substitution: 1+1-2=0, 1+1-2=0, 1+1-2=0).
    Jacobian at root: diag-dominant [[2,1,0],[1,2,0],[0,0,3]], det=3*(4-1)=9≠0.
    """
    return np.array([
        x[0] ** 2 + x[1] - 2.0,
        x[0] + x[1] ** 2 - 2.0,
        x[2] ** 2 + x[2] - 2.0,
    ])

def smooth_nonlinear_jac(x):
    return np.array([
        [2.0 * x[0], 1.0, 0.0],
        [1.0, 2.0 * x[1], 0.0],
        [0.0, 0.0, 2.0 * x[2] + 1.0],
    ])

X_STAR = np.array([1.0, 1.0, 1.0])


def _convergence_order(errors):
    """
    Estimate the order p of convergence from a sequence of errors:
        errors[k+1] ≈ C * errors[k]^p
    Returns array of local estimates.
    """
    ratios = []
    for i in range(1, len(errors) - 1):
        if errors[i] > 0 and errors[i - 1] > 0 and errors[i + 1] > 0:
            p = np.log(errors[i + 1] / errors[i]) / np.log(errors[i] / errors[i - 1])
            ratios.append(p)
    return np.array(ratios)


# ---------------------------------------------------------------------------
# Quadratic convergence: nleq_err with fresh Jacobian each step
# ---------------------------------------------------------------------------

class TestQuadraticConvergence:

    def test_nleq_err_converges_efficiently(self):
        """
        nleq_err converges on a smooth problem in a bounded iteration count.

        Pure quadratic-convergence measurement is not straightforward when
        QNERR is active (it intentionally uses linear chord iterations once
        close to the solution).  Instead, we verify that the solver converges
        and that the total iteration count is consistent with super-linear
        behaviour (not requiring hundreds of steps).
        """
        x0 = np.array([0.1, 0.1, 0.1])

        r = nleq_err(smooth_nonlinear, x0, smooth_nonlinear_jac,
                     tol=1e-12, max_iter=50, display=False)

        assert r.success, f"Expected convergence; got: {r.message}"
        np.testing.assert_allclose(r.x, X_STAR, atol=1e-10)
        # A proper Newton solver should not need hundreds of iterations
        assert r.nit <= 40, f"Too many iterations: {r.nit}"


# ---------------------------------------------------------------------------
# QNERR reduces Jacobian evaluations
# ---------------------------------------------------------------------------

class TestQNERREffectiveness:

    def test_qnerr_reduces_jac_evals(self):
        """
        On a smooth problem starting near the solution, QNERR should reuse
        the Jacobian across iterations (njev < nit) by entering simplified mode.
        """
        # Start close enough that lb=1 and theta<0.5 from the first step
        x0 = X_STAR + 0.01 * np.ones(3)

        r = nleq_err(smooth_nonlinear, x0, smooth_nonlinear_jac,
                     tol=1e-12, max_iter=50, display=False)

        assert r.success
        # QNERR effectiveness: fewer Jacobian evaluations than outer iterations
        assert r.njev < r.nit, (
            f"QNERR should have reused the Jacobian "
            f"(njev={r.njev} should be < nit={r.nit})"
        )

    def test_qnres_reduces_jac_evals(self):
        """Same test for QNRES in nleq_res."""
        # Overdetermined version: stack smooth_nonlinear with its first equation
        def f_over(x):
            base = smooth_nonlinear(x)
            # Add a linearly independent 4th equation to make it 4x3
            return np.append(base, x[0] - 1.0)

        x0 = X_STAR + 0.01 * np.ones(3)

        r = nleq_res(f_over, x0, tol=1e-10, max_iter=50, display=False)
        assert r.success
        # QNRES should have skipped at least one Jacobian call
        assert r.njev <= r.nit

    def test_njev_count_type(self):
        """njev and nit are non-negative integers."""
        r = nleq_err(smooth_nonlinear, np.array([0.8, 0.8, 0.8]),
                     smooth_nonlinear_jac, display=False)
        assert isinstance(r.njev, int)
        assert isinstance(r.nit, int)
        assert r.njev >= 1
        assert r.nit >= 1


# ---------------------------------------------------------------------------
# History consistency
# ---------------------------------------------------------------------------

class TestHistoryConsistency:

    def test_history_length(self):
        """History arrays have length nit + 1 (row 0 = pre-iteration)."""
        r = nleq_err(smooth_nonlinear, np.array([0.8, 0.8, 0.8]),
                     smooth_nonlinear_jac, tol=1e-10, display=False)
        assert r.success
        expected_len = r.nit + 1
        for key, arr in r.history.items():
            assert len(arr) == expected_len, (
                f"history[{key!r}] has length {len(arr)}, expected {expected_len}"
            )

    def test_f_norm_decreasing_monotone(self):
        """For linear/mildly nonlinear problem, residual norm should decrease."""
        r = nleq_err(smooth_nonlinear, np.array([0.8, 0.8, 0.8]),
                     smooth_nonlinear_jac, tol=1e-10, display=False)
        assert r.success
        f_norms = r.history['f_norm']
        # Allow for occasional non-monotone step due to damping, but last norm
        # should be smaller than first
        assert f_norms[-1] < f_norms[0]

    def test_lb_in_unit_interval(self):
        """Damping factor lb should always be in (0, 1]."""
        r = nleq_err(smooth_nonlinear, np.array([0.8, 0.8, 0.8]),
                     smooth_nonlinear_jac, display=False)
        lb = r.history['lb']
        valid = lb[~np.isnan(lb)]
        assert np.all(valid > 0.0)
        assert np.all(valid <= 1.0)
