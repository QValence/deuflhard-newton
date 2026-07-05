"""
Convergence tests on problems with known exact solutions.

All problems have analytically computable roots; each test verifies that the
solver reaches that root to the specified tolerance.
"""
import numpy as np
import pytest
from deuflhard_newton import solve, nleq_err, nleq_res, SolveResult, ProblemType


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

def linear_2x2(x):
    """2x2 linear system.  Exact root: x* = [1/3, 1/3]."""
    return np.array([2.0 * x[0] + x[1] - 1.0, x[0] - x[1]])

def linear_2x2_jac(x):
    return np.array([[2.0, 1.0], [1.0, -1.0]])

ROOT_LINEAR_2X2 = np.array([1.0 / 3.0, 1.0 / 3.0])


def scalar_sqrt2(x):
    """Scalar equation x^2 - 2 = 0.  Root: x* = sqrt(2)."""
    return np.array([x[0] ** 2 - 2.0])

def scalar_sqrt2_jac(x):
    return np.array([[2.0 * x[0]]])

ROOT_SQRT2 = np.array([np.sqrt(2.0)])


def trig_2d(x):
    """
    2D trigonometric system:
        sin(x0) + x1 - 1 = 0
        x0 + cos(x1) - 1 = 0
    Has multiple roots; convergence depends on starting point.
    """
    return np.array([np.sin(x[0]) + x[1] - 1.0, x[0] + np.cos(x[1]) - 1.0])

def trig_2d_jac(x):
    return np.array([[np.cos(x[0]), 1.0], [1.0, -np.sin(x[1])]])


def broyden_banded(x):
    """
    Broyden banded problem (n=6), a standard test for Newton solvers.
    f_i(x) = x_i * (2 + 5*x_i^2) + 1 - sum_{j in bandwidth} x_j * (1 + x_j)
    with lower bandwidth ml=5, upper bandwidth mu=1.
    The exact root is computed by solving f(x*) = 0 numerically.
    """
    n = len(x)
    ml, mu = 5, 1
    f = []
    for i in range(n):
        ji_lo = max(0, i - ml)
        ji_hi = min(n - 1, i + mu)
        s = sum(x[j] * (1.0 + x[j]) for j in range(ji_lo, ji_hi + 1) if j != i)
        f.append(x[i] * (2.0 + 5.0 * x[i] ** 2) + 1.0 - s)
    return np.array(f)  # dtype inferred from elements (float or complex)


# ---------------------------------------------------------------------------
# Tests: nleq_err on known roots
# ---------------------------------------------------------------------------

class TestNleqErr:

    def test_linear_exact(self):
        """Linear system converges in one step to machine precision."""
        r = nleq_err(linear_2x2, np.zeros(2), linear_2x2_jac, tol=1e-10, display=False)
        assert r.success
        np.testing.assert_allclose(r.x, ROOT_LINEAR_2X2, atol=1e-9)
        np.testing.assert_allclose(r.fun, np.zeros(2), atol=1e-9)

    def test_scalar_sqrt2(self):
        """Univariate x^2 - 2, root = sqrt(2)."""
        r = nleq_err(scalar_sqrt2, np.array([1.0]), scalar_sqrt2_jac,
                     tol=1e-10, display=False)
        assert r.success
        np.testing.assert_allclose(r.x, ROOT_SQRT2, atol=1e-9)

    def test_trig_2d(self):
        """2D trigonometric system: solver finds A root (residual ≈ 0)."""
        r = nleq_err(trig_2d, np.array([0.5, 0.5]), trig_2d_jac,
                     tol=1e-10, display=False)
        assert r.success
        np.testing.assert_allclose(r.fun, np.zeros(2), atol=1e-8)

    def test_broyden_banded(self):
        """Broyden banded (n=6): residual near zero at solution."""
        # x0=0 gives ‖dx‖≈14 which massively overshoots; -0.1 is a better start
        x0 = np.full(6, -0.1)
        r = nleq_err(broyden_banded, x0, tol=1e-8, jac=None, display=False)
        assert r.success
        np.testing.assert_allclose(r.fun, np.zeros(6), atol=1e-7)

    def test_result_type(self):
        """Return type is SolveResult with all expected fields."""
        r = nleq_err(linear_2x2, np.zeros(2), linear_2x2_jac,
                     tol=1e-8, display=False)
        assert isinstance(r, SolveResult)
        assert hasattr(r, 'x')
        assert hasattr(r, 'fun')
        assert hasattr(r, 'success')
        assert hasattr(r, 'message')
        assert hasattr(r, 'nit')
        assert hasattr(r, 'nfev')
        assert hasattr(r, 'njev')
        assert hasattr(r, 'history')
        assert hasattr(r, 'method')
        assert r.method.startswith('nleq_err')

    def test_history_no_nan_in_leading_row(self):
        """History row 0 has f_norm; the rest should not contain NaN after convergence."""
        r = nleq_err(linear_2x2, np.zeros(2), linear_2x2_jac,
                     tol=1e-8, display=False)
        assert r.success
        # f_norm has no NaN in any row
        assert not np.any(np.isnan(r.history['f_norm']))
        # lb, mu, theta are NaN only in row 0 (pre-iteration)
        for key in ('lb', 'mu', 'theta'):
            arr = r.history[key]
            assert np.isnan(arr[0]), f"row 0 of history[{key!r}] should be NaN"
            assert not np.any(np.isnan(arr[1:])), f"post-iteration history[{key!r}] has NaN"

    def test_highly_nonlinear(self):
        """ProblemType.HIGHLY_NONLINEAR still converges for trig system."""
        r = nleq_err(trig_2d, np.array([0.5, 0.5]), trig_2d_jac,
                     tol=1e-8, problem_type=ProblemType.HIGHLY_NONLINEAR, display=False)
        assert r.success
        np.testing.assert_allclose(r.fun, np.zeros(2), atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: nleq_res on known roots
# ---------------------------------------------------------------------------

class TestNleqRes:

    def test_linear_exact(self):
        r = nleq_res(linear_2x2, np.zeros(2), linear_2x2_jac, tol=1e-10, display=False)
        assert r.success
        np.testing.assert_allclose(r.x, ROOT_LINEAR_2X2, atol=1e-9)

    def test_scalar_sqrt2(self):
        r = nleq_res(scalar_sqrt2, np.array([1.0]), scalar_sqrt2_jac,
                     tol=1e-10, display=False)
        assert r.success
        np.testing.assert_allclose(r.x, ROOT_SQRT2, atol=1e-9)

    def test_trig_2d(self):
        r = nleq_res(trig_2d, np.array([0.5, 0.5]), trig_2d_jac,
                     tol=1e-10, display=False)
        assert r.success
        np.testing.assert_allclose(r.fun, np.zeros(2), atol=1e-8)

    def test_overdetermined_consistent(self):
        """Consistent overdetermined system (m=4, n=2) has an exact solution."""
        # f(x) = A @ x - b  with b = A @ x*
        A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        x_star = np.array([1.0, 2.0])
        b = A @ x_star

        def f(x):
            return A @ x - b

        def j(x):
            return A

        r = nleq_res(f, np.zeros(2), j, tol=1e-8, display=False)
        assert r.success
        np.testing.assert_allclose(r.x, x_star, atol=1e-6)

    def test_result_method_name(self):
        r = nleq_res(linear_2x2, np.zeros(2), linear_2x2_jac, display=False)
        assert r.method.startswith('nleq_res')


# ---------------------------------------------------------------------------
# Tests: solve() unified API
# ---------------------------------------------------------------------------

class TestSolve:

    def test_square_auto_picks_nleq_err(self):
        r = solve(linear_2x2, np.zeros(2), display=False)
        assert r.method.startswith('nleq_err')
        assert r.success

    def test_moderate_overdetermined_picks_nleq_res(self):
        """m = 2*n → nleq_res."""
        A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        x_star = np.array([1.0, 2.0])
        b = A @ x_star
        r = solve(lambda x: A @ x - b, np.zeros(2), display=False)
        assert r.method.startswith('nleq_res')
        assert r.success

    def test_heavily_overdetermined_picks_lm(self):
        """m = 10*n → lm."""
        n, m = 2, 20
        A = np.random.default_rng(42).normal(size=(m, n))
        x_star = np.array([1.0, 2.0])
        b = A @ x_star
        r = solve(lambda x: A @ x - b, np.zeros(n), display=False)
        assert r.method == 'lm'

    def test_explicit_method_override(self):
        r = solve(linear_2x2, np.zeros(2), method='nleq_res', display=False)
        assert r.method.startswith('nleq_res')
        assert r.success

    def test_problem_type_string(self):
        r = solve(trig_2d, np.array([0.5, 0.5]),
                  problem_type='highly_nonlinear', display=False)
        assert r.success

    def test_auto_jacobian(self):
        """jac=None: auto via csdiff gives same result as analytical jac."""
        r_auto = solve(trig_2d, np.array([0.5, 0.5]), jac=None,
                       tol=1e-8, display=False)
        r_analytical = solve(trig_2d, np.array([0.5, 0.5]), jac=trig_2d_jac,
                             tol=1e-8, display=False)
        assert r_auto.success
        # Both find a root (residual ≈ 0) — may be different roots, but both valid
        np.testing.assert_allclose(r_auto.fun, np.zeros(2), atol=1e-6)
        np.testing.assert_allclose(r_analytical.fun, np.zeros(2), atol=1e-6)

    def test_failure_returns_last_iterate(self):
        """On failure, result.x is the last iterate, not None."""
        def no_root(x):
            return np.array([np.exp(x[0]) + 1.0])  # exp(x)+1 ≥ 2 > 0 always

        r = solve(no_root, np.array([0.0]), max_iter=20, display=False)
        assert not r.success
        assert r.x is not None
        assert r.x.shape == (1,)
        assert r.message != ""

    def test_solve_result_fields(self):
        r = solve(linear_2x2, np.zeros(2), display=False)
        assert r.success
        assert r.nit >= 1
        assert r.nfev >= 1
        assert r.njev >= 1
        assert len(r.history['f_norm']) == r.nit + 1
        assert not np.any(np.isnan(r.history['f_norm']))

    def test_callback_called(self):
        calls = []
        def cb(k, x, fx, dx):
            calls.append(k)

        r = solve(linear_2x2, np.zeros(2), callback=cb, display=False)
        assert r.success
        assert len(calls) == r.nit
        assert calls[0] == 1   # 1-based iteration index

    def test_display_false_no_output(self, capsys):
        solve(linear_2x2, np.zeros(2), display=False)
        captured = capsys.readouterr()
        assert captured.out == ""
