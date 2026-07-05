"""
Tests for the Levenberg-Marquardt solver.

LM converges to a local minimum of ½‖F‖², not necessarily a root.
"""
import numpy as np
import pytest
from deuflhard_newton import lm, SolveResult


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------

def linear_ls(x):
    """
    Linear least squares: F(x) = A @ x - b  with an exact solution.
    m=6, n=2; x* = [1, 2] is the exact solution.
    """
    A = np.array([[1, 0], [0, 1], [1, 1], [2, -1], [1, 3], [0, 2]], dtype=float)
    x_star = np.array([1.0, 2.0])
    return A @ x - A @ x_star

def linear_ls_jac(x):
    return np.array([[1, 0], [0, 1], [1, 1], [2, -1], [1, 3], [0, 2]], dtype=float)


def rosenbrock_resid(x):
    """
    Rosenbrock expressed as a least-squares residual (m=2, n=2).
    F(x) = [10*(x1 - x0^2), (1 - x0)]
    Minimum at x* = [1, 1], F(x*) = [0, 0].
    """
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def inconsistent_ls(x):
    """
    Overdetermined system with no exact root — forces LM to find min-norm residual.
    F(x) = A @ x - b  where b is NOT in the column space of A.
    """
    A = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    b = np.array([1.0, 1.0, 3.0])   # not consistent: [1,0]+[0,1] ≠ 3
    return A @ x - b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLM:

    def test_linear_ls_exact_solution(self):
        """Linear LS with an exact solution: LM finds it."""
        r = lm(linear_ls, np.zeros(2), linear_ls_jac, tol=1e-8, display=False)
        assert r.success
        # Exact solution → residual should be near zero
        np.testing.assert_allclose(r.fun, np.zeros(6), atol=1e-6)
        np.testing.assert_allclose(r.x, [1.0, 2.0], atol=1e-6)

    def test_rosenbrock_minimum(self):
        """Rosenbrock as LS: converges to x* = [1, 1], F(x*) = 0."""
        r = lm(rosenbrock_resid, np.array([-1.0, 1.0]), tol=1e-8,
               max_iter=5_000, display=False)
        assert r.success
        np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-4)
        np.testing.assert_allclose(r.fun, np.zeros(2), atol=1e-4)

    def test_inconsistent_ls_converges(self):
        """No exact root exists — LM should converge to the minimum of ½‖F‖²."""
        r = lm(inconsistent_ls, np.zeros(2), tol=1e-8, display=False)
        assert r.success
        # The minimum-norm solution via lstsq
        A = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
        b = np.array([1.0, 1.0, 3.0])
        x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        np.testing.assert_allclose(r.x, x_lstsq, atol=1e-4)

    def test_result_type(self):
        r = lm(linear_ls, np.zeros(2), linear_ls_jac, display=False)
        assert isinstance(r, SolveResult)
        assert r.method == 'lm'

    def test_message_indicates_minimum(self):
        """Success message should say 'minimum', not 'root' or just 'converged'."""
        r = lm(linear_ls, np.zeros(2), linear_ls_jac, tol=1e-6, display=False)
        assert r.success
        assert "minimum" in r.message.lower() or "optimal" in r.message.lower()

    def test_auto_jacobian(self):
        """LM with jac=None (auto via csdiff) converges."""
        r = lm(rosenbrock_resid, np.array([0.5, 0.5]), tol=1e-6,
               max_iter=2_000, display=False)
        assert r.success
        np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-3)

    def test_history_fields(self):
        """History should contain f_norm, grad_norm, dx_norm, lam, rho."""
        r = lm(linear_ls, np.zeros(2), linear_ls_jac, display=False)
        assert r.success
        for key in ('f_norm', 'grad_norm', 'dx_norm', 'lam', 'rho'):
            assert key in r.history, f"Missing history key: {key!r}"
        # Length: row 0 (initial) + nit rows
        assert len(r.history['f_norm']) == r.nit + 1

    def test_rank_deficient_jacobian(self):
        """Rank-deficient Jacobian: LM should not raise LinAlgError."""
        # F(x) = [x0, x0]  — Jacobian [[1,0],[1,0]] has rank 1
        def f(x):
            return np.array([x[0], x[0]])
        r = lm(f, np.array([1.0, 1.0]), tol=1e-6, max_iter=200, display=False)
        # Should not raise; may or may not converge
        assert isinstance(r, SolveResult)
