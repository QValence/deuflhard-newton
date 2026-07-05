"""
Classic easy tests — readable, self-documenting Newton method verification.

Every problem here has a known closed-form root so failure is immediately
obvious without needing to understand domain specifics.  These complement
the numerical-precision tests elsewhere: they answer "does it find the right
root at all?"
"""

import numpy as np
import pytest

from deuflhard_newton import nleq_err, nleq_res, solve


# ---------------------------------------------------------------------------
# Scalar problems (n = m = 1)
# ---------------------------------------------------------------------------

class TestScalarSqrt2:
    """x² − 2 = 0  →  x* = √2 ≈ 1.41421356…"""

    def _f(self, x):
        return np.array([x[0]**2 - 2.0])

    def test_nleq_err(self):
        r = nleq_err(self._f, np.array([1.0]), tol=1e-10, display=False)
        assert r.success
        assert np.isclose(r.x[0], np.sqrt(2), atol=1e-8)

    def test_nleq_res(self):
        r = nleq_res(self._f, np.array([1.0]), tol=1e-10, display=False)
        assert r.success
        assert np.isclose(r.x[0], np.sqrt(2), atol=1e-8)

    def test_solve(self):
        r = solve(self._f, np.array([1.0]), tol=1e-10, display=False)
        assert r.success
        assert np.isclose(r.x[0], np.sqrt(2), atol=1e-8)

    def test_auto_jac(self):
        """jac=None → complex-step auto-differentiation."""
        r = solve(self._f, np.array([1.0]), jac=None, tol=1e-10, display=False)
        assert r.success
        assert np.isclose(r.x[0], np.sqrt(2), atol=1e-8)


class TestScalarCubic:
    """x³ − x − 2 = 0  →  x* ≈ 1.52137971…   (only real root)"""

    def _f(self, x):
        return np.array([x[0]**3 - x[0] - 2.0])

    def _jac(self, x):
        return np.array([[3.0 * x[0]**2 - 1.0]])

    def test_nleq_err(self):
        r = nleq_err(self._f, np.array([1.0]), jac=self._jac, tol=1e-10, display=False)
        assert r.success
        expected = 1.5213797068045676
        assert np.isclose(r.x[0], expected, atol=1e-8)

    def test_residual_at_solution(self):
        r = solve(self._f, np.array([1.0]), tol=1e-10, display=False)
        assert r.success
        assert np.abs(self._f(r.x)[0]) < 1e-8


# ---------------------------------------------------------------------------
# 2-D problems
# ---------------------------------------------------------------------------

class TestTwoDIntegerRoots:
    """[x₀² − 1,  x₁³ − 8] = 0  →  x* = [1, 2]"""

    def _f(self, x):
        return np.array([x[0]**2 - 1.0, x[1]**3 - 8.0])

    def _jac(self, x):
        return np.array([[2.0 * x[0], 0.0],
                         [0.0,        3.0 * x[1]**2]])

    def test_nleq_err(self):
        r = nleq_err(self._f, np.array([0.5, 1.0]), jac=self._jac,
                     tol=1e-10, display=False)
        assert r.success
        assert np.allclose(r.x, [1.0, 2.0], atol=1e-8)

    def test_solve(self):
        r = solve(self._f, np.array([0.5, 1.0]), tol=1e-10, display=False)
        assert r.success
        assert np.allclose(r.x, [1.0, 2.0], atol=1e-8)


# ---------------------------------------------------------------------------
# n-D diagonal problems (scales with n)
# ---------------------------------------------------------------------------

def _diagonal_problem(n: int):
    """x[i]² − (i+2) = 0  →  x*[i] = √(i+2),  i = 0..n-1"""
    expected = np.sqrt(np.arange(2, n + 2, dtype=float))

    def f(x):
        return np.array([x[i]**2 - (i + 2) for i in range(n)])

    def jac(x):
        return np.diag([2.0 * x[i] for i in range(n)])

    return f, jac, expected


class TestDiagonalND:

    def test_n5_nleq_err(self):
        f, jac, expected = _diagonal_problem(5)
        r = nleq_err(f, np.ones(5), jac=jac, tol=1e-10, display=False)
        assert r.success
        assert np.allclose(r.x, expected, atol=1e-8)

    def test_n10_solve(self):
        f, jac, expected = _diagonal_problem(10)
        r = solve(f, np.ones(10), jac=jac, tol=1e-10, display=False)
        assert r.success
        assert np.allclose(r.x, expected, atol=1e-8)

    def test_n10_auto_jac(self):
        """Large n with auto-diff Jacobian (no analytical jac supplied)."""
        f, _, expected = _diagonal_problem(10)
        r = solve(f, np.ones(10), jac=None, tol=1e-10, display=False)
        assert r.success
        assert np.allclose(r.x, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Simple linear system (one Newton step expected)
# ---------------------------------------------------------------------------

class TestLinear3x3:
    """A x = b  →  x* = A⁻¹ b  (should converge in exactly 1 iteration)."""

    A = np.array([[2.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]])
    b = np.array([5.0, 10.0, 7.0])

    def _f(self, x):
        return self.A @ x - self.b

    def _jac(self, x):
        return self.A

    def test_nleq_err_one_step(self):
        r = nleq_err(self._f, np.zeros(3), jac=self._jac, display=False)
        assert r.success
        expected = np.linalg.solve(self.A, self.b)
        assert np.allclose(r.x, expected, atol=1e-10)
        assert r.nit == 1   # linear → converges in a single Newton step
