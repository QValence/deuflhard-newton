"""
Exception and error-handling tests.

Verifies that the solvers fail gracefully with informative messages when
given invalid inputs or problems that cannot be solved.
"""
import numpy as np
import pytest
from deuflhard_newton import solve, nleq_err, nleq_res, lm, ProblemType
from deuflhard_newton.jac import resolve_jacobian


# ---------------------------------------------------------------------------
# Problems with no root
# ---------------------------------------------------------------------------

def no_real_root(x):
    """f(x) = exp(x) + 1 > 1 for all real x; Jacobian exp(x) > 0 always."""
    return np.array([np.exp(x[0]) + 1.0])

def no_real_root_jac(x):
    return np.array([[np.exp(x[0])]])


# ---------------------------------------------------------------------------
# Tests: failure modes
# ---------------------------------------------------------------------------

class TestFailureModes:

    def test_no_root_returns_failure(self):
        """f(x) = x^2 + 1 has no real root — solver should return failure, not raise."""
        r = nleq_err(no_real_root, np.array([0.0]), no_real_root_jac,
                     max_iter=50, display=False)
        assert not r.success
        assert r.x is not None
        assert r.x.shape == (1,)
        assert r.message != ""

    def test_failure_message_informative(self):
        """Failure message should mention why (damping / max_iter)."""
        r = nleq_err(no_real_root, np.array([0.0]), no_real_root_jac,
                     max_iter=50, display=False)
        assert not r.success
        msg_lower = r.message.lower()
        assert any(kw in msg_lower for kw in ("damping", "threshold", "maximum")), (
            f"Failure message is not informative: {r.message!r}"
        )

    def test_max_iter_zero(self):
        """max_iter=0 should return immediately without raising."""
        r = nleq_err(no_real_root, np.array([0.0]), no_real_root_jac,
                     max_iter=0, display=False)
        assert not r.success
        assert isinstance(r.x, np.ndarray)

    def test_solve_failure_no_root(self):
        """solve() wrapping a problem with no root also returns graceful failure."""
        r = solve(no_real_root, np.array([0.0]), max_iter=50, display=False)
        assert not r.success
        assert r.x is not None


# ---------------------------------------------------------------------------
# Tests: invalid inputs
# ---------------------------------------------------------------------------

class TestInvalidInputs:

    def test_unknown_method_string(self):
        with pytest.raises(ValueError, match="Unknown method"):
            solve(lambda x: x, np.array([0.0]), method='newton_raphson',
                  display=False)

    def test_unknown_problem_type_string(self):
        with pytest.raises(ValueError, match="Unknown problem_type"):
            solve(lambda x: x, np.array([0.0]),
                  problem_type='super_hard', display=False)

    def test_problem_type_wrong_type(self):
        with pytest.raises(TypeError):
            solve(lambda x: x, np.array([0.0]),
                  problem_type=3.14, display=False)

    def test_jac_wrong_type(self):
        with pytest.raises(TypeError, match="callable or None"):
            from deuflhard_newton.jac import resolve_jacobian
            resolve_jacobian(lambda x: x, "finite_diff")

    def test_problem_type_string_variants(self):
        """Various string forms are all accepted."""
        for pt in ("linear", "mildly_nonlinear", "highly_nonlinear", "extremely_nonlinear"):
            r = solve(lambda x: x - 1.0, np.array([0.0]),
                      problem_type=pt, display=False)
            # Should parse successfully (may or may not converge for this trivial fn)
            assert isinstance(r.success, bool)

    def test_problem_type_enum(self):
        """ProblemType enum values are accepted directly."""
        for pt in ProblemType:
            r = solve(lambda x: x - 1.0, np.array([0.0]),
                      problem_type=pt, display=False)
            assert isinstance(r.success, bool)


# ---------------------------------------------------------------------------
# Tests: auto-Jacobian error handling
# ---------------------------------------------------------------------------

class TestAutoJacobian:

    def test_jac_none_converges(self):
        """jac=None with csdiff gives a correct result for a smooth function."""
        def f(x):
            return np.array([x[0] ** 2 - 2.0])

        # x0=0 gives J=[[0]] (singular); use x0=1.5 where J=[[3]] is nonsingular
        r = nleq_err(f, np.array([1.5]), jac=None, tol=1e-10, display=False)
        assert r.success
        np.testing.assert_allclose(r.x[0], np.sqrt(2.0), atol=1e-9)

    def test_jac_callable_used_verbatim(self):
        """Explicit jac callable is not modified by resolve_jacobian."""
        call_count = [0]

        def my_jac(x):
            call_count[0] += 1
            return np.array([[2.0 * x[0]]])

        fn = resolve_jacobian(lambda x: np.array([x[0] ** 2 - 2.0]), my_jac)
        assert fn is my_jac

    def test_jac_none_scalar_output(self):
        """jac=None works when func: R^n -> R^1 (gradient path in csdiff)."""
        def f(x):
            # m=1, n=2: result is a 1-element vector
            return np.array([x[0] ** 2 + x[1] ** 2 - 1.0])

        r = nleq_err(f, np.array([0.5, 0.5]), jac=None, tol=1e-8, display=False)
        assert r.success
        np.testing.assert_allclose(np.linalg.norm(r.x), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: singular Jacobian (rank-deficient)
# ---------------------------------------------------------------------------

class TestSingularJacobian:

    def test_rank_deficient_does_not_crash(self):
        """Rank-1 Jacobian must not raise LinAlgError — fall back to lstsq."""
        # f(x) = [x[0], 0]  has Jacobian [[1,0],[0,0]] — rank 1 at every x.
        # The system has no root (second equation is always 0 ≠ 0... wait,
        # f = [x0, 0] has root x0=0, x1=anything).
        # Use a system with no true root but a well-defined pseudoinverse step.
        def f_rank1(x):
            return np.array([x[0] + x[1] - 1.0, x[0] + x[1] - 2.0])

        def j_rank1(x):
            return np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = nleq_err(f_rank1, np.array([0.5, 0.5]), j_rank1,
                         max_iter=20, display=False)
        # Must not raise; must return a result object
        assert r.x is not None
        assert r.x.shape == (2,)
        # A RuntimeWarning about rank deficiency should have been issued
        rank_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)
                         and "rank" in str(x.message).lower()]
        assert len(rank_warnings) >= 1

    def test_singular_jac_warning_message(self):
        """Warning message must mention rank and pseudoinverse."""
        def f_sing(x):
            return np.array([x[0] - x[1], 0.0])

        def j_sing(x):
            return np.array([[1.0, -1.0], [0.0, 0.0]])   # rank 1

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nleq_err(f_sing, np.array([1.0, 0.0]), j_sing,
                     max_iter=5, display=False)

        rank_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert any("rank" in str(x.message).lower() or "pseudoinverse" in str(x.message).lower()
                   for x in rank_warnings)


# ---------------------------------------------------------------------------
# Tests: display=False produces no output
# ---------------------------------------------------------------------------

class TestDisplay:

    def test_no_output_when_display_false(self, capsys):
        solve(lambda x: x - 1.0, np.array([0.0]),
              problem_type='mildly_nonlinear', display=False)
        out, err = capsys.readouterr()
        assert out == ""
        assert err == ""

    def test_output_when_display_true(self, capsys):
        solve(lambda x: x - 1.0, np.array([0.0]),
              problem_type='mildly_nonlinear', display=True)
        out, _ = capsys.readouterr()
        assert len(out) > 0

    def test_table_rows_equal_width(self, capsys):
        """All table data/header rows (those with column separators) have equal width."""
        solve(lambda x: np.array([x[0] ** 2 - 2.0, x[1] - 1.0]),
              np.array([1.0, 0.5]), display=True)
        out, _ = capsys.readouterr()
        lines = out.splitlines()
        # Keep only column-aligned rows: must have many '|' characters (header + data).
        # Info lines and footer lines use fewer '|' separators.
        table_lines = [l for l in lines if l.count("|") >= 5]
        widths = [len(l) for l in table_lines]
        if widths:
            assert len(set(widths)) == 1, (
                f"Table rows have inconsistent widths: {set(widths)}"
            )
