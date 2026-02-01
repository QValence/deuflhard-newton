import numpy as np

from deuflhard_newton import nleq_res


def test_non_convergence_returns_none():
    def f(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2 + 1.0])

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array([[2.0 * x[0]]])

    x0 = np.array([0.0])

    x, info = nleq_res(
        f,
        x0,
        jac,
        max_iter=5,
        display=False,
    )

    assert x is None
    assert not info.success
