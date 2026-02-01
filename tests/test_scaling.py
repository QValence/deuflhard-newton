import numpy as np

from deuflhard_newton import nleq_err


def test_user_scaling_vector():
    def f(x: np.ndarray) -> np.ndarray:
        return np.array([
            10.0 * x[0] - 1.0,
            0.1 * x[1] - 1.0,
        ])

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array([
            [10.0, 0.0],
            [0.0, 0.1],
        ])

    x0 = np.ones(2)
    scaling = np.array([10.0, 0.1])

    x, info = nleq_err(
        f,
        x0,
        jac,
        tol=1e-8,
        user_scaling=scaling,
        display=False,
    )

    assert x is not None
    assert info.success

    np.testing.assert_allclose(x, [0.1, 10.0], atol=1e-8)
