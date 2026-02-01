import numpy as np

from deuflhard_newton import nleq_err, nleq_res


def test_linear_system():
    def f(x: np.ndarray) -> np.ndarray:
        return np.array([
            2.0 * x[0] + x[1] - 1.0,
            x[0] - x[1],
        ])

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array([
            [2.0, 1.0],
            [1.0, -1.0],
        ])

    x0 = np.zeros(2)

    x_err, res_err = nleq_err(f, x0, jac, tol=1e-10, display=False)
    x_res, res_res = nleq_res(f, x0, jac, tol=1e-10, display=False)

    assert x_err is not None
    assert res_err.success
    assert x_res is not None
    assert res_res.success

    np.testing.assert_allclose(x_err, [1 / 3, 1 / 3], atol=1e-8)
    np.testing.assert_allclose(x_res, [1 / 3, 1 / 3], atol=1e-8)
