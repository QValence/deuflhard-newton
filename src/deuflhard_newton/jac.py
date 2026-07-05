"""
Jacobian resolution: wraps user-supplied callables or auto-computes via csdiff.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def resolve_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    jac: Optional[Callable[[np.ndarray], np.ndarray]],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable ``jac_fn(x) -> ndarray shape (m, n)``.

    Parameters
    ----------
    func:
        The function being solved, f: R^n -> R^m.
    jac:
        If a callable, returned as-is — no wrapping or validation.
        If None, the Jacobian is computed automatically via ``csdiff``
        (complex step differentiation, machine-precision accuracy).

    Returns
    -------
    callable:
        A function ``jac_fn(x)`` returning the Jacobian as shape ``(m, n)``.

    Raises
    ------
    ImportError:
        If ``jac=None`` and ``csdiff`` is not installed.
    TypeError:
        If ``jac`` is neither None nor callable.

    Notes
    -----
    When ``jac=None``, the complex step method is used:

        J[i, j] = Im(f_i(x + i·h·e_j)) / h,   h ≈ ε^(3/2) ≈ 1e-23

    This achieves ~1e-16 relative accuracy without any step-size tuning and
    is strictly superior to finite differences for smooth functions.

    Supply an explicit ``jac`` when:
    - The function cannot accept complex inputs (external C code, integer ops,
      functions using ``numpy.abs`` on the result, etc.)
    - You have an analytical Jacobian and need maximum throughput.
    - You are exploiting sparsity.
    """
    if jac is None:
        return _make_auto_jac(func)

    if callable(jac):
        return jac

    raise TypeError(
        f"jac must be a callable or None, got {type(jac).__name__!r}. "
        "Pass a function jac(x) -> ndarray of shape (m, n), "
        "or None to use automatic differentiation via csdiff."
    )


def _make_auto_jac(
    func: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Build an auto-differentiation Jacobian wrapper using csdiff."""
    try:
        import csdiff
    except ImportError as exc:
        raise ImportError(
            "Automatic Jacobian computation requires the 'csdiff' package.\n"
            "Install it with:\n"
            "    pip install 'deuflhard-newton[autodiff]'\n"
            "Or supply an explicit jac= callable."
        ) from exc

    def auto_jac(x: np.ndarray) -> np.ndarray:
        n = x.size
        m = np.asarray(func(x)).ravel().size
        if m == 1:
            if n == 1:
                # f: R→R — gradient() requires n>1, use derivative() instead.
                # _scalar must NOT cast to float: csdiff passes complex t to
                # extract Im(f(x+ih))/h, and float() would discard imaginary part.
                def _scalar(t):
                    return np.asarray(func(np.array([t]))).ravel()[0]
                return np.array([[csdiff.derivative(_scalar, float(x[0]))]])
            # gradient() expects f: R^n → scalar, not 1-element array.
            def _scalar_func(x_):
                return np.asarray(func(x_)).ravel()[0]
            return csdiff.gradient(_scalar_func, x).reshape(1, -1)
        return csdiff.jacobian(func, x)

    return auto_jac
