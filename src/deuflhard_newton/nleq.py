"""
Implementation of global Newton-type methods for nonlinear problems.

This code is an independent implementation based on the algorithmic descriptions found in:

    Deuflhard, P. (2011).
    Newton Methods for Nonlinear Problems.
    Springer Series in Computational Mathematics, Vol. 35.
    https://doi.org/10.1007/978-3-642-23899-4

Important notes:
- This implementation is written from scratch.
- No text, pseudocode, or figures from the book are reproduced.
- The code reflects my interpretation of the underlying mathematical ideas.
- This software is not affiliated with or endorsed by the author or the publisher.

License:
- This code is licensed under the MIT License (see LICENSE file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional, Tuple

import numpy as np

from deuflhard_newton.scaling import Scale

Array = np.ndarray


class ProblemType(IntEnum):
    LINEAR = 1
    MILDLY_NONLINEAR = 2
    HIGHLY_NONLINEAR = 3
    EXTREMELY_NONLINEAR = 4


def nleq_err(func: Callable[[Array], Array],
             x0: Array,
             jac: Callable[[Array], Array],
             tol: float = 1e-3,
             max_iter: int = 1_000,
             user_scaling: float | np.ndarray = None,
             problem_type: ProblemType = ProblemType.MILDLY_NONLINEAR,
             display: bool = True) -> Tuple[Optional[Array], 'NLEQResults']:
    """
    Global Newton method with error oriented convergence criterion and adaptive trust region strategy.

    Parameters
    ----------
    func:
        Function of R^n -> R^m to find the root of.
    x0:
        Initial guess, in R^n.
    jac:
        Jacobian.
    tol:
        Wished tolerance.
    max_iter:
        Maximum number of iterations.
    user_scaling:
        Scaling factor for x. Scaling will retain, component-wise, max(x0_i, user_scaling_i).
        If an array is given, it must be the same shape as x0.
        If None, unit scale is applied.
    problem_type:
        Problem type in order to choose damping behavior according to Deuflhard's recommendation.
    display:
        Whether to display the iterations.
            |f(x)|: norm of the function
            |dx|: norm of the step in the Newton correction
            |dx_bar|: norm of the step in the simplified Newton correction
            |lb|: damping factor
            |mu|: Lipschitz's estimate
            |theta|: condition number

    Returns
    -------
    tuple:
        Array:
            Root, if convergence occurred.
        NLEQResults:
            Intermediate results of the algorithm.

    """

    # Storage
    max_iter = int(max_iter)
    results = NLEQResults.allocate(max_iter)

    # State-space dimension
    n = x0.size

    # Solution state: updated inplace through the algorithm until convergence
    x_sol = x0.copy().astype(float)

    # First evaluation
    f_x = func(x_sol)
    results.n_func_evals += 1

    # Function-space dimension
    m = f_x.size

    # Initial damping parameters (Deuflhard recommendations)
    if problem_type <= ProblemType.MILDLY_NONLINEAR:
        lb = 1.
        lb_min = 1e-4
        restricted = False
    elif problem_type == ProblemType.HIGHLY_NONLINEAR:
        lb = 1e-2
        lb_min = 1e-4
        restricted = False
    else:
        # In hell, we get
        lb = 1e-4
        lb_min = 1e-8
        restricted = True

    # Allocation
    x_prev = x_sol.copy()
    x_candidate = np.zeros(n, dtype=float)
    dx = np.zeros(n, dtype=float)
    dx_bar = np.zeros(n, dtype=float)
    w = np.zeros(n, dtype=float)

    # Scaling
    unit_scale_x = Scale(dimension=n, tol=tol)
    unit_scale_f = Scale(dimension=m, tol=tol)

    if user_scaling is None:
        user_scaling = np.ones(n, dtype=float)
    elif not isinstance(user_scaling, np.ndarray):
        user_scaling = np.tile(float(user_scaling), n)

    init_weight_x = np.maximum(x0, user_scaling)
    scale_x = Scale(dimension=n, tol=tol, init_weight=init_weight_x)

    # First storage
    results.f_norm[0] = unit_scale_f.evaluate_norm(f_x)

    if display:
        results.print_iteration(0, header_stride=10)

    # Let's start...
    for k in range(max_iter):
        results.iter += 1
        current_iter = results.iter

        # First thing, update scaling with current (intermediate) solution
        scale_x.update(x_sol, x_prev)

        if k > 0:
            # Before dx gets updated, store the norms for later Lipschitz estimate
            norm_dx_prev = scale_x.evaluate_norm(dx)
            norm_dx_bar_prev = scale_x.evaluate_norm(dx_bar)

        # Newton correction
        J_x = jac(x_sol)  # Jacobian will be re-used in the simplified Newton correction later, for regularity loop
        results.n_jac_evals += 1

        dx[:], _, _, _ = np.linalg.lstsq(J_x, -f_x)  # dx updated
        results.n_jac_inv += 1

        # Convergence test
        norm_dx = scale_x.evaluate_norm(dx)

        unit_scaled_norm_dx = unit_scale_x.evaluate_norm(dx)
        results.dx_norm[current_iter] = unit_scaled_norm_dx

        if norm_dx <= scale_x.tol:
            # Success!
            x_sol += dx

            # Storing the last up-to-date statistics
            results.f_norm[current_iter] = unit_scale_f.evaluate_norm(func(x_sol))
            results.success = True

            if display:
                results.print_iteration(current_iter, header_stride=10)

            return x_sol, results

        # Lipschitz estimate (prediction value for damping factor)
        if k > 0:
            w[:] = dx_bar - dx
            s = scale_x.evaluate_norm(w)
            mu = lb * (norm_dx_prev * norm_dx_bar_prev) / (s * norm_dx)  # does not crash: defined above when k > 0
            lb = min(1., mu)

        # Regularity test loop
        has_failed_once = False

        while True:
            # Regularity test
            if lb < lb_min:
                # Ouch, convergence failure...
                # Storing the last up-to-date statistics
                results.f_norm[current_iter] = unit_scale_f.evaluate_norm(f_x)
                # "results.dx_norm" is already up to date, as it is not modified in the regularity loop
                results.dx_bar_norm[current_iter] = unit_scale_x.evaluate_norm(dx_bar)
                results.lb[current_iter] = lb

                # Those two lines are not supposed to crash, as the first passage goes with lb > lb_min (cf. initial
                # damping parameters before algorithm loop). Actually there is a world in which theta is not defined,
                # e.g. mu < lb_min directly at second passage. I'll go with a safe try/except.
                try:
                    results.mu[current_iter] = mu  # should not crash, after quick look
                    results.theta[current_iter] = theta  # could crash, theoretically, if everything fails at first iter
                except NameError:
                    pass

                if display:
                    results.print_iteration(current_iter, header_stride=10)

                return None, results

            # Building candidate, hopefully passing monotonicity test
            x_candidate[:] = x_sol + lb * dx
            f_x_candidate = func(x_candidate)
            results.n_func_evals += 1

            # Simplified Newton correction ('old' Jacobian, 'new' right hand side)
            dx_bar[:], _, _, _ = np.linalg.lstsq(J_x, -f_x_candidate)
            results.n_jac_inv += 1

            norm_dx_bar = scale_x.evaluate_norm(dx_bar)

            # Update candidature criteria
            theta = norm_dx_bar / norm_dx

            w[:] = dx_bar - (1. - lb) * dx
            mu = (0.5 * norm_dx * lb * lb) / scale_x.evaluate_norm(w)

            # Monotonicity test, making us loop again and again...
            if (not restricted and theta >= 1.) or (restricted and theta > 1. - lb / 4.):
                # Candidate was not good enough, one more time!
                lb = min(mu, 0.5 * lb)

                if lb < lb_min and not has_failed_once:
                    # Last chance, matching exactly lb to lb_min: no more joker
                    lb = lb_min
                    has_failed_once = True

                continue

            # We got out of the while loop
            lb_new = min(1., mu)

            if lb == 1. and lb_new == 1.:
                # Convergence test
                if norm_dx_bar <= scale_x.tol:
                    # Success!
                    x_sol[:] = x_candidate + dx_bar

                    # Storing the last up-to-date statistics
                    results.f_norm[current_iter] = unit_scale_f.evaluate_norm(func(x_sol))
                    results.dx_bar_norm[current_iter] = unit_scale_x.evaluate_norm(dx_bar)
                    results.success = True

                    if display:
                        results.print_iteration(current_iter, header_stride=10)

                    return x_sol, results

                if theta < 0.5:
                    # TODO: switch to QNERR algorithm (paragraph 2.1.4)
                    pass

            elif lb_new >= 4. * lb:
                # Seems like a safer zone is at sight!
                lb = lb_new
                continue

            # Not good enough for definitive success, let's see the next iteration...
            x_prev[:] = x_sol
            x_sol[:] = x_candidate
            f_x[:] = f_x_candidate
            break

        results.f_norm[current_iter] = unit_scale_f.evaluate_norm(f_x)
        results.dx_bar_norm[current_iter] = unit_scale_x.evaluate_norm(dx_bar)
        results.lb[current_iter] = lb
        results.mu[current_iter] = mu
        results.theta[current_iter] = theta

        if display:
            results.print_iteration(current_iter, header_stride=10)

    return None, results


def nleq_res(func: Callable[[Array], Array],
             x0: Array,
             jac: Callable[[Array], Array],
             tol: float = 1e-3,
             max_iter: int = 1_000,
             user_scaling: float | np.ndarray = None,
             problem_type: ProblemType = ProblemType.MILDLY_NONLINEAR,
             display: bool = True) -> Tuple[Optional[Array], 'NLEQResults']:
    """
    Global Newton method with residual oriented convergence criterion and adaptive trust region strategy.

        Parameters
    ----------
    func:
        Function of R^n -> R^m to find the root of.
    x0:
        Initial guess, in R^n.
    jac:
        Jacobian.
    tol:
        Wished tolerance.
    max_iter:
        Maximum number of iterations.
    user_scaling:
        Scaling factor for f(x). Scaling will retain, component-wise, max(f(x0)_i, user_scaling_i).
        If an array is given, it must be the same shape as f(x0).
        If None, unit scale is applied.
    problem_type:
        Problem type in order to choose damping behavior according to Deuflhard's recommendation.
    display:
        Whether to display the iterations.
            |f(x)|: norm of the function
            |dx|: norm of the step in the Newton correction
            |dx_bar|: unused in this method (cf. nleq_err)
            |lb|: damping factor
            |mu|: Lipschitz's estimate
            |theta|: condition number

    Returns
    -------
    tuple:
        Array:
            Root, if convergence occurred.
        NLEQResults:
            Intermediate results of the algorithm.

    """

    # Storage
    max_iter = int(max_iter)
    results = NLEQResults.allocate(max_iter)

    # State-space dimension
    n = x0.size

    # Solution state: updated inplace through the algorithm until convergence
    x_sol = x0.copy().astype(float)

    # First evaluation
    f_x = func(x_sol)
    results.n_func_evals += 1

    # Function-space dimension
    m = f_x.size

    # Initial damping parameters (Deuflhard recommendations)
    if problem_type <= ProblemType.MILDLY_NONLINEAR:
        lb = 1.
        lb_min = 1e-4
        restricted = False
    elif problem_type == ProblemType.HIGHLY_NONLINEAR:
        lb = 1e-2
        lb_min = 1e-4
        restricted = False
    else:
        lb = 1e-4
        lb_min = 1e-8
        restricted = True

    # Allocation
    f_x_prev = f_x.copy()
    f_x_candidate = np.zeros(m, dtype=float)
    x_candidate = np.zeros(n, dtype=float)
    dx = np.zeros(n, dtype=float)
    w = np.zeros(m, dtype=float)

    # Scaling
    unit_scale_x = Scale(dimension=n, tol=tol)
    unit_scale_f = Scale(dimension=m, tol=tol)

    if user_scaling is None:
        user_scaling = np.ones(n, dtype=float)
    elif not isinstance(user_scaling, np.ndarray):
        user_scaling = np.tile(float(user_scaling), n)

    init_weight_f = np.maximum(f_x, user_scaling)
    scale_f = Scale(dimension=m, tol=tol, init_weight=init_weight_f)

    # First norm evaluation
    norm_f_x = scale_f.evaluate_norm(f_x)
    results.f_norm[0] = unit_scale_f.evaluate_norm(f_x)

    if display:
        results.print_iteration(0, header_stride=10)

    # Let's start...
    for k in range(max_iter):
        results.iter += 1
        current_iter = results.iter

        # Convergence test
        if norm_f_x <= scale_f.tol:
            # Success!
            results.success = True
            return x_sol, results

        # Newton correction
        J_x = jac(x_sol)
        results.n_func_evals += 1

        dx[:], _, _, _ = np.linalg.lstsq(J_x, -f_x)
        results.n_jac_inv += 1

        # Lipschitz estimate (prediction value for damping factor)
        if k > 0:
            mu = mu * norm_f_x_prev / norm_f_x  # does not crash: defined at first passage i.e. k = 0
            lb = min(1., mu)

        # Regularity test loop
        has_failed_once = False

        while True:
            # Regularity test
            if lb < lb_min:
                # Ouch, convergence failure...
                # Storing the last up-to-date statistics
                results.f_norm[current_iter] = unit_scale_f.evaluate_norm(f_x)
                results.dx_norm[current_iter] = unit_scale_x.evaluate_norm(dx)
                results.lb[current_iter] = lb

                # Those two lines are not supposed to crash, as the first passage goes with lb > lb_min (cf. initial
                # damping parameters before algorithm loop). Actually there is a world in which theta is not defined,
                # e.g. mu < lb_min directly at second passage. I'll go with a safe try/except.
                try:
                    results.mu[current_iter] = mu  # should not crash, after quick look
                    results.theta[current_iter] = theta  # could crash, theoretically, if everything fails at first iter
                except NameError:
                    pass

                if display:
                    results.print_iteration(current_iter, header_stride=10)

                return None, results

            # Building candidate, hopefully passing monotonicity test
            x_candidate[:] = x_sol + lb * dx
            f_x_candidate[:] = func(x_candidate)
            results.n_func_evals += 1

            # Update candidature criteria
            theta = scale_f.evaluate_norm(f_x_candidate) / norm_f_x

            w[:] = f_x_candidate - (1. - lb) * f_x
            mu = (0.5 * norm_f_x * lb * lb) / scale_f.evaluate_norm(w)

            # Monotonicity test, making us loop again and again...
            if (not restricted and theta >= 1.) or (restricted and theta > 1. - lb / 4.):
                # Candidate was not good enough, one more time!
                lb = min(mu, 0.5 * lb)

                if lb < lb_min and not has_failed_once:
                    # Last chance, matching exactly lb to lb_min: no more joker
                    lb = lb_min
                    has_failed_once = True

                continue

            # We got out of the while loop
            lb_new = min(1., mu)

            if lb == 1. and lb_new == 1.:
                # TODO: switch to QNRES algorithm (paragraph 2.2.3)
                pass

            elif lb_new >= 4. * lb:
                # Seems like a safer zone is at sight!
                lb = lb_new
                continue

            # Not good enough for definitive success, let's see the next iteration...
            x_sol[:] = x_candidate
            f_x_prev[:] = f_x
            f_x[:] = f_x_candidate
            break

        # Update scaling with update (intermediate) solution and compute new norms
        scale_f.update(f_x, f_x_prev)

        norm_f_x = scale_f.evaluate_norm(f_x)
        norm_f_x_prev = scale_f.evaluate_norm(f_x_prev)

        # Store results
        results.f_norm[current_iter] = unit_scale_f.evaluate_norm(f_x)
        results.dx_norm[current_iter] = unit_scale_x.evaluate_norm(dx)
        results.lb[current_iter] = lb
        results.mu[current_iter] = mu
        results.theta[current_iter] = theta

        if display:
            results.print_iteration(current_iter, header_stride=10)

    return None, results


@dataclass
class NLEQResults:
    """
    Dataclass to store intermediate results from NLEQ algorithm.
    """

    f_norm: np.ndarray
    dx_norm: np.ndarray
    dx_bar_norm: np.ndarray
    lb: np.ndarray
    mu: np.ndarray
    theta: np.ndarray

    max_iter: int
    iter: int = 0
    n_func_evals: int = 0
    n_jac_evals: int = 0
    n_jac_inv: int = 0

    success: bool = False

    _COL_WIDTH: int = field(default=14, init=False)

    @classmethod
    def allocate(cls, max_iter: int) -> 'NLEQResults':
        """
        Allocation of NLEQ results based on a priori known max_iter.
        """
        return cls(
            f_norm=np.full(max_iter + 1, np.nan),
            dx_norm=np.full(max_iter + 1, np.nan),
            dx_bar_norm=np.full(max_iter + 1, np.nan),
            lb=np.full(max_iter + 1, np.nan),
            mu=np.full(max_iter + 1, np.nan),
            theta=np.full(max_iter + 1, np.nan),
            max_iter=max_iter,
        )

    def print_header(self) -> None:
        """
        Print header of NLEQ results table.
        """
        headers = ["iter", "|f(x)|", "|dx|", "|dx_bar|", "lb", "mu", "theta"]
        sep = " | "

        header = "|" + sep.join(f"{h:^{self._COL_WIDTH}}" for h in headers) + " |"
        line = "+" + "-" * (len(header) - 2) + "+"

        print(line)
        print(header)
        print(line)

    def print_iteration(self, k: int, header_stride: int = None) -> None:
        """
        Print the values of iteration k.
        If "header_stride" is given, the header will be printed if k % header_stride == 0.
        """

        values = [
            k,
            self.f_norm[k],
            self.dx_norm[k],
            self.dx_bar_norm[k],
            self.lb[k],
            self.mu[k],
            self.theta[k],
        ]

        sep = " | "

        def fmt(val) -> str:
            if np.isnan(val):
                return f"{'-':^{self._COL_WIDTH}}"
            elif isinstance(val, int):
                val = f'{val:0{len(str(self.max_iter))}}'
                return f"{val:^{self._COL_WIDTH}}"
            return f"{val:^{self._COL_WIDTH}.6e}"

        row = "|" + sep.join(fmt(v) for v in values) + " |"

        if header_stride is not None and k % header_stride == 0:
            self.print_header()

        print(row)
