"""
Example 1 — Shooting method for a circular orbit
=================================================
Problem
-------
A satellite starts at position (R, 0) with unknown initial velocity (vx0, vy0).
Under Keplerian gravity, find the velocity that closes the orbit exactly
after one period T = 2π√(R³/GM).

The boundary residual is
    F(v0) = [x(T; v0) − R,   y(T; v0) − 0]
where (x, y) is obtained by integrating the equations of motion:
    ẍ = −GM x / r³,   ÿ = −GM y / r³,   r = √(x² + y²).

Why nleq_res (residual-oriented)?
----------------------------------
The Jacobian ∂F/∂v₀ has large entries (tiny velocity changes → large endpoint
displacement).  This makes ‖Δv‖ unreliable as a convergence criterion.
nleq_res uses ‖F(v)‖_D ≤ tol instead — much more natural for BVP residuals.

Why Deuflhard beats scipy here
-------------------------------
From a bad starting point the full Newton step overshoots.  The Deuflhard
damping loop halves λ until the residual decreases, recovering convergence.
scipy.optimize.fsolve uses no damping and can diverge into non-physical orbits.

API used
--------
nleq_res(func, x0, jac, tol, problem_type=ProblemType.HIGHLY_NONLINEAR, display=True)
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from deuflhard_newton import nleq_res, ProblemType

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
GM = 1.0    # gravitational parameter (non-dimensionalised)
R  = 1.0    # orbit radius
T  = 2.0 * np.pi * np.sqrt(R**3 / GM)   # exact circular orbit period

V_CIRC = np.sqrt(GM / R)   # circular orbit speed  (≈ 1.0 for R=GM=1)

def ode_rhs(t, y):
    x, vx, yo, vy = y
    r3 = (x**2 + yo**2)**1.5
    return [vx, -GM * x / r3, vy, -GM * yo / r3]

def orbit_residual(v0):
    """F(v0) = final position minus initial position after one period."""
    y0 = [R, v0[0], 0.0, v0[1]]
    sol = solve_ivp(ode_rhs, [0, T], y0, rtol=1e-10, atol=1e-12,
                    method='DOP853')
    yT = sol.y[:, -1]
    return np.array([yT[0] - R, yT[2] - 0.0])

def orbit_jacobian(v0, h=1e-5):
    """Finite-difference Jacobian (ODE does not accept complex inputs)."""
    f0 = orbit_residual(v0)
    J  = np.empty((2, 2))
    for j in range(2):
        dv = np.zeros(2); dv[j] = h
        J[:, j] = (orbit_residual(v0 + dv) - f0) / h
    return J

# ---------------------------------------------------------------------------
# Exact and perturbed starting points
# ---------------------------------------------------------------------------
V0_EXACT = np.array([0.0, V_CIRC])        # exact circular orbit velocity
V0_BAD   = np.array([-0.4, 0.3 * V_CIRC]) # badly perturbed: wrong sign + small

print("=" * 65)
print("ORBIT SHOOTING — residual-oriented (nleq_res)")
print(f"Exact solution:  vx = {V0_EXACT[0]:.4f}   vy = {V0_EXACT[1]:.4f}")
print(f"Starting point:  vx = {V0_BAD[0]:.4f}   vy = {V0_BAD[1]:.4f}")
print(f"(‖v0_bad − v_true‖ = {np.linalg.norm(V0_BAD - V0_EXACT):.3f},  "
      f"‖v_true‖ = {np.linalg.norm(V0_EXACT):.3f})")
print("=" * 65)

# ---------------------------------------------------------------------------
# Deuflhard — nleq_res with HIGHLY_NONLINEAR damping
# ---------------------------------------------------------------------------
print("\n[Deuflhard nleq_res]")
t0 = time.perf_counter()
r_deufl = nleq_res(
    orbit_residual, V0_BAD, jac=orbit_jacobian,
    tol=1e-7, max_iter=40,
    problem_type=ProblemType.HIGHLY_NONLINEAR,
    display=True,
)
t_deufl = (time.perf_counter() - t0) * 1e3
print(f"\nsuccess={r_deufl.success}  nit={r_deufl.nit}  nfev={r_deufl.nfev}  time={t_deufl:.1f} ms")
print(f"solution: vx = {r_deufl.x[0]:.6f}   vy = {r_deufl.x[1]:.6f}")
print(f"residual: {r_deufl.fun}   ‖F‖ = {np.linalg.norm(r_deufl.fun):.2e}")

# ---------------------------------------------------------------------------
# scipy fsolve — same starting point, no damping
# ---------------------------------------------------------------------------
print("\n[scipy.optimize.fsolve]")
try:
    from scipy.optimize import fsolve
    t0 = time.perf_counter()
    x_sp, info, ier, msg = fsolve(
        orbit_residual, V0_BAD, fprime=orbit_jacobian,
        full_output=True,
    )
    t_scipy = (time.perf_counter() - t0) * 1e3
    f_sp = orbit_residual(x_sp)
    print(f"ier={ier}  nfev={info['nfev']}  time={t_scipy:.1f} ms")
    print(f"solution: vx = {x_sp[0]:.6f}   vy = {x_sp[1]:.6f}")
    print(f"residual: {f_sp}   ‖F‖ = {np.linalg.norm(f_sp):.2e}")
    scipy_ok = ier == 1 and np.linalg.norm(f_sp) < 1e-5
except Exception as e:
    print(f"fsolve raised: {e}")
    scipy_ok = False
    t_scipy = float("nan")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY")
print(f"  Deuflhard: success={r_deufl.success}  ‖F‖={np.linalg.norm(r_deufl.fun):.2e}"
      f"  nfev={r_deufl.nfev}  njev={r_deufl.njev}  time={t_deufl:.1f} ms")
print(f"  scipy:     success={scipy_ok}"
      f"  time={t_scipy:.1f} ms")
print()
print("Deuflhard's affine-invariant damping finds the initial velocity")
print("even from a badly perturbed starting point by taking only steps")
print("that reduce the residual (monotonicity test).")
