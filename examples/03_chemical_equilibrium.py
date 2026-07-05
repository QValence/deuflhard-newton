"""
Example 3 — Chemical equilibrium: water-gas shift reaction
===========================================================
Reaction
--------
    CO + H₂O ⇌ CO₂ + H₂       K(T) = exp(-ΔG°/RT)

At T = 1000 K the equilibrium constant K ≈ 1.44.

Given initial mole fractions (before reaction), find equilibrium
concentrations using the law of mass action + atom-balance constraints.

Variables
---------
x = [n_CO, n_H2O, n_CO2, n_H2]   (moles)

Equations
---------
1. Mass action:  (n_CO2 * n_H2) / (n_CO * n_H2O)  =  K  (in moles, fixed total)
2. Carbon balance:   n_CO + n_CO2 = C_tot
3. Oxygen balance:   n_CO + n_H2O + 2*n_CO2 = O_tot
4. Hydrogen balance: 2*n_H2O + 2*n_H2 = H_tot

This is a 4×4 nonlinear system.  The equilibrium constant makes it
highly nonlinear: a 10× change in initial conditions can move the
equilibrium dramatically.

QNERR demonstration
--------------------
We compare nfev and njev for use_qn=True vs use_qn=False.

API used
--------
solve(func, x0, problem_type='highly_nonlinear', display=True)
solve(func, x0, use_qn=False)    # pure quadratic Newton
"""

import time
import numpy as np
from deuflhard_newton import solve, ProblemType

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_gas = 8.314     # J/(mol·K)
T     = 1000.0    # K
# Approximate ΔG° for  CO + H₂O → CO₂ + H₂  at 1000 K
DG0   = -3300.0   # J/mol  (slightly exothermic at this T)
K_eq  = np.exp(-DG0 / (R_gas * T))   # ≈ 1.48

print(f"T = {T} K,  K_eq = {K_eq:.4f}")

# ---------------------------------------------------------------------------
# Initial composition (moles): 1 mol CO + 1 mol H₂O, 0 CO₂, 0 H₂
# ---------------------------------------------------------------------------
n0 = np.array([1.0, 1.0, 0.0, 0.0])   # [CO, H2O, CO2, H2]
C_tot = n0[0] + n0[2]    # carbon conservation
O_tot = n0[0] + n0[1] + 2*n0[2]   # oxygen conservation
H_tot = 2*n0[1] + 2*n0[3]         # hydrogen conservation

def equilibrium_residual(x):
    n_CO, n_H2O, n_CO2, n_H2 = x
    # Guard against non-physical (negative) concentrations during iteration
    eps = 1e-30
    n_CO   = max(n_CO,  eps)
    n_H2O  = max(n_H2O, eps)
    n_CO2  = max(n_CO2, eps)
    n_H2   = max(n_H2,  eps)
    f1 = n_CO2 * n_H2 - K_eq * n_CO * n_H2O   # mass action (rearranged)
    f2 = n_CO + n_CO2 - C_tot                  # carbon balance
    f3 = n_CO + n_H2O + 2.0*n_CO2 - O_tot     # oxygen balance
    f4 = 2.0*n_H2O + 2.0*n_H2 - H_tot         # hydrogen balance
    return np.array([f1, f2, f3, f4])

# Starting point: pure reactants (no products yet)
x0 = np.array([1.0, 1.0, 1e-6, 1e-6])

# ---------------------------------------------------------------------------
# Solve with auto-Jacobian (csdiff) + QNERR
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("Solving with jac=None (auto-diff) + use_qn=True (QNERR active)")
print("=" * 65)
t0 = time.perf_counter()
r_qn = solve(
    equilibrium_residual, x0,
    problem_type=ProblemType.HIGHLY_NONLINEAR,
    tol=1e-10, display=True,
)
t_qn = (time.perf_counter() - t0) * 1e3
print(f"\nsuccess={r_qn.success}  nit={r_qn.nit}  nfev={r_qn.nfev}  njev={r_qn.njev}  time={t_qn:.1f} ms")
print("Equilibrium concentrations [CO, H2O, CO2, H2]:")
print(f"  {r_qn.x}")
K_check = r_qn.x[2] * r_qn.x[3] / (r_qn.x[0] * r_qn.x[1])
print(f"  Verified K = {K_check:.4f}  (target {K_eq:.4f})")

# ---------------------------------------------------------------------------
# Solve without QNERR — pure quadratic Newton
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("Solving with use_qn=False (pure quadratic Newton)")
print("=" * 65)
t0 = time.perf_counter()
r_full = solve(
    equilibrium_residual, x0,
    problem_type=ProblemType.HIGHLY_NONLINEAR,
    tol=1e-10, display=False, use_qn=False,
)
t_full = (time.perf_counter() - t0) * 1e3
print(f"success={r_full.success}  nit={r_full.nit}  nfev={r_full.nfev}  njev={r_full.njev}  time={t_full:.1f} ms")

# ---------------------------------------------------------------------------
# Compare with scipy
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("scipy.optimize.fsolve comparison")
print("=" * 65)
try:
    from scipy.optimize import fsolve
    t0 = time.perf_counter()
    x_sp, info, ier, msg = fsolve(
        equilibrium_residual, x0, full_output=True,
    )
    t_scipy = (time.perf_counter() - t0) * 1e3
    print(f"ier={ier}  nfev={info['nfev']}  time={t_scipy:.1f} ms")
    print(f"scipy solution: {x_sp}")
    K_sp = x_sp[2] * x_sp[3] / (x_sp[0] * x_sp[1])
    print(f"Verified K = {K_sp:.4f}")
except ImportError:
    print("(scipy not available)")
    t_scipy = float("nan")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY: function evaluations and wall time")
print(f"  QNERR  (use_qn=True)  :  nfev={r_qn.nfev:3d}  njev={r_qn.njev:2d}  time={t_qn:.1f} ms")
print(f"  Full Newton (use_qn=False): nfev={r_full.nfev:3d}  njev={r_full.njev:2d}  time={t_full:.1f} ms")
if not np.isnan(t_scipy):
    print(f"  scipy fsolve:               nfev={info['nfev']:3d}            time={t_scipy:.1f} ms")
print()
print("QNERR: fewer Jacobian evaluations (njev), each costing ~n f-evals via auto-diff.")
print("Benefit grows with n; for small n the Python overhead can dominate wall time.")
