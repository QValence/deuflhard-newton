"""
Example 4 — Nonlinear least squares: Michaelis-Menten kinetics
===============================================================
Model
-----
Enzyme kinetics follow the Michaelis-Menten equation:

    v(S) = Vmax * S / (Km + S)

where  Vmax  is the maximum reaction rate and  Km  is the half-saturation
constant.  We fit these two parameters to noisy experimental data.

This is an overdetermined system (m=15 data points, n=2 parameters).
No exact zero of the residual exists; the LM solver minimises ½‖r(p)‖²
instead.

Key distinction from root-finding
-----------------------------------
result.fun is NOT zero at the solution — it is the residual vector whose
2-norm squared represents the measurement noise.  The gradient ‖Jᵀr‖ → 0
signals that we are at a local minimum.

API used
--------
lm(func, x0, jac=None, tol=1e-8, display=True)

Comparison
----------
We verify against scipy.optimize.curve_fit (which uses the same LM algorithm
under the hood).  Results should be nearly identical.
"""

import time
import numpy as np
from deuflhard_newton import lm

np.random.seed(0)

# ---------------------------------------------------------------------------
# True parameters and synthetic data
# ---------------------------------------------------------------------------
VMAX_TRUE = 1.8    # μmol / (min · mg enzyme)
KM_TRUE   = 0.25   # mM

S_data = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                   0.60, 0.80, 1.00, 1.50, 2.00, 3.00, 5.00, 10.0])   # mM

noise = 0.04 * np.random.randn(len(S_data))
v_data = VMAX_TRUE * S_data / (KM_TRUE + S_data) + noise

print("=" * 65)
print("MICHAELIS-MENTEN CURVE FITTING")
print(f"True parameters: Vmax={VMAX_TRUE:.3f}, Km={KM_TRUE:.3f}")
print("=" * 65)

# ---------------------------------------------------------------------------
# Residual function (m=15 equations, n=2 parameters)
# ---------------------------------------------------------------------------
def residuals(params):
    Vmax, Km = params[0], params[1]
    return Vmax * S_data / (Km + S_data) - v_data

def jac_analytic(params):
    """Analytical Jacobian for maximum speed (2 parameters → 2 columns)."""
    Vmax, Km = params[0], params[1]
    denom = (Km + S_data)**2
    dr_dVmax = S_data / (Km + S_data)
    dr_dKm   = -Vmax * S_data / denom
    return np.column_stack([dr_dVmax, dr_dKm])

# Initial guess
p0 = np.array([1.0, 0.5])   # rough starting estimate

# ---------------------------------------------------------------------------
# Fit with LM + auto-Jacobian
# ---------------------------------------------------------------------------
print("\n[LM with jac=None (auto-diff)]")
t0 = time.perf_counter()
r_auto = lm(residuals, p0, jac=None, tol=1e-10, display=True)
t_auto = (time.perf_counter() - t0) * 1e3
print(f"\nsuccess={r_auto.success}  nit={r_auto.nit}  nfev={r_auto.nfev}  njev={r_auto.njev}  time={t_auto:.1f} ms")
print(f"Fitted Vmax = {r_auto.x[0]:.4f}  (true {VMAX_TRUE})")
print(f"Fitted Km   = {r_auto.x[1]:.4f}  (true {KM_TRUE})")
print(f"Residual RMS = {np.sqrt(np.mean(r_auto.fun**2)):.4f}  (noise level ≈ {0.04:.4f})")

# ---------------------------------------------------------------------------
# Fit with LM + analytical Jacobian
# ---------------------------------------------------------------------------
print("\n[LM with analytical Jacobian]")
t0 = time.perf_counter()
r_ana = lm(residuals, p0, jac=jac_analytic, tol=1e-10, display=False)
t_ana = (time.perf_counter() - t0) * 1e3
print(f"success={r_ana.success}  nit={r_ana.nit}  nfev={r_ana.nfev}  njev={r_ana.njev}  time={t_ana:.1f} ms")
print(f"Fitted Vmax = {r_ana.x[0]:.4f}")
print(f"Fitted Km   = {r_ana.x[1]:.4f}")

# ---------------------------------------------------------------------------
# Compare with scipy.optimize.curve_fit
# ---------------------------------------------------------------------------
print("\n[scipy.optimize.curve_fit]")
try:
    from scipy.optimize import curve_fit
    def mm_model(S, Vmax, Km):
        return Vmax * S / (Km + S)

    t0 = time.perf_counter()
    popt, pcov = curve_fit(mm_model, S_data, v_data, p0=p0)
    t_scipy = (time.perf_counter() - t0) * 1e3
    print(f"Fitted Vmax = {popt[0]:.4f}")
    print(f"Fitted Km   = {popt[1]:.4f}")
    print(f"Parameter std: Vmax±{np.sqrt(pcov[0,0]):.4f}, Km±{np.sqrt(pcov[1,1]):.4f}")
    print(f"time={t_scipy:.1f} ms")
except ImportError:
    print("(scipy not available)")
    t_scipy = float("nan")

# ---------------------------------------------------------------------------
# Convergence history plot (text)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("Convergence of ‖f‖ vs iteration (LM + auto-diff)")
print(f"  {'iter':>4}  {'‖f‖':>12}  {'‖Jᵀf‖':>12}")
f_hist  = r_auto.history["f_norm"]
gn_hist = r_auto.history["grad_norm"]
for i, (fn, gn) in enumerate(zip(f_hist, gn_hist)):
    if i <= 4 or i >= len(f_hist) - 3:
        print(f"  {i:>4}  {fn:12.6f}  {gn:12.2e}")
    elif i == 5:
        print("  ...")

print("\nNOTE: ‖f‖ ≠ 0 at the solution — this is expected for noisy data.")
print("The solver minimises ‖f‖², not roots of f=0.")

print("\n" + "=" * 65)
print("SUMMARY")
print(f"  LM auto-diff:     nfev={r_auto.nfev:3d}  njev={r_auto.njev:2d}  time={t_auto:.1f} ms")
print(f"  LM analytic jac:  nfev={r_ana.nfev:3d}  njev={r_ana.njev:2d}  time={t_ana:.1f} ms")
if not np.isnan(t_scipy):
    print(f"  scipy curve_fit:                      time={t_scipy:.1f} ms")
print()
print("Analytical Jacobian reduces nfev (no finite-difference column probes)")
print("and typically cuts wall time when m (number of data points) is large.")
