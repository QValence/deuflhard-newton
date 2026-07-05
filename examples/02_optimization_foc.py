"""
Example 2 — Portfolio optimisation via KKT / first-order conditions
====================================================================
Problem (linear case)
---------------------
Minimise  ½ xᵀΣx − μᵀx   subject to  1ᵀx = 1  (fully invested).

The Lagrangian FOC gives a linear KKT system:
    [Σ   −1] [x]   [μ]
    [1ᵀ   0] [λ] = [1]

One Newton step from any starting point reaches the exact solution.
This demonstrates the LINEAR problem type and shows that scaling matters
when asset returns span different orders of magnitude.

Problem (nonlinear extension)
------------------------------
Replace the quadratic utility with a CRRA (constant relative risk aversion)
utility: maximise  E[u(W)] ≈ μᵀx − (γ/2) xᵀΣx − (γ²/6)(σ² corrections).

For illustration we use a simple proxy: maximise  μᵀx − λ/2 xᵀΣx − δ ‖x‖⁴
where the quartic term creates genuine nonlinearity.  Multiple Newton steps
are now required.

API used
--------
solve(func, x0, jac, problem_type='linear', display=True)
nleq_err(func, x0, jac, problem_type=ProblemType.MILDLY_NONLINEAR)
"""

import numpy as np
from deuflhard_newton import solve, nleq_err, ProblemType

np.random.seed(42)

# ---------------------------------------------------------------------------
# Part A — Linear KKT system
# ---------------------------------------------------------------------------

n_assets = 5
# Covariance matrix (positive definite)
A = np.random.randn(n_assets, n_assets)
Sigma = A.T @ A / n_assets + 0.1 * np.eye(n_assets)
# Expected returns (heterogeneous scale: some are 1e-3, some are 1e0)
mu = np.array([0.001, 0.002, 0.5, 0.8, 1.2])

# KKT matrix and right-hand side
n_kkt = n_assets + 1
K = np.zeros((n_kkt, n_kkt))
K[:n_assets, :n_assets] = Sigma
K[:n_assets,  n_assets] = -1.0
K[n_assets,  :n_assets] =  1.0
rhs = np.concatenate([mu, [1.0]])

def kkt_residual(z):
    return K @ z - rhs

def kkt_jacobian(z):
    return K   # constant (linear problem)

x0 = np.zeros(n_kkt)

print("=" * 65)
print("PART A — Linear KKT system (one Newton step expected)")
print("=" * 65)
r_lin = solve(
    kkt_residual, x0, jac=kkt_jacobian,
    problem_type=ProblemType.LINEAR, display=True,
)
print(f"\nsuccess={r_lin.success}  nit={r_lin.nit}")
w = r_lin.x[:n_assets]
lam = r_lin.x[n_assets]
print(f"Optimal weights: {w}")
print(f"  sum = {w.sum():.6f}  (must be 1)")
print(f"  expected return = {mu @ w:.4f}")
print(f"  portfolio std = {np.sqrt(w @ Sigma @ w):.4f}")
print(f"Lagrange multiplier λ = {lam:.4f}")

# Show that poor scaling leads to slower convergence
print("\n[With poor scaling (unit default)]")
r_noscale = nleq_err(
    kkt_residual, x0, jac=kkt_jacobian,
    tol=1e-8, display=False,
)
print(f"  nit={r_noscale.nit}  success={r_noscale.success}")

print("\n[With user scaling matching solution magnitude]")
scale_hint = np.abs(r_lin.x) + 1e-6
r_scaled = nleq_err(
    kkt_residual, x0, jac=kkt_jacobian,
    user_scaling=scale_hint, tol=1e-8, display=False,
)
print(f"  nit={r_scaled.nit}  success={r_scaled.success}")

# ---------------------------------------------------------------------------
# Part B — Nonlinear extension: quartic regularisation
# ---------------------------------------------------------------------------

delta = 0.5   # quartic penalty coefficient
gamma = 2.0   # risk aversion

def portfolio_foc(z):
    """First-order conditions for: max μᵀx − γ/2 xᵀΣx − δ ‖x‖⁴ s.t. 1ᵀx=1."""
    x, lam = z[:n_assets], z[n_assets]
    grad_obj  = mu - gamma * (Sigma @ x) - 4.0 * delta * (x @ x) * x
    foc_x     = grad_obj - lam * np.ones(n_assets)
    foc_lam   = np.array([x.sum() - 1.0])
    return np.concatenate([foc_x, foc_lam])

def portfolio_jac(z):
    x = z[:n_assets]
    xx_norm = x @ x
    H = -gamma * Sigma - 4.0 * delta * (2.0 * np.outer(x, x) + xx_norm * np.eye(n_assets))
    J = np.zeros((n_kkt, n_kkt))
    J[:n_assets, :n_assets] = H
    J[:n_assets,  n_assets] = -1.0
    J[n_assets,  :n_assets] =  1.0
    return J

print("\n" + "=" * 65)
print("PART B — Nonlinear FOC (quartic penalty, γ=2, δ=0.5)")
print("=" * 65)
r_nl = nleq_err(
    portfolio_foc, np.zeros(n_kkt), jac=portfolio_jac,
    tol=1e-10, problem_type=ProblemType.MILDLY_NONLINEAR,
    display=True,
)
print(f"\nsuccess={r_nl.success}  nit={r_nl.nit}  njev={r_nl.njev}")
w_nl = r_nl.x[:n_assets]
print(f"Optimal weights (nonlinear): {w_nl}")
print(f"  sum = {w_nl.sum():.6f}")
print(f"  expected return = {mu @ w_nl:.4f}")
print(f"  ‖FOC residual‖ = {np.linalg.norm(r_nl.fun):.2e}")
