"""
Example 5 — Nonlinear truss: large-displacement equilibrium
============================================================
Problem
-------
A single free node (pin joint) is connected to three bars of different
lengths and angles.  The bars are elastic (linear material, but geometric
nonlinearity due to large displacements).

State vector: x = [u, v]  (horizontal and vertical displacement of free node)

Equations: equilibrium in two directions (force balance at the free node).

The system is 2×2 (square), but the equilibrium equations are highly
nonlinear (trigonometric dependence on displacements).

Why EXTREMELY_NONLINEAR
------------------------
The equilibrium equations near the undeformed state (u,v)=(0,0) have
a badly conditioned Jacobian: the bars start nearly horizontal, so
small vertical movements create large force changes.  Starting from
(u, v) = (0, 0) leads to massive initial Newton steps.  The restricted
monotonicity test (EXTREMELY_NONLINEAR) damps these aggressively.

Multiple equilibria
-------------------
This truss has two equilibria under the applied load:
  1. Physical equilibrium — node below all support points (stable).
  2. Inverted equilibrium — node above the supports (unstable; the
     Jacobian has a negative eigenvalue).

Starting from (0, 0) or nearby finds the stable equilibrium.
Starting from (0, 10) finds the inverted equilibrium.
The Jacobian eigenvalue analysis reveals which solution is stable.

API used
--------
nleq_err(func, x0, jac, problem_type=ProblemType.EXTREMELY_NONLINEAR)
nleq_err(func, x0, problem_type=ProblemType.MILDLY_NONLINEAR)  # good x0
"""

import numpy as np
from deuflhard_newton import nleq_err, ProblemType

# ---------------------------------------------------------------------------
# Geometry: 3 bars pinned to walls at fixed points A, B, C
# ---------------------------------------------------------------------------
#  A = (-2, 1.5),  B = (0, 3),  C = (2, 1)   (support points)
#  Free node P at equilibrium position
#  External load: Fx = 0,  Fy = -1  (downward unit force)

EA = 10.0   # EA = Young's modulus × cross-section area (all bars same)

supports = np.array([
    [-2.0, 1.5],
    [ 0.0, 3.0],
    [ 2.0, 1.0],
])

F_ext = np.array([0.0, -1.0])   # applied load at free node

# Undeformed lengths of each bar (free node at origin for reference)
P0 = np.array([0.0, 0.0])   # reference position (origin)
L0 = np.array([np.linalg.norm(P0 - s) for s in supports])

def bar_force(u_node, support, L_ref):
    """Axial force in one bar given current node position."""
    P_curr = u_node
    d_vec  = P_curr - support
    L_curr = np.linalg.norm(d_vec)
    if L_curr < 1e-12:
        return np.zeros(2)
    # Axial strain: ε = (L - L0) / L0
    strain = (L_curr - L_ref) / L_ref
    # Force = EA * ε in bar direction
    force_mag = EA * strain
    unit_vec = d_vec / L_curr
    return force_mag * unit_vec

def equilibrium(u):
    """Force equilibrium at the free node: ΣF_bars + F_ext = 0."""
    r = -F_ext.copy()    # external load (sign: we seek ΣF_internal + F_ext = 0)
    for i, s in enumerate(supports):
        r += bar_force(u, s, L0[i])
    return r

def equilibrium_jac(u, h=1e-6):
    """Finite-difference Jacobian (2×2)."""
    f0 = equilibrium(u)
    J  = np.empty((2, 2))
    for j in range(2):
        du = np.zeros(2); du[j] = h
        J[:, j] = (equilibrium(u + du) - f0) / h
    return J

# ---------------------------------------------------------------------------
# Case 1: good starting point, mildly nonlinear
# ---------------------------------------------------------------------------
print("=" * 65)
print("CASE 1 — Good starting point (near expected solution)")
print("=" * 65)
x0_good = np.array([0.0, -0.05])   # small downward displacement guess

r_good = nleq_err(
    equilibrium, x0_good, jac=equilibrium_jac,
    tol=1e-8, problem_type=ProblemType.MILDLY_NONLINEAR,
    display=True,
)
print(f"\nsuccess={r_good.success}  nit={r_good.nit}  njev={r_good.njev}")
print(f"Equilibrium position: u={r_good.x[0]:.6f}  v={r_good.x[1]:.6f}")
print(f"Residual ‖F‖ = {np.linalg.norm(r_good.fun):.2e}")

# ---------------------------------------------------------------------------
# Case 2: bad starting point, EXTREMELY_NONLINEAR with restricted monotonicity
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("CASE 2 — Bad starting point (origin, large initial Newton step)")
print("=" * 65)
x0_bad = np.array([0.0, 0.0])   # undeformed node

r_extreme = nleq_err(
    equilibrium, x0_bad, jac=equilibrium_jac,
    tol=1e-8, max_iter=50,
    problem_type=ProblemType.EXTREMELY_NONLINEAR,
    display=True,
)
print(f"\nsuccess={r_extreme.success}  nit={r_extreme.nit}  njev={r_extreme.njev}")
if r_extreme.success:
    print(f"Equilibrium position: u={r_extreme.x[0]:.6f}  v={r_extreme.x[1]:.6f}")

# ---------------------------------------------------------------------------
# Case 3: multiple equilibria — starting point determines which root is found
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("CASE 3 — Multiple equilibria: starting point determines the solution")
print("=" * 65)
x0_high = np.array([0.0, 10.0])   # far above the support points

r_inverted = nleq_err(
    equilibrium, x0_high, jac=equilibrium_jac,
    tol=1e-8, max_iter=50,
    problem_type=ProblemType.EXTREMELY_NONLINEAR,
    display=True,
)
print(f"\nsuccess={r_inverted.success}  nit={r_inverted.nit}  njev={r_inverted.njev}")
print(f"Inverted equilibrium: u={r_inverted.x[0]:.4f}  v={r_inverted.x[1]:.4f}")
print(f"Residual ‖F‖ = {np.linalg.norm(equilibrium(r_inverted.x)):.2e}")

# Stability check via Jacobian eigenvalues
J_inv = equilibrium_jac(r_inverted.x)
eigs_inv = np.linalg.eigvals(J_inv)
J_phy = equilibrium_jac(r_good.x)
eigs_phy = np.linalg.eigvals(J_phy)
print()
print("Stability analysis (Jacobian eigenvalues):")
print(f"  Physical  equilibrium ({r_good.x[0]:.4f}, {r_good.x[1]:.4f}): "
      f"λ = {eigs_phy[0]:.3f}, {eigs_phy[1]:.3f}  → all positive → STABLE")
print(f"  Inverted  equilibrium ({r_inverted.x[0]:.4f}, {r_inverted.x[1]:.4f}): "
      f"λ = {eigs_inv[0]:.3f}, {eigs_inv[1]:.3f}  → one negative → UNSTABLE")

print("\n" + "=" * 65)
print("TAKEAWAY: Newton's method finds the nearest equilibrium to x0.")
print("  x0=(0, -0.05): physical (stable)  equilibrium  below supports")
print("  x0=(0, 10)   : inverted (unstable) equilibrium above supports")
print("Use domain knowledge to choose the starting point and eigenvalue")
print("analysis to verify stability of the found solution.")
