# deuflhard-newton

Affine-invariant Newton methods for nonlinear systems of equations, based on the
framework of Peter Deuflhard.  Three algorithms cover the full range of problem shapes:

| Problem shape | Algorithm | Convergence criterion |
|---|---|---|
| n = m (square) | **NLEQ-ERR** | `‖Δx‖_D ≤ tol` (Newton step small) |
| n < m ≤ 4n (overdetermined) | **NLEQ-RES** | `‖f(x)‖_D ≤ tol` (residual small) |
| m > 4n (least squares) | **LM** (Levenberg-Marquardt) | `‖Jᵀf‖_D ≤ tol` (gradient small) |

All algorithms include:
- Automatic Jacobian via complex-step differentiation (`jac=None`)
- Adaptive damping (global convergence from imperfect starting points)
- Quasi-Newton Jacobian reuse near the solution (fewer expensive J evaluations)
- Graceful handling of singular Jacobians (pseudoinverse fallback with warning)

---

## Mathematical framework

### Problem statement

Find `x* ∈ ℝⁿ` such that `F(x*) = 0`, where `F: ℝⁿ → ℝᵐ`.

### Why classic Newton fails

Classic Newton iteration `x_{k+1} = x_k − J(x_k)⁻¹ F(x_k)` has quadratic local
convergence but **no global convergence guarantee**.  Starting from a "bad" `x₀` that
is outside the basin of attraction, the iteration typically diverges.

Two root causes:
1. The full Newton step may overshoot the root.
2. Convergence behaviour changes under a coordinate transformation (not affine-invariant).

### Affine-invariant formulation

Deuflhard's key insight: the Newton direction `Δx = -J⁻¹ F(x)` is already
affine-invariant (independent of coordinate changes), but the *step size* must also
be chosen in an affine-invariant way.

**Scaled norm:**
```
‖v‖_D = ‖D⁻¹ v‖₂   where D = diag(w₁, …, wₙ)
```
`w_i` are adaptive weights tracking the natural scale of each component of `x`.

**Natural monotonicity test (NLEQ-ERR, §2.1.2):**
```
θ_k = ‖Δx̄_k‖_D / ‖Δx_k‖_D   where Δx̄_k = -J(x_k)⁻¹ F(x_k + λ_k Δx_k)
```
`θ_k < 1` guarantees affine-invariant contraction toward a root.

**Damping path:**
```
x_{k+1} = x_k + λ_k Δx_k
```
`λ_k ∈ (0, 1]` is the largest value s.t. the monotonicity test holds.

### Lipschitz predictor (avoids line search)

Instead of evaluating `θ` at each trial `λ`, the algorithm predicts the optimal
damping factor from the previous iterate:

```
μ_k ≈ λ_{k-1} · ‖Δx_{k-1}‖_D · ‖Δx̄_{k-1}‖_D / (‖Δx̄_{k-1} − Δx_k‖_D · ‖Δx_k‖_D)
```

`λ_k = min(1, μ_k)` typically requires only one or two inner evaluations per outer
iteration, even for strongly nonlinear problems.

### Convergence theorem (informal)

If the Jacobian satisfies a Lipschitz condition in a neighbourhood of the root,
and `λ₀ · ‖Δx₀‖_D` is sufficiently small (ensured by problem-type settings),
then NLEQ-ERR converges globally to a root `x*` with:
- **Quadratic convergence** once the full step `λ=1` is accepted,
- **Linear convergence** in the quasi-Newton (QNERR) simplified phase.

### QNERR / QNRES: quasi-Newton acceleration

Once `θ < 0.5` and `λ = 1` (fast-convergence regime), the Jacobian is **frozen** for
the next iteration.  This saves one expensive Jacobian evaluation per iteration at the
cost of reverting from quadratic to linear convergence rate.

Break-even analysis for auto-diff Jacobians (cost ~n function evaluations):

| n (unknowns) | Full Newton total evals | QNERR total evals | Winner |
|---|---|---|---|
| 1 | ~6 | ~22 | Full Newton |
| 7 | ~63 | ~22 | QNERR |
| 20 | ~180 | ~22 | QNERR |
| 100 | ~900 | ~22 | QNERR |

Use `use_qn=False` when the Jacobian is analytical (cheap) and quadratic convergence
is preferred.

### Adaptive scaling

Weights are updated after each accepted step:
```
w_i = max(w₀_i, ½(|x_new_i| + |x_old_i|), ε)
```
Three properties:
- Weights never shrink (robust to oscillation)
- Weights track the natural magnitude of the iterates
- Floor `ε = 1e-30` prevents division by zero for zero-valued variables

### Singular Jacobians

When `J(x)` is singular, `_linear_solve` falls back to the **minimum-norm pseudoinverse
step** `Δx = J⁺(-F(x))` and emits a `RuntimeWarning`.  The affine-invariant damping loop
then finds a `λ` s.t. the monotonicity test holds, or declares failure gracefully if none
exists.  The solver never raises `LinAlgError`.

---

## Installation

```bash
pip install deuflhard-newton          # core (numpy only)
pip install 'deuflhard-newton[autodiff]'   # + csdiff for automatic Jacobians
```

Or from source:

```bash
git clone https://github.com/QValence/deuflhard-newton.git
cd deuflhard-newton
pip install -e '.[autodiff]'
```

**Requirements:** Python ≥ 3.10, numpy.  Optional: scipy (LU reuse), csdiff (auto-Jacobian).

---

## Quick start

### Square system — `solve()` (automatic algorithm selection)

```python
import numpy as np
from deuflhard_newton import solve

def f(x):
    return np.array([x[0]**2 + x[1] - 2.0,
                     x[0] + x[1]**2 - 2.0])

r = solve(f, np.array([0.5, 0.5]))
print(r.x, r.success)   # [1. 1.] True
```

### Square system — `nleq_err()` with explicit Jacobian

```python
from deuflhard_newton import nleq_err

def jac(x):
    return np.array([[2*x[0], 1], [1, 2*x[1]]])

r = nleq_err(f, np.array([0.5, 0.5]), jac=jac, tol=1e-10, display=True)
```

### Overdetermined system — `nleq_res()`

```python
from deuflhard_newton import nleq_res

# m=4 equations, n=2 unknowns
A = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=float)
x_star = np.array([1.0, 2.0])
b = A @ x_star

r = nleq_res(lambda x: A @ x - b, np.zeros(2))
print(r.x)   # [1. 2.]
```

### Nonlinear least squares — `lm()`

```python
from deuflhard_newton import lm

# Fit y = Vmax * S / (Km + S) to data
S_data = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
v_data = np.array([0.12, 0.21, 0.42, 0.63, 0.82, 0.95, 1.01])

def residuals(params):
    Vmax, Km = params
    return Vmax * S_data / (Km + S_data) - v_data

r = lm(residuals, np.array([1.0, 0.5]))   # jac=None uses auto-diff
print(f"Vmax={r.x[0]:.3f}, Km={r.x[1]:.3f}")
```

---

## API reference

### `solve(func, x0, jac=None, tol=1e-3, max_iter=1000, ...)`

Unified entry point.  Selects NLEQ-ERR, NLEQ-RES, or LM based on the shape of `f(x0)`:

```
m == n       → nleq_err
n < m ≤ 4n  → nleq_res
m > 4n       → lm
```

Override with `method='nleq_err'`, `method='nleq_res'`, or `method='lm'`.

### `nleq_err(func, x0, jac=None, tol=1e-3, max_iter=1000, user_scaling=None, problem_type=ProblemType.MILDLY_NONLINEAR, display=True, *, use_qn=True, callback=None)`

Error-oriented Newton.  Recommended for square systems.

### `nleq_res(func, x0, jac=None, tol=1e-3, max_iter=1000, user_scaling=None, problem_type=ProblemType.MILDLY_NONLINEAR, display=True, *, use_qn=True, callback=None)`

Residual-oriented Newton.  Recommended for overdetermined systems (m > n).

### `lm(func, x0, jac=None, tol=1e-6, max_iter=1000, lam0=1e-3, scaling=None, display=True, callback=None)`

Levenberg-Marquardt.  Minimises `½‖F(x)‖²`.  Use when m >> n or no exact root exists.

### `SolveResult` fields

| Field | Type | Description |
|---|---|---|
| `x` | `ndarray` | Solution (or last iterate on failure) |
| `fun` | `ndarray` | Residual `F(x)` at solution |
| `success` | `bool` | True if convergence criterion was met |
| `message` | `str` | Human-readable convergence/failure reason |
| `nit` | `int` | Number of outer iterations |
| `nfev` | `int` | Number of function evaluations |
| `njev` | `int` | Number of Jacobian evaluations |
| `history` | `dict` | Per-iteration arrays: `f_norm`, `dx_norm`, `lb`, `mu`, `theta` |
| `method` | `str` | Algorithm used: `'nleq_err'`, `'nleq_res'`, or `'lm'` |

### `ProblemType` enum

Controls initial damping `λ₀` and minimum threshold `λ_min`:

| Level | λ₀ | λ_min | restricted | Use when |
|---|---|---|---|---|
| `LINEAR` | 1 | 1e-4 | No | Affine or near-affine problems |
| `MILDLY_NONLINEAR` | 1 | 1e-4 | No | Default; smooth, not too curved |
| `HIGHLY_NONLINEAR` | 1e-2 | 1e-4 | No | Strong curvature, Lipschitz constant large |
| `EXTREMELY_NONLINEAR` | 1e-4 | 1e-8 | Yes | Near-singular, highly curved, far from root |

---

## Scaling guide

**Why scaling matters:** `tol` is interpreted in the scaled norm `‖·‖_D`.  With unit
weights (default), convergence is `‖Δx‖₂ ≤ tol`.  With adaptive weights tracking
`|x|`, convergence is `‖Δx / x‖₂ ≤ tol` (relative error criterion).

**Best practice:**
```python
# If the order of magnitude of x* is known:
r = solve(f, x0, user_scaling=np.abs(x0))

# If unknowns span many orders of magnitude (e.g. concentrations + temperatures):
r = nleq_err(f, x0, user_scaling=np.array([1e-6, 1e-6, 300.0, 300.0]))
```

**Pitfalls:**
- Too-small `user_scaling` → over-tight criterion for large-magnitude variables
- Too-large `user_scaling` → solver declares convergence before actually converging
- `user_scaling=None` (default) → safe choice; converges to higher accuracy for large-magnitude variables

---

## Comparison with scipy

| Scenario | scipy.optimize.fsolve | deuflhard_newton |
|---|---|---|
| Good starting point, smooth problem | Similar speed | Similar speed |
| Bad starting point, smooth problem | Often diverges | Converges (affine-invariant damping) |
| Highly nonlinear, far from root | Often fails | Use `EXTREMELY_NONLINEAR` |
| Auto-diff Jacobian needed | Manual or finite-diff | `jac=None` via csdiff |
| Overdetermined system | Not designed for this | `nleq_res` or `lm` |
| Singular Jacobian | Raises `LinAlgError` | Pseudoinverse fallback + warning |

Deuflhard's solver is strictly superior when:
1. The starting point is imperfect (the main engineering use case)
2. The function is highly nonlinear with large Lipschitz constant
3. The Jacobian may be rank-deficient along the Newton path

scipy's fsolve is competitive on easy problems and requires no extra dependencies.

---

## Examples

See the [`examples/`](examples/) directory:

| Script | Topic | API entry point |
|---|---|---|
| `01_orbit_shooting.py` | Boundary value problem via shooting method | `nleq_err()` vs scipy |
| `02_optimization_foc.py` | Portfolio KKT / Lagrangian FOC | `solve()`, `ProblemType.LINEAR` |
| `03_chemical_equilibrium.py` | Water-gas shift reaction equilibrium | `solve()`, `ProblemType.HIGHLY_NONLINEAR` |
| `04_curve_fitting_lm.py` | Michaelis-Menten enzyme kinetics | `lm()`, auto-Jacobian |
| `05_structural_mechanics.py` | Nonlinear truss (large displacement) | `nleq_res()`, `EXTREMELY_NONLINEAR` |

---

## Running the test suite

```bash
pytest -v
```

55 tests covering convergence, convergence order, exception handling, display output,
singular Jacobians, and auto-differentiation accuracy.

---

## References

> P. Deuflhard,
> *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms*,
> Springer Series in Computational Mathematics, Vol. 35, 2011.
> DOI: [10.1007/978-3-642-23899-4](https://doi.org/10.1007/978-3-642-23899-4)

This package is an independent implementation based on the algorithmic descriptions in
the above reference.  No text, pseudocode, or figures from the book are reproduced.

---

## License

MIT — see [`LICENSE`](LICENSE).
