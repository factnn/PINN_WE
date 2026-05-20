#!/usr/bin/env python3
"""
Guderley eigenvalue - collocation/relaxation method.

Instead of integrating the ODE (which fails at the sonic singularity),
discretize the equations on a grid and solve the nonlinear system.

This is equivalent to a relaxation method and naturally handles singularities.

System:
  Δ·V' = N_R/f        (V-equation)
  (f-V)·Z' = Z·[β·(V'+(s-1)V/f) - (n-1)/n]   (Z-equation)

Rewrite without division by Δ (which vanishes at sonic):
  [(f-V)²-Z²/γ]·V' = [Z²(s-1)V/γ + V(n-1)(f-V)/n] / f

BCs:
  V(1) = V_shock, Z(1) = Z_shock  (RH at shock)
  V(0) = 0  (symmetry at center)

Unknown parameter: n (= alpha)
"""
import numpy as np
from scipy.optimize import fsolve, root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gamma = 1.4
s = 3
beta = (gamma - 1.0) / 2.0

V_shock = 2.0 / (gamma + 1.0)
Z_shock = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)

print(f"Guderley collocation method")
print(f"gamma={gamma}, s={s}, V_shock={V_shock:.6f}, Z_shock={Z_shock:.6f}")

# Grid: use Chebyshev-like clustering near both ends
N = 100  # number of interior points
# Map [0,1] to f values: f = f_min + (f_max-f_min) * (1 - cos(pi*i/N))/2
f_min = 0.001
f_max = 1.0
idx = np.arange(N+1)
f_grid = f_min + (f_max - f_min) * 0.5 * (1 - np.cos(np.pi * idx / N))

# Finite difference: central differences for interior, one-sided at boundaries
def build_residual(x, N):
    """
    x = [V_0, V_1, ..., V_N, Z_0, Z_1, ..., Z_N, n]
    Total unknowns: 2*(N+1) + 1

    Equations:
    - At i=0: V(f_min) = 0 (center BC)
    - At i=N: V(f_max) = V_shock (shock BC)
    - At i=N: Z(f_max) = Z_shock (shock BC)
    - For i=1..N-1: V-equation and Z-equation at interior points
    - Total: 1 + 2*(N-1) + 2 + 1 = 2*N + 2 = 2*(N+1)
    - But we have 2*(N+1) + 1 unknowns. Extra equation: smoothness at sonic?

    Actually: 2*(N+1) + 1 unknowns, need 2*(N+1) + 1 equations.
    - 2*(N-1) interior equations (V and Z at points 1..N-1)
    - 3 boundary conditions (V(0)=0, V(1)=V_shock, Z(1)=Z_shock)
    - Total: 2N-2+3 = 2N+1 equations
    - Unknowns: 2N+2 + 1 = 2N+3
    - Need 2 more equations. Use Z at center: dZ/df|_{f=0} = 0 (regularity)
    - And one more. Actually we don't need Z(0) to be specified.

    Let me restructure:
    Unknowns at each grid point: V_i, Z_i for i=0,...,N, plus n. Total = 2(N+1)+1.
    Equations:
    - V ODE at i=1,...,N-1: N-1 equations
    - Z ODE at i=1,...,N-1: N-1 equations
    - V(0) ≈ 0: 1 equation
    - V(N) = V_shock: 1 equation
    - Z(N) = Z_shock: 1 equation
    Total so far: 2(N-1) + 3 = 2N+1 equations, need 2(N+1)+1 = 2N+3.
    Need 2 more.

    Additional conditions:
    - Z'(0) = 0 (regularity at center): 1 equation
    - V equation at i=0 or i=N (one-sided difference): 1 equation
    Total: 2N+3. Matches!
    """
    V = x[:N+1]
    Z = x[N+1:2*(N+1)]
    n = x[-1]

    res = np.zeros(2*(N+1) + 1)
    h = np.diff(f_grid)

    # Interior V equations (i=1,...,N-1)
    for i in range(1, N):
        fi = f_grid[i]
        Vi = V[i]
        Zi = Z[i]

        # Central differences
        dVdf = (V[i+1] - V[i-1]) / (f_grid[i+1] - f_grid[i-1])

        fmV = fi - Vi
        Delta = fmV**2 - Zi**2 / gamma
        N_R = Zi**2 * (s-1) * Vi / gamma + Vi * (n-1) * fmV / n

        # Equation: Delta * dV/df = N_R / f
        # Residual: Delta * dV/df - N_R/f = 0
        res[i-1] = Delta * dVdf - N_R / fi

    # Interior Z equations (i=1,...,N-1)
    for i in range(1, N):
        fi = f_grid[i]
        Vi = V[i]
        Zi = Z[i]

        dVdf = (V[i+1] - V[i-1]) / (f_grid[i+1] - f_grid[i-1])
        dZdf = (Z[i+1] - Z[i-1]) / (f_grid[i+1] - f_grid[i-1])

        fmV = fi - Vi

        # Z equation: (f-V)Z' = Z[β(V'+(s-1)V/f) - (n-1)/n]
        # Residual: (f-V)Z' - Z[β(V'+(s-1)V/f) - (n-1)/n] = 0
        rhs_z = Zi * (beta * (dVdf + (s-1) * Vi / fi) - (n-1.0) / n)
        res[N-1 + i-1] = fmV * dZdf - rhs_z

    # Boundary conditions
    offset = 2*(N-1)

    # V(0) = 0 (center symmetry)
    res[offset] = V[0]

    # V(N) = V_shock
    res[offset+1] = V[N] - V_shock

    # Z(N) = Z_shock
    res[offset+2] = Z[N] - Z_shock

    # Z'(0) = 0 (regularity at center) - forward difference
    res[offset+3] = (Z[1] - Z[0]) / (f_grid[1] - f_grid[0])

    # V equation at i=0 (forward difference, use L'Hôpital at f=0)
    # At f=0, V=0: the V equation Δ·V' = N_R/f becomes
    # f²·V' ≈ 0 (since V~0, Z~const, and N_R ~ Z²(s-1)V/γ ~ 0)
    # Use: V'(0) = V(1)/(f(1)-f(0)) approximately
    # Or: the original ODE at f_min with forward difference
    dVdf_0 = (V[1] - V[0]) / (f_grid[1] - f_grid[0])
    fmV_0 = f_grid[0] - V[0]
    Delta_0 = fmV_0**2 - Z[0]**2 / gamma
    N_R_0 = Z[0]**2 * (s-1) * V[0] / gamma + V[0] * (n-1) * fmV_0 / n
    res[offset+4] = Delta_0 * dVdf_0 - N_R_0 / max(f_grid[0], 1e-10)

    return res


# ========== Initial guess ==========
# V: linear from 0 to V_shock
V_init = V_shock * (f_grid - f_min) / (f_max - f_min)
# Z: constant at Z_shock (crude but OK)
Z_init = Z_shock * np.ones(N+1)
# n: initial guess
n_init = 0.7

x0 = np.concatenate([V_init, Z_init, [n_init]])

print(f"\nGrid: {N+1} points from f={f_min} to f={f_max}")
print(f"Unknowns: {len(x0)}")
print(f"Initial n = {n_init}")

# ========== Solve ==========
print("\nSolving nonlinear system...")

# Try multiple initial guesses
best_result = None
best_residual = 1e10

for n_try in [0.5, 0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.8]:
    x0_try = np.concatenate([V_init, Z_init, [n_try]])

    try:
        result = root(lambda x: build_residual(x, N), x0_try,
                     method='hybr', options={'maxfev': 50000})

        res_norm = np.max(np.abs(result.fun))
        n_found = result.x[-1]

        status = "OK" if result.success else "FAIL"
        print(f"  n0={n_try:.2f}: {status}, n={n_found:.8f}, max_res={res_norm:.2e}")

        if res_norm < best_residual:
            best_residual = res_norm
            best_result = result

    except Exception as e:
        print(f"  n0={n_try:.2f}: ERROR - {e}")

if best_result is not None and best_residual < 1e-3:
    V_sol = best_result.x[:N+1]
    Z_sol = best_result.x[N+1:2*(N+1)]
    n_sol = best_result.x[-1]

    print(f"\n*** Best solution: alpha = {n_sol:.10f} ***")
    print(f"  Max residual: {best_residual:.4e}")
    print(f"  Literature: ~0.6884 (Lazarus 1981, spherical gamma=1.4)")

    # Plot
    Delta_sol = (f_grid - V_sol)**2 - Z_sol**2/gamma

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(f_grid, V_sol, 'b-', lw=2, label='V')
    axes[0].plot(f_grid, Z_sol, 'r-', lw=2, label='Z')
    axes[0].set_xlabel('f = r/R')
    axes[0].set_title(f'Solution: alpha={n_sol:.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_grid, Delta_sol, 'k-', lw=2)
    axes[1].axhline(0, color='r', ls='--')
    axes[1].set_xlabel('f')
    axes[1].set_ylabel('Delta')
    axes[1].set_title('Sonic line')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(V_sol, Z_sol, 'b-', lw=2)
    axes[2].plot(V_shock, Z_shock, 'ro', ms=8, label='shock')
    axes[2].set_xlabel('V')
    axes[2].set_ylabel('Z')
    axes[2].set_title('Phase portrait')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/share/project/zpy/PINN_WE/cases/self-discovery/guderley_collocation.png',
                dpi=150, bbox_inches='tight')
    print("Figure saved.")
else:
    print(f"\nBest residual: {best_residual:.4e} - not converged")
    if best_result is not None:
        print(f"Best n: {best_result.x[-1]:.6f}")
