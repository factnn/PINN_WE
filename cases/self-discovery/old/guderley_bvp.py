#!/usr/bin/env python3
"""
Guderley problem: compute eigenvalue using the SIMPLEST approach.

Instead of shooting, use the fact that at the sonic point:
  (f-V)^2 = Z^2/gamma  AND  N_R = 0

Combined with the RH conditions at f=1, this gives us constraints.

The approach: parameterize the solution by alpha. For each alpha,
integrate from BOTH ends toward the sonic point and check if they match.

But simpler: use the (V, Z/V) phase plane approach from Whitham.

Actually the simplest: just use a BVP solver (scipy.integrate.solve_bvp)
which handles singular ODEs naturally.
"""
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gamma = 1.4
s = 3
beta = (gamma - 1.0) / 2.0

V_shock = 2.0 / (gamma + 1.0)
Z_shock = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)

print(f"Guderley BVP solver")
print(f"gamma={gamma}, s={s}")
print(f"V_shock={V_shock:.6f}, Z_shock={Z_shock:.6f}")


def ode_rhs(f, y, p):
    """
    ODE right-hand side. y = [V, Z, G], p = [alpha].
    Uses the 3-variable system to avoid the artificial V=f singularity.

    From Ramsey:
    (f-V)G'/G = V' + (s-1)V/f
    (f-V)V' = Z^2*G'/(gamma*G) + V*(alpha-1)/(alpha*f)
    (f-V)Z'/Z = beta*(f-V)*G'/G - (alpha-1)/alpha

    Rewrite as matrix system:
    Let h = G'/G.
    From continuity: h = (V' + (s-1)V/f) / (f-V)
    From momentum: (f-V)V' = Z^2/(gamma)*h + V*(alpha-1)/(alpha*f)

    Substituting h:
    (f-V)V' = Z^2/(gamma(f-V)) * (V' + (s-1)V/f) + V*(alpha-1)/(alpha*f)

    Let Delta = (f-V)^2 - Z^2/gamma
    Delta * V' = Z^2*(s-1)*V/(gamma*f) + V*(alpha-1)*(f-V)/(alpha*f)
    Delta * V' = [Z^2*(s-1)*V/gamma + V*(alpha-1)*(f-V)/alpha] / f

    So: V' = N_R / (f * Delta)
    where N_R = Z^2*(s-1)*V/gamma + V*(alpha-1)*(f-V)/alpha

    For the BVP, we regularize by multiplying through by Delta:
    f * Delta * V' = N_R

    Actually, for solve_bvp it's better to use the smooth (non-singular) formulation.
    The standard trick: replace f as independent variable with arc-length or use
    a variable substitution.

    Simple approach: just solve for [V, Z] with the regularized system
    (multiply by Delta on both sides so we don't divide by zero).
    """
    alpha = p[0]
    V, Z = y[0], y[1]

    # Safety
    f_safe = np.maximum(np.abs(f), 1e-12)

    fmV = f - V
    Delta = fmV**2 - Z**2 / gamma
    N_R = Z**2 * (s-1) * V / gamma + V * (alpha - 1) * fmV / alpha

    # V equation: Delta * V' = N_R / f
    # But this is still singular at Delta=0.
    # Instead, use the non-singular form:
    # V' = N_R / (f * Delta)  -- singular at sonic
    # For BVP, we can add a regularization.

    # Alternative: use (V, W) where W = Z^2 system
    # Or use the full 3-var system which doesn't have the Delta singularity.

    # Full 3-var: y = [V, Z, lnG]
    lnG = y[2]

    # h = G'/G
    # From matrix: solve for V' and h simultaneously
    # [Delta]V' = N_R/f  -- from above
    # h = (V' + (s-1)*V/f) / (f-V)  -- from continuity
    # Z' = Z * (beta*h - (alpha-1)/(alpha*fmV))  -- from entropy

    # Regularize: if |Delta| < eps, use a smooth approximation
    eps_D = 1e-8
    Delta_reg = np.where(np.abs(Delta) > eps_D, Delta,
                         np.sign(Delta) * eps_D + Delta * 0)

    dV = N_R / (f_safe * Delta_reg)

    # h and Z' - also need regularization for fmV=0
    fmV_safe = np.where(np.abs(fmV) > eps_D, fmV,
                        np.sign(fmV) * eps_D + fmV * 0)

    h = (dV + (s-1) * V / f_safe) / fmV_safe
    dZ = Z * (beta * h - (alpha - 1) / (alpha * fmV_safe))
    dlnG = h

    return np.vstack([dV, dZ, dlnG])


def bc_residual(ya, yb, p):
    """
    Boundary conditions:
    At f=f_min (near center): V should be small, smooth
    At f=1 (shock): V=V_shock, Z=Z_shock, G=G_shock
    """
    alpha = p[0]
    G_shock = (gamma + 1) / (gamma - 1)

    # At f=1: RH conditions
    res_V = yb[0] - V_shock
    res_Z = yb[1] - Z_shock
    res_G = yb[2] - np.log(G_shock)

    # We have 3 BCs + 1 unknown parameter (alpha) = 4 conditions needed for 3 ODEs + 1 param
    # Additional condition at f_min: regularity (V' finite or V proportional to f)
    # Near center: V ~ const*f (from symmetry), so V(f_min)/f_min should be finite
    # Use: V(f_min) ~ 0 for small f_min, or more precisely V/f -> finite
    # Simple condition: V(f_min) = 0 or dV/df|_center is finite

    # For the center BC, we need V(0) = 0 (from symmetry: u=0 at r=0 implies V=0 at f=0)
    # But we can't evaluate at exactly 0, so use f_min small and V(f_min) ≈ 0
    res_center = ya[0]  # V(f_min) ≈ 0

    return np.array([res_V, res_Z, res_G, res_center])


# ========== Set up mesh and initial guess ==========
f_min = 0.01
f_max = 1.0
N = 200
f_mesh = np.linspace(f_min, f_max, N)

# Initial guess for alpha
alpha0 = 0.7

# Linear initial guess for V (from 0 at center to V_shock at shock)
V_guess = V_shock * (f_mesh - f_min) / (f_max - f_min)
Z_guess = Z_shock * np.ones(N)  # constant
lnG_guess = np.log(6.0) * np.ones(N)

y_guess = np.vstack([V_guess, Z_guess, lnG_guess])
p_guess = np.array([alpha0])

print(f"\nSolving BVP with initial alpha={alpha0}...")

try:
    sol = solve_bvp(ode_rhs, bc_residual, f_mesh, y_guess, p=p_guess,
                    tol=1e-6, max_nodes=5000, verbose=2)

    if sol.success:
        alpha_found = sol.p[0]
        print(f"\n*** BVP CONVERGED ***")
        print(f"  alpha = {alpha_found:.10f}")
        print(f"  Literature (Lazarus 1981, spherical gamma=1.4): ~0.68838")
        print(f"  Difference: {abs(alpha_found - 0.68838):.6e}")

        # Plot solution
        f_plot = np.linspace(f_min, f_max, 500)
        V_plot = sol.sol(f_plot)[0]
        Z_plot = sol.sol(f_plot)[1]
        G_plot = np.exp(sol.sol(f_plot)[2])

        Delta_plot = (f_plot - V_plot)**2 - Z_plot**2/gamma

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0,0].plot(f_plot, V_plot, 'b-', lw=2, label='V(f)')
        axes[0,0].plot(f_plot, Z_plot, 'r-', lw=2, label='Z(f)')
        axes[0,0].axhline(y=V_shock, color='b', ls='--', alpha=0.5)
        axes[0,0].axhline(y=Z_shock, color='r', ls='--', alpha=0.5)
        axes[0,0].set_xlabel('f = r/R')
        axes[0,0].set_title(f'Solution: alpha={alpha_found:.6f}')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(f_plot, G_plot, 'g-', lw=2)
        axes[0,1].set_xlabel('f')
        axes[0,1].set_ylabel('G (density ratio)')
        axes[0,1].set_title('Density profile')
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(f_plot, Delta_plot, 'k-', lw=2)
        axes[1,0].axhline(y=0, color='r', ls='--')
        axes[1,0].set_xlabel('f')
        axes[1,0].set_ylabel('Delta')
        axes[1,0].set_title('Sonic line (Delta=0)')
        axes[1,0].grid(True, alpha=0.3)

        # Phase portrait
        axes[1,1].plot(V_plot, Z_plot, 'b-', lw=2)
        axes[1,1].plot(V_shock, Z_shock, 'ro', ms=8, label='shock')
        sonic_idx = np.argmin(np.abs(Delta_plot))
        axes[1,1].plot(V_plot[sonic_idx], Z_plot[sonic_idx], 'gs', ms=8, label='sonic')
        axes[1,1].set_xlabel('V')
        axes[1,1].set_ylabel('Z')
        axes[1,1].set_title('Phase portrait')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/share/project/zpy/PINN_WE/cases/self-discovery/guderley_bvp.png',
                    dpi=150, bbox_inches='tight')
        print("Figure saved.")
    else:
        print(f"BVP did not converge: {sol.message}")
        print(f"  Best alpha: {sol.p[0]:.6f}")
        print(f"  Residual norm: {np.max(np.abs(sol.rms_residuals)):.4e}")

except Exception as e:
    print(f"BVP failed: {e}")
    import traceback
    traceback.print_exc()

# ========== Try different initial guesses ==========
print("\n\nTrying multiple initial guesses:")
for alpha0 in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
    V_guess = V_shock * (f_mesh - f_min) / (f_max - f_min)
    Z_guess = Z_shock * np.ones(N)
    lnG_guess = np.log(6.0) * np.ones(N)
    y_guess = np.vstack([V_guess, Z_guess, lnG_guess])

    try:
        sol = solve_bvp(ode_rhs, bc_residual, f_mesh, y_guess,
                       p=np.array([alpha0]), tol=1e-4, max_nodes=5000)
        status = "OK" if sol.success else "FAIL"
        print(f"  alpha0={alpha0:.2f}: {status}, alpha={sol.p[0]:.8f}, "
              f"rms_max={np.max(np.abs(sol.rms_residuals)):.2e}")
    except Exception as e:
        print(f"  alpha0={alpha0:.2f}: ERROR - {e}")
