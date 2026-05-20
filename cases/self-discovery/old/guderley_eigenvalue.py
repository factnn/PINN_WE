#!/usr/bin/env python3
"""
Guderley eigenvalue computation - proper sonic point expansion method.

Strategy:
1. At the sonic point ξ_s, both Δ=0 and N_V=0 must hold simultaneously.
2. Given α, these two conditions plus the 2 unknowns (V_s, Z_s) at the sonic point
   form a system. But ξ_s is also unknown (3 unknowns: V_s, Z_s, ξ_s).
3. We only have 2 algebraic conditions (Δ=0, N_V=0), so one parameter is free.
   We parameterize by ξ_s and find V_s, Z_s from the algebraic conditions.
4. Then we integrate outward from ξ_s to ξ=1 (shock) using L'Hôpital limiting slopes.
5. At ξ=1, we check if the RH conditions are satisfied.
6. α and ξ_s are adjusted until the RH conditions match.

Reference: Lazarus (1981), Ramsey et al. (2012) Appendix A.

Equations (Ramsey notation, f = ξ, n = α):
  Delta_R * V' = N_R / f
  where:
    Delta_R = (f-V)^2 - Z^2/gamma
    N_R = Z^2*(s-1)*V/gamma + V*(n-1)*(f-V)/n

  Z' = Z/(f-V) * [beta*(V' + (s-1)*V/f) - (n-1)/n]

At the sonic point (f = f_s):
  Condition 1: (f_s - V_s)^2 = Z_s^2 / gamma  =>  f_s - V_s = Z_s / sqrt(gamma)
  Condition 2: N_R = 0  =>  Z_s^2*(s-1)*V_s/gamma + V_s*(n-1)*(f_s-V_s)/n = 0

From Cond 1: Z_s = sqrt(gamma)*(f_s - V_s)
From Cond 2: (s-1)*V_s*(f_s-V_s)^2 + V_s*(n-1)*(f_s-V_s)/n = 0
  Dividing by V_s*(f_s-V_s) [assuming both nonzero]:
  (s-1)*(f_s-V_s) + (n-1)/n = 0
  f_s - V_s = -(n-1)/[n*(s-1)] = (1-n)/[n*(s-1)]

For n < 1 (converging shock): (1-n) > 0, so f_s - V_s > 0.  Good.

So: f_s - V_s = (1-n) / [n*(s-1)]
And: V_s = f_s - (1-n)/[n*(s-1)]
And: Z_s = sqrt(gamma) * (1-n)/[n*(s-1)]

The ONLY free parameters are n (=α) and f_s (= ξ_s).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gamma = 1.4
s = 3  # spherical
beta = (gamma - 1.0) / 2.0

# RH shock conditions at xi=1
V_shock = 2.0 / (gamma + 1.0)
Z_shock = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)

print("Guderley eigenvalue by sonic point shooting")
print(f"gamma={gamma}, s={s} (spherical)")
print(f"RH: V(1)={V_shock:.6f}, Z(1)={Z_shock:.6f}")
print()


def sonic_point_values(n, f_s):
    """Given alpha=n and sonic point location f_s, return V_s, Z_s."""
    fmV_s = (1.0 - n) / (n * (s - 1))
    V_s = f_s - fmV_s
    Z_s = np.sqrt(gamma) * fmV_s
    return V_s, Z_s


def compute_sonic_slopes(n, f_s, V_s, Z_s):
    """
    Compute V'(f_s) and Z'(f_s) using L'Hôpital's rule at the sonic point.

    At Δ=0 and N_R=0: V' = lim (dN_R/df) / (f * dΔ/df)
    This requires careful expansion. Use the approach from Lazarus (1981).

    The limiting slope V'_s satisfies a quadratic equation obtained by
    Taylor-expanding both numerator and denominator around the sonic point.
    """
    fmV_s = f_s - V_s

    # From the ODe: Delta * V' = N_R / f
    # Taylor expand around f = f_s:
    # Delta(f) ≈ dDelta/df|_s * (f - f_s) + ...
    # N_R(f) ≈ dN_R/df|_s * (f - f_s) + ...
    # So V'_s = (dN_R/df|_s) / (f_s * dDelta/df|_s)

    # dDelta/df = 2*(f-V)*(1-V') - 2*Z*Z'/(gamma)
    # dN_R/df = [2*Z*Z'*(s-1)*V/gamma + Z^2*(s-1)*V'/gamma]
    #         + [V'*(n-1)*(f-V)/n + V*(n-1)*(1-V')/n]

    # But V' and Z' appear in the derivatives, which is what we want to find.
    # This gives us a self-consistent equation.

    # Let p = V'_s. Then from the Z equation:
    # Z' = Z/(f-V) * [beta*(V' + (s-1)*V/f) - (n-1)/n]
    # At the sonic point:
    # Z'_s = Z_s/fmV_s * [beta*(p + (s-1)*V_s/f_s) - (n-1)/n]

    # Let q = Z'_s
    # Then:
    # dDelta/df|_s = 2*fmV_s*(1-p) - 2*Z_s*q/gamma
    # dN_R/df|_s = 2*Z_s*q*(s-1)*V_s/gamma + Z_s^2*(s-1)*p/gamma
    #            + p*(n-1)*fmV_s/n + V_s*(n-1)*(1-p)/n

    # Using Z_s^2/gamma = fmV_s^2:
    # dDelta/df|_s = 2*fmV_s*(1-p) - 2*Z_s*q/gamma
    # dN_R/df|_s = 2*Z_s*q*(s-1)*V_s/gamma + fmV_s^2*(s-1)*p
    #            + p*(n-1)*fmV_s/n + V_s*(n-1)*(1-p)/n

    # From the self-consistency: p * dDelta/df = (1/f_s) * dN_R/df
    # This is a quadratic in p (since q is linear in p).

    # Express q in terms of p:
    q_coeff = Z_s / fmV_s * beta  # coefficient of p in q
    q_const = Z_s / fmV_s * (beta * (s-1) * V_s / f_s - (n-1.0)/n)
    # q = q_coeff * p + q_const

    # Now substitute into the self-consistency equation:
    # p * [2*fmV_s*(1-p) - 2*Z_s*(q_coeff*p + q_const)/gamma]
    # = (1/f_s) * [2*Z_s*(q_coeff*p + q_const)*(s-1)*V_s/gamma + fmV_s^2*(s-1)*p
    #            + p*(n-1)*fmV_s/n + V_s*(n-1)*(1-p)/n]

    # This is quadratic in p. Let's expand:
    # Left side:
    # 2*fmV_s*p - 2*fmV_s*p^2 - 2*Z_s*q_coeff*p^2/gamma - 2*Z_s*q_const*p/gamma

    # Right side (divided by 1/f_s):
    # 2*Z_s*q_coeff*(s-1)*V_s*p/(gamma*f_s) + 2*Z_s*q_const*(s-1)*V_s/(gamma*f_s)
    # + fmV_s^2*(s-1)*p/f_s + p*(n-1)*fmV_s/(n*f_s) + V_s*(n-1)/(n*f_s)
    # - V_s*(n-1)*p/(n*f_s)

    # Collect as a*p^2 + b*p + c = 0:
    a = -2*fmV_s - 2*Z_s*q_coeff/gamma

    b = (2*fmV_s - 2*Z_s*q_const/gamma
         - 2*Z_s*q_coeff*(s-1)*V_s/(gamma*f_s)
         - fmV_s**2*(s-1)/f_s
         - (n-1)*fmV_s/(n*f_s)
         + V_s*(n-1)/(n*f_s))

    c = (-2*Z_s*q_const*(s-1)*V_s/(gamma*f_s)
         - V_s*(n-1)/(n*f_s))

    # Solve quadratic
    disc = b**2 - 4*a*c
    if disc < 0:
        return None, None

    p1 = (-b + np.sqrt(disc)) / (2*a)
    p2 = (-b - np.sqrt(disc)) / (2*a)

    # Choose the physical branch (usually the one where V' is finite and reasonable)
    q1 = q_coeff * p1 + q_const
    q2 = q_coeff * p2 + q_const

    return (p1, q1), (p2, q2)


def integrate_from_sonic(n, f_s, direction='outward', f_end=1.0):
    """
    Integrate from the sonic point outward (toward shock at f=1)
    or inward (toward center at f=0).

    direction: 'outward' (f_s -> 1) or 'inward' (f_s -> 0)
    """
    V_s, Z_s = sonic_point_values(n, f_s)

    if V_s <= 0 or Z_s <= 0:
        return None

    slopes = compute_sonic_slopes(n, f_s, V_s, Z_s)
    if slopes[0] is None:
        return None

    (p1, q1), (p2, q2) = slopes

    # For outward integration (toward f=1), we want the branch where
    # Delta becomes negative (subsonic) as we move away from sonic point.
    # At sonic: Delta=0. For f > f_s: Delta ≈ dDelta/df * (f - f_s)
    # dDelta/df = 2*(f-V)*(1-V') - 2*Z*Z'/gamma
    fmV_s = f_s - V_s

    dD_1 = 2*fmV_s*(1-p1) - 2*Z_s*q1/gamma
    dD_2 = 2*fmV_s*(1-p2) - 2*Z_s*q2/gamma

    # For outward: we want Delta < 0 for f > f_s, so dDelta/df < 0
    # For inward: we want Delta > 0 for f < f_s, so dDelta/df < 0 as well
    # Actually, behind the shock: subsonic (Delta < 0), between shock and center
    # At center and ahead of sonic toward center: supersonic (Delta > 0)

    if direction == 'outward':
        # Want subsonic after sonic point (toward shock): dDelta/df < 0 means
        # Delta goes from 0 to negative as f increases
        if dD_1 < 0:
            p, q = p1, q1
        else:
            p, q = p2, q2
    else:
        # Want supersonic toward center: dDelta/df > 0 means
        # Delta goes from 0 to positive as f decreases (dDelta/df < 0)
        if dD_1 > 0:
            p, q = p1, q1
        else:
            p, q = p2, q2

    # Start slightly away from sonic point
    eps_f = 1e-6 * (1 if direction == 'outward' else -1)
    f0 = f_s + eps_f
    V0 = V_s + p * eps_f
    Z0 = Z_s + q * eps_f

    def rhs(f, y):
        V, Z = y
        f_safe = max(abs(f), 1e-14)
        fmV = f - V
        Delta = fmV**2 - Z**2/gamma
        N_R = Z**2*(s-1)*V/gamma + V*(n-1)*fmV/n

        if abs(Delta) < 1e-14:
            return [p, q]  # use sonic slopes

        dV = N_R / (f_safe * Delta)

        if abs(fmV) < 1e-14:
            return [dV, 0.0]

        dZ = Z/fmV * (beta*(dV + (s-1)*V/f_safe) - (n-1.0)/n)
        return [dV, dZ]

    try:
        sol = solve_ivp(rhs, [f0, f_end], [V0, Z0],
                       method='RK45', max_step=0.0002,
                       rtol=1e-10, atol=1e-12)
        return sol
    except:
        return None


def shooting_residual(params, verbose=False):
    """
    Given (n, f_s), integrate from sonic point to shock.
    Return residual = [V(1) - V_shock, Z(1) - Z_shock].
    """
    n, f_s = params

    if n <= 0.1 or n >= 1.0 or f_s <= 0.01 or f_s >= 0.99:
        return [100, 100]

    sol = integrate_from_sonic(n, f_s, direction='outward', f_end=1.0)

    if sol is None or not sol.success:
        return [100, 100]

    V_end = sol.y[0, -1]
    Z_end = sol.y[1, -1]
    f_end = sol.t[-1]

    if verbose:
        print(f"  n={n:.6f}, f_s={f_s:.6f}")
        print(f"  Sonic: V_s={sonic_point_values(n, f_s)[0]:.6f}, Z_s={sonic_point_values(n, f_s)[1]:.6f}")
        print(f"  At f={f_end:.4f}: V={V_end:.6f}, Z={Z_end:.6f}")
        print(f"  Target: V={V_shock:.6f}, Z={Z_shock:.6f}")
        print(f"  Residual: dV={V_end-V_shock:.6e}, dZ={Z_end-Z_shock:.6e}")

    return [V_end - V_shock, Z_end - Z_shock]


# ========== Scan ==========
print("Phase 1: Scan (n, f_s) space")
print()

n_values = np.linspace(0.5, 0.9, 41)
fs_values = np.linspace(0.3, 0.9, 31)

best_residual = 1e10
best_n = None
best_fs = None

print(f"  {'n':>6s} {'f_s':>6s} {'|res_V|':>10s} {'|res_Z|':>10s} {'|res|':>10s}")
for n in n_values:
    for f_s in fs_values:
        try:
            res = shooting_residual([n, f_s])
            total_res = np.sqrt(res[0]**2 + res[1]**2)
            if total_res < best_residual:
                best_residual = total_res
                best_n = n
                best_fs = f_s
                if total_res < 0.1:
                    print(f"  {n:6.4f} {f_s:6.4f} {abs(res[0]):10.4e} {abs(res[1]):10.4e} {total_res:10.4e}")
        except:
            pass

print(f"\nBest: n={best_n:.4f}, f_s={best_fs:.4f}, |residual|={best_residual:.6e}")

# ========== Refine ==========
if best_residual < 1.0:
    print("\nPhase 2: Refine with Newton's method")
    try:
        result = fsolve(shooting_residual, [best_n, best_fs], full_output=True)
        sol, info, ier, msg = result
        if ier == 1:
            n_star, fs_star = sol
            print(f"\n*** CONVERGED ***")
            print(f"  alpha (n) = {n_star:.10f}")
            print(f"  f_s (sonic) = {fs_star:.10f}")

            V_s, Z_s = sonic_point_values(n_star, fs_star)
            print(f"  V_s = {V_s:.10f}")
            print(f"  Z_s = {Z_s:.10f}")

            # Verify
            print("\nVerification:")
            shooting_residual([n_star, fs_star], verbose=True)

            # Compare with known values
            print(f"\nComparison:")
            print(f"  Computed alpha = {n_star:.8f}")
            print(f"  Lazarus (1981) = 0.68838...")
            print(f"  Difference = {abs(n_star - 0.6884):.6e}")
        else:
            print(f"fsolve did not converge: {msg}")
    except Exception as e:
        print(f"fsolve failed: {e}")
else:
    print("Could not find good initial guess. Trying wider scan...")
    # Do a finer scan
    n_values = np.linspace(0.4, 0.95, 111)
    fs_values = np.linspace(0.1, 0.95, 85)

    for n in n_values:
        for f_s in fs_values:
            try:
                res = shooting_residual([n, f_s])
                total_res = np.sqrt(res[0]**2 + res[1]**2)
                if total_res < best_residual:
                    best_residual = total_res
                    best_n = n
                    best_fs = f_s
            except:
                pass

    print(f"Best from wider scan: n={best_n:.4f}, f_s={best_fs:.4f}, |res|={best_residual:.6e}")

# ========== Plot solution if converged ==========
if best_residual < 0.01:
    n_final = best_n
    fs_final = best_fs

    # Try to refine
    try:
        sol_opt = fsolve(shooting_residual, [n_final, fs_final])
        n_final, fs_final = sol_opt
    except:
        pass

    sol_out = integrate_from_sonic(n_final, fs_final, 'outward', 1.0)
    sol_in = integrate_from_sonic(n_final, fs_final, 'inward', 0.01)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if sol_out is not None:
        axes[0].plot(sol_out.t, sol_out.y[0], 'b-', label='V (outward)')
        axes[0].plot(sol_out.t, sol_out.y[1], 'r-', label='Z (outward)')
    if sol_in is not None:
        axes[0].plot(sol_in.t, sol_in.y[0], 'b--', label='V (inward)')
        axes[0].plot(sol_in.t, sol_in.y[1], 'r--', label='Z (inward)')
    axes[0].axvline(x=fs_final, color='g', ls=':', label=f'sonic f_s={fs_final:.3f}')
    axes[0].set_xlabel('f (= xi)')
    axes[0].set_ylabel('V, Z')
    axes[0].set_title(f'Solution for alpha={n_final:.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Delta plot
    if sol_out is not None:
        D_out = (sol_out.t - sol_out.y[0])**2 - sol_out.y[1]**2/gamma
        axes[1].plot(sol_out.t, D_out, 'b-', label='outward')
    if sol_in is not None:
        D_in = (sol_in.t - sol_in.y[0])**2 - sol_in.y[1]**2/gamma
        axes[1].plot(sol_in.t, D_in, 'r-', label='inward')
    axes[1].axhline(y=0, color='k', ls='-', lw=0.5)
    axes[1].set_xlabel('f (= xi)')
    axes[1].set_ylabel('Delta')
    axes[1].set_title('Sonic line (Delta=0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Phase portrait
    if sol_out is not None:
        axes[2].plot(sol_out.y[0], sol_out.y[1], 'b-', label='outward')
    if sol_in is not None:
        axes[2].plot(sol_in.y[0], sol_in.y[1], 'r-', label='inward')
    axes[2].set_xlabel('V')
    axes[2].set_ylabel('Z')
    axes[2].set_title('Phase portrait')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/share/project/zpy/PINN_WE/cases/self-discovery/guderley_eigenvalue.png',
                dpi=150, bbox_inches='tight')
    print("Figure saved.")
