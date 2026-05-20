#!/usr/bin/env python3
"""
Guderley eigenvalue - sonic point shooting with proper L'Hopital slopes.
Uses 'stiff' method (Radau) to handle near-singular regions.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

gamma = 1.4
s = 3  # spherical
beta = (gamma - 1.0) / 2.0

V_shock = 2.0 / (gamma + 1.0)
Z_shock = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)

print(f"gamma={gamma}, s={s}, V_shock={V_shock:.6f}, Z_shock={Z_shock:.6f}")


def sonic_values(n):
    """For given alpha=n, get (f-V) at sonic and Z at sonic."""
    fmV_s = (1.0 - n) / (n * (s - 1))
    Z_s = np.sqrt(gamma) * fmV_s
    return fmV_s, Z_s


def integrate_outward(n, f_s, verbose=False):
    """
    From sonic point at f_s, integrate to f=1 (shock).

    At sonic: f-V = (1-n)/(n*(s-1)), Z = sqrt(gamma)*(1-n)/(n*(s-1))

    Need proper starting slopes. Use numerical perturbation approach:
    start eps above sonic with V_s + eps*p, Z_s + eps*q where p,q chosen
    to be on the subsonic branch.
    """
    fmV_s, Z_s = sonic_values(n)
    V_s = f_s - fmV_s

    if V_s <= 0 or Z_s <= 0 or fmV_s <= 0:
        return None, None

    def rhs(f, y):
        V, Z = y
        f_safe = max(abs(f), 1e-15)
        fmV = f - V
        Delta = fmV**2 - Z**2 / gamma
        N_R = Z**2 * (s-1) * V / gamma + V * (n-1) * fmV / n

        if abs(Delta) < 1e-14:
            return [0.0, 0.0]

        dV = N_R / (f_safe * Delta)

        if abs(fmV) < 1e-14:
            return [dV, 0.0]

        dZ = Z / fmV * (beta * (dV + (s-1) * V / f_safe) - (n-1.0) / n)
        return [dV, dZ]

    # Start slightly above sonic (subsonic side)
    eps = 1e-5

    # On the subsonic side (f > f_s), Delta < 0
    # We need (f-V)^2 < Z^2/gamma
    # Perturb: V = V_s + p*eps, Z = Z_s + q*eps
    # f = f_s + eps
    # f-V = (f_s+eps) - (V_s+p*eps) = fmV_s + (1-p)*eps
    # Need (fmV_s+(1-p)*eps)^2 < (Z_s+q*eps)^2/gamma
    # At eps=0: fmV_s^2 = Z_s^2/gamma (exactly). So need (1-p)*fmV_s < q*Z_s/gamma

    # Simple approach: just perturb f by eps, keep V and Z at sonic values
    # This puts us slightly supersonic (f-V increased, Z unchanged)
    # OR slightly subsonic depending on direction.

    # Actually, on the f > f_s side:
    # fmV = fmV_s + eps*(1 - dV/df at sonic)
    # We need to figure out dV/df at sonic...

    # Simplest: just try two initial conditions and pick the one that works
    results = []
    for trial_p in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5]:
        f0 = f_s + eps
        V0 = V_s + trial_p * eps
        Z0 = Z_s  # Z doesn't change much at first order

        try:
            sol = solve_ivp(rhs, [f0, 1.0], [V0, Z0],
                           method='Radau', max_step=0.01,
                           rtol=1e-8, atol=1e-10)
            if sol.success and sol.t[-1] > 0.999:
                V_end, Z_end = sol.y[0, -1], sol.y[1, -1]
                res = np.sqrt((V_end - V_shock)**2 + (Z_end - Z_shock)**2)
                results.append((trial_p, V_end, Z_end, res, sol))
        except:
            pass

    if not results:
        return None, None

    # Find best trial
    best = min(results, key=lambda x: x[3])
    if verbose:
        print(f"  n={n:.6f}, f_s={f_s:.4f}: best_p={best[0]:.1f}, "
              f"V(1)={best[1]:.6f}, Z(1)={best[2]:.6f}, res={best[3]:.4e}")

    return best[1] - V_shock, best[2] - Z_shock


# ========== Quick scan ==========
print("\nScanning (n, f_s) space:")

best_res = 1e10
best_n = best_fs = None

for n in np.linspace(0.5, 0.95, 46):
    fmV_s, Z_s = sonic_values(n)
    for f_s in np.linspace(max(0.3, fmV_s + 0.05), 0.95, 30):
        res = integrate_outward(n, f_s)
        if res[0] is not None:
            total = np.sqrt(res[0]**2 + res[1]**2)
            if total < best_res:
                best_res = total
                best_n, best_fs = n, f_s
                if total < 0.05:
                    print(f"  n={n:.4f} f_s={f_s:.4f}: dV={res[0]:+.4e} dZ={res[1]:+.4e} |r|={total:.4e}")

print(f"\nBest: n={best_n}, f_s={best_fs}, |res|={best_res:.6e}")

if best_res < 0.1 and best_n is not None:
    print("\nRefining...")
    # 1D search: fix f_s, vary n
    def residual_n(n):
        res = integrate_outward(n, best_fs)
        if res[0] is None:
            return 1e5
        return res[0]  # match V at shock

    # Find sign change
    ns = np.linspace(max(0.4, best_n - 0.1), min(0.99, best_n + 0.1), 40)
    vals = [residual_n(nn) for nn in ns]

    for i in range(len(vals) - 1):
        if vals[i] * vals[i+1] < 0 and abs(vals[i]) < 10 and abs(vals[i+1]) < 10:
            n_root = brentq(residual_n, ns[i], ns[i+1], xtol=1e-10)
            print(f"  Root: n = {n_root:.10f}")
            integrate_outward(n_root, best_fs, verbose=True)

print("\nDone.")
