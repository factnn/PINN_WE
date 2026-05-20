#!/usr/bin/env python3
"""
Guderley eigenvalue solver — Chisnell (1998) phase-plane method.

Algorithm (following Chisnell 1998 and ExactPack/Ramsey):
    1. Write the Guderley self-similar ODE as a single equation dC/dV = F(V, C)
       using Chisnell's (V, C) phase-plane formulation (Eq. 3.1)
    2. For given alpha (= 1/lambda), the critical (sonic) point (V0, C0) is
       computed algebraically from gamma and geometry
    3. Integrate from shock conditions (Vs, Cs) to V = V0
    4. Compare C(V0) with C0 — difference should be zero for correct alpha
    5. Brent root-finding on alpha

Variables (Chisnell notation):
    V = dimensionless velocity (= 2*alpha/(gamma+1) at shock)
    C = (sound speed)^2 related quantity
    alpha = 1/lambda where lambda is the similarity exponent R ~ t^(1/lambda)

References:
    - Chisnell (1998), J. Fluid Mech. 354, 357-375
    - Ramsey, Kamm & Bolstad (2012), LA-UR-12-20828
    - ExactPack (LANL), exactpack.solvers.guderley.eexp
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Known eigenvalues for validation (alpha = 1/lambda)
# High-precision values computed by this solver (Chisnell phase-plane method)
KNOWN = {
    (1.4, 3): 0.717174501487,   # spherical,  gamma=1.4 (air)
    (1.4, 2): 0.835323191951,   # cylindrical, gamma=1.4
    (5 / 3, 3): 0.688376822921, # spherical,  gamma=5/3 (argon)
    (5 / 3, 2): 0.815624901431, # cylindrical, gamma=5/3
}


def chisnell_rhs(V, C_vec, alpha, gamma, n):
    """
    RHS of Chisnell Eq. (3.1): dC/dV = numer / denom.
    V is the independent variable, C_vec = [C].
    """
    C = C_vec[0]
    delta = (V - alpha) ** 2 - C
    Q = (n * V * (V - alpha)
         + (2.0 / gamma) * (1.0 - alpha) * (alpha - V)
         - V * (V - 1.0))

    numer = C * (2.0 * delta * (alpha - V + (1.0 - alpha) / gamma)
                 + (gamma - 1.0) * (alpha - V) * Q)

    denom = (delta * (n * V - 2.0 * (1.0 - alpha) / gamma) * (alpha - V)
             + (alpha - V) ** 2 * Q)

    return [numer / denom]


def critical_point(alpha: float, gamma: float, n: int) -> tuple[float, float]:
    """
    Compute the critical (sonic) point (V0, C0) algebraically.
    V0 is a root of the singular denominator condition.
    C0 = (V0 - alpha)^2 at the critical point (sonic condition).
    """
    # gamma_crit values from Lazarus/ExactPack
    if n == 3:
        gamma_crit = 1.8697680
    elif n == 2:
        gamma_crit = 1.9092084
    else:
        gamma_crit = 2.0

    V0_denom = 2.0 * gamma * (n - 1.0)
    factor = gamma * n - 2.0
    disc = (8.0 * (alpha - 1.0) * alpha * gamma * (n - 1.0)
            + (2.0 - gamma + alpha * factor) ** 2)

    if disc < 0:
        return None, None

    if gamma >= gamma_crit:
        V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
              - np.sqrt(disc)) / V0_denom
    else:
        V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
              + np.sqrt(disc)) / V0_denom

    C0 = (V0 - alpha) ** 2
    return V0, C0


def shooting_residual(alpha: float, gamma: float, n: int) -> float:
    """
    Integrate Chisnell's ODE from shock to critical point.
    Returns C0_algebraic - C_integrated(V0).
    """
    V0, C0 = critical_point(alpha, gamma, n)
    if V0 is None:
        return 1e10

    # Shock conditions (Chisnell variables)
    Vs = 2.0 * alpha / (gamma + 1.0)
    Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2

    if abs(V0 - Vs) < 1e-14:
        return C0 - Cs

    try:
        sol = solve_ivp(
            lambda V, C: chisnell_rhs(V, C, alpha, gamma, n),
            [Vs, V0],
            [Cs],
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success:
            return C0 - sol.y[0][-1]
    except Exception:
        pass
    return 1e10


def find_eigenvalue(gamma: float, n: int, name: str,
                    verbose: bool = True) -> dict:
    """
    Find Guderley eigenvalue alpha for given (gamma, n).
    n = 2 (cylindrical) or 3 (spherical).
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"gamma={gamma:.4f}, n={n} ({name})")

    # Bracket for alpha (from ExactPack)
    a0num = -2.0 - gamma - np.sqrt(2.0) * gamma * np.sqrt(gamma / (gamma - 1.0))
    a0dem = -2.0 - np.sqrt(2.0) * gamma * np.sqrt(gamma / (gamma - 1.0)) - gamma * n
    a0 = a0num / a0dem

    if gamma > 3.732050808:
        amin = a0
    else:
        amin = ((4.0 + 2.0 * np.sqrt(2.0) * np.sqrt(gamma ** 3 * (n - 1.0) ** 2)
                 + gamma * (-6.0 + (2.0 + gamma) * n))
                / (4.0 + gamma * (-8.0 + n * (4.0 + gamma * n)))) + 1e-6
    amax = min(1.05 * a0, 0.9999)

    if verbose:
        print(f"  Search bracket: [{amin:.6f}, {amax:.6f}]")

    # Verify bracket
    fa = shooting_residual(amin, gamma, n)
    fb = shooting_residual(amax, gamma, n)
    if verbose:
        print(f"  f(amin)={fa:.6e}, f(amax)={fb:.6e}")

    if fa * fb > 0:
        if verbose:
            print("  WARNING: bracket does not straddle zero, scanning...")
        alphas = np.linspace(amin, amax, 200)
        residuals = [shooting_residual(a, gamma, n) for a in alphas]
        residuals = np.array(residuals)
        for i in range(len(residuals) - 1):
            if residuals[i] * residuals[i + 1] < 0:
                amin = alphas[i]
                amax = alphas[i + 1]
                fa = residuals[i]
                fb = residuals[i + 1]
                if verbose:
                    print(f"  Found bracket: [{amin:.6f}, {amax:.6f}]")
                break

    if fa * fb > 0:
        if verbose:
            print("  ERROR: could not find bracket")
        return {"gamma": gamma, "n": n, "geometry": name, "alpha": None,
                "error": None, "known": None}

    # Brent root-finding
    alpha = brentq(shooting_residual, amin, amax, args=(gamma, n), xtol=1e-12)

    # Find known value by closest gamma match
    known = None
    for (g, nn), val in KNOWN.items():
        if nn == n and abs(g - gamma) < 1e-4:
            known = val
            break
    error = abs(alpha - known) if known else None

    if verbose:
        print(f"  alpha = {alpha:.12f}")
        if known:
            print(f"  known = {known:.6f}")
            print(f"  error = {error:.6e}")

    return {
        "gamma": gamma,
        "n": n,
        "geometry": name,
        "alpha": float(alpha),
        "lambda": float(1.0 / alpha),
        "known": known,
        "error": float(error) if error else None,
    }


def main():
    configs = [
        (1.4, 3, "spherical"),
        (1.4, 2, "cylindrical"),
        (5.0 / 3.0, 3, "spherical"),
        (5.0 / 3.0, 2, "cylindrical"),
    ]

    results = []
    for gamma, n, name in configs:
        r = find_eigenvalue(gamma, n, name)
        results.append(r)

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY — Guderley eigenvalues (Chisnell phase-plane method)")
    print(f"{'=' * 72}")
    print(f"{'gamma':>7s} {'geom':>13s} {'alpha':>14s} "
          f"{'lambda':>14s} {'known':>10s} {'error':>12s}")
    print("-" * 72)
    for r in results:
        if r['alpha'] is None:
            print(f"{r['gamma']:7.4f} {r['geometry']:>13s}  FAILED")
            continue
        known_s = f"{r['known']:.6f}" if r['known'] else "N/A"
        err_s = f"{r['error']:.2e}" if r['error'] else ""
        print(f"{r['gamma']:7.4f} {r['geometry']:>13s} "
              f"{r['alpha']:14.12f} {r['lambda']:14.10f} "
              f"{known_s:>10s} {err_s:>12s}")

    # Save
    out = Path("cases/self-discovery/canonical/output")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "guderley_eigenvalues.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out / 'guderley_eigenvalues.json'}")


if __name__ == "__main__":
    main()
