#!/usr/bin/env python3
"""Debug the Guderley integration to understand why it stops."""

import numpy as np
from scipy.integrate import solve_ivp

gamma = 1.4
s = 3
beta = (gamma - 1.0) / 2.0
V_bc = 2.0 / (gamma + 1.0)
Z_bc = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)

def rhs_detail(f, y, n=0.6884):
    V, Z, lnD = y
    eps = 1e-14
    f_s = max(abs(f), eps)
    fmV = f - V
    Delta = fmV**2 - Z**2 / gamma

    if abs(Delta) < 1e-12:
        return [0.0, 0.0, 0.0]

    h = V / f_s * ((s-1) * fmV + (n-1.0)/n) / Delta
    dV = h * fmV - (s-1) * V / f_s

    if abs(fmV) < 1e-12:
        dZ = 0.0
    else:
        dZ = Z * (beta * h - (n-1.0) / (n * fmV))

    dlnD = h
    return [dV, dZ, dlnD]

sol = solve_ivp(rhs_detail, [0.9999, 0.001], [V_bc, Z_bc, np.log(6.0)],
                method='RK45', max_step=0.0001, rtol=1e-12, atol=1e-14,
                dense_output=True)

V, Z = sol.y[0], sol.y[1]
f_arr = sol.t
fmV = f_arr - V
Delta = fmV**2 - Z**2/gamma

header = f"{'f':>8s} {'V':>10s} {'Z':>10s} {'f-V':>12s} {'Delta':>12s} {'dV/df':>12s}"
print(header)
step = max(1, len(f_arr)//20)
for i in range(0, len(f_arr), step):
    if i > 0:
        dVdf = (V[i]-V[i-1])/(f_arr[i]-f_arr[i-1])
    else:
        dVdf = 0
    print(f"{f_arr[i]:8.5f} {V[i]:10.6f} {Z[i]:10.6f} {fmV[i]:12.6e} {Delta[i]:12.6e} {dVdf:12.4f}")

print(f"\nV: {V[0]:.6f} -> {V[-1]:.6f}")
print(f"f: {f_arr[0]:.6f} -> {f_arr[-1]:.6f}")
print(f"V-f: {V[0]-f_arr[0]:.6f} -> {V[-1]-f_arr[-1]:.10f}")
print(f"Z: {Z[0]:.6f} -> {Z[-1]:.6f}")
print(f"f-V always positive: {np.all(fmV > 0)}")
print(f"f-V goes to 0: min(f-V) = {fmV.min():.2e}")

# Is V approaching f from below, with dV/df -> 1 (i.e., V tracks f)?
# In that case V = f + epsilon, with epsilon -> 0
# This means the fluid velocity = coordinate velocity, i.e., the fluid is "riding" the expansion
# This is actually a PHYSICAL critical point where the flow matches the self-similar expansion.

# The key question: does the PHYSICAL sonic line lie at smaller xi than this V=f point?
# If so, we can never reach it with forward shooting.

# Let's check: where would Delta=0 require us to be?
# Delta = (f-V)^2 - Z^2/gamma = 0 => f-V = Z/sqrt(gamma)
# At the end: f-V ~ 0, Z ~ 0.009 => Z/sqrt(gamma) ~ 0.0075
# So we'd need f-V ~ 0.0075, which happens just before the integration fails

# Actually, if both f-V and Z go to 0 at the same rate, Delta could cross 0!
# Let's check the ratio:
ratio = np.abs(fmV) / (Z / np.sqrt(gamma) + 1e-15)
print(f"\n|f-V| / (Z/sqrt(gamma)) ratio:")
print(f"  Start: {ratio[0]:.4f}")
print(f"  End: {ratio[-1]:.4f}")
print(f"  Min: {ratio.min():.4f} at f={f_arr[np.argmin(ratio)]:.4f}")

# If ratio > 1 always, Delta < 0 always (subsonic)
# If ratio crosses 1, Delta=0 (sonic line)
print(f"  Ratio > 1 everywhere: {np.all(ratio > 1)}")
crosses = np.where(np.diff(np.sign(ratio - 1)) != 0)[0]
print(f"  Ratio=1 crossings: {len(crosses)}")
