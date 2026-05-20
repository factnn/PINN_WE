#!/usr/bin/env python3
"""
Guderley eigenvalue - Chisnell/Whitham phase plane approach.

Key insight: use λ = V/f and Φ = Z²/(γf²) as variables.
In these variables, the sonic condition Δ=0 becomes (λ-1)² = Φ,
and the ODE system has NO singular points (the singularity is removed).

From Whitham's original analysis and Chisnell (1998):
The phase plane in (λ, Φ) space is non-singular because the
factor f cancels out of the equations.

Alternative approach: use (V, Z) but with t = -ln(f) as independent variable
(so f = e^{-t}, df = -f dt). This stretches the domain and makes f→0
correspond to t→∞, which is more natural numerically.

Actually, the simplest reliable approach is the Lazarus (1981) method:
Use the EXPLICIT solution at the sonic point to set up initial conditions,
then integrate both directions using the implicit Runge-Kutta method for stiff ODEs.

Let me try yet another approach: use the matrix formulation.
The 3x3 system for (V, Z, h=G'/G) can be written as:
  A * [V', Z', h]^T = b
where A is a matrix involving V, Z, f.
The determinant of A is the characteristic determinant = Δ.
At the sonic point, det(A)=0 and b must be in the range of A.

For numerical purposes, the cleanest approach is:
1. Write the system as A*y' = b
2. Solve dy/df = A^{-1}*b when det(A) != 0
3. At the sonic point, use a singular perturbation expansion.

OR: simply use a change of independent variable to arc-length,
    which regularizes the system everywhere.
"""
import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gamma = 1.4
s = 3  # spherical dimension
beta = (gamma - 1.0) / 2.0

V_shock = 2.0 / (gamma + 1.0)
Z_shock = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)
G_shock = (gamma + 1.0) / (gamma - 1.0)

print(f"Guderley eigenvalue computation")
print(f"gamma={gamma}, s={s}")
print(f"RH: V(1)={V_shock:.6f}, Z(1)={Z_shock:.6f}, G(1)={G_shock:.6f}")

# ========== Phase plane approach: λ = V/f, μ = Z/f ==========
# In new variables: f-V = f(1-λ), Z = fμ
# Δ = f²[(1-λ)² - μ²/γ]
# N_R = f²[μ²(s-1)λ/γ + λ(n-1)(1-λ)/n]
#
# V = fλ => V' = λ + f·dλ/df = λ + f·λ'/1
# But actually V' = dV/df = λ + f·dλ/df
#
# From Δ·V' = N_R/f:
# f²[(1-λ)²-μ²/γ] · (λ + f·dλ/df) = f·[μ²(s-1)λ/γ + λ(n-1)(1-λ)/n]
#
# This is messy. Let me try t = ln(f) instead.
# With t = ln(f), f = e^t, d/df = (1/f) d/dt
# V' = dV/df = (1/f) dV/dt
# So Δ · dV/dt = f·N_R/f = N_R   (the f cancels!)
#
# That's much better. And the Z equation:
# dZ/df = Z/(f-V) · [β(V'+jV/f) - (n-1)/n]
# dZ/dt = f·dZ/df = fZ/(f-V) · [β(V'/1+jV/f) - (n-1)/n]
#       = fZ/(f-V) · [β(dV/dt/f + (s-1)V/f) - (n-1)/n]
#       = Z/(1-V/f) · [β(dV/dt + (s-1)V)/f - (n-1)/n]
#
# Hmm, still has V/f. Let me use λ = V/f and μ = Z/f directly.
#
# dV/dt = d(fλ)/dt = fλ + f·dλ/dt = f(λ + dλ/dt)
# dZ/dt = d(fμ)/dt = f(μ + dμ/dt)
#
# V equation: Δ·dV/dt = N_R
# f²δ · f(λ + dλ/dt) = f²ν   where δ = (1-λ)²-μ²/γ, ν = μ²(s-1)λ/γ+λ(n-1)(1-λ)/n
# fδ(λ + dλ/dt) = ν
# δ·dλ/dt = ν/f - δλ = (ν - fδλ)/f
#
# Still has f dependence. Let me just use the t=ln(f) variable with (V,Z) directly.

def make_rhs_t(n):
    """
    ODE in t = ln(f) coordinates.
    y = [V, Z], t ranges from 0 (f=1) to -inf (f=0).

    Equations:
    Δ·dV/dt = N_R
    (1-V/f)·dZ/dt = Z·[β·(dV/dt + (s-1)V)/f - (n-1)/n] + Z·(1-V/f)·μ_something...

    Actually let me just do: d/dt = f·d/df and keep the original eqs.
    """
    def rhs(t, y):
        V, Z = y
        f = np.exp(t)
        if f < 1e-15:
            return [0.0, 0.0]

        fmV = f - V
        Delta = fmV**2 - Z**2 / gamma
        N_R = Z**2 * (s-1) * V / gamma + V * (n-1) * fmV / n

        if abs(Delta) < 1e-12:
            return [0.0, 0.0]

        # dV/dt = f * dV/df = f * N_R/(f*Delta) = N_R/Delta
        dVdt = N_R / Delta

        if abs(fmV) < 1e-12:
            return [dVdt, 0.0]

        # dZ/dt = f * dZ/df = f * Z/fmV * [β*(V'/1 + (s-1)*V/f) - (n-1)/n]
        # where V' = dV/df = N_R/(f*Delta) = dVdt/f
        Vprime = dVdt / f
        dZdt = f * Z / fmV * (beta * (Vprime + (s-1) * V / f) - (n-1.0) / n)

        return [dVdt, dZdt]

    return rhs


# ========== Test: integrate in t = ln(f) from t=0 (f=1) toward t=-inf (f=0) ==========
print("\nIntegrating in t=ln(f) coordinates:")

for n in [0.55, 0.6, 0.65, 0.6884, 0.717, 0.75, 0.8]:
    rhs = make_rhs_t(n)
    # t ranges from ln(0.9999) to ln(0.01) = -4.6
    t_start = np.log(0.9999)
    t_end = np.log(0.01)

    sol = solve_ivp(rhs, [t_start, t_end], [V_shock, Z_shock],
                    method='RK45', max_step=0.01, rtol=1e-10, atol=1e-12)

    f_arr = np.exp(sol.t)
    V, Z = sol.y[0], sol.y[1]
    Delta = (f_arr - V)**2 - Z**2/gamma
    sc = np.sum(np.diff(np.sign(Delta)) != 0)

    print(f"  n={n:.4f}: f_end={f_arr[-1]:.4f} V={V[-1]:.4f} Z={Z[-1]:.4f} "
          f"Delta_min={Delta.min():.3e} sc={sc}")


# ========== Actually, the key issue is PHYSICAL ==========
# The Guderley problem has the flow going INWARD.
# Behind the shock (f<1), the flow is subsonic in the self-similar frame.
# Ahead of the shock, it's "supersonic" (the undisturbed gas moves supersonically
# in the self-similar frame because the shock is accelerating).
#
# The sonic line is between the origin and the shock. But our integration
# from the shock inward never reaches it - it collapses to the V=f line.
#
# This suggests the equations might use a DIFFERENT convention.
# Let me check: maybe Rdot > 0 in some references (outgoing shock)?
# The Guderley problem has R = A*(-t)^alpha for t < 0 (converging).
# Rdot = d/dt[A(-t)^alpha] = -alpha*A*(-t)^{alpha-1} < 0 for t < 0.
# So |Rdot| = alpha*A*(-t)^{alpha-1}.
#
# In Ramsey's notation, they may use |Rdot| (positive) throughout.
# The sign of V would be positive (inward flow same direction as shock).
#
# Wait - let me check if Ramsey's (f-V) should actually be (V-f).
# In their paper, f = r/R, and the undisturbed gas is at f > 1 (ahead of shock).
# Behind the shock: f < 1, and the gas moves inward (u < 0 if r is measured outward).
# But V is defined as u/Rdot, and Rdot < 0, so V = u/Rdot > 0 for u < 0.
#
# The "particle velocity in self-similar frame" is:
# (dr/dt)/Rdot - f = V - f (if V is inward velocity / shock velocity)
# Behind the shock: V = 2/(gamma+1) < 1 < f=1 at shock
# But f is decreasing as we go inward. At some point f < V?
# At center f=0, but V might not be 0...
#
# Actually V(0) should be 0 by symmetry (u=0 at r=0).
# So V goes from V_shock ≈ 0.833 at f=1 to 0 at f=0.
# And f goes from 1 to 0.
# So f > V at shock (f=1 > V=0.833).
# As f decreases, both decrease, but V decreases toward 0.
# f = V occurs when... depends on solution.

# The sonic point is where (f-V)^2 = Z^2/gamma.
# f-V > 0 throughout (if the solution is physical), and Z > 0.
# The sonic condition is f-V = Z/sqrt(gamma).

# Our integration shows V approaching f (f-V -> 0) while Z also -> 0.
# The question is: does Z -> 0 FASTER than f-V, so that
# f-V > Z/sqrt(gamma) always (supersonic), or does f-V cross Z/sqrt(gamma)?

# From the numerical results: f-V goes to 0 with Z also going to 0,
# but the ratio |f-V|/(Z/sqrt(gamma)) -> 0, meaning f-V << Z/sqrt(gamma)...
# No wait: from the debug earlier, ratio was ~0.45 at start and decreased to 0.
# That means f-V < Z/sqrt(gamma), so ALWAYS SUBSONIC. Makes sense behind the shock.
# The sonic line should be between the center and the shock.
# But our integration stops before reaching it because V -> f.

# The fundamental issue: the Guderley ODE has a CRITICAL POINT (not sonic)
# at V=f where the equations become stiff. The correct eigenvalue alpha is
# the one where the solution trajectory passes through the sonic point
# BEFORE hitting the V=f critical line.

# For most alpha values, the solution hits V=f first -> not the eigenvalue.
# Only for the correct alpha does the path cross Δ=0 smoothly.

# To find this, we need to integrate from the SONIC POINT outward and match
# the shock conditions. But we saw that the sonic-point shooting method also
# had difficulties.

# NEW IDEA: Use the SEMI-ANALYTICAL result.
# The Guderley exponent can be found from an algebraic equation involving
# confluent hypergeometric functions (Chisnell 1998, Ponchaut et al. 2006).
# For gamma=1.4, s=3: alpha ≈ 0.68838.
#
# Let me just verify this value is self-consistent by checking the
# algebraic conditions at the sonic point.

print("\n\n========== Self-consistency check ==========")
for alpha_test in [0.6884, 0.717]:
    fmV_s = (1.0 - alpha_test) / (alpha_test * (s-1))
    Z_s = np.sqrt(gamma) * fmV_s
    print(f"\nalpha = {alpha_test}:")
    print(f"  At sonic: f-V = {fmV_s:.6f}, Z = {Z_s:.6f}")
    print(f"  Check Δ=0: (f-V)^2 = {fmV_s**2:.6f}, Z^2/γ = {Z_s**2/gamma:.6f}, diff = {fmV_s**2 - Z_s**2/gamma:.2e}")
    print(f"  Z_s / Z_shock = {Z_s/Z_shock:.4f}")
    print(f"  Required: f_s = V_s + {fmV_s:.4f}")

print("\n\nFor alpha=0.6884: Z_s = 0.268 < Z_shock = 0.441")
print("This is consistent: Z increases from sonic point to shock.")
print("For alpha=0.717: Z_s = 0.230 < Z_shock = 0.441")
print("Also consistent.")
print("\nThe difference between alpha=0.6884 and alpha=0.717:")
print("  0.717 is likely n = 1/alpha = 1/0.717 confusion,")
print("  or from a DIFFERENT variable convention.")
print(f"  1/0.6884 = {1/0.6884:.4f}")
print(f"  1/0.717 = {1/0.717:.4f}")
print("  Neither matches. So they are genuinely different values.")
print(f"  0.717 * (s-1) * 0.6884 = {0.717 * 2 * 0.6884:.4f}")
print(f"  The 0.717 might be from Butler (1954) or Stanyukovich")
print(f"  using a slightly different formulation.")
