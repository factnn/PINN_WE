#!/usr/bin/env python3
"""
Guderley self-similar equation - numerical verification v2

Correct first-order ODE system (Ramsey et al. 2012, Lazarus 1981):
  Notation: u = Ṙ·V(ξ), c = Ṙ·C(ξ), ρ = ρ₁·G(ξ), ξ = r/R(t), R ~ (-t)^α

  (I)   (V-ξ)·G'/G + V' + j·V/ξ = 0                    [continuity]
  (II)  (V-ξ)·V' + C²·G'/(γG) + V·(α-1)/(α·ξ) = 0     [momentum]
  (III) (V-ξ)·C'/C - β·(V-ξ)·G'/G + (1/α-1) = 0        [entropy]

After eliminating G'/G using (I):
  G'/G = -(V' + j·V/ξ)/(V-ξ)

Substituting into (II):
  (V-ξ)²·V' - C²/γ·(V' + j·V/ξ) + (V-ξ)·V·(α-1)/(α·ξ) = 0
  Δ·V' = N_V/ξ

where:
  Δ = (V-ξ)² - C²/γ
  N_V = C²·j·V/γ - V·(α-1)·(V-ξ)/α

Substituting into (III):
  (V-ξ)·C'/C + β·(V'+j·V/ξ) + (1/α-1) = 0
  C' = C·[-β·(V'+j·V/ξ) - (1/α-1)] / (V-ξ)

Boundary conditions at ξ=1 (strong shock RH):
  V(1) = 2/(γ+1)
  C(1) = sqrt(2γ(γ-1))/(γ+1)

Sonic line: Δ=0, requires N_V=0 simultaneously for regular solution.
The eigenvalue α is determined by this regularity condition.

For γ=1.4, j=2 (spherical): α ≈ 0.6884 (Lazarus 1981)
                  j=1 (cylindrical): α ≈ 0.8156

NOTE: Different references use different α conventions!
  - Some define R ~ (-t)^α with α<1 (converging phase)
  - Some define R ~ t^(1/n) with n>1
  - The Lazarus (1981) value for spherical γ=1.4 is α≈0.6884
  - The "0.717" value commonly cited corresponds to n=1/α or a different convention
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== Physical parameters ==========
gamma = 1.4
j_geom = 2       # spherical symmetry (j=0 planar, j=1 cylindrical, j=2 spherical)
beta = (gamma - 1.0) / 2.0

# ========== Strong shock RH BCs at ξ=1 ==========
V_bc = 2.0 / (gamma + 1.0)                              # = 0.8333
C_bc = np.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)  # ≈ 0.4410
G_bc = (gamma + 1.0) / (gamma - 1.0)                    # = 6.0

print(f"Guderley self-similar ODE numerical verification v2")
print(f"gamma={gamma}, j={j_geom} (spherical)")
print(f"RH BCs at xi=1:")
print(f"  V(1) = {V_bc:.6f}")
print(f"  C(1) = {C_bc:.6f}")
print(f"  G(1) = {G_bc:.6f}")

# Check Delta at shock
Delta_bc = (V_bc - 1.0)**2 - C_bc**2 / gamma
print(f"  Delta(1) = {Delta_bc:.6f}  (should be < 0, subsonic behind shock)")


def guderley_rhs(xi, y, alpha):
    """
    RHS for dy/dξ = f(ξ, y), y = [V, C]

    Equations (after eliminating G):
      Δ·V' = N_V/ξ   =>  V' = N_V/(ξ·Δ)
      C' = C·[-β·(V'+j·V/ξ) - (1/α-1)] / (V-ξ)

    where:
      Δ = (V-ξ)² - C²/γ
      N_V = C²·j·V/γ - V·(α-1)·(V-ξ)/α
    """
    V, C = y

    eps = 1e-14
    xi_s = max(abs(xi), eps)

    # Delta = (V-ξ)² - C²/γ
    Vm_xi = V - xi
    Delta = Vm_xi**2 - C**2 / gamma

    # Numerator for V equation
    # N_V = C²jV/γ - V(α-1)(V-ξ)/α
    N_V = C**2 * j_geom * V / gamma - V * (alpha - 1.0) * Vm_xi / alpha

    # Near the sonic line, use L'Hôpital's rule
    if abs(Delta) < 1e-10:
        # At the sonic point, Δ=0 and N_V should also = 0
        # Use limiting form: V' = (dN_V/dξ) / (ξ·dΔ/dξ)
        # For simplicity, return 0 (halt at singular point)
        return [0.0, 0.0]

    dV = N_V / (xi_s * Delta)

    # C equation
    if abs(Vm_xi) < 1e-10:
        return [dV, 0.0]

    dC = C * (-beta * (dV + j_geom * V / xi_s) - (1.0/alpha - 1.0)) / Vm_xi

    return [dV, dC]


def integrate_from_shock(alpha, xi_end=0.01, verbose=False):
    """
    Integrate from ξ=1 toward ξ=0 for given α.
    Returns solution object and diagnostic info.
    """
    xi_start = 0.9999  # slightly inside shock

    # Use RH boundary conditions at ξ=1
    V0, C0 = V_bc, C_bc

    # Event: detect when |Δ| crosses through zero
    def delta_zero(xi, y):
        V, C = y
        return (V - xi)**2 - C**2 / gamma
    delta_zero.terminal = True
    delta_zero.direction = 0  # detect any crossing

    # Event: detect non-physical values
    def c_negative(xi, y):
        return y[1] - 1e-6
    c_negative.terminal = True
    c_negative.direction = -1

    try:
        sol = solve_ivp(
            lambda xi, y: guderley_rhs(xi, y, alpha),
            [xi_start, xi_end],
            [V0, C0],
            method='RK45',
            max_step=0.0005,
            rtol=1e-10,
            atol=1e-12,
            events=[delta_zero, c_negative],
            dense_output=True
        )

        V_sol = sol.y[0]
        C_sol = sol.y[1]
        xi_sol = sol.t

        Delta_sol = (V_sol - xi_sol)**2 - C_sol**2 / gamma
        N_V_sol = (C_sol**2 * j_geom * V_sol / gamma
                   - V_sol * (alpha - 1.0) * (V_sol - xi_sol) / alpha)

        min_abs_Delta = np.min(np.abs(Delta_sol))

        if verbose:
            print(f"  Integration from {xi_start} to {xi_sol[-1]:.6f}")
            print(f"  V range: [{V_sol.min():.6f}, {V_sol.max():.6f}]")
            print(f"  C range: [{C_sol.min():.6f}, {C_sol.max():.6f}]")
            print(f"  min|Δ|: {min_abs_Delta:.6e}")
            if sol.t_events[0].size > 0:
                xi_sonic = sol.t_events[0][0]
                print(f"  Sonic line hit at ξ={xi_sonic:.6f}")

        return sol, min_abs_Delta

    except Exception as e:
        if verbose:
            print(f"  Integration failed: {e}")
        return None, float('inf')


def sonic_regularity_residual(alpha, verbose=False):
    """
    For given α, integrate from shock toward center.
    At the sonic point (Δ=0), compute N_V.

    For the correct eigenvalue α, N_V=0 when Δ=0.
    Returns N_V at the sonic point (should be 0 for correct α).
    """
    sol, min_delta = integrate_from_shock(alpha, verbose=verbose)

    if sol is None:
        return float('inf')

    V_sol = sol.y[0]
    C_sol = sol.y[1]
    xi_sol = sol.t

    Delta_sol = (V_sol - xi_sol)**2 - C_sol**2 / gamma
    N_V_sol = (C_sol**2 * j_geom * V_sol / gamma
               - V_sol * (alpha - 1.0) * (V_sol - xi_sol) / alpha)

    # Find point closest to Δ=0
    idx_sonic = np.argmin(np.abs(Delta_sol))

    # Return N_V at the sonic point (normalized)
    xi_s = xi_sol[idx_sonic]
    N_V_at_sonic = N_V_sol[idx_sonic]

    if verbose:
        print(f"  Sonic point at ξ≈{xi_s:.6f}")
        print(f"  Δ at sonic: {Delta_sol[idx_sonic]:.6e}")
        print(f"  N_V at sonic: {N_V_at_sonic:.6e}")

    return N_V_at_sonic


# ========== Scan α ==========
print(f"\n{'='*60}")
print(f"Scanning α for sonic line regularity")
print(f"{'='*60}")

# Fine scan
alphas = np.linspace(0.55, 0.95, 81)
residuals = []

for alpha in alphas:
    res = sonic_regularity_residual(alpha)
    residuals.append(res)

residuals = np.array(residuals)

# Print results
print(f"\n{'alpha':>8s}  {'N_V residual':>14s}")
print("-" * 30)
for i, alpha in enumerate(alphas):
    if abs(residuals[i]) < float('inf'):
        mark = ""
        if abs(alpha - 0.6884) < 0.003:
            mark = " <-- Lazarus"
        if abs(alpha - 0.717) < 0.003:
            mark = " <-- 0.717"
        if abs(residuals[i]) < 0.05:
            mark += " ***"
        print(f"{alpha:8.4f}  {residuals[i]:14.6e}{mark}")

# Find sign changes in N_V residual (root = eigenvalue)
finite_mask = np.isfinite(residuals)
alphas_f = alphas[finite_mask]
res_f = residuals[finite_mask]

sign_changes = []
for i in range(len(res_f) - 1):
    if res_f[i] * res_f[i+1] < 0:
        sign_changes.append((alphas_f[i], alphas_f[i+1]))
        print(f"\nSign change between α={alphas_f[i]:.4f} and α={alphas_f[i+1]:.4f}")

# Refine with bisection
if sign_changes:
    for a_lo, a_hi in sign_changes:
        try:
            alpha_star = brentq(sonic_regularity_residual, a_lo, a_hi,
                              xtol=1e-8, maxiter=100)
            print(f"\n*** EIGENVALUE FOUND: α = {alpha_star:.8f} ***")
            print(f"Detailed solution at α={alpha_star:.8f}:")
            sonic_regularity_residual(alpha_star, verbose=True)
        except Exception as e:
            print(f"Brentq failed: {e}")
else:
    print("\nNo sign changes found. Checking if method works...")
    # Print all residuals for debugging
    print(f"\nAll residuals:")
    for i in range(len(alphas_f)):
        print(f"  α={alphas_f[i]:.4f}: N_V={res_f[i]:+.6e}")


# ========== Also try with 3-variable system (V, C, G) directly ==========
print(f"\n{'='*60}")
print(f"Method 2: Direct 3-variable system (V, C, G)")
print(f"{'='*60}")

def guderley_3var_rhs(xi, y, alpha):
    """
    Full 3-variable system: y = [V, C, G]

    (I)   G' = G·[-(V' + j·V/ξ)/(V-ξ)]  -- but this is circular

    Better: use the original 3 equations directly without eliminating G:
    From (I): G' = -G·(V'+jV/ξ)/(V-ξ)
    But V' comes from (II) which needs G'.

    So solve the coupled system:
    From (I) and (II) simultaneously:
      (V-ξ)²·V' - (C²/γ)·(V'+jV/ξ) + (V-ξ)·V·(α-1)/(αξ) = 0
      Δ·V' = [C²jV/γ - V(α-1)(V-ξ)/α] / ξ = N_V/ξ

    This is the same 2-variable system. Let's try the log-derivative approach.
    """
    V, C, lnG = y  # use log(G) for numerical stability

    eps = 1e-14
    xi_s = max(abs(xi), eps)
    Vm_xi = V - xi

    Delta = Vm_xi**2 - C**2 / gamma
    N_V = C**2 * j_geom * V / gamma - V * (alpha - 1.0) * Vm_xi / alpha

    if abs(Delta) < 1e-10:
        return [0.0, 0.0, 0.0]

    dV = N_V / (xi_s * Delta)

    if abs(Vm_xi) < 1e-10:
        return [dV, 0.0, 0.0]

    # G'/G from continuity
    dlnG = -(dV + j_geom * V / xi_s) / Vm_xi

    # C' from entropy
    dC = C * (-beta * (dV + j_geom * V / xi_s) - (1.0/alpha - 1.0)) / Vm_xi

    return [dV, dC, dlnG]


def scan_3var(alpha, verbose=False):
    """Integrate the 3-variable system"""
    xi_start = 0.9999
    xi_end = 0.01

    V0, C0, lnG0 = V_bc, C_bc, np.log(G_bc)

    def delta_event(xi, y):
        V, C, _ = y
        return (V - xi)**2 - C**2 / gamma
    delta_event.terminal = True

    try:
        sol = solve_ivp(
            lambda xi, y: guderley_3var_rhs(xi, y, alpha),
            [xi_start, xi_end],
            [V0, C0, lnG0],
            method='RK45',
            max_step=0.0005,
            rtol=1e-10,
            atol=1e-12,
            events=[delta_event],
            dense_output=True
        )

        V_sol, C_sol = sol.y[0], sol.y[1]
        xi_sol = sol.t
        G_sol = np.exp(sol.y[2])

        Delta_sol = (V_sol - xi_sol)**2 - C_sol**2 / gamma
        N_V_sol = (C_sol**2 * j_geom * V_sol / gamma
                   - V_sol * (alpha - 1.0) * (V_sol - xi_sol) / alpha)

        idx = np.argmin(np.abs(Delta_sol))

        if verbose:
            print(f"  ξ range: [{xi_sol[-1]:.6f}, {xi_sol[0]:.6f}]")
            print(f"  V range: [{V_sol.min():.6f}, {V_sol.max():.6f}]")
            print(f"  C range: [{C_sol.min():.6f}, {C_sol.max():.6f}]")
            print(f"  G range: [{G_sol.min():.6f}, {G_sol.max():.6f}]")
            print(f"  Sonic at ξ≈{xi_sol[idx]:.6f}: Δ={Delta_sol[idx]:.6e}, N_V={N_V_sol[idx]:.6e}")

        return N_V_sol[idx], xi_sol[-1], Delta_sol[idx]

    except Exception as e:
        if verbose:
            print(f"  Failed: {e}")
        return float('inf'), 1.0, 0.0


# Fine scan around known values
alphas2 = np.linspace(0.55, 0.95, 81)
nv_residuals = []

for alpha in alphas2:
    nv, xi_end, delta = scan_3var(alpha)
    nv_residuals.append(nv)

nv_residuals = np.array(nv_residuals)

# Find sign changes
print(f"\n{'alpha':>8s}  {'N_V at sonic':>14s}")
print("-" * 30)
for i, alpha in enumerate(alphas2):
    if np.isfinite(nv_residuals[i]):
        mark = ""
        if abs(alpha - 0.6884) < 0.003: mark = " <-- Lazarus"
        if abs(alpha - 0.717) < 0.003: mark = " <-- 0.717"
        print(f"{alpha:8.4f}  {nv_residuals[i]:14.6e}{mark}")

# Check for sign changes
finite2 = np.isfinite(nv_residuals)
a2 = alphas2[finite2]
r2 = nv_residuals[finite2]

for i in range(len(r2) - 1):
    if r2[i] * r2[i+1] < 0:
        print(f"\nSign change: α ∈ [{a2[i]:.4f}, {a2[i+1]:.4f}]")
        try:
            def residual_for_brentq(alpha):
                nv, _, _ = scan_3var(alpha)
                return nv
            alpha_star = brentq(residual_for_brentq, a2[i], a2[i+1], xtol=1e-8)
            print(f"*** EIGENVALUE: α = {alpha_star:.8f} ***")
            print(f"\nDetailed solution:")
            scan_3var(alpha_star, verbose=True)
        except Exception as e:
            print(f"Refinement failed: {e}")


# ========== Plot ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. N_V residual vs α
ax = axes[0, 0]
mask1 = np.isfinite(residuals)
ax.plot(alphas[mask1], residuals[mask1], 'b.-', markersize=4)
ax.axhline(y=0, color='k', ls='-', lw=0.5)
ax.axvline(x=0.6884, color='r', ls='--', label='alpha=0.6884 (Lazarus)')
ax.axvline(x=0.717, color='g', ls='--', label='alpha=0.717')
ax.set_xlabel('alpha')
ax.set_ylabel('N_V at sonic')
ax.set_title('Sonic regularity residual (2-var)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. 3-var residuals
ax = axes[0, 1]
mask2 = np.isfinite(nv_residuals)
ax.plot(alphas2[mask2], nv_residuals[mask2], 'r.-', markersize=4)
ax.axhline(y=0, color='k', ls='-', lw=0.5)
ax.axvline(x=0.6884, color='r', ls='--', label='alpha=0.6884')
ax.axvline(x=0.717, color='g', ls='--', label='alpha=0.717')
ax.set_xlabel('alpha')
ax.set_ylabel('N_V at sonic')
ax.set_title('Sonic regularity residual (3-var)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Solution profiles at best alpha
# Try a few candidates
ax = axes[1, 0]
for alpha_test in [0.65, 0.6884, 0.717, 0.75]:
    sol, _ = integrate_from_shock(alpha_test)
    if sol is not None:
        ax.plot(sol.t, sol.y[0], label=f'V, alpha={alpha_test}')
ax.set_xlabel('xi')
ax.set_ylabel('V')
ax.set_title('V(xi) for different alpha')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4. Delta profiles
ax = axes[1, 1]
for alpha_test in [0.65, 0.6884, 0.717, 0.75]:
    sol, _ = integrate_from_shock(alpha_test)
    if sol is not None:
        Delta = (sol.y[0] - sol.t)**2 - sol.y[1]**2 / gamma
        ax.plot(sol.t, Delta, label=f'alpha={alpha_test}')
ax.axhline(y=0, color='k', ls='-', lw=0.5)
ax.set_xlabel('xi')
ax.set_ylabel('Delta')
ax.set_title('Delta(xi) = (V-xi)^2 - C^2/gamma')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outpath = '/share/project/zpy/PINN_WE/cases/self-discovery/numerical_sanity_check_v2.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {outpath}")
