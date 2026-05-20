#!/usr/bin/env python3
"""Legacy shooting search for the old self-discovery branch.

Prefer `cases/self-discovery/canonical/nonpinn_fit.py` for the current mainline.

Rigorous shooting search for Guderley self-similar exponent alpha.

Uses scipy.integrate.solve_ivp and scipy.optimize.minimize_scalar to find the
alpha that minimizes a regularity residual near the sonic line.

Run inside the PINN_WE conda env:
  source /share/project/zhaohuxing/anaconda3/bin/activate PINN_WE
  python cases/self-discovery/rigorous_shooting.py
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def shock_bc_strong(gamma: float) -> Tuple[float, float]:
    V1 = 2.0 / (gamma + 1.0)
    C1 = math.sqrt(2.0 * gamma * (gamma - 1.0)) / (gamma + 1.0)
    return V1, C1


def numerators(alpha: float, gamma: float, m: int, V: float, C: float) -> Tuple[float, float]:
    vm1 = V - 1.0
    Nv = (vm1 * (V * vm1 * (V - 1.0 / alpha) - 2.0 * C * C / gamma)
          - m * C * C * vm1 / (gamma * alpha))
    Nc = (C * (vm1 * (V - 1.0 / alpha) - C * C / gamma)
          - m * C / (gamma * alpha))
    return Nv, Nc


def rhs(xi, y, alpha, gamma, m):
    V, C = y
    delta = (V - 1.0) ** 2 - C ** 2
    Nv, Nc = numerators(alpha, gamma, m, V, C)
    if abs(delta) < 1e-12:
        sign = 1.0 if delta >= 0.0 else -1.0
        delta = sign * 1e-12
    dV = Nv / (xi * delta)
    dC = Nc / (xi * delta)
    return [dV, dC]


def _delta(xi: float, y: np.ndarray) -> float:
    V, C = y
    return (V - 1.0) ** 2 - C ** 2


def _nonphysical_c(xi: float, y: np.ndarray) -> float:
    return y[1]


def integrate_to_sonic(alpha: float, gamma: float = 1.4, m: int = 2,
                       xi_span=(1.0, 0.05), start_eps: float = 1e-6):
    V1, C1 = shock_bc_strong(gamma)
    xi0 = 1.0 - start_eps
    try:
        dV0, dC0 = rhs(1.0, [V1, C1], alpha, gamma, m)
    except Exception:
        dV0, dC0 = 0.0, 0.0
    V0 = V1 - dV0 * start_eps
    C0 = C1 - dC0 * start_eps

    def fun_xi(xi, y):
        return rhs(xi, y, alpha, gamma, m)

    def sonic_event(xi, y):
        return _delta(xi, y)

    def c_event(xi, y):
        return _nonphysical_c(xi, y)

    sonic_event.terminal = True
    sonic_event.direction = 0
    c_event.terminal = True
    c_event.direction = -1

    sol = solve_ivp(
        fun_xi,
        xi_span,
        [V0, C0],
        method='Radau',
        rtol=1e-8,
        atol=1e-10,
        max_step=1e-3,
        events=[sonic_event, c_event],
    )

    return sol


def trajectory_residual(alpha: float, gamma: float = 1.4, m: int = 2,
                        xi_span=(1.0, 0.05)) -> float:
    """Integrate inward and use sonic event residual as the eigenvalue score."""
    sol = integrate_to_sonic(alpha=alpha, gamma=gamma, m=m, xi_span=xi_span)

    if not sol.success:
        return 1e6

    sonic_hits = sol.t_events[0]
    c_hits = sol.t_events[1]

    if len(sonic_hits) > 0:
        V_s, C_s = sol.y_events[0][0]
        Nv_s, Nc_s = numerators(alpha, gamma, m, float(V_s), float(C_s))
        return float(abs(Nv_s) + abs(Nc_s))

    if len(c_hits) > 0:
        return 1e5

    V = sol.y[0]
    C = sol.y[1]
    delta = (V - 1.0) ** 2 - C ** 2
    Nv, Nc = numerators(alpha, gamma, m, V, C)
    idx = int(np.argmin(np.abs(delta)))
    return float(abs(Nv[idx]) + abs(Nc[idx]) + 100.0 * abs(delta[idx]) + 1.0)


def find_alpha_bruteforce(grid, **kwargs):
    scores = []
    for a in grid:
        scores.append((a, trajectory_residual(a, **kwargs)))
    scores.sort(key=lambda x: x[1])
    return scores


def find_alpha_opt(alpha_bounds=(0.6, 1.0), **kwargs):
    f = lambda a: trajectory_residual(a, **kwargs)
    res = minimize_scalar(f, bounds=alpha_bounds, method='bounded', options={'xatol': 1e-4})
    return res


def main():
    gamma = 1.4
    # try m=2 (spherical in many conventions) and m=3 (some files used 3)
    out = {}
    for m in (2, 3):
        print(f"Running brute force scan for m={m} ...")
        grid = np.linspace(0.65, 0.9, 26)
        scores = find_alpha_bruteforce(grid, gamma=gamma, m=m)
        best_grid = scores[0]
        print(f"Best on grid for m={m}: alpha={best_grid[0]:.6f}, residual={best_grid[1]:.6e}")
        # refine with bounded optimization
        res = find_alpha_opt(alpha_bounds=(0.65, 0.9), gamma=gamma, m=m)
        print(f"Minimizer for m={m}: alpha={res.x:.6f}, residual={res.fun:.6e}")
        local_grid = [item for item in scores[:8]]
        out[f'm={m}'] = {
            'grid_best': {'alpha': float(best_grid[0]), 'residual': float(best_grid[1])},
            'opt_best': {'alpha': float(res.x), 'residual': float(res.fun)},
            'top_grid': [
                {'alpha': float(alpha), 'residual': float(score)}
                for alpha, score in local_grid
            ],
        }

    p = Path(__file__).resolve().parent / 'output_quickcheck' / 'rigorous_shooting.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print('Saved results to', p)


if __name__ == '__main__':
    main()
