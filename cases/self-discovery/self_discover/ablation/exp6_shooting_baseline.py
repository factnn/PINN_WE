#!/usr/bin/env python3
"""
Baseline: 传统打靶法 (Shooting Method)

使用Chisnell (1998)相平面ODE + brentq二分法求解Guderley alpha。
ODE积分方向 Vs→V0 不穿过V=alpha奇点。

这是经典的数值方法baseline, 对比PINN方法。
打靶法可以达到机器精度(~1e-13), 但:
  - 需要知道正确的bracket区间
  - 需要仔细处理奇点
  - 每次评估需要完整的ODE积分
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import time
import json
from common import *


def chisnell_rhs(V, C_vec, alpha, gamma, n):
    """dC/dV = numer / denom (Chisnell Eq. 3.1)"""
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


def shooting_residual(alpha, gamma, n):
    """
    从shock端积分到sonic端, 返回 C0_algebraic - C_integrated(V0).
    正确alpha使得残差=0。
    """
    V0_C0 = critical_point_np(alpha, gamma, n)
    if V0_C0 is None or V0_C0[0] is None:
        return 1e10
    V0, C0 = V0_C0

    # Shock conditions (Chisnell convention)
    Vs = 2.0 * alpha / (gamma + 1.0)
    Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2

    if abs(V0 - Vs) < 1e-14:
        return C0 - Cs

    try:
        sol = solve_ivp(
            lambda V, C: chisnell_rhs(V, C, alpha, gamma, n),
            [Vs, V0], [Cs],
            method='DOP853', rtol=1e-10, atol=1e-12,
        )
        if sol.success:
            return C0 - sol.y[0][-1]
    except Exception:
        pass
    return 1e10


def find_alpha_shooting(gamma, n, verbose=True):
    """用brentq二分法找alpha"""
    t0 = time.time()
    n_eval = 0

    def f(alpha):
        nonlocal n_eval
        n_eval += 1
        return shooting_residual(alpha, gamma, n)

    # Bracket from ExactPack (Ramsey et al.)
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
        print(f"    bracket: [{amin:.6f}, {amax:.6f}]")

    # 验证bracket
    fa = f(amin)
    fb = f(amax)

    if fa * fb > 0:
        # 扫描找bracket
        if verbose:
            print("    scanning for bracket...")
        alphas = np.linspace(amin, amax, 200)
        residuals = [f(a) for a in alphas]
        for i in range(len(residuals) - 1):
            if residuals[i] * residuals[i + 1] < 0:
                amin = alphas[i]
                amax = alphas[i + 1]
                fa = residuals[i]
                fb = residuals[i + 1]
                break

    if fa * fb > 0:
        elapsed = time.time() - t0
        return {
            "alpha": float(amin),
            "mismatch": float(fa),
            "n_eval": n_eval,
            "time_s": elapsed,
            "converged": False,
        }

    # Brent法精细化
    try:
        alpha_opt = brentq(f, amin, amax, xtol=1e-12, rtol=1e-12, maxiter=200)
        mis = f(alpha_opt)
        elapsed = time.time() - t0
        return {
            "alpha": float(alpha_opt),
            "mismatch": float(mis),
            "n_eval": n_eval,
            "time_s": elapsed,
            "converged": True,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "alpha": float(amin),
            "mismatch": float(fa),
            "n_eval": n_eval,
            "time_s": elapsed,
            "converged": False,
            "error": str(e),
        }


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Baseline: Traditional Shooting Method (Chisnell phase-plane)")
    print("=" * 70)

    all_results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        ref = get_ref_alpha(gamma, n)

        print(f"\n  --- gamma={gamma}, {geo} (ref={ref}) ---")
        result = find_alpha_shooting(gamma, n)
        result["ref_alpha"] = ref
        if ref:
            result["rel_err_pct"] = abs(result["alpha"] - ref) / ref * 100
        all_results[key] = result

        converged = "YES" if result["converged"] else "NO"
        err_s = f"{result.get('rel_err_pct', float('nan')):.8f}%"
        print(f"  alpha = {result['alpha']:.12f}")
        print(f"  ref   = {ref:.12f}")
        print(f"  err   = {err_s}")
        print(f"  evals = {result['n_eval']}")
        print(f"  time  = {result['time_s']:.3f}s")
        print(f"  converged = {converged}")

    print(f"\n{'='*70}")
    print("  SHOOTING METHOD SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'alpha':<18} {'ref':<18} {'err%':<14} {'conv':<6} {'time':<8}")
    print(f"  {'-'*85}")
    for key, r in all_results.items():
        err_s = f"{r.get('rel_err_pct', float('nan')):.8f}"
        conv_s = "YES" if r["converged"] else "NO"
        print(f"  {key:<25} {r['alpha']:<18.12f} {r['ref_alpha']:<18.12f} {err_s:<14} {conv_s:<6} {r['time_s']:<8.3f}")

    with open(out_dir / "exp6_shooting_baseline.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'exp6_shooting_baseline.json'}")
