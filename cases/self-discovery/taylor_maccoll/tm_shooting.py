#!/usr/bin/env python3
"""
Taylor-Maccoll 打靶法基线

给定 (M∞, γ, θ_c), 求激波角 β。
方法: 猜β → 斜激波关系求post-shock → 积分TM ODE → 检查v_θ(θ_c)=0 → brentq。
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import time
import json
from pathlib import Path

from tm_common import (oblique_shock_post, taylor_maccoll_rhs,
                        v1_over_vmax, TEST_CASES)


def shooting_residual(beta_deg, M_inf, gamma, theta_c_deg):
    """
    从θ=β积分到θ=θ_c, 返回 v_θ(θ_c)。
    正确的β使得 v_θ(θ_c) = 0。
    """
    beta_rad = np.radians(beta_deg)
    theta_c_rad = np.radians(theta_c_deg)

    # Mach角检查 (β必须大于Mach角)
    mu = np.arcsin(1.0 / M_inf)
    if beta_rad <= mu or beta_rad >= np.pi / 2:
        return 1e10

    # 激波后条件
    vr_s, vt_s = oblique_shock_post(M_inf, beta_rad, gamma)

    # 检查物理性
    v2 = vr_s**2 + vt_s**2
    if v2 >= 1.0:
        return 1e10  # 超过V_max

    # 从β积分到θ_c (θ递减)
    try:
        sol = solve_ivp(
            lambda theta, y: taylor_maccoll_rhs(theta, y, gamma),
            [beta_rad, theta_c_rad],  # θ从β递减到θ_c
            [vr_s, vt_s],
            method='DOP853',
            rtol=1e-10, atol=1e-12,
            max_step=0.01,
        )
        if sol.success:
            return sol.y[1][-1]  # v_θ at θ_c, 应该=0
    except Exception:
        pass
    return 1e10


def find_beta_shooting(M_inf, gamma, theta_c_deg, verbose=True):
    """用brentq二分法找β"""
    t0 = time.time()
    n_eval = 0

    def f(beta_deg):
        nonlocal n_eval
        n_eval += 1
        return shooting_residual(beta_deg, M_inf, gamma, theta_c_deg)

    # Bracket: β ∈ (Mach角, 90°)
    mu_deg = np.degrees(np.arcsin(1.0 / M_inf))

    # 对于attached shock, β在Mach角和某个上限之间
    # 扫描找到零点bracket
    betas = np.linspace(mu_deg + 0.5, 89.0, 300)
    residuals = []
    for b in betas:
        r = f(b)
        residuals.append(r)
    residuals = np.array(residuals)

    # 找最接近零的bracket
    bracket_found = False
    for i in range(len(residuals) - 1):
        if residuals[i] * residuals[i + 1] < 0:
            beta_lo = betas[i]
            beta_hi = betas[i + 1]
            bracket_found = True
            break

    if not bracket_found:
        elapsed = time.time() - t0
        if verbose:
            print(f"    No bracket found! min|residual| = {np.min(np.abs(residuals)):.3e}")
        return {
            "beta_deg": float('nan'),
            "converged": False,
            "n_eval": n_eval,
            "time_s": elapsed,
        }

    if verbose:
        print(f"    bracket: [{beta_lo:.2f}, {beta_hi:.2f}] deg")

    # Brent法精细化
    try:
        beta_opt = brentq(f, beta_lo, beta_hi, xtol=1e-12, rtol=1e-12, maxiter=200)
        mis = f(beta_opt)
        elapsed = time.time() - t0
        return {
            "beta_deg": float(beta_opt),
            "mismatch": float(mis),
            "n_eval": n_eval,
            "time_s": elapsed,
            "converged": True,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "beta_deg": float(beta_lo),
            "converged": False,
            "n_eval": n_eval,
            "time_s": elapsed,
            "error": str(e),
        }


def get_full_solution(M_inf, gamma, theta_c_deg, beta_deg):
    """给定精确β，积分得到完整流场"""
    beta_rad = np.radians(beta_deg)
    theta_c_rad = np.radians(theta_c_deg)

    vr_s, vt_s = oblique_shock_post(M_inf, beta_rad, gamma)

    sol = solve_ivp(
        lambda theta, y: taylor_maccoll_rhs(theta, y, gamma),
        [beta_rad, theta_c_rad],
        [vr_s, vt_s],
        method='DOP853',
        rtol=1e-12, atol=1e-14,
        dense_output=True,
    )

    thetas = np.linspace(beta_rad, theta_c_rad, 200)
    y = sol.sol(thetas)

    return {
        "theta_deg": np.degrees(thetas),
        "v_r": y[0],
        "v_theta": y[1],
    }


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Taylor-Maccoll 打靶法基线")
    print("=" * 70)

    all_results = {}
    for tc in TEST_CASES:
        M, gamma, tc_deg, label = tc["M_inf"], tc["gamma"], tc["theta_c_deg"], tc["label"]
        print(f"\n--- {label}: M={M}, γ={gamma}, θ_c={tc_deg}° ---")

        result = find_beta_shooting(M, gamma, tc_deg)
        result["M_inf"] = M
        result["gamma"] = gamma
        result["theta_c_deg"] = tc_deg

        if result["converged"]:
            print(f"    β = {result['beta_deg']:.10f}°")
            print(f"    mismatch = {result.get('mismatch', 'N/A')}")
            print(f"    evals = {result['n_eval']}, time = {result['time_s']:.3f}s")
        else:
            print(f"    FAILED: {result}")

        all_results[label] = result

    # 汇总表
    print(f"\n{'='*70}")
    print("  SHOOTING METHOD SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Label':<20} {'M∞':>4} {'γ':>6} {'θ_c':>6} {'β (deg)':<16} {'conv':>5} {'time':>8}")
    print(f"  {'-'*65}")
    for label, r in all_results.items():
        conv = "YES" if r["converged"] else "NO"
        beta_s = f"{r['beta_deg']:.10f}" if r["converged"] else "N/A"
        print(f"  {label:<20} {r['M_inf']:>4.1f} {r['gamma']:>6.3f} {r['theta_c_deg']:>6.1f} "
              f"{beta_s:<16} {conv:>5} {r['time_s']:>8.3f}")

    with open(out_dir / "tm_shooting.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'tm_shooting.json'}")
