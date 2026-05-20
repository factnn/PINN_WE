#!/usr/bin/env python3
"""
双参数反演: α 和 γ 同时未知

基于123框架 (Chisnell + 混合残差 + 双端硬约束), 将γ从常数改为可学习参数。
这是打靶法 (brentq 1D root-finder) 无法直接处理的场景。

关键insight: 纯ODE约束下(α,γ)存在简并——多组(α,γ)都能让ODE残差≈0。
必须加入观测数据约束来打破简并, 这正好演示了PINN"天然加loss项"的优势。

实验设计:
  1. 用打靶法生成ground-truth C(V)曲线作为"合成观测数据"
  2. PINN同时最小化 L_ode + λ_data * L_data
  3. 对比: 仅ODE (简并, 失败) vs ODE+数据 (成功)

用法:
  python exp_dual_param.py --device cuda
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from common import (build_mlp, inv_sigmoid, chisnell_residual,
                    get_ref_alpha, GEOMETRY_N)
import json


# 打靶法高精度参考值
REF_ALPHA = {
    (1.4, 3): 0.717174501487,
    (1.4, 2): 0.835323191951,
    (5/3, 3): 0.688376822922,
    (5/3, 2): 0.815624901431,
}


def generate_observation_data(gamma, n, n_obs=20, noise_std=0.0):
    """
    用打靶法生成合成观测数据: C(V) 在若干V点的值。
    可选加高斯噪声模拟真实观测。
    """
    alpha = REF_ALPHA[(gamma, n)]

    # Chisnell ODE RHS
    def rhs(V, C_vec):
        C = C_vec[0]
        delta = (V - alpha) ** 2 - C
        Q = (n * V * (V - alpha)
             + (2.0 / gamma) * (1.0 - alpha) * (alpha - V)
             - V * (V - 1.0))
        numer = C * (2.0 * delta * (alpha - V + (1.0 - alpha) / gamma)
                     + (gamma - 1.0) * (alpha - V) * Q)
        denom = (delta * (n * V - 2.0 * (1.0 - alpha) / gamma) * (alpha - V)
                 + (alpha - V) ** 2 * Q)
        return [numer / (denom + 1e-30)]

    # Shock conditions (Convention A)
    Vs = 2.0 * alpha / (gamma + 1.0)
    Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2

    # Critical point
    from common import critical_point_np
    V0, C0 = critical_point_np(alpha, gamma, n)

    # Integrate
    sol = solve_ivp(rhs, [Vs, V0], [Cs], method='DOP853',
                    rtol=1e-12, atol=1e-14, dense_output=True)

    # Sample observation points (avoid very close to endpoints)
    V_obs = np.linspace(Vs + 0.05 * (V0 - Vs),
                        V0 - 0.05 * (V0 - Vs), n_obs)
    C_obs = sol.sol(V_obs)[0]

    if noise_std > 0:
        C_obs += np.random.normal(0, noise_std, size=C_obs.shape)

    return V_obs, C_obs


def critical_point_torch_differentiable(alpha, gamma, n):
    """critical_point_torch with differentiable gamma (gamma < gamma_crit only)."""
    V0_denom = 2.0 * gamma * (n - 1.0)
    factor = gamma * n - 2.0
    disc = (8.0 * (alpha - 1.0) * alpha * gamma * (n - 1.0)
            + (2.0 - gamma + alpha * factor) ** 2)
    disc = torch.clamp(disc, min=1e-14)
    V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
          + torch.sqrt(disc)) / V0_denom
    C0 = (V0 - alpha) ** 2
    return V0, C0


class DualParamPINN(nn.Module):
    """α和γ同时可学习的PINN"""

    def __init__(self, n, alpha_init=0.75, gamma_init=1.5,
                 n_layers=5, n_neurons=64):
        super().__init__()
        self.n = n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5),
                         dtype=torch.float64))
        self.raw_gamma = nn.Parameter(
            torch.tensor(inv_sigmoid((gamma_init - 1.05) / 1.45),
                         dtype=torch.float64))
        self.net = build_mlp(1, 1, n_layers, n_neurons).double()

    @property
    def alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    @property
    def gamma(self):
        return 1.05 + 1.45 * torch.sigmoid(self.raw_gamma)

    def shock_cond(self):
        alpha, gamma = self.alpha, self.gamma
        Vs = 2.0 * alpha / (gamma + 1.0)
        Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
        return Vs, Cs

    def sonic_cond(self):
        return critical_point_torch_differentiable(
            self.alpha, self.gamma, self.n)

    def forward(self, V):
        Vs, Cs = self.shock_cond()
        V0, C0 = self.sonic_cond()
        s = (V - Vs) / (V0 - Vs + 1e-14)
        return (1.0 - s) * Cs + s * C0 + s * (1.0 - s) * self.net(s)


def train_dual_param(n, geometry, alpha_init, gamma_init, gamma_ref,
                     V_obs, C_obs, lambda_data=10.0,
                     epochs=15000, lr=1e-3, n_colloc=400, device="cuda"):
    """训练双参数PINN: L_ode + λ_data * L_data"""
    ref_alpha = REF_ALPHA[(gamma_ref, n)]
    model = DualParamPINN(n, alpha_init, gamma_init).double().to(device)

    # 观测数据 → tensor
    V_obs_t = torch.tensor(V_obs, dtype=torch.float64, device=device).reshape(-1, 1)
    C_obs_t = torch.tensor(C_obs, dtype=torch.float64, device=device).reshape(-1, 1)

    history_alpha, history_gamma, history_loss = [], [], []

    def compute_loss():
        alpha = model.alpha
        gamma = model.gamma
        Vs, Cs = model.shock_cond()
        V0, C0 = model.sonic_cond()

        # --- ODE residual (混合残差) ---
        Vs_d, V0_d = Vs.detach(), V0.detach()
        V_colloc = (Vs_d + (V0_d - Vs_d) *
                    torch.linspace(0.01, 0.99, n_colloc,
                                   dtype=torch.float64, device=device)
                    ).reshape(-1, 1)
        V_colloc.requires_grad_(True)

        C_pred = model(V_colloc)
        dC_dV = torch.autograd.grad(
            C_pred, V_colloc, grad_outputs=torch.ones_like(C_pred),
            create_graph=True)[0]

        residual, numer, denom, delta = chisnell_residual(
            V_colloc, C_pred, dC_dV, alpha, gamma, n)

        abs_denom = torch.abs(denom)
        w_far = torch.clamp(abs_denom / 1e-6 - 1.0, min=0.0, max=1.0)
        R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        loss_far = torch.mean(w_far * R_divided ** 2)
        w_sonic = 1.0 - w_far
        loss_sonic = torch.mean(w_sonic * residual ** 2)
        loss_numer = torch.mean(w_sonic * numer ** 2)

        loss_ode = loss_far + loss_sonic + 10.0 * loss_numer

        # --- 观测数据约束 (打破(α,γ)简并的关键!) ---
        C_pred_obs = model(V_obs_t)
        loss_data = torch.mean((C_pred_obs - C_obs_t) ** 2)

        return loss_ode, loss_data

    optimizer = optim.Adam([
        {"params": model.net.parameters(), "lr": lr},
        {"params": [model.raw_alpha], "lr": lr * 0.2},
        {"params": [model.raw_gamma], "lr": lr * 0.2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        loss_ode, loss_data = compute_loss()
        loss = loss_ode + lambda_data * loss_data
        if torch.isnan(loss) or torch.isinf(loss):
            history_alpha.append(float(model.alpha.detach()))
            history_gamma.append(float(model.gamma.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        a = float(model.alpha.detach())
        g = float(model.gamma.detach())
        l = float(loss.detach())
        history_alpha.append(a)
        history_gamma.append(g)
        history_loss.append(l)

        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] ep={ep:5d}  loss_ode={float(loss_ode):.3e}  "
                  f"loss_data={float(loss_data):.3e}  "
                  f"α={a:.6f}(ref={ref_alpha:.6f})  "
                  f"γ={g:.6f}(ref={gamma_ref:.4f})")

    alpha_fit = float(model.alpha.detach())
    gamma_fit = float(model.gamma.detach())
    alpha_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100
    gamma_err = abs(gamma_fit - gamma_ref) / gamma_ref * 100
    print(f"  [{geometry}] DONE  α={alpha_fit:.8f} err={alpha_err:.6f}%  "
          f"γ={gamma_fit:.8f} err={gamma_err:.6f}%")

    return {
        "alpha_fit": alpha_fit, "ref_alpha": ref_alpha,
        "alpha_err_pct": alpha_err,
        "gamma_fit": gamma_fit, "ref_gamma": gamma_ref,
        "gamma_err_pct": gamma_err,
        "final_loss": history_loss[-1],
        "n_obs": len(V_obs),
        "alpha_history": np.array(history_alpha),
        "gamma_history": np.array(history_gamma),
        "loss_history": np.array(history_loss),
    }


# 实验配置: (ref_gamma, n, geometry, alpha_init, gamma_init)
DUAL_CONFIGS = [
    (1.4,   3, "spherical",   0.72, 1.50),
    (1.4,   2, "cylindrical", 0.84, 1.50),
    (5/3,   3, "spherical",   0.69, 1.50),
    (5/3,   2, "cylindrical", 0.82, 1.50),
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--n-obs", type=int, default=20,
                        help="Number of observation points")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Observation noise std (0=clean)")
    parser.add_argument("--lambda-data", type=float, default=10.0,
                        help="Weight for data loss")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  双参数反演: α + γ 同时未知 (123框架 + 观测数据)")
    print(f"  观测点数: {args.n_obs}, 噪声std: {args.noise}, λ_data: {args.lambda_data}")
    print("  打靶法 (brentq) 无法直接处理此问题")
    print("=" * 70)

    results = {}
    for gamma_ref, n, geo, a_init, g_init in DUAL_CONFIGS:
        key = f"g{gamma_ref:.2f}_{geo}"
        print(f"\n{'='*60}")
        print(f"  {key}: α_init={a_init}, γ_init={g_init}")
        print(f"  ref: α={REF_ALPHA[(gamma_ref, n)]}, γ={gamma_ref}")

        # 生成观测数据
        V_obs, C_obs = generate_observation_data(
            gamma_ref, n, n_obs=args.n_obs, noise_std=args.noise)
        print(f"  观测数据: {len(V_obs)}点, V∈[{V_obs[0]:.4f}, {V_obs[-1]:.4f}]")
        print(f"{'='*60}")

        results[key] = train_dual_param(
            n, geo, a_init, g_init, gamma_ref,
            V_obs, C_obs, lambda_data=args.lambda_data,
            epochs=args.epochs, device=args.device)

    # Summary
    print(f"\n{'='*70}")
    print("  DUAL PARAMETER INVERSION SUMMARY (ODE + data)")
    print(f"{'='*70}")
    print(f"  {'Config':<22} {'α_fit':<14} {'α_ref':<14} {'α_err%':<12} "
          f"{'γ_fit':<10} {'γ_ref':<8} {'γ_err%':<10} {'loss':<10}")
    print(f"  {'-'*95}")
    for key, r in results.items():
        print(f"  {key:<22} {r['alpha_fit']:<14.10f} {r['ref_alpha']:<14.10f} "
              f"{r['alpha_err_pct']:<12.6f} {r['gamma_fit']:<10.6f} "
              f"{r['ref_gamma']:<8.4f} {r['gamma_err_pct']:<10.6f} "
              f"{r['final_loss']:<10.3e}")

    # Save (convert numpy arrays for JSON)
    results_save = {}
    for k, v in results.items():
        results_save[k] = {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                           for kk, vv in v.items()}
    with open(out_dir / "exp_dual_param.json", "w") as f:
        json.dump({"experiment": "Dual parameter inversion (alpha + gamma)",
                   "n_obs": args.n_obs, "noise_std": args.noise,
                   "lambda_data": args.lambda_data,
                   "configs": results_save}, f, indent=2)
    print(f"\nSaved: {out_dir / 'exp_dual_param.json'}")
