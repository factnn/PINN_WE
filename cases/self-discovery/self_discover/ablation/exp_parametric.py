#!/usr/bin/env python3
"""
参数化PINN: 一次训练发现整条 α(γ) 曲线

打靶法每次只能解一个 (α, γ) 对。本实验演示PINN可以:
  - 用一个网络学 C(V; γ)，覆盖 γ∈[1.2, 1.8]
  - 用辅助网络 α_net(γ) 输出对应的 eigenvalue
  - 仅靠 ODE 残差训练，不需要任何观测数据 (纯自发现)

训练后, α_net(γ) 给出任意 γ 的 Guderley eigenvalue。

用法:
  python exp_parametric.py --device cuda --geometry spherical
  python exp_parametric.py --device cuda --geometry cylindrical
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common import build_mlp, inv_sigmoid, chisnell_residual, GEOMETRY_N


# 打靶法高精度参考值
REF_ALPHA = {
    (1.4, 3): 0.717174501487,
    (1.4, 2): 0.835323191951,
    (5/3, 3): 0.688376822922,
    (5/3, 2): 0.815624901431,
}


def critical_point_torch_diff(alpha, gamma, n):
    """Differentiable critical point (gamma < gamma_crit branch only)."""
    V0_denom = 2.0 * gamma * (n - 1.0)
    factor = gamma * n - 2.0
    disc = (8.0 * (alpha - 1.0) * alpha * gamma * (n - 1.0)
            + (2.0 - gamma + alpha * factor) ** 2)
    disc = torch.clamp(disc, min=1e-14)
    V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
          + torch.sqrt(disc)) / V0_denom
    C0 = (V0 - alpha) ** 2
    return V0, C0


class AlphaNet(nn.Module):
    """小辅助网络: γ → α(γ)"""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        ).double()

    def forward(self, gamma):
        """gamma: scalar tensor → alpha ∈ (0.5, 1.0)"""
        # 归一化输入: γ∈[1.1, 2.0] → [-1, 1]
        g_norm = (gamma - 1.55) / 0.45
        raw = self.net(g_norm.reshape(-1, 1))
        return 0.5 + 0.5 * torch.sigmoid(raw.squeeze(-1))


class ParametricPINN(nn.Module):
    """
    参数化PINN: C(s, γ) with hard constraint
    s = normalized coordinate [0,1], γ = heat ratio
    """
    def __init__(self, n, n_layers=5, n_neurons=64):
        super().__init__()
        self.n = n
        self.alpha_net = AlphaNet(hidden=32)
        # 主网络: (s, γ_norm) → correction
        self.net = build_mlp(2, 1, n_layers, n_neurons).double()

    def get_alpha(self, gamma):
        return self.alpha_net(gamma)

    def shock_cond(self, alpha, gamma):
        Vs = 2.0 * alpha / (gamma + 1.0)
        Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
        return Vs, Cs

    def sonic_cond(self, alpha, gamma):
        return critical_point_torch_diff(alpha, gamma, self.n)

    def forward(self, s, gamma, alpha=None):
        """
        s: (N, 1) normalized coord
        gamma: scalar tensor
        """
        if alpha is None:
            alpha = self.get_alpha(gamma)
        Vs, Cs = self.shock_cond(alpha, gamma)
        V0, C0 = self.sonic_cond(alpha, gamma)

        # γ归一化作为网络输入
        g_norm = (gamma - 1.55) / 0.45
        g_input = g_norm.expand(s.shape[0], 1)
        net_input = torch.cat([s, g_input], dim=1)
        correction = self.net(net_input)

        C = (1.0 - s) * Cs + s * C0 + s * (1.0 - s) * correction
        return C, Vs, V0, alpha


def train_parametric(n, geometry, gamma_range=(1.2, 1.8), n_gamma=16,
                     epochs=20000, lr=1e-3, n_colloc=200, device="cuda"):
    """
    训练参数化PINN, 覆盖 γ∈gamma_range
    每个epoch随机采样n_gamma个γ值, 对每个γ计算ODE残差
    """
    model = ParametricPINN(n).double().to(device)

    optimizer = optim.Adam([
        {"params": model.net.parameters(), "lr": lr},
        {"params": model.alpha_net.parameters(), "lr": lr * 0.5},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    gamma_lo, gamma_hi = gamma_range
    history_loss = []

    # 固定γ采样点 (含参考值), 避免每epoch重采样
    gammas_fixed = torch.linspace(gamma_lo, gamma_hi, n_gamma,
                                   dtype=torch.float64, device=device)
    # 替换最近的点为精确参考值
    gammas_fixed[2] = 1.4
    gammas_fixed[-4] = 5.0 / 3.0
    s_colloc_base = torch.linspace(0.01, 0.99, n_colloc,
                                    dtype=torch.float64, device=device)

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)

        for i in range(n_gamma):
            gamma = gammas_fixed[i]
            alpha = model.get_alpha(gamma)
            Vs, Cs = model.shock_cond(alpha, gamma)
            V0, C0 = model.sonic_cond(alpha, gamma)

            Vs_d, V0_d = Vs.detach(), V0.detach()
            s_colloc = s_colloc_base.reshape(-1, 1).clone()
            s_colloc.requires_grad_(True)

            C_pred, _, _, _ = model(s_colloc, gamma, alpha)
            V_colloc = Vs_d + s_colloc * (V0_d - Vs_d)

            dC_ds = torch.autograd.grad(
                C_pred, s_colloc, grad_outputs=torch.ones_like(C_pred),
                create_graph=True)[0]
            dV_ds = (V0_d - Vs_d)
            dC_dV = dC_ds / (dV_ds + 1e-30)

            residual, numer, denom, delta = chisnell_residual(
                V_colloc, C_pred, dC_dV, alpha, gamma, n)

            # 混合残差
            abs_denom = torch.abs(denom)
            w_far = torch.clamp(abs_denom / 1e-6 - 1.0, min=0.0, max=1.0)
            R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
            w_sonic = 1.0 - w_far
            loss_i = (torch.mean(w_far * R_divided ** 2)
                      + torch.mean(w_sonic * residual ** 2)
                      + 10.0 * torch.mean(w_sonic * numer ** 2))
            total_loss = total_loss + loss_i

        loss = total_loss / n_gamma

        if torch.isnan(loss) or torch.isinf(loss):
            history_loss.append(1e10)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        history_loss.append(float(loss.detach()))

        if ep % 2000 == 0 or ep == 1:
            # 打印参考点
            with torch.no_grad():
                a14 = float(model.get_alpha(
                    torch.tensor(1.4, dtype=torch.float64, device=device)))
                a53 = float(model.get_alpha(
                    torch.tensor(5/3, dtype=torch.float64, device=device)))
            ref14 = REF_ALPHA.get((1.4, n), 0)
            ref53 = REF_ALPHA.get((5/3, n), 0)
            print(f"  [{geometry}] ep={ep:5d}  loss={float(loss):.4e}  "
                  f"α(1.4)={a14:.6f}(ref={ref14:.6f})  "
                  f"α(5/3)={a53:.6f}(ref={ref53:.6f})")

    # 评估: 在整条γ曲线上
    gammas_eval = torch.linspace(gamma_lo, gamma_hi, 100,
                                  dtype=torch.float64, device=device)
    alphas_eval = []
    with torch.no_grad():
        for g in gammas_eval:
            a = float(model.get_alpha(g))
            alphas_eval.append(a)

    # 检查参考点
    results = {"geometry": geometry, "n": n,
               "gamma_range": list(gamma_range),
               "gamma_curve": gammas_eval.cpu().numpy().tolist(),
               "alpha_curve": alphas_eval,
               "loss_history": history_loss}

    for gamma_ref in [1.4, 5/3]:
        ref_a = REF_ALPHA.get((gamma_ref, n))
        if ref_a:
            with torch.no_grad():
                a_pred = float(model.get_alpha(
                    torch.tensor(gamma_ref, dtype=torch.float64, device=device)))
            err = abs(a_pred - ref_a) / ref_a * 100
            key = f"g{gamma_ref:.4f}"
            results[key] = {"alpha_pred": a_pred, "alpha_ref": ref_a,
                            "err_pct": err}
            print(f"  γ={gamma_ref:.4f}: α_pred={a_pred:.8f}, "
                  f"ref={ref_a:.8f}, err={err:.6f}%")

    return results, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--geometry", type=str, default="spherical",
                        choices=["spherical", "cylindrical"])
    parser.add_argument("--epochs", type=int, default=20000)
    args = parser.parse_args()

    n = GEOMETRY_N[args.geometry]
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  参数化PINN: 一次训练发现 α(γ) 曲线 ({args.geometry})")
    print(f"  γ∈[1.2, 1.8], 纯ODE自发现, 无观测数据")
    print("=" * 70)

    results, model = train_parametric(
        n, args.geometry, epochs=args.epochs, device=args.device)

    fname = f"exp_parametric_{args.geometry}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / fname}")
