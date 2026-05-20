#!/usr/bin/env python3
"""
Ablation 5: ξ-based 原始Guderley方程 (非Chisnell相平面)

原始Guderley自相似ODE:
  dV/dξ = F1(V, C, ξ; alpha, gamma, n)
  dC/dξ = F2(V, C, ξ; alpha, gamma, n)

这是2方程系统，网络需要学 ξ -> (V(ξ), C(ξ))

成功方案用Chisnell单方程 V -> C(V)，消除了自变量ξ，降为1D问题。

预期结果: 更难收敛，因为：
  1. 2个未知函数 vs 1个
  2. 需要处理ξ和V,C的耦合
  3. 奇点结构更复杂
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common import *


class XiBasedPINN(nn.Module):
    """
    ξ-based formulation: 网络输出 (V(ξ), C(ξ))

    Guderley ODE (Chisnell 1998, eq before phase-plane reduction):
      ξ dV/dξ = Numerator_V / Delta
      ξ dC/dξ = Numerator_C / Delta

    where Delta = (V-α)² - C
    """

    def __init__(self, gamma, n, alpha_init=0.75, n_layers=5, n_neurons=64):
        super().__init__()
        self.gamma = gamma
        self.n = n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5), dtype=torch.float64)
        )
        self.net = build_mlp(1, 2, n_layers, n_neurons).double()

    @property
    def alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def forward(self, xi):
        """输出 (V, C)"""
        out = self.net(xi)
        V = out[:, 0:1]
        C = torch.abs(out[:, 1:2]) + 1e-10  # C > 0
        return V, C


def train_xi_based(gamma, n, geometry, alpha_init=0.75, epochs=10000,
                   lr=1e-3, n_colloc=400, device="cuda"):
    ref_alpha = get_ref_alpha(gamma, n)
    model = XiBasedPINN(gamma, n, alpha_init).double().to(device)
    history_alpha, history_loss = [], []

    # 激波条件: 在ξ=1处 (shock)
    # V(1) = Vs = 2α/(γ+1)
    # C(1) = Cs = 2γ(γ-1)α²/(γ+1)²

    def compute_loss():
        alpha = model.alpha

        # 激波值
        Vs = 2.0 * alpha / (gamma + 1.0)
        Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2

        # 音速点值
        V0, C0 = critical_point_torch(alpha, gamma, n)

        # 配点: ξ ∈ (0.01, 1.0)
        xi = torch.linspace(0.01, 1.0, n_colloc, dtype=torch.float64, device=device).reshape(-1, 1)
        xi.requires_grad_(True)

        V_pred, C_pred = model(xi)

        # 导数
        dV_dxi = torch.autograd.grad(
            V_pred, xi, grad_outputs=torch.ones_like(V_pred),
            create_graph=True
        )[0]
        dC_dxi = torch.autograd.grad(
            C_pred, xi, grad_outputs=torch.ones_like(C_pred),
            create_graph=True
        )[0]

        # Guderley ODE (ξ-based)
        delta = (V_pred - alpha) ** 2 - C_pred
        Q = (n * V_pred * (V_pred - alpha)
             + (2.0 / gamma) * (1.0 - alpha) * (alpha - V_pred)
             - V_pred * (V_pred - 1.0))

        # ξ dV/dξ = V * Q / Delta  (simplified from Chisnell)
        numer_V = V_pred * Q
        # ξ dC/dξ 需要从 dC/dV * dV/dξ 推导
        numer_C_over_delta = C_pred * (
            2.0 * delta * (alpha - V_pred + (1.0 - alpha) / gamma)
            + (gamma - 1.0) * (alpha - V_pred) * Q
        )
        denom_C = (delta * (n * V_pred - 2.0 * (1.0 - alpha) / gamma) * (alpha - V_pred)
                   + (alpha - V_pred) ** 2 * Q)

        # 乘法形式 (避免除零):
        # xi * dV/dxi * delta = V * Q
        # dC/dV * denom = numer → dC/dxi = (numer/denom) * dV/dxi
        res_V = xi * dV_dxi * delta - numer_V
        res_C = dC_dxi * denom_C - numer_C_over_delta * dV_dxi

        loss_ode = torch.mean(res_V ** 2) + torch.mean(res_C ** 2)

        # 边界条件 (soft)
        V_at_1, C_at_1 = model(torch.tensor([[1.0]], dtype=torch.float64, device=device))
        loss_bc_shock = (V_at_1 - Vs) ** 2 + (C_at_1 - Cs) ** 2

        # Sonic条件 (soft) — 需要找到ξ*使得delta=0
        # 简化: 用最后一个配点附近加soft约束
        loss_bc_sonic = torch.tensor(0.0, dtype=torch.float64, device=device)

        return loss_ode, loss_bc_shock.squeeze()

    # Phase 1
    warmup = epochs // 3
    model.raw_alpha.requires_grad_(False)
    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(1, warmup + 1):
        opt1.zero_grad()
        l_ode, l_bc = compute_loss()
        loss = l_ode + 100.0 * l_bc
        if torch.isnan(loss):
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        opt1.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] warmup {ep:5d}  loss={loss.item():.4e}  "
                  f"ode={l_ode.item():.3e}  bc={l_bc.item():.3e}  "
                  f"alpha={float(model.alpha):.6f}")

    # Phase 2
    model.raw_alpha.requires_grad_(True)
    main_epochs = epochs - warmup
    opt2 = optim.Adam([
        {"params": model.net.parameters(), "lr": lr * 0.3},
        {"params": [model.raw_alpha], "lr": lr * 0.2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=main_epochs, eta_min=1e-6)

    for ep in range(1, main_epochs + 1):
        opt2.zero_grad()
        l_ode, l_bc = compute_loss()
        loss = l_ode + 100.0 * l_bc
        if torch.isnan(loss):
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt2.step()
        scheduler.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] joint {ep:5d}  loss={loss.item():.4e}  "
                  f"alpha={float(model.alpha):.6f}  (ref={ref_alpha})")

    alpha_fit = float(model.alpha.detach())
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100 if ref_alpha else None
    return {
        "alpha_fit": alpha_fit, "ref_alpha": ref_alpha,
        "rel_err_pct": rel_err, "final_loss": history_loss[-1],
        "alpha_history": np.array(history_alpha),
        "loss_history": np.array(history_loss),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device

    out_dir = Path(__file__).parent / "output"
    results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        print(f"\n{'='*60}")
        print(f"  Exp5 Xi-Based Formulation: gamma={gamma}, {geo}")
        print(f"{'='*60}")
        results[key] = train_xi_based(gamma, n, geo, alpha_init=ALPHA_INIT[(gamma, n)], epochs=10000, device=device)

    print_summary_table(results, "Ablation 5: Xi-Based 2-Equation (not Chisnell)")
    save_results_json(results, "Ablation 5: Xi-Based 2-Equation (not Chisnell)", str(out_dir / "exp5_xi_formulation.json"))
    plot_ablation_result(results, "Ablation 5: Xi-Based 2-Equation Formulation",
                         str(out_dir / "exp5_xi_formulation.png"))
