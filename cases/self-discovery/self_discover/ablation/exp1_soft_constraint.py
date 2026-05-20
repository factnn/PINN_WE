#!/usr/bin/env python3
"""
Ablation 1: Soft Constraint — 用loss惩罚端点，不用硬约束

对比: 成功方案用硬约束 C(V) = (1-s)*Cs + s*C0 + s*(1-s)*net(s)
本实验: 网络直接输出 C(V)，用soft loss约束端点

预期结果: alpha收敛差或不收敛，因为：
  1. 端点误差通过ODE传播放大
  2. alpha的梯度信号弱（只通过ODE残差间接传递）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common import *


class SoftConstraintPINN(nn.Module):
    """无硬约束: 网络直接输出C(V), 端点靠loss惩罚"""

    def __init__(self, gamma, n, alpha_init=0.75, n_layers=5, n_neurons=64):
        super().__init__()
        self.gamma = gamma
        self.n = n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5), dtype=torch.float64)
        )
        self.net = build_mlp(1, 1, n_layers, n_neurons).double()

    @property
    def alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def shock_cond(self):
        alpha = self.alpha
        Vs = 2.0 * alpha / (self.gamma + 1.0)
        Cs = 2.0 * self.gamma * (self.gamma - 1.0) * alpha ** 2 / (self.gamma + 1.0) ** 2
        return Vs, Cs

    def sonic_cond(self):
        return critical_point_torch(self.alpha, self.gamma, self.n)

    def forward(self, V):
        # 直接输出，无硬约束
        return self.net(V)


def train_soft(gamma, n, geometry, alpha_init=0.75, epochs=10000,
               lr=1e-3, n_colloc=400, w_bc=100.0, device="cuda"):
    ref_alpha = get_ref_alpha(gamma, n)
    model = SoftConstraintPINN(gamma, n, alpha_init).double().to(device)

    history_alpha, history_loss = [], []

    def compute_loss():
        alpha = model.alpha
        Vs, Cs = model.shock_cond()
        V0, C0 = model.sonic_cond()

        Vs_d, V0_d = Vs.detach(), V0.detach()
        V_colloc = (Vs_d + (V0_d - Vs_d) *
                    torch.linspace(0.01, 0.99, n_colloc, dtype=torch.float64, device=device)
                    ).reshape(-1, 1)
        V_colloc.requires_grad_(True)

        C_pred = model(V_colloc)
        dC_dV = torch.autograd.grad(
            C_pred, V_colloc, grad_outputs=torch.ones_like(C_pred),
            create_graph=True
        )[0]

        residual, numer, denom, delta = chisnell_residual(
            V_colloc, C_pred, dC_dV, alpha, gamma, n)

        # 混合残差形式 (和成功方案一样)
        abs_denom = torch.abs(denom)
        denom_thr = 1e-6
        w_far = torch.clamp(abs_denom / denom_thr - 1.0, min=0.0, max=1.0)
        R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        loss_ode_far = torch.mean(w_far * R_divided ** 2)
        w_sonic = 1.0 - w_far
        loss_ode_sonic = torch.mean(w_sonic * residual ** 2)
        loss_numer = torch.mean(w_sonic * numer ** 2)
        loss_ode = loss_ode_far + loss_ode_sonic
        loss_regularity = loss_numer

        # Soft端点约束 (替代硬约束)
        C_at_shock = model(Vs.detach().reshape(1, 1))
        C_at_sonic = model(V0.detach().reshape(1, 1))
        loss_bc = (C_at_shock - Cs) ** 2 + (C_at_sonic - C0) ** 2
        loss_bc = loss_bc.squeeze()

        return loss_ode, loss_regularity, loss_bc

    # Phase 1: 固定alpha
    warmup = epochs // 3
    model.raw_alpha.requires_grad_(False)
    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(1, warmup + 1):
        opt1.zero_grad()
        l_ode, l_reg, l_bc = compute_loss()
        loss = l_ode + 10.0 * l_reg + w_bc * l_bc
        loss.backward()
        opt1.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] warmup {ep:5d}  loss={loss.item():.4e}  "
                  f"ode={l_ode.item():.3e}  bc={l_bc.item():.3e}  "
                  f"alpha={float(model.alpha):.6f}")

    # Phase 2: 解冻alpha
    model.raw_alpha.requires_grad_(True)
    main_epochs = epochs - warmup
    opt2 = optim.Adam([
        {"params": model.net.parameters(), "lr": lr * 0.3},
        {"params": [model.raw_alpha], "lr": lr * 0.2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=main_epochs, eta_min=1e-6)

    for ep in range(1, main_epochs + 1):
        opt2.zero_grad()
        l_ode, l_reg, l_bc = compute_loss()
        loss = l_ode + 10.0 * l_reg + w_bc * l_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt2.step()
        scheduler.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] joint {ep:5d}  loss={loss.item():.4e}  "
                  f"ode={l_ode.item():.3e}  bc={l_bc.item():.3e}  "
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
        print(f"  Exp1 Soft Constraint: gamma={gamma}, {geo}")
        print(f"{'='*60}")
        results[key] = train_soft(gamma, n, geo, alpha_init=ALPHA_INIT[(gamma, n)], epochs=10000, device=device)

    print_summary_table(results, "Ablation 1: Soft Constraint (no hard constraint)")
    save_results_json(results, "Ablation 1: Soft Constraint (no hard constraint)", str(out_dir / "exp1_soft_constraint.json"))
    plot_ablation_result(results, "Ablation 1: Soft Constraint Only",
                         str(out_dir / "exp1_soft_constraint.png"))
    print(f"\nPlot: {out_dir / 'exp1_soft_constraint.png'}")
