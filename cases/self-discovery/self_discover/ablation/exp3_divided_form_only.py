#!/usr/bin/env python3
"""
Ablation 3a: 纯除法形式残差 — dC/dV - numer/denom

成功方案用混合形式: 远离奇点用除法，靠近奇点用乘法。
本实验: 全部用除法形式。

预期结果: 奇点附近denom→0导致除法爆炸，训练不稳定或NaN。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common import *


class DualEndpointPINN(nn.Module):
    """和成功方案一样的硬约束网络"""

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
        Vs, Cs = self.shock_cond()
        V0, C0 = self.sonic_cond()
        s = (V - Vs) / (V0 - Vs + 1e-14)
        return (1.0 - s) * Cs + s * C0 + s * (1.0 - s) * self.net(s)


def train_divided_only(gamma, n, geometry, alpha_init=0.75, epochs=10000,
                       lr=1e-3, n_colloc=400, device="cuda"):
    ref_alpha = get_ref_alpha(gamma, n)
    model = DualEndpointPINN(gamma, n, alpha_init).double().to(device)
    history_alpha, history_loss = [], []
    nan_count = 0

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

        # 纯除法形式 — 不区分远近奇点!
        R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        loss_ode = torch.mean(R_divided ** 2)

        return loss_ode

    # Phase 1
    warmup = epochs // 3
    model.raw_alpha.requires_grad_(False)
    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(1, warmup + 1):
        opt1.zero_grad()
        loss = compute_loss()
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        opt1.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] warmup {ep:5d}  loss={loss.item():.4e}  "
                  f"alpha={float(model.alpha):.6f}  NaN={nan_count}")

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
        loss = compute_loss()
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
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
                  f"alpha={float(model.alpha):.6f}  (ref={ref_alpha})  NaN={nan_count}")

    alpha_fit = float(model.alpha.detach())
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100 if ref_alpha else None
    print(f"  [{geometry}] DONE alpha={alpha_fit:.6f} ref={ref_alpha} NaN_total={nan_count}")
    return {
        "alpha_fit": alpha_fit, "ref_alpha": ref_alpha,
        "rel_err_pct": rel_err, "final_loss": history_loss[-1],
        "alpha_history": np.array(history_alpha),
        "loss_history": np.array(history_loss),
        "nan_count": nan_count,
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
        print(f"  Exp3a Divided-Form Only: gamma={gamma}, {geo}")
        print(f"{'='*60}")
        results[key] = train_divided_only(gamma, n, geo, alpha_init=ALPHA_INIT[(gamma, n)], epochs=10000, device=device)

    print_summary_table(results, "Ablation 3a: Divided-Form Only (no mixed residual)")
    save_results_json(results, "Ablation 3a: Divided-Form Only (no mixed residual)", str(out_dir / "exp3a_divided_only.json"))
    plot_ablation_result(results, "Ablation 3a: Divided-Form Only",
                         str(out_dir / "exp3a_divided_only.png"))
