#!/usr/bin/env python3
"""
Ablation 4: 无warmup — alpha从第1个epoch就自由训练

成功方案: Phase1固定alpha训练网络 → Phase2解冻alpha联合训练
本实验: 从头开始alpha和网络一起训

预期结果: alpha可能漂移到错误值，因为：
  1. 网络初始输出随机 → ODE残差无意义 → alpha梯度是噪声
  2. alpha早期漂移后，端点位置(Vs,V0)也变 → 配点域都变了 → 恶性循环
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


def train_no_warmup(gamma, n, geometry, alpha_init=0.75, epochs=10000,
                    lr=1e-3, n_colloc=400, device="cuda"):
    ref_alpha = get_ref_alpha(gamma, n)
    model = DualEndpointPINN(gamma, n, alpha_init).double().to(device)
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

        abs_denom = torch.abs(denom)
        denom_thr = 1e-6
        w_far = torch.clamp(abs_denom / denom_thr - 1.0, min=0.0, max=1.0)
        R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        loss_ode_far = torch.mean(w_far * R_divided ** 2)
        w_sonic = 1.0 - w_far
        loss_ode_sonic = torch.mean(w_sonic * residual ** 2)
        loss_numer = torch.mean(w_sonic * numer ** 2)

        return (loss_ode_far + loss_ode_sonic), loss_numer

    # 无warmup: 从头alpha就自由
    model.raw_alpha.requires_grad_(True)
    optimizer = optim.Adam([
        {"params": model.net.parameters(), "lr": lr},
        {"params": [model.raw_alpha], "lr": lr * 0.2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        l_ode, l_reg = compute_loss()
        loss = l_ode + 10.0 * l_reg
        if torch.isnan(loss) or torch.isinf(loss):
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] ep={ep:5d}  loss={loss.item():.4e}  "
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
        print(f"  Exp4 No Warmup: gamma={gamma}, {geo}")
        print(f"{'='*60}")
        results[key] = train_no_warmup(gamma, n, geo, alpha_init=ALPHA_INIT[(gamma, n)], epochs=10000, device=device)

    print_summary_table(results, "Ablation 4: No Warmup (alpha free from epoch 1)")
    save_results_json(results, "Ablation 4: No Warmup (alpha free from epoch 1)", str(out_dir / "exp4_no_warmup.json"))
    plot_ablation_result(results, "Ablation 4: No Warmup",
                         str(out_dir / "exp4_no_warmup.png"))
