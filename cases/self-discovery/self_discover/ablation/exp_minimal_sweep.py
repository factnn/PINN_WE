#!/usr/bin/env python3
"""
最简方案探索: Chisnell + 混合残差 为基底，测试最少还需要加什么。
通过 --mode 选择组合:
  A: 只加warmup（无任何约束）
  B: 只加音速端硬约束（无warmup）
  C: 只加激波端硬约束（无warmup）
  D: 只加软约束（无warmup）
  E: 纯基底（无约束+无warmup）
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from common import (ALPHA_INIT, DEFAULT_CONFIGS, GEOMETRY_N,
                     get_ref_alpha, inv_sigmoid, critical_point_torch,
                     chisnell_residual)


def shock_conditions_torch(alpha, gamma, n):
    Vs = 2.0 / ((gamma + 1.0) * alpha)
    Cs = (2.0 * gamma * (gamma - 1.0)) / ((gamma + 1.0) ** 2 * alpha ** 2)
    return Vs, Cs


class FlexiblePINN(nn.Module):
    def __init__(self, gamma, n, alpha_init, constraint_mode="none",
                 n_layers=5, n_neurons=64):
        """
        constraint_mode: "none", "sonic", "shock", "soft"
        """
        super().__init__()
        self.gamma = gamma
        self.n = n
        self.constraint_mode = constraint_mode

        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5), dtype=torch.float64)
        )

        layers = [nn.Linear(1, n_neurons, dtype=torch.float64), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons, dtype=torch.float64), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1, dtype=torch.float64)]
        self.net = nn.Sequential(*layers)

    @property
    def alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def forward(self, V):
        alpha = self.alpha
        Vs, Cs = shock_conditions_torch(alpha, self.gamma, self.n)
        V0, C0 = critical_point_torch(alpha, self.gamma, self.n)

        if self.constraint_mode == "sonic":
            # C(V0) = C0, 线性基底 + 网络修正
            # C = C0 + (V - V0) * net(V)
            return C0 + (V - V0) * self.net(V)

        elif self.constraint_mode == "shock":
            # C(Vs) = Cs
            # C = Cs + (V - Vs) * net(V)
            return Cs + (V - Vs) * self.net(V)

        else:  # "none" or "soft"
            return self.net(V)


def train_flexible(gamma, n, geometry, alpha_init, mode,
                   epochs=10000, device="cuda"):
    ref_alpha = get_ref_alpha(gamma, n)

    use_warmup = (mode == "A")
    constraint = {"A": "none", "B": "sonic", "C": "shock", "D": "soft", "E": "none"}[mode]

    model = FlexiblePINN(gamma, n, alpha_init, constraint_mode=constraint).to(device)

    mode_desc = {
        "A": "只加warmup，无约束",
        "B": "只加音速端硬约束，无warmup",
        "C": "只加激波端硬约束，无warmup",
        "D": "只加软约束，无warmup",
        "E": "纯基底（无约束+无warmup）",
    }[mode]
    print(f"  模式: {mode_desc}")

    history_alpha = []
    history_loss = []

    def compute_loss():
        alpha = model.alpha
        Vs, Cs = shock_conditions_torch(alpha.detach(), gamma, n)
        V0, C0 = critical_point_torch(alpha.detach(), gamma, n)

        V_colloc = (Vs + (V0 - Vs) *
                    torch.linspace(0.01, 0.99, 400, dtype=torch.float64, device=device)
                    ).reshape(-1, 1)
        V_colloc.requires_grad_(True)

        C_pred = model(V_colloc)

        dC_dV = torch.autograd.grad(
            C_pred, V_colloc,
            grad_outputs=torch.ones_like(C_pred),
            create_graph=True,
        )[0]

        # 混合残差 (和完整方案一致)
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
        loss_ode = loss_ode_far + loss_ode_sonic + 10.0 * loss_numer

        # 软约束惩罚 (mode D)
        loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
        if constraint == "soft":
            alpha_nd = model.alpha
            Vs_nd, Cs_nd = shock_conditions_torch(alpha_nd, gamma, n)
            V0_nd, C0_nd = critical_point_torch(alpha_nd, gamma, n)
            C_at_shock = model(Vs_nd.detach().reshape(1, 1))
            C_at_sonic = model(V0_nd.detach().reshape(1, 1))
            loss_bc = (C_at_shock - Cs_nd) ** 2 + (C_at_sonic - C0_nd) ** 2

        return loss_ode, loss_bc

    if use_warmup:
        # 阶段1: 固定alpha
        warmup_epochs = epochs // 3
        model.raw_alpha.requires_grad_(False)
        opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        for ep in range(1, warmup_epochs + 1):
            opt1.zero_grad()
            loss_ode, loss_bc = compute_loss()
            loss = loss_ode + 100.0 * loss_bc
            if torch.isnan(loss):
                continue
            loss.backward()
            opt1.step()
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(float(loss.detach()))
            if ep % 2000 == 0 or ep == 1:
                print(f"  [{geometry}] warmup {ep:5d}  loss={loss.item():.4e}  alpha={float(model.alpha.detach()):.6f}")

        model.raw_alpha.requires_grad_(True)
        main_epochs = epochs - warmup_epochs
    else:
        main_epochs = epochs

    # 主训练
    opt2 = optim.Adam([
        {"params": model.net.parameters(), "lr": 1e-3},
        {"params": [model.raw_alpha], "lr": 2e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=main_epochs, eta_min=1e-6)

    for ep in range(1, main_epochs + 1):
        opt2.zero_grad()
        loss_ode, loss_bc = compute_loss()
        loss = loss_ode + 100.0 * loss_bc
        if torch.isnan(loss) or torch.isinf(loss):
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
                  f"alpha={float(model.alpha.detach()):.6f}  (ref={ref_alpha})")

    alpha_fit = float(model.alpha.detach())
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100
    final_loss = history_loss[-1]
    print(f"  [{geometry}] DONE  alpha={alpha_fit:.6f}  ref={ref_alpha}  err={rel_err:.4f}%  loss={final_loss:.3e}")

    return {
        "alpha_fit": alpha_fit, "ref_alpha": ref_alpha,
        "rel_err_pct": rel_err, "final_loss": final_loss,
        "alpha_history": np.array(history_alpha),
        "loss_history": np.array(history_loss),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha-init", type=float, default=None,
                        help="Override alpha_init for all configs (e.g. 0.75 for bad init)")
    args = parser.parse_args()

    mode_desc = {
        "A": "只加warmup，无约束",
        "B": "只加音速端硬约束，无warmup",
        "C": "只加激波端硬约束，无warmup",
        "D": "只加软约束，无warmup",
        "E": "纯基底（无约束+无warmup）",
    }

    print(f"\n{'='*70}")
    print(f"  最简方案探索 Mode {args.mode}: {mode_desc[args.mode]}")
    print(f"{'='*70}")

    results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        ai = args.alpha_init if args.alpha_init else ALPHA_INIT[(gamma, n)]
        print(f"\n--- gamma={gamma}, {geo}, alpha_init={ai} ---")
        results[key] = train_flexible(gamma, n, geo, ai, args.mode,
                                       epochs=10000, device=args.device)

    print(f"\n{'='*70}")
    print(f"  Summary Mode {args.mode}: {mode_desc[args.mode]}")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'alpha_fit':<14} {'ref':<14} {'err%':<12} {'loss':<12}")
    print(f"  {'-'*75}")
    for key, r in results.items():
        print(f"  {key:<25} {r['alpha_fit']:<14.6f} {r['ref_alpha']:<14.6f} "
              f"{r['rel_err_pct']:<12.4f} {r['final_loss']:<12.3e}")

    # 保存结果
    import json
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    init_tag = f"_badinit{args.alpha_init:.2f}" if args.alpha_init else ""
    fname = f"exp_minimal_mode{args.mode}{init_tag}.json"
    save_data = {"experiment": f"Minimal Mode {args.mode}: {mode_desc[args.mode]}", "configs": {}}
    for key, r in results.items():
        save_data["configs"][key] = {
            "alpha_fit": r["alpha_fit"], "ref_alpha": r["ref_alpha"],
            "rel_err_pct": r["rel_err_pct"], "final_loss": r["final_loss"],
            "alpha_history": r["alpha_history"].tolist(),
            "loss_history": r["loss_history"].tolist(),
        }
    with open(out_dir / fname, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {out_dir / fname}")
