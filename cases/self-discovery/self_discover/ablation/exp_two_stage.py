#!/usr/bin/env python3
"""
两阶段策略: 先软约束+warmup粗定位, 再硬约束精细化。

Stage 1: Chisnell + 混合残差 + 软约束 + warmup (= Exp1), 5000 epochs
  → 得到粗alpha
Stage 2: 用粗alpha作为init, Chisnell + 混合残差 + 双端点硬约束, 5000 epochs
  → 精细化
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
                     chisnell_residual, save_results_json)


def shock_conditions_torch(alpha, gamma, n):
    Vs = 2.0 / ((gamma + 1.0) * alpha)
    Cs = (2.0 * gamma * (gamma - 1.0)) / ((gamma + 1.0) ** 2 * alpha ** 2)
    return Vs, Cs


class SoftPINN(nn.Module):
    """Stage 1: 软约束模型"""
    def __init__(self, gamma, n, alpha_init, n_layers=5, n_neurons=64):
        super().__init__()
        self.gamma = gamma
        self.n = n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5), dtype=torch.float64))
        layers = [nn.Linear(1, n_neurons, dtype=torch.float64), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons, dtype=torch.float64), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1, dtype=torch.float64)]
        self.net = nn.Sequential(*layers)

    @property
    def alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def forward(self, V):
        return self.net(V)


class HardPINN(nn.Module):
    """Stage 2: 双端点硬约束模型"""
    def __init__(self, gamma, n, alpha_init, n_layers=5, n_neurons=64):
        super().__init__()
        self.gamma = gamma
        self.n = n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5), dtype=torch.float64))
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
        s = (V - Vs) / (V0 - Vs + 1e-30)
        return (1 - s) * Cs + s * C0 + s * (1 - s) * self.net(V)


def compute_loss(model, gamma, n, device, soft=False):
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

    # 混合残差
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

    loss_bc = torch.tensor(0.0, dtype=torch.float64, device=device)
    if soft:
        alpha_nd = model.alpha
        Vs_nd, Cs_nd = shock_conditions_torch(alpha_nd, gamma, n)
        V0_nd, C0_nd = critical_point_torch(alpha_nd, gamma, n)
        C_at_shock = model(Vs_nd.detach().reshape(1, 1))
        C_at_sonic = model(V0_nd.detach().reshape(1, 1))
        loss_bc = (C_at_shock - Cs_nd) ** 2 + (C_at_sonic - C0_nd) ** 2

    return loss_ode + 100.0 * loss_bc


def train_two_stage(gamma, n, geometry, alpha_init,
                    epochs_s1=5000, epochs_s2=5000, device="cuda"):
    ref_alpha = get_ref_alpha(gamma, n)
    history_alpha = []
    history_loss = []

    # ===== Stage 1: 软约束 + warmup =====
    print(f"  [{geometry}] Stage 1: soft constraint + warmup (alpha_init={alpha_init:.4f})")
    model1 = SoftPINN(gamma, n, alpha_init).to(device)

    # warmup: 固定alpha
    warmup_epochs = epochs_s1 // 3
    model1.raw_alpha.requires_grad_(False)
    opt1w = optim.Adam([p for p in model1.parameters() if p.requires_grad], lr=1e-3)

    for ep in range(1, warmup_epochs + 1):
        opt1w.zero_grad()
        loss = compute_loss(model1, gamma, n, device, soft=True)
        if torch.isnan(loss):
            continue
        loss.backward()
        opt1w.step()
        history_alpha.append(float(model1.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 1000 == 0:
            print(f"    warmup {ep:5d}  loss={loss.item():.4e}  alpha={float(model1.alpha.detach()):.6f}")

    model1.raw_alpha.requires_grad_(True)
    main1 = epochs_s1 - warmup_epochs
    opt1 = optim.Adam([
        {"params": model1.net.parameters(), "lr": 1e-3},
        {"params": [model1.raw_alpha], "lr": 2e-4},
    ])
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=main1, eta_min=1e-6)

    for ep in range(1, main1 + 1):
        opt1.zero_grad()
        loss = compute_loss(model1, gamma, n, device, soft=True)
        if torch.isnan(loss) or torch.isinf(loss):
            history_alpha.append(float(model1.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
        opt1.step()
        sched1.step()
        history_alpha.append(float(model1.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 1000 == 0:
            print(f"    joint  {ep:5d}  loss={loss.item():.4e}  alpha={float(model1.alpha.detach()):.6f}")

    alpha_stage1 = float(model1.alpha.detach())
    print(f"  [{geometry}] Stage 1 DONE: alpha={alpha_stage1:.6f}  (ref={ref_alpha}, err={abs(alpha_stage1-ref_alpha)/ref_alpha*100:.4f}%)")

    # ===== Stage 2: 硬约束, 用stage1的alpha做init =====
    print(f"  [{geometry}] Stage 2: hard constraint (alpha_init={alpha_stage1:.6f})")
    model2 = HardPINN(gamma, n, alpha_stage1).to(device)

    opt2 = optim.Adam([
        {"params": model2.net.parameters(), "lr": 1e-3},
        {"params": [model2.raw_alpha], "lr": 2e-4},
    ])
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_s2, eta_min=1e-6)

    for ep in range(1, epochs_s2 + 1):
        opt2.zero_grad()
        loss = compute_loss(model2, gamma, n, device, soft=False)
        if torch.isnan(loss) or torch.isinf(loss):
            history_alpha.append(float(model2.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        opt2.step()
        sched2.step()
        history_alpha.append(float(model2.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 1000 == 0:
            print(f"    refine {ep:5d}  loss={loss.item():.4e}  alpha={float(model2.alpha.detach()):.6f}")

    alpha_final = float(model2.alpha.detach())
    rel_err = abs(alpha_final - ref_alpha) / ref_alpha * 100
    final_loss = history_loss[-1]
    print(f"  [{geometry}] Stage 2 DONE: alpha={alpha_final:.6f}  ref={ref_alpha}  err={rel_err:.4f}%  loss={final_loss:.3e}")

    return {
        "alpha_fit": alpha_final, "ref_alpha": ref_alpha,
        "rel_err_pct": rel_err, "final_loss": final_loss,
        "alpha_stage1": alpha_stage1,
        "alpha_history": np.array(history_alpha),
        "loss_history": np.array(history_loss),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha-init", type=float, default=0.75,
                        help="Initial alpha guess (default 0.75 = bad init)")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  两阶段策略: 软约束+warmup → 硬约束 (alpha_init={args.alpha_init})")
    print("=" * 70)

    results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        print(f"\n--- gamma={gamma}, {geo} ---")
        results[key] = train_two_stage(gamma, n, geo, args.alpha_init,
                                        device=args.device)

    print(f"\n{'='*70}")
    print(f"  两阶段策略 Summary (alpha_init={args.alpha_init})")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'stage1':<14} {'final':<14} {'ref':<14} {'err%':<12} {'loss':<12}")
    print(f"  {'-'*85}")
    for key, r in results.items():
        print(f"  {key:<25} {r['alpha_stage1']:<14.6f} {r['alpha_fit']:<14.6f} "
              f"{r['ref_alpha']:<14.6f} {r['rel_err_pct']:<12.4f} {r['final_loss']:<12.3e}")

    init_tag = f"_init{args.alpha_init:.2f}"
    fname = f"exp_two_stage{init_tag}.json"
    save_data = {"experiment": f"Two-stage: soft+warmup -> hard (init={args.alpha_init})", "configs": {}}
    for key, r in results.items():
        save_data["configs"][key] = {
            "alpha_fit": r["alpha_fit"], "ref_alpha": r["ref_alpha"],
            "rel_err_pct": r["rel_err_pct"], "final_loss": r["final_loss"],
            "alpha_stage1": r["alpha_stage1"],
            "alpha_history": r["alpha_history"].tolist(),
            "loss_history": r["loss_history"].tolist(),
        }
    with open(out_dir / fname, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {out_dir / fname}")
