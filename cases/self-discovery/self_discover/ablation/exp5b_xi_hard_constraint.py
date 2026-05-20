#!/usr/bin/env python3
"""
Exp5b: ξ两方程 + 双端点硬约束 + warmup + 混合残差

对照实验: 和Exp7(完整方案=Chisnell+硬约束+warmup)做单变量对比。
唯一区别: 用ξ两方程系统替代Chisnell单方程。

实现:
  - 归一化坐标 t ∈ [0,1], t=0→sonic, t=1→shock
  - V(t) = (1-t)*V0(α) + t*Vs(α) + t*(1-t)*net_V(t)  (硬约束)
  - C(t) = (1-t)*C0(α) + t*Cs(α) + t*(1-t)*net_C(t)  (硬约束)
  - ξ(t) = ξ* + t*(1-ξ*), ξ* 可学习
  - 两个ODE残差 (ξ-based multiplied form):
    R1: ξ * dV/dξ * Δ - V*Q = 0
    R2: dC/dξ * D - N * dV/dξ = 0  (Chisnell chain-rule form)
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
                     build_mlp, save_results_json, print_summary_table)


def shock_conditions(alpha, gamma):
    Vs = 2.0 * alpha / (gamma + 1.0)
    Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
    return Vs, Cs


class XiHardPINN(nn.Module):
    def __init__(self, gamma, n, alpha_init, n_layers=5, n_neurons=64):
        super().__init__()
        self.gamma = gamma
        self.n = n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5), dtype=torch.float64))
        # ξ* ∈ (0.01, 0.99)
        self.raw_xi_star = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.net = build_mlp(1, 2, n_layers, n_neurons).double()

    @property
    def alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    @property
    def xi_star(self):
        return 0.01 + 0.98 * torch.sigmoid(self.raw_xi_star)

    def forward(self, t):
        """
        t ∈ [0,1]: t=0 → sonic, t=1 → shock
        Returns V(t), C(t), xi(t)
        """
        alpha = self.alpha
        Vs, Cs = shock_conditions(alpha, self.gamma)
        V0, C0 = critical_point_torch(alpha, self.gamma, self.n)

        raw = self.net(t)
        net_V = raw[:, 0:1]
        net_C = raw[:, 1:2]

        V = (1 - t) * V0 + t * Vs + t * (1 - t) * net_V
        C = (1 - t) * C0 + t * Cs + t * (1 - t) * net_C

        xi = self.xi_star + t * (1 - self.xi_star)
        return V, C, xi


def train_xi_hard(gamma, n, geometry, alpha_init, epochs=10000, device="cuda"):
    ref_alpha = get_ref_alpha(gamma, n)
    model = XiHardPINN(gamma, n, alpha_init).to(device)
    history_alpha, history_loss = [], []

    def compute_loss():
        alpha = model.alpha
        xi_star = model.xi_star

        t_colloc = torch.linspace(0.01, 0.99, 400, dtype=torch.float64,
                                  device=device).reshape(-1, 1)
        t_colloc.requires_grad_(True)

        V, C, xi = model(t_colloc)

        # dV/dt, dC/dt
        dV_dt = torch.autograd.grad(V, t_colloc, grad_outputs=torch.ones_like(V),
                                     create_graph=True)[0]
        dC_dt = torch.autograd.grad(C, t_colloc, grad_outputs=torch.ones_like(C),
                                     create_graph=True)[0]

        # dξ/dt = 1 - ξ*
        dxi_dt = 1.0 - xi_star

        # dV/dξ = dV/dt / dξ/dt,  dC/dξ = dC/dt / dξ/dt
        dV_dxi = dV_dt / (dxi_dt + 1e-30)
        dC_dxi = dC_dt / (dxi_dt + 1e-30)

        # ODE terms
        delta = (V - alpha) ** 2 - C
        Q = (n * V * (V - alpha)
             + (2.0 / gamma) * (1.0 - alpha) * (alpha - V)
             - V * (V - 1.0))

        # Chisnell numer/denom for dC/dV equation
        numer = C * (2.0 * delta * (alpha - V + (1.0 - alpha) / gamma)
                     + (gamma - 1.0) * (alpha - V) * Q)
        denom = (delta * (n * V - 2.0 * (1.0 - alpha) / gamma) * (alpha - V)
                 + (alpha - V) ** 2 * Q)

        # Residual 1 (V-equation): ξ * dV/dξ * Δ - V*Q = 0  (multiplied form)
        res_V = xi * dV_dxi * delta - V * Q

        # Residual 2 (C-equation): dC/dξ * D - N * (dV/dξ)^(-1) ...
        # Actually: dC/dV = dC/dξ / dV/dξ, Chisnell: dC/dV = numer/denom
        # → dC/dxi * denom - numer * (dV/dxi)^(-1) ... no
        # Multiplied form: dC/dxi * denom - numer * dV/dxi^{-1}
        # Better: dC/dt * denom = numer * dV/dt / (denom... no
        # Simplest: dC/dV = (dC/dt)/(dV/dt), and use mixed residual like Chisnell

        dC_dV = dC_dt / (dV_dt + 1e-30 * torch.sign(dV_dt))

        # Mixed residual for C equation (same as our method)
        abs_denom = torch.abs(denom)
        denom_thr = 1e-6
        w_far = torch.clamp(abs_denom / denom_thr - 1.0, min=0.0, max=1.0)

        R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        residual_mult = dC_dV * denom - numer

        loss_C_far = torch.mean(w_far * R_divided ** 2)
        w_sonic = 1.0 - w_far
        loss_C_sonic = torch.mean(w_sonic * residual_mult ** 2)
        loss_C_numer = torch.mean(w_sonic * numer ** 2)

        loss_ode = torch.mean(res_V ** 2) + loss_C_far + loss_C_sonic + 10.0 * loss_C_numer

        return loss_ode

    # Warmup: 固定alpha和xi_star
    warmup = epochs // 3
    model.raw_alpha.requires_grad_(False)
    model.raw_xi_star.requires_grad_(False)
    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    for ep in range(1, warmup + 1):
        opt1.zero_grad()
        loss = compute_loss()
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
                  f"alpha={float(model.alpha.detach()):.6f}  xi*={float(model.xi_star.detach()):.4f}")

    # Joint phase
    model.raw_alpha.requires_grad_(True)
    model.raw_xi_star.requires_grad_(True)
    main_epochs = epochs - warmup
    opt2 = optim.Adam([
        {"params": model.net.parameters(), "lr": 1e-3},
        {"params": [model.raw_alpha], "lr": 2e-4},
        {"params": [model.raw_xi_star], "lr": 1e-3},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=main_epochs, eta_min=1e-6)

    for ep in range(1, main_epochs + 1):
        opt2.zero_grad()
        loss = compute_loss()
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
                  f"alpha={float(model.alpha.detach()):.6f}  xi*={float(model.xi_star.detach()):.4f}  (ref={ref_alpha})")

    alpha_fit = float(model.alpha.detach())
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100
    final_loss = history_loss[-1]
    print(f"  [{geometry}] DONE  alpha={alpha_fit:.6f}  ref={ref_alpha}  err={rel_err:.4f}%")

    return {
        "alpha_fit": alpha_fit, "ref_alpha": ref_alpha,
        "rel_err_pct": rel_err, "final_loss": final_loss,
        "xi_star": float(model.xi_star.detach()),
        "alpha_history": np.array(history_alpha),
        "loss_history": np.array(history_loss),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Exp5b: Xi 2-Equation + Dual Hard Constraint + Warmup")
    print("=" * 70)

    results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        ai = ALPHA_INIT[(gamma, n)]
        print(f"\n--- gamma={gamma}, {geo}, alpha_init={ai} ---")
        results[key] = train_xi_hard(gamma, n, geo, ai, epochs=10000, device=args.device)

    print(f"\n{'='*70}")
    print("  Exp5b Summary: Xi 2-Equation + Hard Constraint + Warmup")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'alpha':<14} {'ref':<14} {'err%':<12} {'loss':<12} {'xi*':<8}")
    print(f"  {'-'*80}")
    for key, r in results.items():
        print(f"  {key:<25} {r['alpha_fit']:<14.6f} {r['ref_alpha']:<14.6f} "
              f"{r['rel_err_pct']:<12.4f} {r['final_loss']:<12.3e} {r['xi_star']:<8.4f}")

    save_results_json(results,
                      "Exp5b: Xi 2-Equation + Hard Constraint + Warmup",
                      str(out_dir / "exp5b_xi_hard.json"))
    print(f"\nSaved: {out_dir / 'exp5b_xi_hard.json'}")
