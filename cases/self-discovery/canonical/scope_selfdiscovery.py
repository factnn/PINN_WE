#!/usr/bin/env python3
"""
参数化SCOPE: 纯自发现版本
输入(s, gamma, n)，ODE loss驱动alpha自发现，无预训练，无打靶法数据。

和case-by-case的唯一区别：alpha是alpha_net(gamma,n)的输出，而不是标量。
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from guderley_ode_solver import find_eigenvalue


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def shock_conditions(alpha, gamma):
    Vs = 2.0 * alpha / (gamma + 1.0)
    Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
    return Vs, Cs


def critical_point(alpha, gamma, n):
    # n can be tensor or int
    if isinstance(n, int):
        gamma_crit = 1.8697680 if n == 3 else 1.9092084
        gamma_crit_t = torch.full_like(gamma, gamma_crit)
    else:
        gamma_crit_t = torch.where(n >= 2.5,
                                    torch.full_like(gamma, 1.8697680),
                                    torch.full_like(gamma, 1.9092084))
    V0_denom = 2.0 * gamma * (n - 1.0)
    factor = gamma * n - 2.0
    disc = (8.0 * (alpha - 1.0) * alpha * gamma * (n - 1.0)
            + (2.0 - gamma + alpha * factor) ** 2)
    disc = torch.clamp(disc, min=0.0)
    sq = torch.sqrt(disc)
    V0p = (2.0 - 2.0*alpha - gamma + alpha*gamma*n + sq) / V0_denom
    V0m = (2.0 - 2.0*alpha - gamma + alpha*gamma*n - sq) / V0_denom
    use_m = (gamma >= gamma_crit_t).float()
    V0 = use_m * V0m + (1.0 - use_m) * V0p
    C0 = (V0 - alpha) ** 2
    return V0, C0


def chisnell_rhs(V, C, alpha, gamma, n):
    delta = (V - alpha) ** 2 - C
    Q = (n * V * (V - alpha)
         + (2.0 / gamma) * (1.0 - alpha) * (alpha - V)
         - V * (V - 1.0))
    N = C * (2.0 * delta * (alpha - V + (1.0 - alpha) / gamma)
             + (gamma - 1.0) * (alpha - V) * Q)
    D = (delta * (n * V - 2.0 * (1.0 - alpha) / gamma) * (alpha - V)
         + (alpha - V) ** 2 * Q)
    return N, D


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ScopePINN(nn.Module):
    """
    参数化SCOPE: 纯自发现
    alpha_net(gamma, n) -> alpha  (靠ODE loss自发现，无预训练)
    C_net(s, gamma, n) -> correction
    """
    def __init__(self, n_hidden=64, n_layers=4):
        super().__init__()
        # alpha_net: (gamma, n) -> raw_alpha
        self.alpha_net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        # C_net: (s, gamma, n) -> correction
        layers = [nn.Linear(3, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 1))
        self.C_net = nn.Sequential(*layers)

        # 初始化: 根据几何给合理初始值（球面~0.71，柱面~0.83）
        # 在train_single里会根据n_geom覆盖这个值
        with torch.no_grad():
            nn.init.zeros_(self.alpha_net[-1].weight)
            self.alpha_net[-1].bias.data.fill_(-0.32)  # sigmoid(-0.32)*0.5+0.5 ≈ 0.71

    def get_alpha(self, gamma, n):
        """gamma, n: (B,1) tensors"""
        inp = torch.cat([gamma, n], dim=1)
        return 0.5 + 0.5 * torch.sigmoid(self.alpha_net(inp))

    def forward(self, s, gamma, n):
        alpha = self.get_alpha(gamma, n)
        Vs, Cs = shock_conditions(alpha, gamma)
        V0, C0 = critical_point(alpha, gamma, n)
        V = Vs + s * (V0 - Vs)
        inp = torch.cat([s, gamma, n], dim=1)
        correction = self.C_net(inp)
        C = (1.0 - s) * Cs + s * C0 + s * (1.0 - s) * correction
        return C, alpha, V, Vs, Cs, V0, C0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

SONIC_THRESH = 1e-3


def compute_loss(model, gamma_batch, n_batch, n_colloc=200):
    B = gamma_batch.shape[0]
    device = gamma_batch.device
    dtype = gamma_batch.dtype

    s_base = torch.linspace(0.01, 0.99, n_colloc, device=device, dtype=dtype)
    s = s_base.unsqueeze(0).expand(B, -1).reshape(-1, 1)
    g = gamma_batch.expand(-1, n_colloc).reshape(-1, 1)
    nv = n_batch.expand(-1, n_colloc).reshape(-1, 1)

    s_grad = s.detach().requires_grad_(True)
    C, alpha, V, Vs, Cs, V0, C0 = model(s_grad, g, nv)

    dC_ds = torch.autograd.grad(C, s_grad,
                                grad_outputs=torch.ones_like(C),
                                create_graph=True)[0]
    dV_ds = (V0 - Vs)
    dC_dV = dC_ds / (dV_ds + 1e-30)

    N, D = chisnell_rhs(V, C, alpha, g, nv)

    abs_D = torch.abs(D)
    w = torch.clamp(abs_D / SONIC_THRESH - 1.0, 0.0, 1.0)
    res_div = dC_dV - N / (D + 1e-30)
    res_mul = dC_dV * D - N
    residual = w * res_div + (1.0 - w) * res_mul
    loss_ode = torch.mean(residual ** 2)

    # 区间长度约束: 防止平凡解 α→1 (此时 |V0-Vs|→0)
    # 物理意义: 激波必须存在，相平面区间不能退化
    span = torch.abs(dV_ds)  # (B*n_colloc, 1)
    span_mean = span.mean()
    loss_span = torch.clamp(0.1 - span_mean, min=0.0) ** 2  # 惩罚 span < 0.1

    return loss_ode + 10.0 * loss_span


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_single(n_geom, gamma_val, epochs=10000, lr=1e-3, device="cuda", seed=42):
    """训练单个gamma值，纯自发现"""
    torch.manual_seed(seed)
    dtype = torch.float64
    model = ScopePINN().to(device).to(dtype)

    # 根据几何设置合理初始值: 球面α∈[0.67,0.76]用0.71, 柱面α∈[0.80,0.86]用0.83
    alpha_init = 0.71 if n_geom == 3 else 0.83
    raw_init = float(np.log((alpha_init - 0.5) / (0.5 - (alpha_init - 0.5) + 1e-8)))
    with torch.no_grad():
        model.alpha_net[-1].bias.data.fill_(raw_init)

    g_fixed = torch.tensor([[gamma_val]], dtype=dtype, device=device)
    n_fixed = torch.tensor([[float(n_geom)]], dtype=dtype, device=device)

    optimizer = torch.optim.Adam([
        {"params": model.C_net.parameters(),     "lr": lr},
        {"params": model.alpha_net.parameters(), "lr": lr * 0.5},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = compute_loss(model, g_fixed, n_fixed, n_colloc=200)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch % 2000 == 0 or epoch == 1:
            with torch.no_grad():
                a = float(model.get_alpha(g_fixed, n_fixed))
            print(f"  epoch {epoch:6d}  loss={loss.item():.4e}  α={a:.6f}")

    with torch.no_grad():
        alpha_final = float(model.get_alpha(g_fixed, n_fixed))
    return alpha_final

    optimizer = torch.optim.Adam([
        {"params": model.C_net.parameters(),     "lr": lr},
        {"params": model.alpha_net.parameters(), "lr": lr * 0.5},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    rng = np.random.default_rng(seed)

    g_fixed = torch.tensor([[gamma_val]], dtype=dtype, device=device)
    n_fixed = torch.tensor([[float(n_geom)]], dtype=dtype, device=device)

    for epoch in range(1, epochs + 1):
        # 每次只用这一个gamma，200个配点
        s = torch.tensor(rng.uniform(0.01, 0.99, (200, 1)), dtype=dtype, device=device)
        g_t = g_fixed.expand(200, 1)
        n_t = n_fixed.expand(200, 1)

        optimizer.zero_grad()
        loss = compute_loss(model, g_fixed, n_fixed, n_colloc=200)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 2000 == 0 or epoch == 1:
            with torch.no_grad():
                a = float(model.get_alpha(g_fixed, n_fixed))
            print(f"  epoch {epoch:6d}  loss={loss.item():.4e}  α={a:.6f}")

    with torch.no_grad():
        alpha_final = float(model.get_alpha(g_fixed, n_fixed))
    return alpha_final


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, gamma_min, gamma_max, device, n_ref=20):
    dtype = torch.float64
    model.eval()
    gammas = np.linspace(gamma_min, gamma_max, n_ref)
    results = {}
    print("\nEvaluating vs shooting reference...")
    for geo, nv in [("spherical", 3), ("cylindrical", 2)]:
        errs = []
        for g in gammas:
            ref = find_eigenvalue(g, nv, geo, verbose=False)
            if not ref["alpha"]:
                errs.append(np.nan); continue
            g_t = torch.tensor([[g]], dtype=dtype, device=device)
            n_t = torch.tensor([[float(nv)]], dtype=dtype, device=device)
            with torch.no_grad():
                a = float(model.get_alpha(g_t, n_t))
            errs.append(abs(a - ref["alpha"]) / ref["alpha"] * 100)
        errs = np.array(errs)
        results[geo] = errs
        print(f"  {geo}: Max={np.nanmax(errs):.4f}%  Mean={np.nanmean(errs):.4f}%")
    return gammas, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-min", type=float, default=1.2)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--epochs",    type=int,   default=15000)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cuda")
    parser.add_argument("--output-dir",type=str,
                        default="/share/project/zpy/PINN_WE/cases/self-discovery/canonical/output")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / f"scope_selfdiscovery_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"spherical": "steelblue", "cylindrical": "tomato"}
    dtype = torch.float64

    test_gammas = np.linspace(args.gamma_min, args.gamma_max, 10)

    for ax, (n_geom, geo) in zip(axes, [(3, "spherical"), (2, "cylindrical")]):
        print(f"\n{'='*50}\n{geo} (n={n_geom}): train one gamma at a time")
        pinn_alphas, ref_alphas, errs = [], [], []

        for g in test_gammas:
            ref = find_eigenvalue(g, n_geom, geo, verbose=False)
            if not ref["alpha"]:
                pinn_alphas.append(np.nan); ref_alphas.append(np.nan); errs.append(np.nan)
                continue
            print(f"\n  γ={g:.3f} (ref α={ref['alpha']:.6f})")
            alpha_pinn = train_single(n_geom, g, args.epochs, args.lr, args.device)
            err = abs(alpha_pinn - ref["alpha"]) / ref["alpha"] * 100
            print(f"  → α_pinn={alpha_pinn:.6f}  err={err:.4f}%")
            pinn_alphas.append(alpha_pinn); ref_alphas.append(ref["alpha"]); errs.append(err)

        errs = np.array(errs)
        mask = ~np.isnan(errs)
        print(f"\n  {geo}: Max={np.nanmax(errs):.4f}%  Mean={np.nanmean(errs):.4f}%")

        ax.semilogy(test_gammas[mask], errs[mask], 'o-', color=colors[geo], lw=2, ms=5)
        ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("α error (%)")
        ax.set_title(f"Self-discovery SCOPE: {geo}\nMax={np.nanmax(errs):.4f}%")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("α error (%)")
        ax.set_title(f"Self-discovery SCOPE: {geo}\nMax={np.nanmax(errs):.4f}%")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Parametric SCOPE — Pure Self-discovery (no pretraining)", fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / "scope_selfdiscovery.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
