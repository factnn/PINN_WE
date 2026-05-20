#!/usr/bin/env python3
"""
Unified Parametric PINN: one network for both geometries.
Inputs: (s, gamma, n) where n in {2, 3} (cylindrical / spherical).
Outputs: alpha(gamma, n) and C(s; gamma, n).

123 framework same as pinn_parametric.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from guderley_ode_solver import find_eigenvalue


# ---------------------------------------------------------------------------
# Physics (identical to pinn_parametric.py)
# ---------------------------------------------------------------------------

def shock_conditions(alpha, gamma):
    Vs = 2.0 * alpha / (gamma + 1.0)
    Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
    return Vs, Cs


def critical_point(alpha, gamma, n):
    """n can be a float tensor (2.0 or 3.0)."""
    gamma_crit = torch.where(n >= 3.0,
                             torch.full_like(n, 1.8697680),
                             torch.full_like(n, 1.9092084))
    V0_denom = 2.0 * gamma * (n - 1.0)
    factor = gamma * n - 2.0
    disc = (8.0 * (alpha - 1.0) * alpha * gamma * (n - 1.0)
            + (2.0 - gamma + alpha * factor) ** 2)
    disc = torch.clamp(disc, min=0.0)
    sq = torch.sqrt(disc)
    V0p = (2.0 - 2.0*alpha - gamma + alpha*gamma*n + sq) / V0_denom
    V0m = (2.0 - 2.0*alpha - gamma + alpha*gamma*n - sq) / V0_denom
    use_m = (gamma >= gamma_crit).float()
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

class UnifiedParametricPINN(nn.Module):
    def __init__(self, n_hidden: int = 64, n_layers: int = 4):
        super().__init__()
        # alpha_net: (gamma, n) -> alpha in (0.5, 1.0)
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

    def get_alpha(self, gamma, n_t):
        """gamma, n_t: (B,1) -> alpha: (B,1)"""
        return 0.5 + 0.5 * torch.sigmoid(self.alpha_net(torch.cat([gamma, n_t], dim=1)))

    def forward(self, s, gamma, n_t):
        alpha = self.get_alpha(gamma, n_t)
        Vs, Cs = shock_conditions(alpha, gamma)
        V0, C0 = critical_point(alpha, gamma, n_t)
        V = Vs + s * (V0 - Vs)
        correction = self.C_net(torch.cat([s, gamma, n_t], dim=1))
        C = (1.0 - s) * Cs + s * C0 + s * (1.0 - s) * correction
        return C, alpha, V, Vs, Cs, V0, C0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

SONIC_THRESH = 1e-3


def compute_loss(model: UnifiedParametricPINN, gamma_batch: torch.Tensor,
                 n_batch: torch.Tensor, n_colloc: int = 200):
    B = gamma_batch.shape[0]
    device = gamma_batch.device
    dtype = gamma_batch.dtype

    s_base = torch.linspace(0.01, 0.99, n_colloc, device=device, dtype=dtype)
    s = s_base.unsqueeze(0).expand(B, -1).reshape(-1, 1)
    g = gamma_batch.expand(-1, n_colloc).reshape(-1, 1)
    n_t = n_batch.expand(-1, n_colloc).reshape(-1, 1)

    s_grad = s.detach().requires_grad_(True)
    C, alpha, V, Vs, Cs, V0, C0 = model(s_grad, g, n_t)

    dC_ds = torch.autograd.grad(C, s_grad, grad_outputs=torch.ones_like(C),
                                create_graph=True)[0]
    dV_ds = V0 - Vs
    dC_dV = dC_ds / (dV_ds + 1e-30)

    N, D = chisnell_rhs(V, C, alpha, g, n_t)

    abs_D = torch.abs(D)
    w = torch.clamp(abs_D / SONIC_THRESH - 1.0, 0.0, 1.0)
    residual = w * (dC_dV - N / (D + 1e-30)) + (1.0 - w) * (dC_dV * D - N)
    loss_ode = torch.mean(residual ** 2)
    loss_reg = torch.mean((abs_D < SONIC_THRESH).float() * N ** 2)

    return loss_ode + 10.0 * loss_reg


# ---------------------------------------------------------------------------
# Pretrain alpha_net
# ---------------------------------------------------------------------------

def pretrain_alpha_net(model: UnifiedParametricPINN,
                       gamma_min: float, gamma_max: float,
                       device: str, dtype, n_pts: int = 16, epochs: int = 3000):
    pairs = []  # (gamma, n_int, alpha)
    for n_int, geometry in [(3, "spherical"), (2, "cylindrical")]:
        print(f"  [pretrain] shooting for {n_pts} gamma values ({geometry})...")
        for g in np.linspace(gamma_min, gamma_max, n_pts):
            r = find_eigenvalue(g, n_int, geometry, verbose=False)
            if r["alpha"] is not None:
                pairs.append((g, float(n_int), r["alpha"]))
                print(f"    gamma={g:.3f}  n={n_int}  alpha={r['alpha']:.6f}")

    g_t = torch.tensor([p[0] for p in pairs], dtype=dtype, device=device).unsqueeze(1)
    n_t = torch.tensor([p[1] for p in pairs], dtype=dtype, device=device).unsqueeze(1)
    a_t = torch.tensor([p[2] for p in pairs], dtype=dtype, device=device).unsqueeze(1)

    opt = torch.optim.Adam(model.alpha_net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        ((model.get_alpha(g_t, n_t) - a_t) ** 2).mean().backward()
        opt.step()

    with torch.no_grad():
        err = (model.get_alpha(g_t, n_t) - a_t).abs().max().item()
    print(f"  [pretrain] done, max alpha err={err:.5f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(gamma_min: float = 1.2, gamma_max: float = 2.0,
          epochs: int = 10000, batch_size: int = 32, lr: float = 1e-3,
          device: str = "cpu", seed: int = 42) -> UnifiedParametricPINN:

    torch.manual_seed(seed)
    dtype = torch.float64
    model = UnifiedParametricPINN().to(device).to(dtype)

    pretrain_alpha_net(model, gamma_min, gamma_max, device, dtype)

    optimizer = torch.optim.Adam([
        {"params": model.C_net.parameters(),     "lr": lr},
        {"params": model.alpha_net.parameters(), "lr": lr * 0.2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        gamma_np = rng.uniform(gamma_min, gamma_max, size=(batch_size,))
        # randomly mix spherical (3) and cylindrical (2)
        n_np = rng.choice([2.0, 3.0], size=(batch_size,))
        gamma_t = torch.tensor(gamma_np, dtype=dtype, device=device).unsqueeze(1)
        n_t = torch.tensor(n_np, dtype=dtype, device=device).unsqueeze(1)

        optimizer.zero_grad()
        loss = compute_loss(model, gamma_t, n_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0 or epoch == 1:
            print(f"  epoch {epoch:6d}  loss={loss.item():.4e}")

    return model


# ---------------------------------------------------------------------------
# Validation & plot
# ---------------------------------------------------------------------------

def validate_and_plot(model: UnifiedParametricPINN,
                      gamma_min: float, gamma_max: float,
                      output_dir: Path, device: str = "cpu"):
    dtype = torch.float64
    model.eval()

    gammas = np.linspace(gamma_min, gamma_max, 100)
    g_t = torch.tensor(gammas, dtype=dtype, device=device).unsqueeze(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (n_int, geometry) in enumerate([(3, "spherical"), (2, "cylindrical")]):
        n_t = torch.full_like(g_t, float(n_int))
        with torch.no_grad():
            alpha_pinn = model.get_alpha(g_t, n_t).cpu().numpy().reshape(-1)

        ref_gammas = np.linspace(gamma_min, gamma_max, 20)
        ref_alphas = []
        for g in ref_gammas:
            r = find_eigenvalue(g, n_int, geometry, verbose=False)
            ref_alphas.append(r["alpha"] if r["alpha"] else np.nan)
        ref_alphas = np.array(ref_alphas)
        mask = ~np.isnan(ref_alphas)

        ax = axes[ax_idx]
        ax.plot(gammas, alpha_pinn, "b-", linewidth=2, label="PINN")
        ax.scatter(ref_gammas[mask], ref_alphas[mask], color="r", s=40,
                   zorder=5, label="shooting")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\alpha$")
        ax.set_title(f"α(γ) — {geometry}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        interp = np.interp(ref_gammas[mask], gammas, alpha_pinn)
        err_pct = np.abs(interp - ref_alphas[mask]) / ref_alphas[mask] * 100
        print(f"  {geometry}: Max error={np.nanmax(err_pct):.4f}%  Mean={np.nanmean(err_pct):.4f}%")

    fig.tight_layout()
    fname = output_dir / "pinn_parametric_unified.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma-min", type=float, default=1.2)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--epochs",     type=int,   default=10000)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--output-dir", type=str,
                        default="/share/project/zpy/PINN_WE/cases/self-discovery/canonical/output")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training unified parametric PINN")
    print(f"  gamma in [{args.gamma_min}, {args.gamma_max}], epochs={args.epochs}")
    model = train(gamma_min=args.gamma_min, gamma_max=args.gamma_max,
                  epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, device=args.device)
    validate_and_plot(model, args.gamma_min, args.gamma_max, out, device=args.device)
    print("\nDone.")


if __name__ == "__main__":
    main()
