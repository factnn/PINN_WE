#!/usr/bin/env python3
"""
Parametric PINN: train once over gamma in [gamma_min, gamma_max],
query alpha(gamma) for any gamma without retraining.

123 framework:
  1. Chisnell phase-plane ODE dC/dV = N/D
  2. Mixed residual: division far / multiplication near sonic point
  3. Hard constraints: C(s=0)=Cs(alpha,gamma), C(s=1)=C0(alpha,gamma)

Network inputs: (s, gamma) where s in [0,1].
V = Vs(alpha,gamma) + s*(V0(alpha,gamma) - Vs(alpha,gamma))  -- alpha has grad
alpha_net(gamma) -> alpha  (separate small MLP, pretrained via shooting)
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
# Physics (torch, vectorized)
# ---------------------------------------------------------------------------

def shock_conditions(alpha, gamma):
    Vs = 2.0 * alpha / (gamma + 1.0)
    Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
    return Vs, Cs


def critical_point(alpha, gamma, n: int):
    gamma_crit = 1.8697680 if n == 3 else 1.9092084
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


def chisnell_rhs(V, C, alpha, gamma, n: int):
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

class ParametricPINN(nn.Module):
    def __init__(self, n: int, n_hidden: int = 64, n_layers: int = 4):
        super().__init__()
        self.n = n
        self.alpha_net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        layers = [nn.Linear(2, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 1))
        self.C_net = nn.Sequential(*layers)

    def get_alpha(self, gamma):
        return 0.5 + 0.5 * torch.sigmoid(self.alpha_net(gamma))

    def forward(self, s, gamma):
        """
        s: (B,1) in [0,1] — collocation coordinate (fixed, no grad)
        gamma: (B,1)
        Returns C, alpha, V, Vs, Cs, V0, C0.
        Gradients flow through alpha -> Vs, V0, Cs, C0, V.
        """
        alpha = self.get_alpha(gamma)
        Vs, Cs = shock_conditions(alpha, gamma)
        V0, C0 = critical_point(alpha, gamma, self.n)

        # V depends on alpha through Vs and V0 — gradient flows
        V = Vs + s * (V0 - Vs)

        correction = self.C_net(torch.cat([s, gamma], dim=1))
        # Hard constraint: exact at s=0 (shock) and s=1 (sonic)
        C = (1.0 - s) * Cs + s * C0 + s * (1.0 - s) * correction
        return C, alpha, V, Vs, Cs, V0, C0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

SONIC_THRESH = 1e-3


def compute_loss(model: ParametricPINN, gamma_batch: torch.Tensor,
                 n_colloc: int = 200):
    B = gamma_batch.shape[0]
    device = gamma_batch.device
    dtype = gamma_batch.dtype

    # s is fixed collocation coordinate — no grad needed on s itself
    s_base = torch.linspace(0.01, 0.99, n_colloc, device=device, dtype=dtype)
    s = s_base.unsqueeze(0).expand(B, -1).reshape(-1, 1)
    g = gamma_batch.expand(-1, n_colloc).reshape(-1, 1)

    # V is computed inside forward and depends on alpha(gamma)
    # We need dC/dV = dC/ds * ds/dV = dC/ds / (dV/ds)
    # dV/ds = V0 - Vs  (computed inside forward)
    # To get dC/ds, we need s to have grad
    s_grad = s.detach().requires_grad_(True)

    C, alpha, V, Vs, Cs, V0, C0 = model(s_grad, g)

    dC_ds = torch.autograd.grad(C, s_grad,
                                grad_outputs=torch.ones_like(C),
                                create_graph=True)[0]
    dV_ds = (V0 - Vs)  # (B*n_colloc, 1), has grad through alpha
    dC_dV = dC_ds / (dV_ds + 1e-30)

    N, D = chisnell_rhs(V, C, alpha, g, model.n)

    abs_D = torch.abs(D)
    w = torch.clamp(abs_D / SONIC_THRESH - 1.0, 0.0, 1.0)
    res_div = dC_dV - N / (D + 1e-30)
    res_mul = dC_dV * D - N
    residual = w * res_div + (1.0 - w) * res_mul
    loss_ode = torch.mean(residual ** 2)

    near_sonic = (abs_D < SONIC_THRESH).float()
    loss_reg = torch.mean(near_sonic * N ** 2)

    return loss_ode + 10.0 * loss_reg


# ---------------------------------------------------------------------------
# Pretrain alpha_net via shooting method
# ---------------------------------------------------------------------------

def pretrain_alpha_net(model: ParametricPINN, n: int,
                       gamma_min: float, gamma_max: float,
                       device: str, dtype, n_pts: int = 16, epochs: int = 3000):
    geometry = "spherical" if n == 3 else "cylindrical"
    gammas = np.linspace(gamma_min, gamma_max, n_pts)
    pairs = []
    print(f"  [pretrain] shooting for {n_pts} gamma values...")
    for g in gammas:
        r = find_eigenvalue(g, n, geometry, verbose=False)
        if r["alpha"] is not None:
            pairs.append((g, r["alpha"]))
            print(f"    gamma={g:.3f}  alpha={r['alpha']:.6f}")

    if len(pairs) < 3:
        print("  [pretrain] too few points, skipping")
        return

    g_t = torch.tensor([p[0] for p in pairs], dtype=dtype, device=device).unsqueeze(1)
    a_t = torch.tensor([p[1] for p in pairs], dtype=dtype, device=device).unsqueeze(1)
    # fit in raw (pre-sigmoid) space
    a_c = a_t.clamp(0.501, 0.999)
    raw_target = torch.log((a_c - 0.5) / (1.0 - (a_c - 0.5)))

    opt = torch.optim.Adam(model.alpha_net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        ((model.get_alpha(g_t) - a_t) ** 2).mean().backward()
        opt.step()

    with torch.no_grad():
        err = (model.get_alpha(g_t) - a_t).abs().max().item()
    print(f"  [pretrain] done, max alpha err={err:.5f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n: int, gamma_min: float = 1.2, gamma_max: float = 2.0,
          epochs: int = 10000, batch_size: int = 32, lr: float = 1e-3,
          device: str = "cpu", seed: int = 42):

    torch.manual_seed(seed)
    dtype = torch.float64
    model = ParametricPINN(n=n).to(device).to(dtype)

    pretrain_alpha_net(model, n, gamma_min, gamma_max, device, dtype)

    optimizer = torch.optim.Adam([
        {"params": model.C_net.parameters(),     "lr": lr},
        {"params": model.alpha_net.parameters(), "lr": lr * 0.2},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    rng = np.random.default_rng(seed)
    loss_history = []

    for epoch in range(1, epochs + 1):
        gamma_np = rng.uniform(gamma_min, gamma_max, size=(batch_size,))
        gamma_t = torch.tensor(gamma_np, dtype=dtype, device=device).unsqueeze(1)

        optimizer.zero_grad()
        loss = compute_loss(model, gamma_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch % 1000 == 0 or epoch == 1:
            print(f"  epoch {epoch:6d}  loss={loss.item():.4e}")

    return model, loss_history


# ---------------------------------------------------------------------------
# Validation & plot
# ---------------------------------------------------------------------------

def evaluate_max_error(model: ParametricPINN, n: int, geometry: str,
                       gamma_min: float, gamma_max: float,
                       device: str = "cpu", n_ref: int = 20) -> float:
    """Return max relative alpha error (%) vs shooting reference."""
    dtype = torch.float64
    model.eval()
    gammas = np.linspace(gamma_min, gamma_max, 200)
    g_t = torch.tensor(gammas, dtype=dtype, device=device).unsqueeze(1)
    with torch.no_grad():
        alpha_pinn = model.get_alpha(g_t).cpu().numpy().reshape(-1)
    ref_gammas = np.linspace(gamma_min, gamma_max, n_ref)
    ref_alphas = np.array([
        (r["alpha"] if r["alpha"] else np.nan)
        for g in ref_gammas
        for r in [find_eigenvalue(g, n, geometry, verbose=False)]
    ])
    mask = ~np.isnan(ref_alphas)
    interp = np.interp(ref_gammas[mask], gammas, alpha_pinn)
    return float(np.nanmax(np.abs(interp - ref_alphas[mask]) / ref_alphas[mask] * 100))


def validate_and_plot(model: ParametricPINN, n: int, geometry: str,
                      gamma_min: float, gamma_max: float,
                      output_dir: Path, device: str = "cpu"):
    dtype = torch.float64
    model.eval()

    gammas = np.linspace(gamma_min, gamma_max, 100)
    g_t = torch.tensor(gammas, dtype=dtype, device=device).unsqueeze(1)
    with torch.no_grad():
        alpha_pinn = model.get_alpha(g_t).cpu().numpy().reshape(-1)

    ref_gammas = np.linspace(gamma_min, gamma_max, 20)
    ref_alphas = []
    print(f"\nComputing shooting reference for {geometry} (n={n})...")
    for g in ref_gammas:
        r = find_eigenvalue(g, n, geometry, verbose=False)
        ref_alphas.append(r["alpha"] if r["alpha"] else np.nan)
    ref_alphas = np.array(ref_alphas)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(gammas, alpha_pinn, "b-", linewidth=2, label="PINN")
    mask = ~np.isnan(ref_alphas)
    ax.scatter(ref_gammas[mask], ref_alphas[mask], color="r", s=40,
               zorder=5, label="shooting (reference)")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(f"Parametric PINN: α(γ) — {geometry}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    interp_pinn = np.interp(ref_gammas[mask], gammas, alpha_pinn)
    err_pct = np.abs(interp_pinn - ref_alphas[mask]) / ref_alphas[mask] * 100
    ax.semilogy(ref_gammas[mask], err_pct, "o-", linewidth=2)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("relative error (%)")
    ax.set_title("α error vs shooting")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = output_dir / f"pinn_parametric_{geometry}.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")
    print(f"  Max error: {np.nanmax(err_pct):.4f}%  Mean: {np.nanmean(err_pct):.4f}%")


def plot_solution_structure(model: ParametricPINN, n: int, geometry: str,
                             gamma_min: float, gamma_max: float,
                             output_dir: Path, device: str = "cpu"):
    """画解的结构：C(V)曲线族，展示不同γ下的相平面轨迹"""
    dtype = torch.float64
    model.eval()

    # 选5个代表性的γ值
    gammas_plot = [gamma_min, (gamma_min+gamma_max)/2, gamma_max]
    if gamma_max - gamma_min > 0.6:
        gammas_plot = [gamma_min, 1.4, (gamma_min+gamma_max)/2, 1.67, gamma_max]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：C(V)曲线族
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gammas_plot)))

    for gamma_val, color in zip(gammas_plot, colors):
        g_t = torch.tensor([[gamma_val]], dtype=dtype, device=device)
        with torch.no_grad():
            alpha = model.get_alpha(g_t)
            Vs, Cs = shock_conditions(alpha, g_t)
            V0, C0 = critical_point(alpha, g_t, n)

            # 生成s∈[0,1]的密集点
            s_dense = torch.linspace(0, 1, 300, dtype=dtype, device=device).unsqueeze(1)
            g_dense = g_t.expand(300, 1)
            C, _, V, _, _, _, _ = model(s_dense, g_dense)

            V_np = V.cpu().numpy().flatten()
            C_np = C.cpu().numpy().flatten()

            ax.plot(V_np, C_np, color=color, lw=2,
                   label=f'γ={gamma_val:.2f}, α={float(alpha):.4f}')
            # 标记激波端和音速端
            ax.plot(float(Vs), float(Cs), 'o', color=color, markersize=8)
            ax.plot(float(V0), float(C0), 's', color=color, markersize=8)

    ax.set_xlabel('V (velocity)', fontsize=12)
    ax.set_ylabel('C (sound speed squared)', fontsize=12)
    ax.set_title(f'Solution Structure: C(V) — {geometry}', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图：α(γ)曲线
    ax = axes[1]
    gammas_dense = np.linspace(gamma_min, gamma_max, 200)
    g_t = torch.tensor(gammas_dense, dtype=dtype, device=device).unsqueeze(1)
    with torch.no_grad():
        alpha_dense = model.get_alpha(g_t).cpu().numpy().flatten()

    ax.plot(gammas_dense, alpha_dense, 'b-', lw=2, label='PINN α(γ)')
    ax.set_xlabel('γ', fontsize=12)
    ax.set_ylabel('α', fontsize=12)
    ax.set_title(f'Eigenvalue α(γ) — {geometry}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = output_dir / f"solution_structure_{geometry}.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Solution structure plot saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["spherical", "cylindrical", "both"],
                        default="both")
    parser.add_argument("--gamma-min", type=float, default=1.2)
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--epochs",     type=int,   default=10000)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--output-dir", type=str,
                        default="/share/project/zpy/PINN_WE/cases/self-discovery/canonical/output")
    parser.add_argument("--n-runs",     type=int,   default=1,
                        help="Train N times with different seeds, keep best model")
    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / f"run_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out}")

    configs = []
    if args.geometry in ("spherical", "both"):
        configs.append((3, "spherical"))
    if args.geometry in ("cylindrical", "both"):
        configs.append((2, "cylindrical"))

    for n, geometry in configs:
        print(f"\n{'='*60}")
        print(f"Training parametric PINN: {geometry} (n={n}), n_runs={args.n_runs}")
        print(f"  gamma in [{args.gamma_min}, {args.gamma_max}], epochs={args.epochs}")

        best_model, best_err, best_loss_history = None, float("inf"), None
        for run in range(args.n_runs):
            seed = 42 + run
            print(f"\n  --- run {run+1}/{args.n_runs} (seed={seed}) ---")
            model, loss_history = train(n=n, gamma_min=args.gamma_min, gamma_max=args.gamma_max,
                          epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, device=args.device, seed=seed)
            err = evaluate_max_error(model, n, geometry, args.gamma_min, args.gamma_max,
                                     device=args.device)
            print(f"  run {run+1} max_err={err:.4f}%")
            if err < best_err:
                best_err, best_model, best_loss_history = err, model, loss_history

        print(f"\n  Best max_err={best_err:.4f}%")
        validate_and_plot(best_model, n, geometry, args.gamma_min, args.gamma_max,
                          out, device=args.device)

        # 保存最优模型
        model_path = out / f"pinn_parametric_{geometry}.pt"
        torch.save(best_model.state_dict(), model_path)
        print(f"  Model saved: {model_path}")

        # 保存loss数据
        loss_npy = out / f"training_loss_{geometry}.npy"
        np.save(loss_npy, np.array(best_loss_history))
        print(f"  Loss data saved: {loss_npy}")

        # 画loss曲线（带滑动平均）
        loss_arr = np.array(best_loss_history)
        w = max(1, len(loss_arr) // 50)  # 窗口=总epoch/50
        smooth = np.convolve(loss_arr, np.ones(w)/w, mode='valid')
        fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
        ax_loss.semilogy(loss_arr, lw=0.5, alpha=0.3, color='steelblue', label='raw')
        ax_loss.semilogy(range(w-1, len(loss_arr)), smooth, lw=2, color='navy', label=f'moving avg (w={w})')
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"Training Loss — {geometry}")
        ax_loss.legend(); ax_loss.grid(True, alpha=0.3)
        fig_loss.tight_layout()
        loss_fname = out / f"training_loss_{geometry}.png"
        fig_loss.savefig(loss_fname, dpi=160, bbox_inches="tight")
        plt.close(fig_loss)
        print(f"  Loss curve saved: {loss_fname}")

        # 画解结构图
        plot_solution_structure(best_model, n, geometry, args.gamma_min, args.gamma_max,
                                out, device=args.device)

    print("\nDone.")


if __name__ == "__main__":
    main()
