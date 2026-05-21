"""Shared utilities for 2D compressible Euler experiments.
Sod shock tube: 1D Riemann problem solved on 2D grid (y-invariant).
Euler equations: ρ_t + (ρu)_x = 0, (ρu)_t + (ρu²+p)_x = 0, E_t + ((E+p)u)_x = 0
EOS: p = (γ-1)(E - 0.5*ρ*u²), γ = 1.4

Initial conditions (Sod):
  Left  (x < 0.5): ρ=1.0, u=0, p=1.0
  Right (x > 0.5): ρ=0.125, u=0, p=0.1
"""
import sys, time, argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Physics & grid ---
gamma = 1.4
Nx, Nt = 200, 50   # fine x, coarse t (1D-like on 2D)
T_final = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent.parent / "output" / "euler_2d"
OUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Network architectures
# ==========================================
class MLP(nn.Module):
    """Point-wise MLP: (x, t) -> (ρ, ρu, E)"""
    def __init__(self, width=256, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_rho = nn.Linear(width, 1)
        self.out_rhou = nn.Linear(width, 1)
        self.out_E = nn.Linear(width, 1)

    def forward(self, xt):
        h = self.net(xt)
        rho = torch.abs(self.out_rho(h).squeeze(-1)) + 1e-6  # ρ > 0
        rhou = self.out_rhou(h).squeeze(-1)
        E = torch.abs(self.out_E(h).squeeze(-1)) + 1e-6       # E > 0
        return rho, rhou, E


class PhyCNN(nn.Module):
    """Grid-based 1D CNN: (X, T) -> (ρ, ρu, E) on [Nt, Nx]"""
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, 3, kernel_size=5, padding=2),  # ρ, ρu, E
        )

    def forward(self, X, T):
        # X, T: [Nt, Nx]
        inputs = torch.stack([X, T], dim=1)  # [Nt, 2, Nx]
        out = self.enc(inputs)  # [Nt, 3, Nx]
        rho = torch.abs(out[:, 0, :]) + 1e-6
        rhou = out[:, 1, :]
        E = torch.abs(out[:, 2, :]) + 1e-6
        return rho, rhou, E


# ==========================================
# 2. Grid & Sod exact solution
# ==========================================
def make_grid():
    x = torch.linspace(0, 1, Nx, device=device)
    t = torch.linspace(0, T_final, Nt, device=device)
    T, X = torch.meshgrid(t, x, indexing='ij')  # [Nt, Nx]
    dx = float(x[1] - x[0]); dt = float(t[1] - t[0])
    return X, T, dx, dt


def sod_ic(X):
    """Sod shock tube initial conditions."""
    rho = torch.where(X < 0.5, torch.ones_like(X), 0.125 * torch.ones_like(X))
    u = torch.zeros_like(X)
    p = torch.where(X < 0.5, torch.ones_like(X), 0.1 * torch.ones_like(X))
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return rho, rho * u, E


# ==========================================
# 3. PDE residual & losses
# ==========================================
def euler_residual_pytorch(rho, rhou, E, dx, dt):
    """1D Euler residual via central differences on [Nt, Nx] grid.
    ρ_t + (ρu)_x = 0
    (ρu)_t + (ρu²+p)_x = 0
    E_t + ((E+p)u)_x = 0
    """
    u = rhou / (rho + 1e-10)
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    # fluxes
    f1 = rhou                     # ρu
    f2 = rhou * u + p             # ρu² + p
    f3 = (E + p) * u              # (E+p)u
    # time derivatives (interior in t)
    rho_t = (rho[2:, 1:-1] - rho[:-2, 1:-1]) / (2*dt)
    rhou_t = (rhou[2:, 1:-1] - rhou[:-2, 1:-1]) / (2*dt)
    E_t = (E[2:, 1:-1] - E[:-2, 1:-1]) / (2*dt)
    # spatial derivatives (interior in x)
    f1_x = (f1[1:-1, 2:] - f1[1:-1, :-2]) / (2*dx)
    f2_x = (f2[1:-1, 2:] - f2[1:-1, :-2]) / (2*dx)
    f3_x = (f3[1:-1, 2:] - f3[1:-1, :-2]) / (2*dx)
    res1 = rho_t + f1_x
    res2 = rhou_t + f2_x
    res3 = E_t + f3_x
    return (res1**2 + res2**2 + res3**2).mean()


def ic_loss(rho, rhou, E, X):
    """Initial condition loss at t=0."""
    rho_ic, rhou_ic, E_ic = sod_ic(X[0])
    return ((rho[0] - rho_ic)**2).mean() + \
           ((rhou[0] - rhou_ic)**2).mean() + \
           ((E[0] - E_ic)**2).mean()


def bc_loss_euler(rho, rhou, E):
    """Outflow BC: zero-gradient at x boundaries."""
    bc = ((rho[:, 0] - rho[:, 1])**2 + (rho[:, -1] - rho[:, -2])**2).mean()
    bc += ((rhou[:, 0] - rhou[:, 1])**2 + (rhou[:, -1] - rhou[:, -2])**2).mean()
    bc += ((E[:, 0] - E[:, 1])**2 + (E[:, -1] - E[:, -2])**2).mean()
    return bc


def unified_loss_fn(model, xt, X, T, dx, dt):
    rho, rhou, E = infer(model, xt, X, T)
    return euler_residual_pytorch(rho, rhou, E, dx, dt) + \
           10 * ic_loss(rho, rhou, E, X) + bc_loss_euler(rho, rhou, E)


# ==========================================
# 4. Inference helper
# ==========================================
def infer(model, xt, X, T):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        rho, rhou, E = model(X, T)
    else:
        rho, rhou, E = model(xt)
        rho = rho.reshape(Nt, Nx)
        rhou = rhou.reshape(Nt, Nx)
        E = E.reshape(Nt, Nx)
    return rho, rhou, E


# ==========================================
# 5. Training & visualization
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=200000, lr=1e-3, loss_threshold=1e-5, runs=1):
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, T, dx, dt = make_grid()
    xt = torch.stack([X.flatten(), T.flatten()], dim=1)

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; t0 = time.time()
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}x{Nt}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, xt, X, T, dx, dt)
            loss.backward(); opt.step(); sch.step()
            torch.cuda.synchronize()
            step_times.append(time.time() - t_step)

            wall = time.time() - t0; lv = loss.item()
            history.append((wall, ep, lv))
            if t2s is None and lv < loss_threshold:
                t2s = wall; t2s_ep = ep
                print(f"  [{backend_name}] run {run_i+1}: T2S={wall:.1f}s at ep {ep}")
                break
            if ep % 5000 == 0:
                print(f"[{backend_name}] run {run_i+1} ep {ep:5d} loss={lv:.3e} t={wall:.1f}s")

        elapsed = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9
        avg_step_ms = np.mean(step_times) * 1000
        all_elapsed.append(elapsed); all_t2s.append(t2s)
        print(f"[{backend_name}] run {run_i+1}: {elapsed:.1f}s  mem={mem:.3f}GB  avg_step={avg_step_ms:.2f}ms  epochs={t2s_ep or ep}")

    ckpt = OUT / f"model_{backend_name}.pt"
    torch.save(getattr(model, '_orig_mod', model).state_dict(), ckpt)

    bytes_per_step = 3 * Nt * Nx * 4 * 10
    mem_bw_gbs = bytes_per_step / (avg_step_ms * 1e-3) / 1e9

    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep,
        avg_step_ms=avg_step_ms, mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT / f"meta_{backend_name}.npy", metrics)

    with torch.no_grad():
        rho, rhou, E = infer(model, xt, X, T)
    _plot(rho.cpu().numpy(), rhou.cpu().numpy(), E.cpu().numpy(),
          X.cpu().numpy(), backend_name)

    print(f"\n[{backend_name}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem:.3f}")
    print(f"  Mem_BW_GBs   : {mem_bw_gbs:.1f}")
    return dict(elapsed=np.median(all_elapsed), mem_gb=mem, t2s=t2s,
                t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
                mem_bw_gbs=mem_bw_gbs, history=history)


def _plot(rho, rhou, E, X, name):
    """Plot density, velocity, pressure at final time."""
    x = X[0]
    u = rhou / (rho + 1e-10)
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    ti = -1  # final time

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x, rho[ti], 'b-', lw=2); axes[0].set_title('Density ρ'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, u[ti], 'r-', lw=2); axes[1].set_title('Velocity u'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(x, p[ti], 'g-', lw=2); axes[2].set_title('Pressure p'); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'Sod Shock Tube ({name}) | t={T_final}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / f"solution_{name}.png", dpi=150)
    plt.close()


def base_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--max-epochs', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=1e-5)
    p.add_argument('--gpu', type=int, default=0)
    return p
