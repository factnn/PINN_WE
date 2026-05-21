"""Shared utilities for 1D steady Burgers experiments.
u * u_x = nu * u_xx on [-1, 1], Dirichlet BC u(-1)=1, u(1)=-1.
Stationary shock at x=0. Exact solution: u(x) = -tanh(x / (2*nu)).
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
nu = 0.01 / np.pi
Nx = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent.parent / "output" / "burgers_1d_steady"
OUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Network architectures
# ==========================================
class MLP(nn.Module):
    """Point-wise MLP: x -> u"""
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)

    def forward(self, x):
        return self.out_u(self.net(x)).squeeze(-1)


class PhyCNN(nn.Module):
    """Grid-based 1D CNN: X -> U"""
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, X):
        # X: [Nx]
        inp = X.unsqueeze(0).unsqueeze(0)  # [1, 1, Nx]
        out = self.enc(inp)  # [1, 1, Nx]
        return out.squeeze(0).squeeze(0)  # [Nx]


# ==========================================
# 2. Grid & exact solution
# ==========================================
def make_grid():
    x = torch.linspace(-1, 1, Nx, device=device)
    dx = float(x[1] - x[0])
    return x, dx


def exact_u(x):
    """Exact solution: u(x) = -tanh(x / (2*nu)).
    Satisfies u*u_x = nu*u_xx exactly, with u(-1)≈1, u(1)≈-1.
    """
    return -torch.tanh(x / (2.0 * nu))


# ==========================================
# 3. PDE residual & losses
# ==========================================
def pde_residual_pytorch(U, dx):
    """Steady Burgers: u*u_x = nu*u_xx via central differences.
    U: [Nx]. Only interior points contribute.
    """
    u_x = (U[2:] - U[:-2]) / (2*dx)
    u_xx = (U[2:] - 2*U[1:-1] + U[:-2]) / dx**2
    uc = U[1:-1]
    res = uc * u_x - nu * u_xx
    return (res**2).mean()


def bc_loss(U):
    """Dirichlet BC: u(-1) = 1, u(1) = -1."""
    return (U[0] - 1.0)**2 + (U[-1] + 1.0)**2


def unified_loss_fn(model, x_inp, X, dx):
    U = infer(model, x_inp, X)
    return pde_residual_pytorch(U, dx) + 10 * bc_loss(U)


# ==========================================
# 4. Inference helper
# ==========================================
def infer(model, x_inp, X):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        U = model(X)
    else:
        U = model(x_inp)
    return U


# ==========================================
# 5. Training & visualization
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=200000, lr=1e-3, loss_threshold=1e-5, runs=1):
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, dx = make_grid()
    x_inp = X.unsqueeze(-1)  # [Nx, 1] for MLP

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; t0 = time.time()
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, x_inp, X, dx)
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

    bytes_per_step = Nx * 4 * 6
    mem_bw_gbs = bytes_per_step / (avg_step_ms * 1e-3) / 1e9

    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep,
        avg_step_ms=avg_step_ms, mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT / f"meta_{backend_name}.npy", metrics)

    with torch.no_grad():
        U_pred = infer(model, x_inp, X).cpu().numpy()
    x_np = X.cpu().numpy()
    U_exact = exact_u(X).cpu().numpy()
    l2 = np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12)
    _plot(U_pred, U_exact, x_np, backend_name, l2)

    print(f"\n[{backend_name}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem:.3f}")
    print(f"  Mem_BW_GBs   : {mem_bw_gbs:.1f}")
    print(f"  L2_err       : {l2:.4e}")
    return dict(elapsed=np.median(all_elapsed), mem_gb=mem, t2s=t2s,
                t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
                mem_bw_gbs=mem_bw_gbs, l2=l2, history=history)


def _plot(U, U_exact, x, name, l2):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, U, 'b-', lw=2, label='PINN')
    ax.plot(x, U_exact, 'r--', lw=2, label='Exact')
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.set_title(f'Steady Burgers ({name}) | L2={l2:.2e}')
    ax.legend(); ax.grid(True, alpha=0.3)
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
