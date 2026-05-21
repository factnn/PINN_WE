"""Shared utilities for 2D Advection-Diffusion experiments.
c_t + u0*c_x + v0*c_y = nu*(c_xx + c_yy)
Constant velocity field (u0, v0), periodic BC on [0, 2π]².
Exact solution: c = sin(x - u0*t) * sin(y - v0*t) * exp(-2*nu*t).
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
nu = 0.01
u0 = 1.0   # advection velocity x
v0 = 1.0   # advection velocity y
Nx, Ny, Nt = 64, 64, 20
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent.parent / "output" / "advdiff_2d"
OUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Network architectures
# ==========================================
class MLP(nn.Module):
    """Point-wise MLP: (x,y,t) -> c"""
    def __init__(self, width=128, depth=5):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_c = nn.Linear(width, 1)

    def forward(self, xyt):
        h = self.net(xyt)
        return self.out_c(h).squeeze(-1)


class PhyCNN(nn.Module):
    """Grid-based CNN: (X, Y, T) -> C"""
    def __init__(self, channels=32):
        super().__init__()
        # Input: 3 channels (X, Y, T coords)
        self.enc = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 1, kernel_size=5, padding=2),  # output C
        )

    def forward(self, X, Y, T):
        inputs = torch.stack([X, Y, T], dim=1)  # [Nt, 3, Nx, Ny]
        out = self.enc(inputs)
        return out[:, 0, :, :]  # [Nt, Nx, Ny]


# ==========================================
# 2. Grid & exact solution
# ==========================================
def make_grid():
    dx = 2.0 * np.pi / Nx
    dy = 2.0 * np.pi / Ny
    x = torch.arange(Nx, device=device) * dx
    y = torch.arange(Ny, device=device) * dy
    t = torch.linspace(0, 1, Nt, device=device)
    T, X, Y = torch.meshgrid(t, x, y, indexing='ij')
    dt = float(t[1] - t[0])
    return X, Y, T, dx, dy, dt


def exact_c(X, Y, T):
    """Exact solution: c = sin(x - u0*t) * sin(y - v0*t) * exp(-2*nu*t)"""
    return torch.sin(X - u0*T) * torch.sin(Y - v0*T) * torch.exp(-2*nu*T)


# ==========================================
# 3. PDE residual & losses
# ==========================================
def pde_residual_pytorch(C, dx, dy, dt):
    """c_t + u0*c_x + v0*c_y - nu*(c_xx + c_yy) via central differences.
    C: [Nt, Nx, Ny]
    """
    c_t  = (C[2:,1:-1,1:-1] - C[:-2,1:-1,1:-1]) / (2*dt)
    c_x  = (C[1:-1,2:,1:-1] - C[1:-1,:-2,1:-1]) / (2*dx)
    c_y  = (C[1:-1,1:-1,2:] - C[1:-1,1:-1,:-2]) / (2*dy)
    c_xx = (C[1:-1,2:,1:-1] - 2*C[1:-1,1:-1,1:-1] + C[1:-1,:-2,1:-1]) / dx**2
    c_yy = (C[1:-1,1:-1,2:] - 2*C[1:-1,1:-1,1:-1] + C[1:-1,1:-1,:-2]) / dy**2
    res = c_t + u0*c_x + v0*c_y - nu*(c_xx + c_yy)
    return (res**2).mean()


def ic_bc_loss(C, C_exact):
    """IC at t=0 + periodic BC."""
    ic = ((C[0] - C_exact[0])**2).mean()
    bc_x = ((C[:, 0, :] - C[:, -1, :])**2).mean()
    bc_y = ((C[:, :, 0] - C[:, :, -1])**2).mean()
    return ic + bc_x + bc_y


def unified_loss_fn(model, xyt, X, Y, T, C_exact, dx, dy, dt):
    C = infer(model, xyt, X, Y, T)
    return pde_residual_pytorch(C, dx, dy, dt) + ic_bc_loss(C, C_exact)


# ==========================================
# 4. Training & visualization
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=200000, lr=1e-3, loss_threshold=1e-5, runs=1):
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, Y, T, dx, dy, dt = make_grid()
    C_exact = exact_c(X, Y, T)
    xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; t0 = time.time()
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}x{Ny}x{Nt}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, xyt, X, Y, T, C_exact, dx, dy, dt)
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

    bytes_per_step = Nt * Nx * Ny * 4 * 10
    mem_bw_gbs = bytes_per_step / (avg_step_ms * 1e-3) / 1e9

    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep,
        avg_step_ms=avg_step_ms, mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT / f"meta_{backend_name}.npy", metrics)

    with torch.no_grad():
        C_pred = infer(model, xyt, X, Y, T)
    Ce = C_exact.cpu().numpy()
    Cp = C_pred.cpu().numpy()
    l2 = np.linalg.norm(Cp - Ce) / (np.linalg.norm(Ce) + 1e-12)
    _plot(Cp, Ce, X, Y, backend_name, l2)

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


def infer(model, xyt, X, Y, T):
    """Infer C from model (MLP or PhyCNN)."""
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        C = model(X, Y, T)
    else:
        c_f = model(xyt)
        C = c_f.reshape(Nt, Nx, Ny)
    return C


def base_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--max-epochs', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=1e-5)
    p.add_argument('--gpu', type=int, default=0)
    return p


def _plot(C, Ce, X, Y, name, l2):
    x_np = X[0, :, 0].cpu().numpy()
    y_np = Y[0, 0, :].cpu().numpy()
    ti = Nt // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].contourf(x_np, y_np, C[ti].T, levels=20, cmap='RdBu_r')
    plt.colorbar(im0, ax=axes[0]); axes[0].set_title(f'PINN c')
    im1 = axes[1].contourf(x_np, y_np, Ce[ti].T, levels=20, cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[1]); axes[1].set_title(f'Exact c')
    err = np.abs(C[ti] - Ce[ti])
    im2 = axes[2].contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
    plt.colorbar(im2, ax=axes[2]); axes[2].set_title(f'|Error|')

    plt.suptitle(f'AdvDiff 2D ({name}) | t={ti}/{Nt} | L2={l2:.2e}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / f"solution_{name}.png", dpi=150)
    plt.close()
