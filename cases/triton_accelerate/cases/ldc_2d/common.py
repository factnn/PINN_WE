"""Shared utilities for 2D Lid-Driven Cavity (LDC) experiments.
Steady-state NS, Re=100. Grid [Nx, Ny], no time dimension.
Reference: Ghia et al. 1982 centerline velocities.
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
Re = 100.0
nu = 1.0 / Re
Nx, Ny = 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent.parent / "output" / "ldc_2d"
OUT.mkdir(parents=True, exist_ok=True)

# Ghia et al. 1982 reference data (Re=100, centerline u at x=0.5)
GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
           0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
GHIA_U = [0.0000,-0.0372,-0.0419,-0.0477,-0.0643,-0.1015,-0.1566,-0.2109,
          -0.2058,-0.1364, 0.0033, 0.2315, 0.6872, 0.7372, 0.7887, 0.8412, 1.0000]


# ==========================================
# 1. Network architectures
# ==========================================
class MLP(nn.Module):
    def __init__(self, width=128, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)
        self.out_v = nn.Linear(width, 1)
        self.out_p = nn.Linear(width, 1)

    def forward(self, xy):
        h = self.net(xy)
        return self.out_u(h).squeeze(-1), self.out_v(h).squeeze(-1), self.out_p(h).squeeze(-1)


class PhyCNN(nn.Module):
    """Grid-based CNN for steady 2D flow."""
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 3, kernel_size=5, padding=2),  # output U, V, P
        )

    def forward(self, X, Y):
        inputs = torch.stack([X, Y], dim=0).unsqueeze(0)  # [1, 2, Nx, Ny]
        out = self.enc(inputs).squeeze(0)  # [3, Nx, Ny]
        return out[0], out[1], out[2]


# ==========================================
# 2. Grid & physics
# ==========================================
def make_grid():
    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    return X, Y, dx, dy


def pde_residual_pytorch(U, V, P, dx, dy):
    """Steady incompressible NS:
    res_u = u*u_x + v*u_y + p_x - nu*(u_xx+u_yy)
    res_v = u*v_x + v*v_y + p_y - nu*(v_xx+v_yy)
    res_div = u_x + v_y
    """
    u_x  = (U[2:,1:-1] - U[:-2,1:-1]) / (2*dx)
    u_y  = (U[1:-1,2:] - U[1:-1,:-2]) / (2*dy)
    u_xx = (U[2:,1:-1] - 2*U[1:-1,1:-1] + U[:-2,1:-1]) / dx**2
    u_yy = (U[1:-1,2:] - 2*U[1:-1,1:-1] + U[1:-1,:-2]) / dy**2
    v_x  = (V[2:,1:-1] - V[:-2,1:-1]) / (2*dx)
    v_y  = (V[1:-1,2:] - V[1:-1,:-2]) / (2*dy)
    v_xx = (V[2:,1:-1] - 2*V[1:-1,1:-1] + V[:-2,1:-1]) / dx**2
    v_yy = (V[1:-1,2:] - 2*V[1:-1,1:-1] + V[1:-1,:-2]) / dy**2
    p_x  = (P[2:,1:-1] - P[:-2,1:-1]) / (2*dx)
    p_y  = (P[1:-1,2:] - P[1:-1,:-2]) / (2*dy)
    u_c = U[1:-1,1:-1]; v_c = V[1:-1,1:-1]
    res_u = u_c*u_x + v_c*u_y + p_x - nu*(u_xx+u_yy)
    res_v = u_c*v_x + v_c*v_y + p_y - nu*(v_xx+v_yy)
    res_div = u_x + v_y
    return (res_u**2 + res_v**2 + res_div**2).mean()


def bc_loss(U, V):
    """Lid: top u=1,v=0; walls u=v=0."""
    top   = (U[:,-1] - 1)**2 + V[:,-1]**2
    bot   = U[:,0]**2 + V[:,0]**2
    left  = U[0,:]**2 + V[0,:]**2
    right = U[-1,:]**2 + V[-1,:]**2
    return top.mean() + bot.mean() + left.mean() + right.mean()


def unified_loss_fn(model, xy, X, Y, dx, dy):
    U, V, P = infer(model, xy, X, Y)
    return pde_residual_pytorch(U, V, P, dx, dy) + 10 * bc_loss(U, V)


def unified_loss_fn_triton(model, xy, X, Y, dx, dy):
    from kernels.stencil_2d_burgers_steady import ldc_residual_triton
    U, V, P = infer(model, xy, X, Y)
    return ldc_residual_triton(U, V, P, dx, dy, nu) + 10 * bc_loss(U, V)


# ==========================================
# 3. Inference helper
# ==========================================
def infer(model, xy, X, Y):
    """Infer U, V, P from model regardless of type (MLP or PhyCNN)."""
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        U, V, P = model(X, Y)
    else:
        u_f, v_f, p_f = model(xy)
        U = u_f.reshape(Nx, Ny)
        V = v_f.reshape(Nx, Ny)
        P = p_f.reshape(Nx, Ny)
    return U, V, P


# ==========================================
# 4. Training & visualization
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=200000, lr=1e-3, loss_threshold=1e-5, runs=1):
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, Y, dx, dy = make_grid()
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; t0 = time.time()
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}x{Ny}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, xy, X, Y, dx, dy)
            loss.backward(); opt.step(); sch.step()
            torch.cuda.synchronize()
            step_times.append(time.time() - t_step)

            wall = time.time() - t0; lv = loss.item()
            history.append((wall, ep, lv))
            if t2s is None and lv < loss_threshold:
                t2s = wall; t2s_ep = ep
                print(f"  [{backend_name}] run {run_i+1}: T2S={wall:.1f}s at ep {ep}")
                break
            if ep % 1000 == 0:
                print(f"[{backend_name}] run {run_i+1} ep {ep:5d} loss={lv:.3e} t={wall:.1f}s")

        elapsed = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9
        avg_step_ms = np.mean(step_times) * 1000
        all_elapsed.append(elapsed); all_t2s.append(t2s)
        print(f"[{backend_name}] run {run_i+1}: {elapsed:.1f}s  mem={mem:.3f}GB  avg_step={avg_step_ms:.2f}ms  epochs={t2s_ep or ep}")

    ckpt = OUT / f"model_{backend_name}.pt"
    torch.save(getattr(model, '_orig_mod', model).state_dict(), ckpt)

    # Memory bandwidth estimate
    bytes_per_step = 3 * Nx * Ny * 4 * 10  # U,V,P x grid x float32 x stencil accesses
    mem_bw_gbs = bytes_per_step / (avg_step_ms * 1e-3) / 1e9

    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep,
        avg_step_ms=avg_step_ms,
        mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT / f"meta_{backend_name}.npy", metrics)

    # Evaluate vs Ghia
    with torch.no_grad():
        U_pred, V_pred, _ = infer(model, xy, X, Y)
    U_np = U_pred.cpu().numpy(); V_np = V_pred.cpu().numpy()
    x_np = X[:, 0].cpu().numpy(); y_np = Y[0, :].cpu().numpy()
    l2 = _ghia_l2(U_np, y_np)
    _plot(U_np, V_np, x_np, y_np, backend_name, l2)

    print(f"\n[{backend_name}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem:.3f}")
    print(f"  Mem_BW_GBs   : {mem_bw_gbs:.1f}")
    print(f"  L2_Ghia      : {l2:.4e}")
    return dict(elapsed=np.median(all_elapsed), mem_gb=mem, t2s=t2s,
                t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
                mem_bw_gbs=mem_bw_gbs, l2=l2, history=history)


def _ghia_l2(U_np, y_np):
    """L2 error vs Ghia centerline data."""
    xi = Nx // 2
    u_center = U_np[xi, :]
    ghia_y = np.array(GHIA_Y); ghia_u = np.array(GHIA_U)
    u_interp = np.interp(ghia_y, y_np, u_center)
    return np.linalg.norm(u_interp - ghia_u) / np.linalg.norm(ghia_u)


def _plot(U, V, x_np, y_np, name, l2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    speed = np.sqrt(U**2 + V**2)
    axes[0].contourf(x_np, y_np, speed.T, levels=20, cmap='viridis')
    axes[0].set_title(f'Speed - {name}')
    axes[1].contourf(x_np, y_np, U.T, levels=20, cmap='RdBu_r')
    axes[1].set_title('u velocity')
    u_center = U[Nx//2, :]
    axes[2].plot(u_center, y_np, 'b-', label='PINN', lw=2)
    axes[2].plot(GHIA_U, GHIA_Y, 'ro', label='Ghia Re=100', ms=5)
    axes[2].set_xlabel('u'); axes[2].set_ylabel('y')
    axes[2].set_title('Centerline u (x=0.5)'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'LDC ({name}) | L2_Ghia={l2:.2e}')
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
