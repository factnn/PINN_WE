"""Shared utilities for 3D Lid-Driven Cavity experiments.
Steady-state 3D NS, Re=100. Grid [Nx, Ny, Nz], no time dimension.
BC: top lid u=1, v=w=0; other walls u=v=w=0.
Reference: Ku et al. 1987 / Albensoeder & Kuhlmann 2005.
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
Nx, Ny, Nz = 32, 32, 32
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent.parent / "output" / "ldc_3d"
OUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Network architectures
# ==========================================
class MLP(nn.Module):
    """Point-wise MLP: (x,y,z) -> (u,v,w)"""
    def __init__(self, width=64, depth=5):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)
        self.out_v = nn.Linear(width, 1)
        self.out_w = nn.Linear(width, 1)
        self.out_p = nn.Linear(width, 1)

    def forward(self, xyz):
        h = self.net(xyz)
        return (self.out_u(h).squeeze(-1), self.out_v(h).squeeze(-1),
                self.out_w(h).squeeze(-1), self.out_p(h).squeeze(-1))


class PhyCNN(nn.Module):
    """Grid-based 3D CNN: (X, Y, Z) -> (U, V, W)"""
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(3, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, 4, kernel_size=5, padding=2),  # U, V, W, P
        )

    def forward(self, X, Y, Z):
        inputs = torch.stack([X, Y, Z], dim=0).unsqueeze(0)  # [1, 3, Nx, Ny, Nz]
        out = self.enc(inputs).squeeze(0)  # [4, Nx, Ny, Nz]
        return out[0], out[1], out[2], out[3]


# ==========================================
# 2. Grid
# ==========================================
def make_grid():
    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    z = torch.linspace(0, 1, Nz, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); dz = float(z[1] - z[0])
    return X, Y, Z, dx, dy, dz


# ==========================================
# 3. PDE residual & losses
# ==========================================
def pde_residual_pytorch(U, V, W, P, dx, dy, dz):
    """Steady 3D incompressible NS: momentum + P + div.
    U, V, W, P: [Nx, Ny, Nz]
    """
    i = slice(1, -1)
    # first derivatives
    u_x = (U[2:,i,i] - U[:-2,i,i]) / (2*dx)
    u_y = (U[i,2:,i] - U[i,:-2,i]) / (2*dy)
    u_z = (U[i,i,2:] - U[i,i,:-2]) / (2*dz)
    v_x = (V[2:,i,i] - V[:-2,i,i]) / (2*dx)
    v_y = (V[i,2:,i] - V[i,:-2,i]) / (2*dy)
    v_z = (V[i,i,2:] - V[i,i,:-2]) / (2*dz)
    w_x = (W[2:,i,i] - W[:-2,i,i]) / (2*dx)
    w_y = (W[i,2:,i] - W[i,:-2,i]) / (2*dy)
    w_z = (W[i,i,2:] - W[i,i,:-2]) / (2*dz)
    p_x = (P[2:,i,i] - P[:-2,i,i]) / (2*dx)
    p_y = (P[i,2:,i] - P[i,:-2,i]) / (2*dy)
    p_z = (P[i,i,2:] - P[i,i,:-2]) / (2*dz)
    # second derivatives
    u_xx = (U[2:,i,i] - 2*U[i,i,i] + U[:-2,i,i]) / dx**2
    u_yy = (U[i,2:,i] - 2*U[i,i,i] + U[i,:-2,i]) / dy**2
    u_zz = (U[i,i,2:] - 2*U[i,i,i] + U[i,i,:-2]) / dz**2
    v_xx = (V[2:,i,i] - 2*V[i,i,i] + V[:-2,i,i]) / dx**2
    v_yy = (V[i,2:,i] - 2*V[i,i,i] + V[i,:-2,i]) / dy**2
    v_zz = (V[i,i,2:] - 2*V[i,i,i] + V[i,i,:-2]) / dz**2
    w_xx = (W[2:,i,i] - 2*W[i,i,i] + W[:-2,i,i]) / dx**2
    w_yy = (W[i,2:,i] - 2*W[i,i,i] + W[i,:-2,i]) / dy**2
    w_zz = (W[i,i,2:] - 2*W[i,i,i] + W[i,i,:-2]) / dz**2
    # center values
    uc = U[i,i,i]; vc = V[i,i,i]; wc = W[i,i,i]
    res_u = uc*u_x + vc*u_y + wc*u_z + p_x - nu*(u_xx + u_yy + u_zz)
    res_v = uc*v_x + vc*v_y + wc*v_z + p_y - nu*(v_xx + v_yy + v_zz)
    res_w = uc*w_x + vc*w_y + wc*w_z + p_z - nu*(w_xx + w_yy + w_zz)
    res_div = u_x + v_y + w_z
    return (res_u**2 + res_v**2 + res_w**2 + res_div**2).mean()


def bc_loss(U, V, W):
    """Lid: top (y=1) u=1, v=w=0; all other walls u=v=w=0."""
    loss = 0.0
    # top: y = Ny-1 -> u=1, v=w=0
    loss += ((U[:,-1,:] - 1)**2 + V[:,-1,:]**2 + W[:,-1,:]**2).mean()
    # bottom: y = 0
    loss += (U[:,0,:]**2 + V[:,0,:]**2 + W[:,0,:]**2).mean()
    # left: x = 0
    loss += (U[0,:,:]**2 + V[0,:,:]**2 + W[0,:,:]**2).mean()
    # right: x = Nx-1
    loss += (U[-1,:,:]**2 + V[-1,:,:]**2 + W[-1,:,:]**2).mean()
    # front: z = 0
    loss += (U[:,:,0]**2 + V[:,:,0]**2 + W[:,:,0]**2).mean()
    # back: z = Nz-1
    loss += (U[:,:,-1]**2 + V[:,:,-1]**2 + W[:,:,-1]**2).mean()
    return loss


def unified_loss_fn(model, xyz, X, Y, Z, dx, dy, dz):
    U, V, W, P = infer(model, xyz, X, Y, Z)
    return pde_residual_pytorch(U, V, W, P, dx, dy, dz) + 10 * bc_loss(U, V, W)


# ==========================================
# 4. Inference helper
# ==========================================
def infer(model, xyz, X, Y, Z):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        U, V, W, P = model(X, Y, Z)
    else:
        u_f, v_f, w_f, p_f = model(xyz)
        U = u_f.reshape(Nx, Ny, Nz)
        V = v_f.reshape(Nx, Ny, Nz)
        W = w_f.reshape(Nx, Ny, Nz)
        P = p_f.reshape(Nx, Ny, Nz)
    return U, V, W, P


# ==========================================
# 5. Training & visualization
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=200000, lr=1e-3, loss_threshold=1e-4, runs=1):
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, Y, Z, dx, dy, dz = make_grid()
    xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; t0 = time.time()
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}x{Ny}x{Nz}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, xyz, X, Y, Z, dx, dy, dz)
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

    bytes_per_step = 4 * Nx * Ny * Nz * 4 * 14
    mem_bw_gbs = bytes_per_step / (avg_step_ms * 1e-3) / 1e9

    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep,
        avg_step_ms=avg_step_ms, mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT / f"meta_{backend_name}.npy", metrics)

    with torch.no_grad():
        U_pred, V_pred, W_pred, _ = infer(model, xyz, X, Y, Z)
    U_np = U_pred.cpu().numpy(); V_np = V_pred.cpu().numpy()
    _plot(U_np, V_np, X, Y, backend_name)

    print(f"\n[{backend_name}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem:.3f}")
    print(f"  Mem_BW_GBs   : {mem_bw_gbs:.1f}")
    return dict(elapsed=np.median(all_elapsed), mem_gb=mem, t2s=t2s,
                t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
                mem_bw_gbs=mem_bw_gbs, history=history)


def _plot(U, V, X, Y, name):
    """Plot z-midplane velocity."""
    x_np = X[:, 0, 0].cpu().numpy()
    y_np = Y[0, :, 0].cpu().numpy()
    zi = Nz // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    speed = np.sqrt(U[:,:,zi]**2 + V[:,:,zi]**2)
    axes[0].contourf(x_np, y_np, speed.T, levels=20, cmap='viridis')
    axes[0].set_title(f'Speed z-mid - {name}')
    axes[1].contourf(x_np, y_np, U[:,:,zi].T, levels=20, cmap='RdBu_r')
    axes[1].set_title('u velocity z-mid')
    plt.suptitle(f'LDC 3D ({name})')
    plt.tight_layout()
    plt.savefig(OUT / f"solution_{name}.png", dpi=150)
    plt.close()


def base_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--max-epochs', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=1e-4)
    p.add_argument('--gpu', type=int, default=0)
    return p
