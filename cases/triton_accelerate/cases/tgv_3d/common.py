"""Shared utilities and models for 3D TGV experiments."""
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
Nx, Ny, Nz, Nt = 32, 32, 32, 10
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent.parent / "output" / "tgv_3d"
OUT.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. Network architectures
# ==========================================
class MLP(nn.Module):
    """Point-wise MLP: (x,y,z,t) -> (u,v,w,p)"""
    def __init__(self, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(4, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)
        self.out_v = nn.Linear(width, 1)
        self.out_w = nn.Linear(width, 1)
        self.out_p = nn.Linear(width, 1)

    def forward(self, xyzt):
        h = self.net(xyzt)
        return (self.out_u(h).squeeze(-1), self.out_v(h).squeeze(-1),
                self.out_w(h).squeeze(-1), self.out_p(h).squeeze(-1))


class PhyCNN(nn.Module):
    """Grid-based 3D conv network: (X,Y,Z,T) -> (U,V,W,P)"""
    def __init__(self, channels=32):
        super().__init__()
        # Input channels: 4 (X, Y, Z, T coords)
        self.enc = nn.Sequential(
            nn.Conv3d(4, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, 4, kernel_size=5, padding=2),  # U, V, W, P
        )

    def forward(self, X, Y, Z, T):
        # Input: [Nt, Nx, Ny, Nz] each -> [Nt, 4, Nx, Ny, Nz]
        inputs = torch.stack([X, Y, Z, T], dim=1)
        out = self.enc(inputs)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


# ==========================================
# 2. Grid & exact solution
# ==========================================
def make_grid():
    dx = 2.0 * np.pi / Nx
    dy = 2.0 * np.pi / Ny
    dz = 2.0 * np.pi / Nz
    x = torch.arange(Nx, device=device) * dx
    y = torch.arange(Ny, device=device) * dy
    z = torch.arange(Nz, device=device) * dz
    t = torch.linspace(0, 1, Nt, device=device)
    T, X, Y, Z = torch.meshgrid(t, x, y, z, indexing='ij')
    dt = float(t[1] - t[0])
    return X, Y, Z, T, dx, dy, dz, dt


def tgv3d_ic(X, Y, Z):
    """3D TGV initial condition at t=0 (exact for Stokes)."""
    U = torch.sin(X) * torch.cos(Y) * torch.cos(Z)
    V = -torch.cos(X) * torch.sin(Y) * torch.cos(Z)
    W = torch.zeros_like(U)
    P = (1.0/16.0) * (torch.cos(2*X) + torch.cos(2*Y)) * (torch.cos(2*Z) + 2)
    return U, V, W, P


# ==========================================
# 3. PDE residual & losses
# ==========================================
def pde_residual_pytorch(U, V, W, P, dx, dy, dz, dt):
    """3D incompressible NS residual via central differences.
    U,V,W,P: [Nt, Nx, Ny, Nz]
    """
    # interior slices
    i = slice(1, -1)
    # time derivatives
    u_t = (U[2:,i,i,i] - U[:-2,i,i,i]) / (2*dt)
    v_t = (V[2:,i,i,i] - V[:-2,i,i,i]) / (2*dt)
    w_t = (W[2:,i,i,i] - W[:-2,i,i,i]) / (2*dt)
    # first spatial derivatives
    u_x = (U[i,2:,i,i] - U[i,:-2,i,i]) / (2*dx)
    u_y = (U[i,i,2:,i] - U[i,i,:-2,i]) / (2*dy)
    u_z = (U[i,i,i,2:] - U[i,i,i,:-2]) / (2*dz)
    v_x = (V[i,2:,i,i] - V[i,:-2,i,i]) / (2*dx)
    v_y = (V[i,i,2:,i] - V[i,i,:-2,i]) / (2*dy)
    v_z = (V[i,i,i,2:] - V[i,i,i,:-2]) / (2*dz)
    w_x = (W[i,2:,i,i] - W[i,:-2,i,i]) / (2*dx)
    w_y = (W[i,i,2:,i] - W[i,i,:-2,i]) / (2*dy)
    w_z = (W[i,i,i,2:] - W[i,i,i,:-2]) / (2*dz)
    # second spatial derivatives
    u_xx = (U[i,2:,i,i] - 2*U[i,i,i,i] + U[i,:-2,i,i]) / dx**2
    u_yy = (U[i,i,2:,i] - 2*U[i,i,i,i] + U[i,i,:-2,i]) / dy**2
    u_zz = (U[i,i,i,2:] - 2*U[i,i,i,i] + U[i,i,i,:-2]) / dz**2
    v_xx = (V[i,2:,i,i] - 2*V[i,i,i,i] + V[i,:-2,i,i]) / dx**2
    v_yy = (V[i,i,2:,i] - 2*V[i,i,i,i] + V[i,i,:-2,i]) / dy**2
    v_zz = (V[i,i,i,2:] - 2*V[i,i,i,i] + V[i,i,i,:-2]) / dz**2
    w_xx = (W[i,2:,i,i] - 2*W[i,i,i,i] + W[i,:-2,i,i]) / dx**2
    w_yy = (W[i,i,2:,i] - 2*W[i,i,i,i] + W[i,i,:-2,i]) / dy**2
    w_zz = (W[i,i,i,2:] - 2*W[i,i,i,i] + W[i,i,i,:-2]) / dz**2
    # pressure gradient
    p_x = (P[i,2:,i,i] - P[i,:-2,i,i]) / (2*dx)
    p_y = (P[i,i,2:,i] - P[i,i,:-2,i]) / (2*dy)
    p_z = (P[i,i,i,2:] - P[i,i,i,:-2]) / (2*dz)
    # center values
    uc = U[i,i,i,i]; vc = V[i,i,i,i]; wc = W[i,i,i,i]
    # NS residuals
    res_u = u_t + uc*u_x + vc*u_y + wc*u_z + p_x - nu*(u_xx + u_yy + u_zz)
    res_v = v_t + uc*v_x + vc*v_y + wc*v_z + p_y - nu*(v_xx + v_yy + v_zz)
    res_w = w_t + uc*w_x + vc*w_y + wc*w_z + p_z - nu*(w_xx + w_yy + w_zz)
    res_div = u_x + v_y + w_z
    return (res_u**2 + res_v**2 + res_w**2 + res_div**2).mean()


def ic_bc_loss(U, V, W, P, U_ic, V_ic, W_ic, P_ic):
    """IC at t=0 + periodic BC in x,y,z. No exact solution for t>0."""
    # IC: only t=0 slice
    ic = ((U[0]-U_ic)**2).mean() + ((V[0]-V_ic)**2).mean() + \
         ((W[0]-W_ic)**2).mean() + ((P[0]-P_ic)**2).mean()
    # Periodic BC: x
    bc_x = ((U[:,0]-U[:,-1])**2 + (V[:,0]-V[:,-1])**2 +
            (W[:,0]-W[:,-1])**2 + (P[:,0]-P[:,-1])**2).mean()
    # Periodic BC: y
    bc_y = ((U[:,:,0]-U[:,:,-1])**2 + (V[:,:,0]-V[:,:,-1])**2 +
            (W[:,:,0]-W[:,:,-1])**2 + (P[:,:,0]-P[:,:,-1])**2).mean()
    # Periodic BC: z
    bc_z = ((U[:,:,:,0]-U[:,:,:,-1])**2 + (V[:,:,:,0]-V[:,:,:,-1])**2 +
            (W[:,:,:,0]-W[:,:,:,-1])**2 + (P[:,:,:,0]-P[:,:,:,-1])**2).mean()
    return ic + bc_x + bc_y + bc_z


def kinetic_energy(U, V, W):
    """Compute volume-averaged kinetic energy Ek(t) = 0.5 * mean(u^2+v^2+w^2) per timestep.
    Returns: [Nt] tensor.
    """
    return 0.5 * (U**2 + V**2 + W**2).mean(dim=(1, 2, 3))


def unified_loss_fn(model, xyzt, X, Y, Z, T, U_ic, V_ic, W_ic, P_ic, dx, dy, dz, dt):
    U, V, W, P = infer(model, xyzt, X, Y, Z, T)
    return pde_residual_pytorch(U, V, W, P, dx, dy, dz, dt) + \
           ic_bc_loss(U, V, W, P, U_ic, V_ic, W_ic, P_ic)


# ==========================================
# 4. Training & visualization
# ==========================================
def train_and_save(backend_name, model_fn, loss_fn=None,
                   max_epochs=50000, lr=1e-3, loss_threshold=1e-3, runs=1):
    if loss_fn is None:
        loss_fn = unified_loss_fn

    X, Y, Z, T, dx, dy, dz, dt = make_grid()
    U_ic, V_ic, W_ic, P_ic = tgv3d_ic(X[0], Y[0], Z[0])
    xyzt = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)

    all_elapsed = []; all_t2s = []

    for run_i in range(runs):
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

        history = []; t2s = None; t2s_ep = None; t0 = time.time()
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        print(f"[{backend_name}] run {run_i+1}/{runs} | Grid: {Nx}x{Ny}x{Nz}x{Nt}")

        for ep in range(1, max_epochs + 1):
            t_step = time.time()
            opt.zero_grad()
            loss = loss_fn(model, xyzt, X, Y, Z, T,
                           U_ic, V_ic, W_ic, P_ic, dx, dy, dz, dt)
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

    bytes_per_step = 4 * Nt * Nx * Ny * Nz * 4 * 20
    mem_bw_gbs = bytes_per_step / (avg_step_ms * 1e-3) / 1e9

    metrics = dict(
        history=history, elapsed=elapsed, mem_gb=mem,
        t2s=t2s, t2s_ep=t2s_ep,
        avg_step_ms=avg_step_ms, mem_bw_gbs=mem_bw_gbs,
        all_elapsed=all_elapsed, all_t2s=all_t2s,
    )
    np.save(OUT / f"meta_{backend_name}.npy", metrics)

    # Kinetic energy evaluation (no exact solution for t>0)
    with torch.no_grad():
        U_pred, V_pred, W_pred, _ = infer(model, xyzt, X, Y, Z, T)
    Ek = kinetic_energy(U_pred, V_pred, W_pred).cpu().numpy()
    t_np = T[:, 0, 0, 0].cpu().numpy()
    Up = U_pred.cpu().numpy(); Vp = V_pred.cpu().numpy()
    # Ek decay rate: -dEk/dt
    dEk_dt = -np.gradient(Ek, t_np)
    _plot(Up, Vp, X, Y, backend_name, Ek, t_np, dEk_dt)

    print(f"\n[{backend_name}] === Summary ===")
    print(f"  T2S          : {t2s:.1f}s" if t2s else "  T2S          : N/A")
    print(f"  Total_Epochs : {t2s_ep or ep}")
    print(f"  Avg_Step_ms  : {avg_step_ms:.2f}")
    print(f"  Peak_Mem_GB  : {mem:.3f}")
    print(f"  Mem_BW_GBs   : {mem_bw_gbs:.1f}")
    print(f"  Ek(t=0)      : {Ek[0]:.4e}")
    print(f"  Ek(t=T)      : {Ek[-1]:.4e}")
    return dict(elapsed=np.median(all_elapsed), mem_gb=mem, t2s=t2s,
                t2s_ep=t2s_ep, avg_step_ms=avg_step_ms,
                mem_bw_gbs=mem_bw_gbs, Ek=Ek, history=history)


def infer(model, xyzt, X, Y, Z, T):
    """Infer U,V,W,P from model (MLP or PhyCNN)."""
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        U, V, W, P = model(X, Y, Z, T)
    else:
        u_f, v_f, w_f, p_f = model(xyzt)
        U = u_f.reshape(Nt, Nx, Ny, Nz)
        V = v_f.reshape(Nt, Nx, Ny, Nz)
        W = w_f.reshape(Nt, Nx, Ny, Nz)
        P = p_f.reshape(Nt, Nx, Ny, Nz)
    return U, V, W, P


def base_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--max-epochs', type=int, default=50000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--threshold', type=float, default=1e-3)
    p.add_argument('--gpu', type=int, default=0)
    return p


def _plot(U, V, X, Y, name, Ek, t_np, dEk_dt):
    """Plot z-midplane slice + Ek(t) decay curve."""
    x_np = X[0, :, 0, 0].cpu().numpy()
    y_np = Y[0, 0, :, 0].cpu().numpy()
    ti = Nt // 2; zi = Nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # velocity slice
    sl_u = U[ti, :, :, zi]
    im = axes[0].contourf(x_np, y_np, sl_u.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=axes[0]); axes[0].set_title(f'u (t={ti}, z-mid)')
    sl_v = V[ti, :, :, zi]
    im2 = axes[1].contourf(x_np, y_np, sl_v.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[1]); axes[1].set_title(f'v (t={ti}, z-mid)')
    # Ek decay
    axes[2].plot(t_np, Ek, 'b-', lw=2, label='Ek(t)')
    axes[2].set_xlabel('t'); axes[2].set_ylabel('Ek')
    axes[2].set_title('Kinetic Energy Decay'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'TGV 3D ({name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUT / f"solution_{name}.png", dpi=150)
    plt.close()

    # Separate dissipation rate plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(t_np, dEk_dt, 'r-', lw=2, label='-dEk/dt (PINN)')
    ax2.set_xlabel('t'); ax2.set_ylabel('-dEk/dt')
    ax2.set_title(f'Kinetic Energy Dissipation Rate ({name})')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"dissipation_{name}.png", dpi=150)
    plt.close()
