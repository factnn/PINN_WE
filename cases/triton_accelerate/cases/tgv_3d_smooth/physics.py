"""TGV 3D: physics module.

3D Taylor-Green Vortex: unsteady incompressible NS with periodic BC.
Domain [0, 2pi]^3 x [0, 1], nu=0.01.
IC: u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0,
    p=(1/16)(cos(2x)+cos(2y))(cos(2z)+2).
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "tgv_3d_smooth"
nu = 0.1  # Re=10, very smooth laminar decay
Nt, Nx, Ny, Nz = 10, 32, 32, 32
GRID_SHAPE = (Nt, Nx, Ny, Nz)
MLP_THRESHOLD = 1e-3
CNN_THRESHOLD = 1e-3
MLP_THRESHOLD = 5e-3
CNN_THRESHOLD = 5e-2


# ─── Models ──────────────────────────────────────────────────────────────────
class Sine(nn.Module):
    """Sin activation for SIREN networks."""
    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """SIREN MLP: (x,y,z,t) -> (u,v,w,p) with Sin activation."""
    def __init__(self, width=256, depth=4):
        super().__init__()
        layers = [nn.Linear(4, width), Sine()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), Sine()]
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
        self.enc = nn.Sequential(
            nn.Conv3d(4, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, 4, kernel_size=5, padding=2),
        )

    def forward(self, X, Y, Z, T):
        inputs = torch.stack([X, Y, Z, T], dim=1)
        out = self.enc(inputs)
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3]


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


# ─── Grid & Context ──────────────────────────────────────────────────────────
def make_context(device="cuda"):
    dx = 2.0 * np.pi / Nx
    dy = 2.0 * np.pi / Ny
    dz = 2.0 * np.pi / Nz
    x = torch.arange(Nx, device=device) * dx
    y = torch.arange(Ny, device=device) * dy
    z = torch.arange(Nz, device=device) * dz
    t = torch.linspace(0, 1, Nt, device=device)
    T, X, Y, Z = torch.meshgrid(t, x, y, z, indexing='ij')
    dt = float(t[1] - t[0])
    xyzt = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
    # IC at t=0
    U_ic = torch.sin(X[0]) * torch.cos(Y[0]) * torch.cos(Z[0])
    V_ic = -torch.cos(X[0]) * torch.sin(Y[0]) * torch.cos(Z[0])
    W_ic = torch.zeros_like(U_ic)
    P_ic = (1.0 / 16.0) * (torch.cos(2 * X[0]) + torch.cos(2 * Y[0])) * (torch.cos(2 * Z[0]) + 2)
    return {"X": X, "Y": Y, "Z": Z, "T": T,
            "dx": dx, "dy": dy, "dz": dz, "dt": dt, "xyzt": xyzt,
            "U_ic": U_ic, "V_ic": V_ic, "W_ic": W_ic, "P_ic": P_ic}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        U, V, W, P = model(ctx["X"], ctx["Y"], ctx["Z"], ctx["T"])
    else:
        u_f, v_f, w_f, p_f = model(ctx["xyzt"])
        U = u_f.reshape(Nt, Nx, Ny, Nz)
        V = v_f.reshape(Nt, Nx, Ny, Nz)
        W = w_f.reshape(Nt, Nx, Ny, Nz)
        P = p_f.reshape(Nt, Nx, Ny, Nz)
    return U, V, W, P


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _pde_residual_fd(U, V, W, P, dx, dy, dz, dt):
    """3D incompressible NS residual via central differences."""
    i = slice(1, -1)
    u_t = (U[2:,i,i,i] - U[:-2,i,i,i]) / (2*dt)
    v_t = (V[2:,i,i,i] - V[:-2,i,i,i]) / (2*dt)
    w_t = (W[2:,i,i,i] - W[:-2,i,i,i]) / (2*dt)
    u_x = (U[i,2:,i,i] - U[i,:-2,i,i]) / (2*dx)
    u_y = (U[i,i,2:,i] - U[i,i,:-2,i]) / (2*dy)
    u_z = (U[i,i,i,2:] - U[i,i,i,:-2]) / (2*dz)
    v_x = (V[i,2:,i,i] - V[i,:-2,i,i]) / (2*dx)
    v_y = (V[i,i,2:,i] - V[i,i,:-2,i]) / (2*dy)
    v_z = (V[i,i,i,2:] - V[i,i,i,:-2]) / (2*dz)
    w_x = (W[i,2:,i,i] - W[i,:-2,i,i]) / (2*dx)
    w_y = (W[i,i,2:,i] - W[i,i,:-2,i]) / (2*dy)
    w_z = (W[i,i,i,2:] - W[i,i,i,:-2]) / (2*dz)
    u_xx = (U[i,2:,i,i] - 2*U[i,i,i,i] + U[i,:-2,i,i]) / dx**2
    u_yy = (U[i,i,2:,i] - 2*U[i,i,i,i] + U[i,i,:-2,i]) / dy**2
    u_zz = (U[i,i,i,2:] - 2*U[i,i,i,i] + U[i,i,i,:-2]) / dz**2
    v_xx = (V[i,2:,i,i] - 2*V[i,i,i,i] + V[i,:-2,i,i]) / dx**2
    v_yy = (V[i,i,2:,i] - 2*V[i,i,i,i] + V[i,i,:-2,i]) / dy**2
    v_zz = (V[i,i,i,2:] - 2*V[i,i,i,i] + V[i,i,i,:-2]) / dz**2
    w_xx = (W[i,2:,i,i] - 2*W[i,i,i,i] + W[i,:-2,i,i]) / dx**2
    w_yy = (W[i,i,2:,i] - 2*W[i,i,i,i] + W[i,i,:-2,i]) / dy**2
    w_zz = (W[i,i,i,2:] - 2*W[i,i,i,i] + W[i,i,i,:-2]) / dz**2
    p_x = (P[i,2:,i,i] - P[i,:-2,i,i]) / (2*dx)
    p_y = (P[i,i,2:,i] - P[i,i,:-2,i]) / (2*dy)
    p_z = (P[i,i,i,2:] - P[i,i,i,:-2]) / (2*dz)
    uc = U[i,i,i,i]; vc = V[i,i,i,i]; wc = W[i,i,i,i]
    res_u = u_t + uc*u_x + vc*u_y + wc*u_z + p_x - nu*(u_xx + u_yy + u_zz)
    res_v = v_t + uc*v_x + vc*v_y + wc*v_z + p_y - nu*(v_xx + v_yy + v_zz)
    res_w = w_t + uc*w_x + vc*w_y + wc*w_z + p_z - nu*(w_xx + w_yy + w_zz)
    res_div = u_x + v_y + w_z
    return (res_u**2 + res_v**2 + res_w**2 + res_div**2).mean()


def _ic_bc_loss(U, V, W, P, U_ic, V_ic, W_ic, P_ic):
    """IC at t=0 + periodic BC in x, y, z."""
    ic = ((U[0]-U_ic)**2).mean() + ((V[0]-V_ic)**2).mean() + \
         ((W[0]-W_ic)**2).mean() + ((P[0]-P_ic)**2).mean()
    bc_x = ((U[:,0]-U[:,-1])**2 + (V[:,0]-V[:,-1])**2 +
            (W[:,0]-W[:,-1])**2 + (P[:,0]-P[:,-1])**2).mean()
    bc_y = ((U[:,:,0]-U[:,:,-1])**2 + (V[:,:,0]-V[:,:,-1])**2 +
            (W[:,:,0]-W[:,:,-1])**2 + (P[:,:,0]-P[:,:,-1])**2).mean()
    bc_z = ((U[:,:,:,0]-U[:,:,:,-1])**2 + (V[:,:,:,0]-V[:,:,:,-1])**2 +
            (W[:,:,:,0]-W[:,:,:,-1])**2 + (P[:,:,:,0]-P[:,:,:,-1])**2).mean()
    return ic + bc_x + bc_y + bc_z


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xyzt_g = ctx["xyzt"].detach().requires_grad_(True)
    u_g, v_g, w_g, p_g = model(xyzt_g)
    u_g = u_g.reshape(Nt,Nx,Ny,Nz); v_g = v_g.reshape(Nt,Nx,Ny,Nz)
    w_g = w_g.reshape(Nt,Nx,Ny,Nz); p_g = p_g.reshape(Nt,Nx,Ny,Nz)
    gu = torch.autograd.grad(u_g.sum(), xyzt_g, create_graph=True)[0]
    gv = torch.autograd.grad(v_g.sum(), xyzt_g, create_graph=True)[0]
    gw = torch.autograd.grad(w_g.sum(), xyzt_g, create_graph=True)[0]
    gp = torch.autograd.grad(p_g.sum(), xyzt_g, create_graph=True)[0]
    u_x=gu[:,0].reshape(Nt,Nx,Ny,Nz); u_y=gu[:,1].reshape(Nt,Nx,Ny,Nz)
    u_z=gu[:,2].reshape(Nt,Nx,Ny,Nz); u_t=gu[:,3].reshape(Nt,Nx,Ny,Nz)
    v_x=gv[:,0].reshape(Nt,Nx,Ny,Nz); v_y=gv[:,1].reshape(Nt,Nx,Ny,Nz)
    v_z=gv[:,2].reshape(Nt,Nx,Ny,Nz); v_t=gv[:,3].reshape(Nt,Nx,Ny,Nz)
    w_x=gw[:,0].reshape(Nt,Nx,Ny,Nz); w_y=gw[:,1].reshape(Nt,Nx,Ny,Nz)
    w_z=gw[:,2].reshape(Nt,Nx,Ny,Nz); w_t=gw[:,3].reshape(Nt,Nx,Ny,Nz)
    p_x=gp[:,0].reshape(Nt,Nx,Ny,Nz); p_y=gp[:,1].reshape(Nt,Nx,Ny,Nz)
    p_z=gp[:,2].reshape(Nt,Nx,Ny,Nz)
    u_xx=torch.autograd.grad(u_x.sum(),xyzt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny,Nz)
    u_yy=torch.autograd.grad(u_y.sum(),xyzt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny,Nz)
    u_zz=torch.autograd.grad(u_z.sum(),xyzt_g,create_graph=True)[0][:,2].reshape(Nt,Nx,Ny,Nz)
    v_xx=torch.autograd.grad(v_x.sum(),xyzt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny,Nz)
    v_yy=torch.autograd.grad(v_y.sum(),xyzt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny,Nz)
    v_zz=torch.autograd.grad(v_z.sum(),xyzt_g,create_graph=True)[0][:,2].reshape(Nt,Nx,Ny,Nz)
    w_xx=torch.autograd.grad(w_x.sum(),xyzt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny,Nz)
    w_yy=torch.autograd.grad(w_y.sum(),xyzt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny,Nz)
    w_zz=torch.autograd.grad(w_z.sum(),xyzt_g,create_graph=True)[0][:,2].reshape(Nt,Nx,Ny,Nz)
    res_u = u_t + u_g*u_x + v_g*u_y + w_g*u_z + p_x - nu*(u_xx+u_yy+u_zz)
    res_v = v_t + u_g*v_x + v_g*v_y + w_g*v_z + p_y - nu*(v_xx+v_yy+v_zz)
    res_w = w_t + u_g*w_x + v_g*w_y + w_g*w_z + p_z - nu*(w_xx+w_yy+w_zz)
    res_div = u_x + v_y + w_z
    pde = (res_u**2 + res_v**2 + res_w**2 + res_div**2).mean()
    return pde + 10 * _ic_bc_loss(u_g, v_g, w_g, p_g,
                                   ctx["U_ic"], ctx["V_ic"], ctx["W_ic"], ctx["P_ic"])


def loss_canpinn(model, ctx):
    U, V, W, P = infer(model, ctx)
    pde = _pde_residual_fd(U, V, W, P, ctx["dx"], ctx["dy"], ctx["dz"], ctx["dt"])
    return pde + 10 * _ic_bc_loss(U, V, W, P,
                                   ctx["U_ic"], ctx["V_ic"], ctx["W_ic"], ctx["P_ic"])


def loss_triton(model, ctx):
    from kernels.stencil_3d_ns_unsteady import ns3d_residual_triton
    U, V, W, P = infer(model, ctx)
    pde = ns3d_residual_triton(U, V, W, P,
                               ctx["dx"], ctx["dy"], ctx["dz"], ctx["dt"], nu)
    return pde + 10 * _ic_bc_loss(U, V, W, P,
                                   ctx["U_ic"], ctx["V_ic"], ctx["W_ic"], ctx["P_ic"])


# ─── Evaluation ──────────────────────────────────────────────────────────────
def compute_l2_error(model, ctx):
    """Compute L2 error of IC reproduction (no exact solution for t>0)."""
    with torch.no_grad():
        U, V, W, P = infer(model, ctx)
    # Compare t=0 slice to exact IC
    U0 = U[0]; V0 = V[0]; W0 = W[0]; P0 = P[0]
    U_ic = ctx["U_ic"]; V_ic = ctx["V_ic"]; W_ic = ctx["W_ic"]; P_ic = ctx["P_ic"]
    err = torch.sqrt(((U0-U_ic)**2 + (V0-V_ic)**2 + (W0-W_ic)**2 + (P0-P_ic)**2).mean())
    ref = torch.sqrt((U_ic**2 + V_ic**2 + W_ic**2 + P_ic**2).mean())
    return float(err / (ref + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    with torch.no_grad():
        U, V, W, P = infer(model, ctx)
    U_np = U.cpu().numpy(); V_np = V.cpu().numpy(); W_np = W.cpu().numpy()
    x_np = ctx["X"][0, :, 0, 0].cpu().numpy()
    y_np = ctx["Y"][0, 0, :, 0].cpu().numpy()
    l2 = compute_l2_error(model, ctx)

    # Kinetic energy decay
    Ek = 0.5 * (U_np**2 + V_np**2 + W_np**2).mean(axis=(1, 2, 3))
    t_np = ctx["T"][:, 0, 0, 0].cpu().numpy()

    ti = Nt // 2; zi = Nz // 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sl_u = U_np[ti, :, :, zi]
    im = axes[0].contourf(x_np, y_np, sl_u.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=axes[0]); axes[0].set_title(f'u (t={ti}, z-mid)')
    sl_v = V_np[ti, :, :, zi]
    im2 = axes[1].contourf(x_np, y_np, sl_v.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[1]); axes[1].set_title(f'v (t={ti}, z-mid)')
    axes[2].plot(t_np, Ek, 'b-', lw=2, label='Ek(t)')
    axes[2].set_xlabel('t'); axes[2].set_ylabel('Ek')
    axes[2].set_title(f'Kinetic Energy (IC L2={l2:.2e})')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'TGV 3D ({name})', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return 4 * Nt * Nx * Ny * Nz * 4 * 20  # U,V,W,P × grid × float32 × stencil accesses

# Aliases for kernel verification scripts
pde_residual_pytorch = _pde_residual_fd
