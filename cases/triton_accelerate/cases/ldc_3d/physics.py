"""LDC 3D: physics module.

Steady-state 3D incompressible NS with P+div, Re=100.
BC: top lid u=1, v=w=0; other walls u=v=w=0.
Reference: Ku et al. 1987 / Albensoeder & Kuhlmann 2005.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "ldc_3d"
Re = 100.0
nu = 1.0 / Re
Nx, Ny, Nz = 48, 48, 48
GRID_SHAPE = (Nx, Ny, Nz)


# ─── Models ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, width=128, depth=5):
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
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(3, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, 4, kernel_size=5, padding=2),
        )

    def forward(self, X, Y, Z):
        inputs = torch.stack([X, Y, Z], dim=0).unsqueeze(0)  # [1, 3, Nx, Ny, Nz]
        out = self.enc(inputs).squeeze(0)  # [4, Nx, Ny, Nz]
        return out[0], out[1], out[2], out[3]


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


# ─── Grid & Context ──────────────────────────────────────────────────────────
def make_context(device="cuda"):
    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    z = torch.linspace(0, 1, Nz, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    return {"X": X, "Y": Y, "Z": Z, "dx": dx, "dy": dy, "dz": dz, "xyz": xyz}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"], ctx["Y"], ctx["Z"])
    else:
        u_f, v_f, w_f, p_f = model(ctx["xyz"])
        return (u_f.reshape(Nx, Ny, Nz), v_f.reshape(Nx, Ny, Nz),
                w_f.reshape(Nx, Ny, Nz), p_f.reshape(Nx, Ny, Nz))


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _pde_residual_fd(U, V, W, P, dx, dy, dz):
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


def _bc_loss(U, V, W):
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


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xyz_g = ctx["xyz"].detach().requires_grad_(True)
    u_g, v_g, w_g, p_g = model(xyz_g)
    u_g = u_g.reshape(Nx,Ny,Nz); v_g = v_g.reshape(Nx,Ny,Nz)
    w_g = w_g.reshape(Nx,Ny,Nz); p_g = p_g.reshape(Nx,Ny,Nz)
    gu = torch.autograd.grad(u_g.sum(), xyz_g, create_graph=True)[0]
    gv = torch.autograd.grad(v_g.sum(), xyz_g, create_graph=True)[0]
    gw = torch.autograd.grad(w_g.sum(), xyz_g, create_graph=True)[0]
    gp = torch.autograd.grad(p_g.sum(), xyz_g, create_graph=True)[0]
    u_x=gu[:,0].reshape(Nx,Ny,Nz); u_y=gu[:,1].reshape(Nx,Ny,Nz); u_z=gu[:,2].reshape(Nx,Ny,Nz)
    v_x=gv[:,0].reshape(Nx,Ny,Nz); v_y=gv[:,1].reshape(Nx,Ny,Nz); v_z=gv[:,2].reshape(Nx,Ny,Nz)
    w_x=gw[:,0].reshape(Nx,Ny,Nz); w_y=gw[:,1].reshape(Nx,Ny,Nz); w_z=gw[:,2].reshape(Nx,Ny,Nz)
    p_x=gp[:,0].reshape(Nx,Ny,Nz); p_y=gp[:,1].reshape(Nx,Ny,Nz); p_z=gp[:,2].reshape(Nx,Ny,Nz)
    u_xx=torch.autograd.grad(u_x.sum(),xyz_g,create_graph=True)[0][:,0].reshape(Nx,Ny,Nz)
    u_yy=torch.autograd.grad(u_y.sum(),xyz_g,create_graph=True)[0][:,1].reshape(Nx,Ny,Nz)
    u_zz=torch.autograd.grad(u_z.sum(),xyz_g,create_graph=True)[0][:,2].reshape(Nx,Ny,Nz)
    v_xx=torch.autograd.grad(v_x.sum(),xyz_g,create_graph=True)[0][:,0].reshape(Nx,Ny,Nz)
    v_yy=torch.autograd.grad(v_y.sum(),xyz_g,create_graph=True)[0][:,1].reshape(Nx,Ny,Nz)
    v_zz=torch.autograd.grad(v_z.sum(),xyz_g,create_graph=True)[0][:,2].reshape(Nx,Ny,Nz)
    w_xx=torch.autograd.grad(w_x.sum(),xyz_g,create_graph=True)[0][:,0].reshape(Nx,Ny,Nz)
    w_yy=torch.autograd.grad(w_y.sum(),xyz_g,create_graph=True)[0][:,1].reshape(Nx,Ny,Nz)
    w_zz=torch.autograd.grad(w_z.sum(),xyz_g,create_graph=True)[0][:,2].reshape(Nx,Ny,Nz)
    res_u = u_g*u_x + v_g*u_y + w_g*u_z + p_x - nu*(u_xx+u_yy+u_zz)
    res_v = u_g*v_x + v_g*v_y + w_g*v_z + p_y - nu*(v_xx+v_yy+v_zz)
    res_w = u_g*w_x + v_g*w_y + w_g*w_z + p_z - nu*(w_xx+w_yy+w_zz)
    res_div = u_x + v_y + w_z
    return (res_u**2+res_v**2+res_w**2+res_div**2).mean() + 10*_bc_loss(u_g, v_g, w_g)


def loss_canpinn(model, ctx):
    U, V, W, P = infer(model, ctx)
    return _pde_residual_fd(U, V, W, P, ctx["dx"], ctx["dy"], ctx["dz"]) + 10 * _bc_loss(U, V, W)


def loss_triton(model, ctx):
    from kernels.stencil_3d_ns_steady import ldc_residual_triton
    U, V, W, P = infer(model, ctx)
    return ldc_residual_triton(U, V, W, P, ctx["dx"], ctx["dy"], ctx["dz"], nu) + 10 * _bc_loss(U, V, W)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def compute_l2_error(model, ctx):
    """L2 error of u-velocity centerline (z-midplane) vs zero reference.
    No exact reference available; report PDE residual norm as proxy."""
    with torch.no_grad():
        U, V, W, P = infer(model, ctx)
    res = _pde_residual_fd(U, V, W, P, ctx["dx"], ctx["dy"], ctx["dz"])
    return float(res.item())


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    with torch.no_grad():
        U, V, W, P = infer(model, ctx)
    U_np = U.cpu().numpy(); V_np = V.cpu().numpy()
    x_np = ctx["X"][:, 0, 0].cpu().numpy()
    y_np = ctx["Y"][0, :, 0].cpu().numpy()
    zi = Nz // 2
    l2 = compute_l2_error(model, ctx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    speed = np.sqrt(U_np[:,:,zi]**2 + V_np[:,:,zi]**2)
    axes[0].contourf(x_np, y_np, speed.T, levels=20, cmap='viridis')
    axes[0].set_title(f'Speed z-mid - {name}')
    axes[1].contourf(x_np, y_np, U_np[:,:,zi].T, levels=20, cmap='RdBu_r')
    axes[1].set_title('u velocity z-mid')
    u_center = U_np[Nx // 2, :, zi]
    axes[2].plot(u_center, y_np, 'b-', label='PINN', lw=2)
    axes[2].set_xlabel('u'); axes[2].set_ylabel('y')
    axes[2].set_title(f'Centerline (res={l2:.2e})')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'LDC 3D ({name})')
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return 4 * Nx * Ny * Nz * 4 * 14  # U,V,W,P × grid × float32 × stencil accesses

# Aliases for kernel verification scripts
pde_residual_pytorch = _pde_residual_fd
