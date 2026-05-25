"""TGV 2D: physics module.

Unsteady 2D incompressible NS (Taylor-Green Vortex), nu=0.01.
Domain: [0,2pi]^2 x [0,1]. Periodic BC. Exact solution available.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "tgv_2d"
nu = 0.01
Nx, Ny, Nt = 64, 64, 20
GRID_SHAPE = (Nt, Nx, Ny)
MLP_THRESHOLD = 1e-4
CNN_THRESHOLD = 5e-4


# ─── Models ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, width=128, depth=6):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)
        self.out_v = nn.Linear(width, 1)
        self.out_p = nn.Linear(width, 1)

    def forward(self, xyt):
        h = self.net(xyt)
        return self.out_u(h).squeeze(-1), self.out_v(h).squeeze(-1), self.out_p(h).squeeze(-1)


class PhyCNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 3, kernel_size=5, padding=2),
        )

    def forward(self, X, Y, T):
        inputs = torch.stack([X, Y, T], dim=1)  # [Nt, 3, Nx, Ny]
        out = self.enc(inputs)
        return out[:, 0, :, :], out[:, 1, :, :], out[:, 2, :, :]


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


# ─── Exact solution ──────────────────────────────────────────────────────────
def _exact_uvp(X, Y, T):
    U = torch.sin(X) * torch.cos(Y) * torch.exp(-2*nu*T)
    V = -torch.cos(X) * torch.sin(Y) * torch.exp(-2*nu*T)
    P = 0.25 * (torch.cos(2*X) + torch.cos(2*Y)) * torch.exp(-4*nu*T)
    return U, V, P


# ─── Grid & Context ──────────────────────────────────────────────────────────
def make_context(device="cuda"):
    dx = 2.0 * np.pi / Nx
    dy = 2.0 * np.pi / Ny
    x = torch.arange(Nx, device=device) * dx
    y = torch.arange(Ny, device=device) * dy
    t = torch.linspace(0, 1, Nt, device=device)
    T, X, Y = torch.meshgrid(t, x, y, indexing='ij')
    dt = float(t[1] - t[0])
    xyt = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)
    U_exact, V_exact, P_exact = _exact_uvp(X, Y, T)
    return {"X": X, "Y": Y, "T": T, "dx": dx, "dy": dy, "dt": dt,
            "xyt": xyt, "U_exact": U_exact, "V_exact": V_exact, "P_exact": P_exact}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"], ctx["Y"], ctx["T"])
    else:
        u_f, v_f, p_f = model(ctx["xyt"])
        return u_f.reshape(Nt, Nx, Ny), v_f.reshape(Nt, Nx, Ny), p_f.reshape(Nt, Nx, Ny)


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _pde_residual_fd(U, V, P, dx, dy, dt):
    u_t = (U[2:,1:-1,1:-1] - U[:-2,1:-1,1:-1]) / (2*dt)
    v_t = (V[2:,1:-1,1:-1] - V[:-2,1:-1,1:-1]) / (2*dt)
    u_x = (U[1:-1,2:,1:-1] - U[1:-1,:-2,1:-1]) / (2*dx)
    u_y = (U[1:-1,1:-1,2:] - U[1:-1,1:-1,:-2]) / (2*dy)
    v_x = (V[1:-1,2:,1:-1] - V[1:-1,:-2,1:-1]) / (2*dx)
    v_y = (V[1:-1,1:-1,2:] - V[1:-1,1:-1,:-2]) / (2*dy)
    u_xx = (U[1:-1,2:,1:-1] - 2*U[1:-1,1:-1,1:-1] + U[1:-1,:-2,1:-1]) / dx**2
    u_yy = (U[1:-1,1:-1,2:] - 2*U[1:-1,1:-1,1:-1] + U[1:-1,1:-1,:-2]) / dy**2
    v_xx = (V[1:-1,2:,1:-1] - 2*V[1:-1,1:-1,1:-1] + V[1:-1,:-2,1:-1]) / dx**2
    v_yy = (V[1:-1,1:-1,2:] - 2*V[1:-1,1:-1,1:-1] + V[1:-1,1:-1,:-2]) / dy**2
    p_x = (P[1:-1,2:,1:-1] - P[1:-1,:-2,1:-1]) / (2*dx)
    p_y = (P[1:-1,1:-1,2:] - P[1:-1,1:-1,:-2]) / (2*dy)
    uc = U[1:-1,1:-1,1:-1]; vc = V[1:-1,1:-1,1:-1]
    res_u = u_t + uc*u_x + vc*u_y + p_x - nu*(u_xx + u_yy)
    res_v = v_t + uc*v_x + vc*v_y + p_y - nu*(v_xx + v_yy)
    res_div = u_x + v_y
    return (res_u**2 + res_v**2 + res_div**2).mean()


def _ic_bc_loss(U, V, P, U_exact, V_exact, P_exact):
    """IC at t=0 + periodic BC in x and y."""
    # IC
    ic_u = ((U[0] - U_exact[0])**2).mean()
    ic_v = ((V[0] - V_exact[0])**2).mean()
    ic_p = ((P[0] - P_exact[0])**2).mean()
    # Periodic BC
    bc_u = ((U[:, 0, :] - U[:, -1, :])**2 + (U[:, :, 0] - U[:, :, -1])**2).mean()
    bc_v = ((V[:, 0, :] - V[:, -1, :])**2 + (V[:, :, 0] - V[:, :, -1])**2).mean()
    bc_p = ((P[:, 0, :] - P[:, -1, :])**2 + (P[:, :, 0] - P[:, :, -1])**2).mean()
    return ic_u + ic_v + ic_p + bc_u + bc_v + bc_p


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xyt_g = ctx["xyt"].detach().requires_grad_(True)
    u_g, v_g, p_g = model(xyt_g)
    u_g = u_g.reshape(Nt,Nx,Ny); v_g = v_g.reshape(Nt,Nx,Ny); p_g = p_g.reshape(Nt,Nx,Ny)
    gu = torch.autograd.grad(u_g.sum(), xyt_g, create_graph=True)[0]
    gv = torch.autograd.grad(v_g.sum(), xyt_g, create_graph=True)[0]
    u_x=gu[:,0].reshape(Nt,Nx,Ny); u_y=gu[:,1].reshape(Nt,Nx,Ny); u_t=gu[:,2].reshape(Nt,Nx,Ny)
    v_x=gv[:,0].reshape(Nt,Nx,Ny); v_y=gv[:,1].reshape(Nt,Nx,Ny); v_t=gv[:,2].reshape(Nt,Nx,Ny)
    u_xx=torch.autograd.grad(u_x.sum(),xyt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny)
    u_yy=torch.autograd.grad(u_y.sum(),xyt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny)
    v_xx=torch.autograd.grad(v_x.sum(),xyt_g,create_graph=True)[0][:,0].reshape(Nt,Nx,Ny)
    v_yy=torch.autograd.grad(v_y.sum(),xyt_g,create_graph=True)[0][:,1].reshape(Nt,Nx,Ny)
    gp = torch.autograd.grad(p_g.sum(), xyt_g, create_graph=True)[0]
    p_x=gp[:,0].reshape(Nt,Nx,Ny); p_y=gp[:,1].reshape(Nt,Nx,Ny)
    res_u = u_t + u_g*u_x + v_g*u_y + p_x - nu*(u_xx+u_yy)
    res_v = v_t + u_g*v_x + v_g*v_y + p_y - nu*(v_xx+v_yy)
    res_div = u_x + v_y
    return (res_u**2+res_v**2+res_div**2).mean() + _ic_bc_loss(u_g, v_g, p_g,
            ctx["U_exact"], ctx["V_exact"], ctx["P_exact"])


def loss_canpinn(model, ctx):
    U, V, P = infer(model, ctx)
    return (_pde_residual_fd(U, V, P, ctx["dx"], ctx["dy"], ctx["dt"])
            + _ic_bc_loss(U, V, P, ctx["U_exact"], ctx["V_exact"], ctx["P_exact"]))


def loss_triton(model, ctx):
    from kernels.stencil_2d_ns_unsteady import ns2d_residual_triton
    U, V, P = infer(model, ctx)
    return (ns2d_residual_triton(U, V, P, ctx["dx"], ctx["dy"], ctx["dt"], nu)
            + _ic_bc_loss(U, V, P, ctx["U_exact"], ctx["V_exact"], ctx["P_exact"]))


# ─── Evaluation ──────────────────────────────────────────────────────────────
def compute_l2_error(model, ctx):
    with torch.no_grad():
        U, V, P = infer(model, ctx)
    U_np = U.cpu().numpy(); V_np = V.cpu().numpy()
    Ue = ctx["U_exact"].cpu().numpy(); Ve = ctx["V_exact"].cpu().numpy()
    err = np.sqrt(np.sum((U_np - Ue)**2 + (V_np - Ve)**2))
    ref = np.sqrt(np.sum(Ue**2 + Ve**2)) + 1e-12
    return float(err / ref)


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    with torch.no_grad():
        U, V, P = infer(model, ctx)
    U_np = U.cpu().numpy(); V_np = V.cpu().numpy()
    Ue = ctx["U_exact"].cpu().numpy(); Ve = ctx["V_exact"].cpu().numpy()
    x_np = ctx["X"][0, :, 0].cpu().numpy()
    y_np = ctx["Y"][0, 0, :].cpu().numpy()
    l2 = compute_l2_error(model, ctx)
    ti = Nt // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (f, fe, label) in enumerate([(U_np, Ue, 'u'), (V_np, Ve, 'v')]):
        im = axes[row, 0].contourf(x_np, y_np, f[ti].T, levels=20, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[row, 0]); axes[row, 0].set_title(f'PINN {label}')
        im2 = axes[row, 1].contourf(x_np, y_np, fe[ti].T, levels=20, cmap='RdBu_r')
        plt.colorbar(im2, ax=axes[row, 1]); axes[row, 1].set_title(f'Exact {label}')
        err = np.abs(f[ti] - fe[ti])
        im3 = axes[row, 2].contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
        plt.colorbar(im3, ax=axes[row, 2]); axes[row, 2].set_title('|Error|')
    plt.suptitle(f'TGV 2D ({name}) | t={ti}/{Nt} | L2={l2:.2e}')
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return 3 * Nt * Nx * Ny * 4 * 14  # U,V,P × grid × float32 × stencil accesses

# Aliases for kernel verification scripts
pde_residual_pytorch = _pde_residual_fd
