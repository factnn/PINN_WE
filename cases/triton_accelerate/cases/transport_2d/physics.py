"""Transport 2D: physics module.

2D Advection-Diffusion: c_t + u0*c_x + v0*c_y = nu*(c_xx + c_yy)
Domain [0, 2pi]^2 x [0, 1], periodic BC.
Constant velocity u0=v0=1, nu=0.01.
Exact solution: c = sin(x - u0*t) * sin(y - v0*t) * exp(-2*nu*t).
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "transport_2d"
nu = 0.01
u0 = 1.0
v0 = 1.0
Nx, Ny, Nt = 64, 64, 20
GRID_SHAPE = (Nt, Nx, Ny)
MLP_THRESHOLD = 1e-3
CNN_THRESHOLD = 1e-3


# ─── Models ──────────────────────────────────────────────────────────────────
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
        self.enc = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, X, Y, T):
        inputs = torch.stack([X, Y, T], dim=1)  # [Nt, 3, Nx, Ny]
        out = self.enc(inputs)
        return out[:, 0, :, :]  # [Nt, Nx, Ny]


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


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
    C_exact = torch.sin(X - u0*T) * torch.sin(Y - v0*T) * torch.exp(-2*nu*T)
    return {"X": X, "Y": Y, "T": T, "dx": dx, "dy": dy, "dt": dt,
            "xyt": xyt, "C_exact": C_exact}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"], ctx["Y"], ctx["T"])
    else:
        return model(ctx["xyt"]).reshape(Nt, Nx, Ny)


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _pde_residual_fd(C, dx, dy, dt):
    """c_t + u0*c_x + v0*c_y - nu*(c_xx + c_yy) via central differences."""
    c_t  = (C[2:,1:-1,1:-1] - C[:-2,1:-1,1:-1]) / (2*dt)
    c_x  = (C[1:-1,2:,1:-1] - C[1:-1,:-2,1:-1]) / (2*dx)
    c_y  = (C[1:-1,1:-1,2:] - C[1:-1,1:-1,:-2]) / (2*dy)
    c_xx = (C[1:-1,2:,1:-1] - 2*C[1:-1,1:-1,1:-1] + C[1:-1,:-2,1:-1]) / dx**2
    c_yy = (C[1:-1,1:-1,2:] - 2*C[1:-1,1:-1,1:-1] + C[1:-1,1:-1,:-2]) / dy**2
    res = c_t + u0*c_x + v0*c_y - nu*(c_xx + c_yy)
    return (res**2).mean()


def _ic_bc_loss(C, C_exact):
    """IC at t=0 + periodic BC."""
    ic = ((C[0] - C_exact[0])**2).mean()
    bc_x = ((C[:, 0, :] - C[:, -1, :])**2).mean()
    bc_y = ((C[:, :, 0] - C[:, :, -1])**2).mean()
    return ic + bc_x + bc_y


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xyt_g = ctx["xyt"].detach().requires_grad_(True)
    c_g = model(xyt_g).reshape(Nt, Nx, Ny)
    gc = torch.autograd.grad(c_g.sum(), xyt_g, create_graph=True)[0]
    c_x = gc[:, 0].reshape(Nt, Nx, Ny)
    c_y = gc[:, 1].reshape(Nt, Nx, Ny)
    c_t = gc[:, 2].reshape(Nt, Nx, Ny)
    c_xx = torch.autograd.grad(c_x.sum(), xyt_g, create_graph=True)[0][:, 0].reshape(Nt, Nx, Ny)
    c_yy = torch.autograd.grad(c_y.sum(), xyt_g, create_graph=True)[0][:, 1].reshape(Nt, Nx, Ny)
    res = c_t + u0*c_x + v0*c_y - nu*(c_xx + c_yy)
    return (res**2).mean() + 10 * _ic_bc_loss(c_g, ctx["C_exact"])


def loss_canpinn(model, ctx):
    C = infer(model, ctx)
    return _pde_residual_fd(C, ctx["dx"], ctx["dy"], ctx["dt"]) + \
           10 * _ic_bc_loss(C, ctx["C_exact"])


def loss_triton(model, ctx):
    from kernels.stencil_2d_transport import advdiff_residual_triton
    C = infer(model, ctx)
    pde = advdiff_residual_triton(C, ctx["dx"], ctx["dy"], ctx["dt"], u0, v0, nu)
    return pde + 10 * _ic_bc_loss(C, ctx["C_exact"])


# ─── Evaluation ──────────────────────────────────────────────────────────────
def compute_l2_error(model, ctx):
    with torch.no_grad():
        C_pred = infer(model, ctx)
    C_exact = ctx["C_exact"]
    err = torch.sqrt(((C_pred - C_exact)**2).sum())
    ref = torch.sqrt((C_exact**2).sum())
    return float(err / (ref + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    with torch.no_grad():
        C_pred = infer(model, ctx)
    C_exact = ctx["C_exact"]
    Cp = C_pred.cpu().numpy()
    Ce = C_exact.cpu().numpy()
    l2 = compute_l2_error(model, ctx)

    x_np = ctx["X"][0, :, 0].cpu().numpy()
    y_np = ctx["Y"][0, 0, :].cpu().numpy()
    ti = Nt // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].contourf(x_np, y_np, Cp[ti].T, levels=20, cmap='RdBu_r')
    plt.colorbar(im0, ax=axes[0]); axes[0].set_title('PINN c')
    im1 = axes[1].contourf(x_np, y_np, Ce[ti].T, levels=20, cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[1]); axes[1].set_title('Exact c')
    err = np.abs(Cp[ti] - Ce[ti])
    im2 = axes[2].contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
    plt.colorbar(im2, ax=axes[2]); axes[2].set_title('|Error|')

    plt.suptitle(f'Transport 2D ({name}) | t={ti}/{Nt} | L2={l2:.2e}', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return Nt * Nx * Ny * 4 * 10  # C × grid × float32 × stencil accesses

# Aliases for kernel verification scripts
pde_residual_pytorch = _pde_residual_fd
