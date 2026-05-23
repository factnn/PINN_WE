"""LDC 2D: physics module.

Steady-state incompressible NS with P+div, Re=100.
BC: top lid u=1,v=0; other walls u=v=0.
Reference: Ghia et al. 1982.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "ldc_2d"
Re = 100.0
nu = 1.0 / Re
Nx, Ny = 128, 128
GRID_SHAPE = (Nx, Ny)

GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
           0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
GHIA_U = [0.0000,-0.0372,-0.0419,-0.0477,-0.0643,-0.1015,-0.1566,-0.2109,
          -0.2058,-0.1364, 0.0033, 0.2315, 0.6872, 0.7372, 0.7887, 0.8412, 1.0000]


# ─── Models ──────────────────────────────────────────────────────────────────
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
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 3, kernel_size=5, padding=2),
        )

    def forward(self, X, Y):
        inputs = torch.stack([X, Y], dim=0).unsqueeze(0)
        out = self.enc(inputs).squeeze(0)
        return out[0], out[1], out[2]


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


# ─── Grid & Context ──────────────────────────────────────────────────────────
def make_context(device="cuda"):
    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return {"X": X, "Y": Y, "dx": dx, "dy": dy, "xy": xy}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"], ctx["Y"])
    else:
        u_f, v_f, p_f = model(ctx["xy"])
        return u_f.reshape(Nx, Ny), v_f.reshape(Nx, Ny), p_f.reshape(Nx, Ny)


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _pde_residual_fd(U, V, P, dx, dy):
    u_x = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dx)
    u_y = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dy)
    u_xx = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dx**2
    u_yy = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dy**2
    v_x = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * dx)
    v_y = (V[1:-1, 2:] - V[1:-1, :-2]) / (2 * dy)
    v_xx = (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / dx**2
    v_yy = (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1, :-2]) / dy**2
    p_x = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2 * dx)
    p_y = (P[1:-1, 2:] - P[1:-1, :-2]) / (2 * dy)
    uc = U[1:-1, 1:-1]; vc = V[1:-1, 1:-1]
    res_u = uc * u_x + vc * u_y + p_x - nu * (u_xx + u_yy)
    res_v = uc * v_x + vc * v_y + p_y - nu * (v_xx + v_yy)
    res_div = u_x + v_y
    return (res_u**2 + res_v**2 + res_div**2).mean()


def _bc_loss(U, V):
    top = (U[:, -1] - 1)**2 + V[:, -1]**2
    bot = U[:, 0]**2 + V[:, 0]**2
    left = U[0, :]**2 + V[0, :]**2
    right = U[-1, :]**2 + V[-1, :]**2
    return top.mean() + bot.mean() + left.mean() + right.mean()


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xy_g = ctx["xy"].detach().requires_grad_(True)
    u_g, v_g, p_g = model(xy_g)
    u_g = u_g.reshape(Nx, Ny); v_g = v_g.reshape(Nx, Ny); p_g = p_g.reshape(Nx, Ny)
    gu = torch.autograd.grad(u_g.sum(), xy_g, create_graph=True)[0]
    gv = torch.autograd.grad(v_g.sum(), xy_g, create_graph=True)[0]
    gp = torch.autograd.grad(p_g.sum(), xy_g, create_graph=True)[0]
    u_x = gu[:, 0].reshape(Nx, Ny); u_y = gu[:, 1].reshape(Nx, Ny)
    v_x = gv[:, 0].reshape(Nx, Ny); v_y = gv[:, 1].reshape(Nx, Ny)
    p_x = gp[:, 0].reshape(Nx, Ny); p_y = gp[:, 1].reshape(Nx, Ny)
    u_xx = torch.autograd.grad(u_x.sum(), xy_g, create_graph=True)[0][:, 0].reshape(Nx, Ny)
    u_yy = torch.autograd.grad(u_y.sum(), xy_g, create_graph=True)[0][:, 1].reshape(Nx, Ny)
    v_xx = torch.autograd.grad(v_x.sum(), xy_g, create_graph=True)[0][:, 0].reshape(Nx, Ny)
    v_yy = torch.autograd.grad(v_y.sum(), xy_g, create_graph=True)[0][:, 1].reshape(Nx, Ny)
    res_u = u_g * u_x + v_g * u_y + p_x - nu * (u_xx + u_yy)
    res_v = u_g * v_x + v_g * v_y + p_y - nu * (v_xx + v_yy)
    res_div = u_x + v_y
    return (res_u**2 + res_v**2 + res_div**2).mean() + 10 * _bc_loss(u_g, v_g)


def loss_canpinn(model, ctx):
    U, V, P = infer(model, ctx)
    return _pde_residual_fd(U, V, P, ctx["dx"], ctx["dy"]) + 10 * _bc_loss(U, V)


def loss_triton(model, ctx):
    from kernels.stencil_2d_burgers_steady import ldc_residual_triton
    U, V, P = infer(model, ctx)
    return ldc_residual_triton(U, V, P, ctx["dx"], ctx["dy"], nu) + 10 * _bc_loss(U, V)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def compute_l2_error(model, ctx):
    with torch.no_grad():
        U, V, P = infer(model, ctx)
    U_np = U.cpu().numpy()
    y_np = ctx["Y"][0, :].cpu().numpy()
    xi = Nx // 2
    u_center = U_np[xi, :]
    ghia_y = np.array(GHIA_Y)
    ghia_u = np.array(GHIA_U)
    u_interp = np.interp(ghia_y, y_np, u_center)
    return float(np.linalg.norm(u_interp - ghia_u) / np.linalg.norm(ghia_u))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    with torch.no_grad():
        U, V, P = infer(model, ctx)
    U_np = U.cpu().numpy(); V_np = V.cpu().numpy()
    x_np = ctx["X"][:, 0].cpu().numpy()
    y_np = ctx["Y"][0, :].cpu().numpy()
    l2 = compute_l2_error(model, ctx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    speed = np.sqrt(U_np**2 + V_np**2)
    axes[0].contourf(x_np, y_np, speed.T, levels=20, cmap='viridis')
    axes[0].set_title(f'Speed - {name}')
    axes[1].contourf(x_np, y_np, U_np.T, levels=20, cmap='RdBu_r')
    axes[1].set_title('u velocity')
    u_center = U_np[Nx // 2, :]
    axes[2].plot(u_center, y_np, 'b-', label='PINN', lw=2)
    axes[2].plot(GHIA_U, GHIA_Y, 'ro', label='Ghia Re=100', ms=5)
    axes[2].set_xlabel('u'); axes[2].set_ylabel('y')
    axes[2].set_title(f'Centerline (L2={l2:.2e})')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'LDC 2D ({name})')
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return 3 * Nx * Ny * 4 * 10  # U,V,P × grid × float32 × stencil accesses

# Aliases for kernel verification scripts
pde_residual_pytorch = _pde_residual_fd
