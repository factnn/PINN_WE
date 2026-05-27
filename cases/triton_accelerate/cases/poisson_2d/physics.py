"""2D Poisson Equation: Δu = f.

Simplest elliptic PDE. Manufactured solution:
  u_exact = sin(πx) * sin(πy)
  f = -2π² * sin(πx) * sin(πy)
Domain [0,1]², Dirichlet BC (u=0 on boundary).
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "poisson_2d"
Nx, Ny = 64, 64
GRID_SHAPE = (Nx, Ny)
MLP_THRESHOLD = 1e-5
CNN_THRESHOLD = 1e-4


# ─── Models ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, width=128, depth=5):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)

    def forward(self, xy):
        return self.out_u(self.net(xy)).squeeze(-1)


class PhyCNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv2d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, X, Y):
        inputs = torch.stack([X, Y], dim=0).unsqueeze(0)
        return self.enc(inputs).squeeze(0).squeeze(0)


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
    # Source term f = -2π²sin(πx)sin(πy)
    F = -2 * np.pi**2 * torch.sin(np.pi * X) * torch.sin(np.pi * Y)
    return {"X": X, "Y": Y, "dx": dx, "dy": dy, "xy": xy, "F": F}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"], ctx["Y"])
    else:
        return model(ctx["xy"]).reshape(Nx, Ny)


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _pde_residual_fd(U, F, dx, dy):
    """Δu - f via central differences."""
    u_xx = (U[2:, 1:-1] - 2*U[1:-1, 1:-1] + U[:-2, 1:-1]) / dx**2
    u_yy = (U[1:-1, 2:] - 2*U[1:-1, 1:-1] + U[1:-1, :-2]) / dy**2
    res = u_xx + u_yy - F[1:-1, 1:-1]
    return (res**2).mean()


def _bc_loss(U):
    """Dirichlet BC: u=0 on all boundaries."""
    return (U[0, :]**2 + U[-1, :]**2 + U[:, 0]**2 + U[:, -1]**2).mean()


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    """Autograd-based."""
    xy_g = ctx["xy"].detach().requires_grad_(True)
    u_g = model(xy_g).reshape(Nx, Ny)
    # Compute Laplacian via autograd
    gu = torch.autograd.grad(u_g.sum(), xy_g, create_graph=True)[0]
    u_x = gu[:, 0].reshape(Nx, Ny)
    u_y = gu[:, 1].reshape(Nx, Ny)
    u_xx = torch.autograd.grad(u_x.sum(), xy_g, create_graph=True)[0][:, 0].reshape(Nx, Ny)
    u_yy = torch.autograd.grad(u_y.sum(), xy_g, create_graph=True)[0][:, 1].reshape(Nx, Ny)
    res = u_xx + u_yy - ctx["F"]
    return (res**2).mean() + 100 * _bc_loss(u_g)


def loss_canpinn(model, ctx):
    """PyTorch finite-difference."""
    U = infer(model, ctx)
    return _pde_residual_fd(U, ctx["F"], ctx["dx"], ctx["dy"]) + 100 * _bc_loss(U)


def loss_triton(model, ctx):
    """Triton fused kernel."""
    from kernels.stencil_2d_poisson import poisson2d_residual_triton
    U = infer(model, ctx)
    return poisson2d_residual_triton(U, ctx["F"], ctx["dx"], ctx["dy"]) + 100 * _bc_loss(U)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def _exact_u(X, Y):
    return torch.sin(np.pi * X) * torch.sin(np.pi * Y)


def compute_l2_error(model, ctx):
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    U_exact = _exact_u(ctx["X"], ctx["Y"]).cpu().numpy()
    return float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    U_exact = _exact_u(ctx["X"], ctx["Y"]).cpu().numpy()
    x_np = ctx["X"][:, 0].cpu().numpy()
    y_np = ctx["Y"][0, :].cpu().numpy()
    l2 = float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].contourf(x_np, y_np, U_pred.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=axes[0]); axes[0].set_title(f'Pred — {name}')
    im2 = axes[1].contourf(x_np, y_np, U_exact.T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[1]); axes[1].set_title('Exact')
    err = np.abs(U_pred - U_exact)
    im3 = axes[2].contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
    plt.colorbar(im3, ax=axes[2]); axes[2].set_title(f'|Error| (L2={l2:.2e})')
    plt.suptitle(f'2D Poisson — {name}')
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return Nx * Ny * 4 * 6  # U + F + neighbors


# Aliases for kernel verification
pde_residual_pytorch = _pde_residual_fd
