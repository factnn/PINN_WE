"""Burgers 1D Steady: physics module.

PDE: u * u_x = nu * u_xx on [-1, 1]
BC: u(-1) = 1, u(1) = -1
Exact: u(x) = -tanh(x / (2*nu))
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "burgers_1d_steady"
nu = 0.01 / np.pi
Nx = 1024
GRID_SHAPE = (Nx,)


# ─── Models ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)

    def forward(self, x):
        return self.out_u(self.net(x)).squeeze(-1)


class PhyCNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, X):
        inp = X.unsqueeze(0).unsqueeze(0)
        return self.enc(inp).squeeze(0).squeeze(0)


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


# ─── Grid & Context ──────────────────────────────────────────────────────────
def make_context(device="cuda"):
    x = torch.linspace(-1, 1, Nx, device=device)
    dx = float(x[1] - x[0])
    x_inp = x.unsqueeze(-1)  # [Nx, 1] for MLP
    return {"X": x, "dx": dx, "x_inp": x_inp}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"])
    else:
        return model(ctx["x_inp"])


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _pde_residual_fd(U, dx):
    """Central FD residual: u*u_x - nu*u_xx."""
    u_x = (U[2:] - U[:-2]) / (2 * dx)
    u_xx = (U[2:] - 2 * U[1:-1] + U[:-2]) / dx**2
    uc = U[1:-1]
    return uc * u_x - nu * u_xx


def _bc_loss(U):
    return (U[0] - 1.0)**2 + (U[-1] + 1.0)**2


# ─── Loss functions (standard interface) ─────────────────────────────────────
def loss_vanilla(model, ctx):
    """Autograd-based (no FD)."""
    x_g = ctx["x_inp"].detach().requires_grad_(True)
    u_g = model(x_g)
    u_x = torch.autograd.grad(u_g.sum(), x_g, create_graph=True)[0].squeeze(-1)
    u_xx = torch.autograd.grad(u_x.sum(), x_g, create_graph=True)[0].squeeze(-1)
    res = u_g * u_x - nu * u_xx
    return (res**2).mean() + 10 * _bc_loss(u_g)


def loss_canpinn(model, ctx):
    """PyTorch finite-difference."""
    U = infer(model, ctx)
    res = _pde_residual_fd(U, ctx["dx"])
    return (res**2).mean() + 10 * _bc_loss(U)


def loss_triton(model, ctx):
    """Triton fused kernel."""
    from kernels.stencil_1d_steady import burgers_steady_loss_triton
    U = infer(model, ctx)
    return burgers_steady_loss_triton(U, ctx["dx"], nu) + 10 * _bc_loss(U)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def _exact_u(x):
    return -torch.tanh(x / (2.0 * nu))


def compute_l2_error(model, ctx):
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    U_exact = _exact_u(ctx["X"]).cpu().numpy()
    return float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    x_np = ctx["X"].cpu().numpy()
    U_exact = _exact_u(ctx["X"]).cpu().numpy()
    l2 = float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x_np, U_pred, 'b-', lw=2, label='PINN')
    ax.plot(x_np, U_exact, 'r--', lw=2, label='Exact')
    ax.set_xlabel('x'); ax.set_ylabel('u')
    ax.set_title(f'Steady Burgers ({name}) | L2={l2:.2e}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    """Estimate bytes read+written per training step."""
    return Nx * 4 * 6  # 1 field × Nx × float32 × ~6 stencil accesses

# Aliases for kernel verification scripts
pde_residual_pytorch = _pde_residual_fd

def make_grid(device="cuda"):
    x = torch.linspace(-1, 1, Nx, device=device)
    dx = float(x[1] - x[0])
    return x, dx
