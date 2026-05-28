"""1D Heat Equation (pure diffusion): u_t = nu * u_xx.

Simplest parabolic PDE. No convection, no nonlinearity.
Domain [-1,1] x [0,1], Dirichlet BC u(-1,t)=u(1,t)=0.
IC: u(x,0) = sin(πx).
Exact: u(x,t) = sin(πx) * exp(-ν*π²*t).
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "diffusion_1d"
nu = 0.5
Nx, Nt = 128, 50
GRID_SHAPE = (Nt, Nx)
MLP_THRESHOLD = 1e-5
CNN_THRESHOLD = 1e-4


# ─── Models ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)

    def forward(self, xt):
        return self.out_u(self.net(xt)).squeeze(-1)


class PhyCNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, X, T):
        inputs = torch.stack([X, T], dim=1)
        return self.enc(inputs)[:, 0, :]


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


# ─── Grid & Context ──────────────────────────────────────────────────────────
def make_context(device="cuda"):
    x = torch.linspace(-1, 1, Nx, device=device)
    t = torch.linspace(0, 1, Nt, device=device)
    T, X = torch.meshgrid(t, x, indexing='ij')
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    xt = torch.stack([X.flatten(), T.flatten()], dim=1)
    return {"X": X, "T": T, "dx": dx, "dt": dt, "xt": xt}


# ─── Inference ───────────────────────────────────────────────────────────────
def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"], ctx["T"])
    else:
        return model(ctx["xt"]).reshape(Nt, Nx)


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _ic_loss(U, X):
    return (U[0] - torch.sin(np.pi * X[0])).pow(2).mean()

def _bc_loss(U):
    return U[:, 0].pow(2).mean() + U[:, -1].pow(2).mean()

def _pde_residual_fd(U, dx, dt):
    u_t = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dt)
    u_xx = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
    res = u_t - nu * u_xx
    return (res**2).mean()


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xt_g = ctx["xt"].detach().requires_grad_(True)
    u_g = model(xt_g).reshape(Nt, Nx)
    gu = torch.autograd.grad(u_g.sum(), xt_g, create_graph=True)[0]
    u_x = gu[:, 0].reshape(Nt, Nx)
    u_t = gu[:, 1].reshape(Nt, Nx)
    u_xx = torch.autograd.grad(u_x.sum(), xt_g, create_graph=True)[0][:, 0].reshape(Nt, Nx)
    res = u_t - nu * u_xx
    return (res**2).mean() + 10 * _ic_loss(u_g, ctx["X"]) + 10 * _bc_loss(u_g)


def loss_canpinn(model, ctx):
    U = infer(model, ctx)
    return _pde_residual_fd(U, ctx["dx"], ctx["dt"]) + 10 * _ic_loss(U, ctx["X"]) + 10 * _bc_loss(U)


def loss_triton(model, ctx):
    from kernels.stencil_1d_heat import heat1d_residual_triton
    U = infer(model, ctx)
    return heat1d_residual_triton(U, ctx["dx"], ctx["dt"], nu) + 10 * _ic_loss(U, ctx["X"]) + 10 * _bc_loss(U)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def _exact_u(X, T):
    return torch.sin(np.pi * X) * torch.exp(-nu * np.pi**2 * T)


def compute_l2_error(model, ctx):
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    U_exact = _exact_u(ctx["X"], ctx["T"]).cpu().numpy()
    return float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    U_exact = _exact_u(ctx["X"], ctx["T"]).cpu().numpy()
    x_np = ctx["X"][0].cpu().numpy()
    t_np = ctx["T"][:, 0].cpu().numpy()
    l2 = float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].contourf(x_np, t_np, U_pred, levels=30, cmap='RdBu_r')
    axes[0].set_title(f'Pred — {name}')
    axes[1].contourf(x_np, t_np, U_exact, levels=30, cmap='RdBu_r')
    axes[1].set_title('Exact')
    axes[2].contourf(x_np, t_np, np.abs(U_pred - U_exact), levels=30, cmap='hot_r')
    axes[2].set_title(f'|Error| L2={l2:.2e}')
    plt.suptitle(f'1D Heat Equation — {name}')
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return Nt * Nx * 4 * 5


pde_residual_pytorch = _pde_residual_fd
