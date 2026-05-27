"""1D Diffusion-dominated Burgers (large viscosity, smooth solution).

PDE: u_t + u*u_x = nu*u_xx, nu=0.5 (diffusion dominates).
Smooth solution, no shock. Tests kernel on parabolic PDE.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "diffusion_1d"
nu = 0.5  # large viscosity, smooth diffusion-dominated solution
Nx, Nt = 1024, 100
GRID_SHAPE = (Nt, Nx)
MLP_THRESHOLD = 1e-4
CNN_THRESHOLD = 1e-4
MLP_THRESHOLD = 1e-3
CNN_THRESHOLD = 5e-4


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
        inputs = torch.stack([X, T], dim=1)  # [Nt, 2, Nx]
        return self.enc(inputs)[:, 0, :]  # [Nt, Nx]


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
    return (U[0] - (-torch.sin(np.pi * X[0]))).pow(2).mean()

def _bc_loss(U):
    return U[:, 0].pow(2).mean() + U[:, -1].pow(2).mean()

def _pde_residual_fd(U, dx, dt):
    u_t = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dt)
    u_x = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dx)
    u_xx = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
    uc = U[1:-1, 1:-1]
    res = u_t + uc * u_x - nu * u_xx
    return (res**2).mean()


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xt_g = ctx["xt"].detach().requires_grad_(True)
    u_g = model(xt_g).reshape(Nt, Nx)
    gu = torch.autograd.grad(u_g.sum(), xt_g, create_graph=True)[0]
    u_x = gu[:, 0].reshape(Nt, Nx)
    u_t = gu[:, 1].reshape(Nt, Nx)
    u_xx = torch.autograd.grad(u_x.sum(), xt_g, create_graph=True)[0][:, 0].reshape(Nt, Nx)
    res = u_t + u_g * u_x - nu * u_xx
    return (res**2).mean() + 10 * _ic_loss(u_g, ctx["X"]) + 10 * _bc_loss(u_g)


def loss_canpinn(model, ctx):
    U = infer(model, ctx)
    return _pde_residual_fd(U, ctx["dx"], ctx["dt"]) + 10 * _ic_loss(U, ctx["X"]) + 10 * _bc_loss(U)


def loss_triton(model, ctx):
    from kernels.stencil_1d_unsteady import burgers_2d_loss_triton_autograd
    U = infer(model, ctx)
    return burgers_2d_loss_triton_autograd(U, ctx["dx"], ctx["dt"], nu) + 10 * _ic_loss(U, ctx["X"]) + 10 * _bc_loss(U)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def _exact_solution(x_np, t_np):
    dx = x_np[1] - x_np[0]
    diag_ = -2 * np.ones(Nx - 2)
    off = np.ones(Nx - 3)
    D2 = diags([off, diag_, off], [-1, 0, 1]).toarray() / dx**2

    def rhs(t, u_int):
        u = np.concatenate([[0.], u_int, [0.]])
        u_x = (u[2:] - u[:-2]) / (2 * dx)
        return -u_int * u_x + nu * D2 @ u_int

    u0 = -np.sin(np.pi * x_np[1:-1])
    t_eval = t_np[t_np > 0]
    sol = solve_ivp(rhs, [0, t_np[-1]], u0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10)
    U = np.zeros((len(t_np), Nx))
    U[0] = -np.sin(np.pi * x_np)
    U[t_np > 0, 1:-1] = sol.y.T
    return U


def compute_l2_error(model, ctx):
    x_np = ctx["X"][0].cpu().numpy()
    t_np = ctx["T"][:, 0].cpu().numpy()
    U_exact = _exact_solution(x_np, t_np)
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    return float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    x_np = ctx["X"][0].cpu().numpy()
    t_np = ctx["T"][:, 0].cpu().numpy()
    U_exact = _exact_solution(x_np, t_np)
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    l2 = float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))

    slices = [(0, 't=0'), (25, 't=0.25'), (50, 't=0.5'), (75, 't=0.75'), (99, 't=1.0')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].contourf(x_np, t_np, U_pred, levels=50, cmap='RdBu_r')
    axes[0].set_title(f'PINN - {name}')
    axes[1].contourf(x_np, t_np, U_exact, levels=50, cmap='RdBu_r')
    axes[1].set_title('Exact')
    colors = plt.cm.viridis(np.linspace(0, 1, len(slices)))
    for (ti, label), c in zip(slices, colors):
        axes[2].plot(x_np, U_pred[ti], color=c, lw=2, label=f'PINN {label}')
        axes[2].plot(x_np, U_exact[ti], color=c, lw=1.5, ls='--')
    axes[2].set_title(f'Slices (L2={l2:.2e})')
    axes[2].legend(fontsize=6, ncol=2)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return Nt * Nx * 4 * 6

# Aliases for kernel verification scripts
pde_residual_pytorch = _pde_residual_fd
