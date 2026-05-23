"""Sod 2D: physics module.

1D Sod shock tube on 2D grid (compressible Euler equations).
Euler: rho_t + (rho*u)_x = 0, (rho*u)_t + (rho*u^2+p)_x = 0, E_t + ((E+p)*u)_x = 0
EOS: p = (gamma-1)(E - 0.5*rho*u^2), gamma = 1.4
Domain [0,1] x [0, 0.2], IC: left (rho=1,u=0,p=1), right (rho=0.125,u=0,p=0.1).
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Constants ───────────────────────────────────────────────────────────────
CASE_NAME = "sod_2d"
gamma = 1.4
Nx, Nt = 200, 50
T_final = 0.2
GRID_SHAPE = (Nt, Nx)


# ─── Models ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Point-wise MLP: (x, t) -> (rho, rhou, E)"""
    def __init__(self, width=256, depth=4):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_rho = nn.Linear(width, 1)
        self.out_rhou = nn.Linear(width, 1)
        self.out_E = nn.Linear(width, 1)

    def forward(self, xt):
        h = self.net(xt)
        rho = torch.abs(self.out_rho(h).squeeze(-1)) + 1e-6
        rhou = self.out_rhou(h).squeeze(-1)
        E = torch.abs(self.out_E(h).squeeze(-1)) + 1e-6
        return rho, rhou, E


class PhyCNN(nn.Module):
    """Grid-based 1D CNN: (X, T) -> (rho, rhou, E) on [Nt, Nx]"""
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv1d(channels, 3, kernel_size=5, padding=2),
        )

    def forward(self, X, T):
        inputs = torch.stack([X, T], dim=1)  # [Nt, 2, Nx]
        out = self.enc(inputs)  # [Nt, 3, Nx]
        rho = torch.abs(out[:, 0, :]) + 1e-6
        rhou = out[:, 1, :]
        E = torch.abs(out[:, 2, :]) + 1e-6
        return rho, rhou, E


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


# ─── Grid & Context ──────────────────────────────────────────────────────────
def make_context(device="cuda"):
    x = torch.linspace(0, 1, Nx, device=device)
    t = torch.linspace(0, T_final, Nt, device=device)
    T, X = torch.meshgrid(t, x, indexing='ij')  # [Nt, Nx]
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
        rho, rhou, E = model(ctx["xt"])
        return rho.reshape(Nt, Nx), rhou.reshape(Nt, Nx), E.reshape(Nt, Nx)


# ─── PDE helpers ─────────────────────────────────────────────────────────────
def _sod_ic(X):
    """Sod shock tube initial conditions."""
    rho = torch.where(X < 0.5, torch.ones_like(X), 0.125 * torch.ones_like(X))
    u = torch.zeros_like(X)
    p = torch.where(X < 0.5, torch.ones_like(X), 0.1 * torch.ones_like(X))
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return rho, rho * u, E


def _euler_residual_fd(rho, rhou, E, dx, dt):
    """1D Euler residual via central differences on [Nt, Nx] grid."""
    u = rhou / (rho + 1e-10)
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    f1 = rhou
    f2 = rhou * u + p
    f3 = (E + p) * u
    rho_t = (rho[2:, 1:-1] - rho[:-2, 1:-1]) / (2*dt)
    rhou_t = (rhou[2:, 1:-1] - rhou[:-2, 1:-1]) / (2*dt)
    E_t = (E[2:, 1:-1] - E[:-2, 1:-1]) / (2*dt)
    f1_x = (f1[1:-1, 2:] - f1[1:-1, :-2]) / (2*dx)
    f2_x = (f2[1:-1, 2:] - f2[1:-1, :-2]) / (2*dx)
    f3_x = (f3[1:-1, 2:] - f3[1:-1, :-2]) / (2*dx)
    res1 = rho_t + f1_x
    res2 = rhou_t + f2_x
    res3 = E_t + f3_x
    return (res1**2 + res2**2 + res3**2).mean()


def _ic_loss(rho, rhou, E, X):
    """Initial condition loss at t=0."""
    rho_ic, rhou_ic, E_ic = _sod_ic(X[0])
    return ((rho[0] - rho_ic)**2).mean() + \
           ((rhou[0] - rhou_ic)**2).mean() + \
           ((E[0] - E_ic)**2).mean()


def _bc_loss(rho, rhou, E):
    """Outflow BC: zero-gradient at x boundaries."""
    bc = ((rho[:, 0] - rho[:, 1])**2 + (rho[:, -1] - rho[:, -2])**2).mean()
    bc += ((rhou[:, 0] - rhou[:, 1])**2 + (rhou[:, -1] - rhou[:, -2])**2).mean()
    bc += ((E[:, 0] - E[:, 1])**2 + (E[:, -1] - E[:, -2])**2).mean()
    return bc


# ─── Loss functions ──────────────────────────────────────────────────────────
def loss_vanilla(model, ctx):
    xt_g = ctx["xt"].detach().requires_grad_(True)
    rho, rhou, E = model(xt_g)
    rho = rho.reshape(Nt, Nx); rhou = rhou.reshape(Nt, Nx); E = E.reshape(Nt, Nx)
    u = rhou / (rho + 1e-10)
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    g_rho = torch.autograd.grad(rho.sum(), xt_g, create_graph=True)[0]
    rho_x = g_rho[:, 0].reshape(Nt, Nx); rho_t = g_rho[:, 1].reshape(Nt, Nx)
    g_rhou = torch.autograd.grad(rhou.sum(), xt_g, create_graph=True)[0]
    rhou_x = g_rhou[:, 0].reshape(Nt, Nx); rhou_t = g_rhou[:, 1].reshape(Nt, Nx)
    g_E = torch.autograd.grad(E.sum(), xt_g, create_graph=True)[0]
    E_x = g_E[:, 0].reshape(Nt, Nx); E_t = g_E[:, 1].reshape(Nt, Nx)
    g_p = torch.autograd.grad(p.sum(), xt_g, create_graph=True)[0]
    p_x = g_p[:, 0].reshape(Nt, Nx)
    res1 = rho_t + rhou_x
    res2 = rhou_t + torch.autograd.grad((rhou*u).sum(), xt_g, create_graph=True)[0][:, 0].reshape(Nt, Nx) + p_x
    res3 = E_t + torch.autograd.grad(((E+p)*u).sum(), xt_g, create_graph=True)[0][:, 0].reshape(Nt, Nx)
    pde = (res1**2 + res2**2 + res3**2).mean()
    return pde + 10 * _ic_loss(rho, rhou, E, ctx["X"]) + _bc_loss(rho, rhou, E)


def loss_canpinn(model, ctx):
    rho, rhou, E = infer(model, ctx)
    return _euler_residual_fd(rho, rhou, E, ctx["dx"], ctx["dt"]) + \
           10 * _ic_loss(rho, rhou, E, ctx["X"]) + _bc_loss(rho, rhou, E)


def loss_triton(model, ctx):
    from kernels.stencil_2d_compressible import compressible_sod_residual_triton
    rho, rhou, E = infer(model, ctx)
    return compressible_sod_residual_triton(rho, rhou, E, ctx["dx"], ctx["dt"], gamma) + \
           10 * _ic_loss(rho, rhou, E, ctx["X"]) + _bc_loss(rho, rhou, E)


# ─── Evaluation ──────────────────────────────────────────────────────────────
def compute_l2_error(model, ctx):
    """L2 error of IC reproduction (no closed-form for t>0)."""
    with torch.no_grad():
        rho, rhou, E = infer(model, ctx)
    rho_ic, rhou_ic, E_ic = _sod_ic(ctx["X"][0])
    err = torch.sqrt(((rho[0]-rho_ic)**2 + (rhou[0]-rhou_ic)**2 + (E[0]-E_ic)**2).mean())
    ref = torch.sqrt((rho_ic**2 + rhou_ic**2 + E_ic**2).mean())
    return float(err / (ref + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    with torch.no_grad():
        rho, rhou, E = infer(model, ctx)
    rho_np = rho.cpu().numpy()
    rhou_np = rhou.cpu().numpy()
    E_np = E.cpu().numpy()
    x_np = ctx["X"][0].cpu().numpy()

    u_np = rhou_np / (rho_np + 1e-10)
    p_np = (gamma - 1) * (E_np - 0.5 * rho_np * u_np**2)
    l2 = compute_l2_error(model, ctx)
    ti = -1  # final time

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(x_np, rho_np[ti], 'b-', lw=2)
    axes[0].set_title('Density rho'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(x_np, u_np[ti], 'r-', lw=2)
    axes[1].set_title('Velocity u'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(x_np, p_np[ti], 'g-', lw=2)
    axes[2].set_title('Pressure p'); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'Sod Shock Tube ({name}) | t={T_final} | IC L2={l2:.2e}', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return 3 * Nt * Nx * 4 * 10  # rho,rhou,E × grid × float32 × stencil accesses
