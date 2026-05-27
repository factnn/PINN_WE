"""3D Poisson Equation: Δu = f.

Elliptic PDE on [0,1]³. Manufactured solution:
  u_exact = sin(πx) * sin(πy) * sin(πz)
  f = -3π² * sin(πx) * sin(πy) * sin(πz)
Dirichlet BC (u=0 on all faces).
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_NAME = "poisson_3d"
Nx, Ny, Nz = 32, 32, 32
GRID_SHAPE = (Nx, Ny, Nz)
MLP_THRESHOLD = 1e-4
CNN_THRESHOLD = 1e-3


class MLP(nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.out_u = nn.Linear(width, 1)

    def forward(self, xyz):
        return self.out_u(self.net(xyz)).squeeze(-1)


class PhyCNN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(3, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2), nn.Tanh(),
            nn.Conv3d(channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, X, Y, Z):
        inputs = torch.stack([X, Y, Z], dim=0).unsqueeze(0)
        return self.enc(inputs).squeeze(0).squeeze(0)


def make_mlp():
    return MLP()

def make_cnn():
    return PhyCNN()


def make_context(device="cuda"):
    x = torch.linspace(0, 1, Nx, device=device)
    y = torch.linspace(0, 1, Ny, device=device)
    z = torch.linspace(0, 1, Nz, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])
    xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    F = -3 * np.pi**2 * torch.sin(np.pi * X) * torch.sin(np.pi * Y) * torch.sin(np.pi * Z)
    return {"X": X, "Y": Y, "Z": Z, "dx": dx, "dy": dy, "dz": dz, "xyz": xyz, "F": F}


def infer(model, ctx):
    inner = getattr(model, '_orig_mod', model)
    if isinstance(inner, PhyCNN):
        return model(ctx["X"], ctx["Y"], ctx["Z"])
    else:
        return model(ctx["xyz"]).reshape(Nx, Ny, Nz)


def _pde_residual_fd(U, F, dx, dy, dz):
    i = slice(1, -1)
    u_xx = (U[2:, i, i] - 2*U[i, i, i] + U[:-2, i, i]) / dx**2
    u_yy = (U[i, 2:, i] - 2*U[i, i, i] + U[i, :-2, i]) / dy**2
    u_zz = (U[i, i, 2:] - 2*U[i, i, i] + U[i, i, :-2]) / dz**2
    res = u_xx + u_yy + u_zz - F[i, i, i]
    return (res**2).mean()


def _bc_loss(U):
    return (U[0]**2 + U[-1]**2 + U[:, 0]**2 + U[:, -1]**2 + U[:, :, 0]**2 + U[:, :, -1]**2).mean()


def loss_vanilla(model, ctx):
    xyz_g = ctx["xyz"].detach().requires_grad_(True)
    u_g = model(xyz_g).reshape(Nx, Ny, Nz)
    gu = torch.autograd.grad(u_g.sum(), xyz_g, create_graph=True)[0]
    u_x = gu[:, 0].reshape(Nx, Ny, Nz)
    u_y = gu[:, 1].reshape(Nx, Ny, Nz)
    u_z = gu[:, 2].reshape(Nx, Ny, Nz)
    u_xx = torch.autograd.grad(u_x.sum(), xyz_g, create_graph=True)[0][:, 0].reshape(Nx, Ny, Nz)
    u_yy = torch.autograd.grad(u_y.sum(), xyz_g, create_graph=True)[0][:, 1].reshape(Nx, Ny, Nz)
    u_zz = torch.autograd.grad(u_z.sum(), xyz_g, create_graph=True)[0][:, 2].reshape(Nx, Ny, Nz)
    res = u_xx + u_yy + u_zz - ctx["F"]
    return (res**2).mean() + 10 * _bc_loss(u_g)


def loss_canpinn(model, ctx):
    U = infer(model, ctx)
    return _pde_residual_fd(U, ctx["F"], ctx["dx"], ctx["dy"], ctx["dz"]) + 10 * _bc_loss(U)


def loss_triton(model, ctx):
    from kernels.stencil_3d_poisson import poisson3d_residual_triton
    U = infer(model, ctx)
    return poisson3d_residual_triton(U, ctx["F"], ctx["dx"], ctx["dy"], ctx["dz"]) + 10 * _bc_loss(U)


def _exact_u(X, Y, Z):
    return torch.sin(np.pi * X) * torch.sin(np.pi * Y) * torch.sin(np.pi * Z)


def compute_l2_error(model, ctx):
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    U_exact = _exact_u(ctx["X"], ctx["Y"], ctx["Z"]).cpu().numpy()
    return float(np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12))


def plot_solution(model, ctx, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        U_pred = infer(model, ctx).cpu().numpy()
    U_exact = _exact_u(ctx["X"], ctx["Y"], ctx["Z"]).cpu().numpy()
    x_np = ctx["X"][:, 0, 0].cpu().numpy()
    y_np = ctx["Y"][0, :, 0].cpu().numpy()
    zi = Nz // 2
    l2 = compute_l2_error(model, ctx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].contourf(x_np, y_np, U_pred[:, :, zi].T, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=axes[0]); axes[0].set_title(f'Pred z-mid — {name}')
    im2 = axes[1].contourf(x_np, y_np, U_exact[:, :, zi].T, levels=20, cmap='RdBu_r')
    plt.colorbar(im2, ax=axes[1]); axes[1].set_title('Exact z-mid')
    err = np.abs(U_pred[:, :, zi] - U_exact[:, :, zi])
    im3 = axes[2].contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
    plt.colorbar(im3, ax=axes[2]); axes[2].set_title('|Error|')
    plt.suptitle(f'3D Poisson z-mid ({name}) L2={l2:.2e}')
    plt.tight_layout()
    plt.savefig(out_dir / f"solution_{name}.png", dpi=150)
    plt.close()


def bytes_per_step():
    return Nx * Ny * Nz * 4 * 8


pde_residual_pytorch = _pde_residual_fd
