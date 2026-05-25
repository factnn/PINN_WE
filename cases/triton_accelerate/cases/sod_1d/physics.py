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
CASE_NAME = "sod_1d"
gamma = 1.4
Nx, Nt = 1000, 200
T_final = 0.2
GRID_SHAPE = (Nt, Nx)
MLP_THRESHOLD = 5e-2
CNN_THRESHOLD = 5e-2


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


# ─── Riemann Exact Solver ─────────────────────────────────────────────────────
def _riemann_exact(x_arr, t, rhoL=1.0, uL=0.0, pL=1.0,
                   rhoR=0.125, uR=0.0, pR=0.1, x0=0.5, gam=1.4):
    """Exact Riemann solution for Sod shock tube at time t.

    Returns rho, u, p arrays on x_arr grid.
    Standard algorithm: find p_star by Newton iteration, then sample waves.
    """
    if t <= 0:
        rho = np.where(x_arr < x0, rhoL, rhoR)
        u = np.where(x_arr < x0, uL, uR)
        p = np.where(x_arr < x0, pL, pR)
        return rho, u, p

    g = gam
    g1 = (g - 1) / (2 * g)
    g2 = (g + 1) / (2 * g)
    g3 = 2 * g / (g - 1)
    g4 = 2 / (g - 1)
    g5 = 2 / (g + 1)
    g6 = (g - 1) / (g + 1)
    g7 = (g - 1) / 2

    aL = np.sqrt(g * pL / rhoL)
    aR = np.sqrt(g * pR / rhoR)

    # Newton iteration for p_star
    def f(p, rho_k, p_k, a_k):
        if p > p_k:  # shock
            A = g5 / rho_k
            B = g6 * p_k
            return (p - p_k) * np.sqrt(A / (p + B))
        else:  # rarefaction
            return g4 * a_k * ((p / p_k)**g1 - 1)

    def df(p, rho_k, p_k, a_k):
        if p > p_k:
            A = g5 / rho_k
            B = g6 * p_k
            sq = np.sqrt(A / (p + B))
            return sq * (1 - (p - p_k) / (2 * (p + B)))
        else:
            return (1 / (rho_k * a_k)) * (p / p_k)**(-(g + 1) / (2 * g))

    # Initial guess (Two-Rarefaction approximation)
    p_star = ((aL + aR - g7 * (uR - uL)) /
              (aL / pL**g1 + aR / pR**g1))**(1 / g1)

    for _ in range(50):
        fL = f(p_star, rhoL, pL, aL)
        fR = f(p_star, rhoR, pR, aR)
        fp = fL + fR + (uR - uL)
        dfp = df(p_star, rhoL, pL, aL) + df(p_star, rhoR, pR, aR)
        dp = -fp / dfp
        p_star = max(p_star + dp, 1e-15)
        if abs(dp) < 1e-12 * p_star:
            break

    u_star = 0.5 * (uL + uR) + 0.5 * (f(p_star, rhoR, pR, aR) - f(p_star, rhoL, pL, aL))

    # Sample solution
    rho = np.empty_like(x_arr)
    u = np.empty_like(x_arr)
    p = np.empty_like(x_arr)

    S = (x_arr - x0) / t  # similarity variable

    for i in range(len(x_arr)):
        s = S[i]
        if s < u_star:  # Left of contact
            if p_star <= pL:  # Left rarefaction
                aL_star = aL * (p_star / pL)**g1
                sHL = uL - aL
                sTL = u_star - aL_star
                if s <= sHL:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                elif s >= sTL:
                    rho[i] = rhoL * (p_star / pL)**(1/g)
                    u[i] = u_star
                    p[i] = p_star
                else:
                    u[i] = g5 * (aL + g7 * uL + s)
                    a = g5 * (aL - g7 * (uL - s))
                    rho[i] = rhoL * (a / aL)**g4
                    p[i] = pL * (a / aL)**g3
            else:  # Left shock
                sL = uL - aL * np.sqrt(g2 * p_star / pL + g1)
                if s <= sL:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                else:
                    rho[i] = rhoL * (p_star/pL + g6) / (g6 * p_star/pL + 1)
                    u[i] = u_star
                    p[i] = p_star
        else:  # Right of contact
            if p_star <= pR:  # Right rarefaction
                aR_star = aR * (p_star / pR)**g1
                sHR = uR + aR
                sTR = u_star + aR_star
                if s >= sHR:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                elif s <= sTR:
                    rho[i] = rhoR * (p_star / pR)**(1/g)
                    u[i] = u_star
                    p[i] = p_star
                else:
                    u[i] = g5 * (-aR + g7 * uR + s)
                    a = g5 * (aR - g7 * (s - uR))
                    rho[i] = rhoR * (a / aR)**g4
                    p[i] = pR * (a / aR)**g3
            else:  # Right shock
                sR = uR + aR * np.sqrt(g2 * p_star / pR + g1)
                if s >= sR:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                else:
                    rho[i] = rhoR * (p_star/pR + g6) / (g6 * p_star/pR + 1)
                    u[i] = u_star
                    p[i] = p_star

    return rho, u, p


# ─── Evaluation ──────────────────────────────────────────────────────────────
def compute_l2_error(model, ctx):
    """L2 error vs exact Riemann solution at final time."""
    with torch.no_grad():
        rho_pred, rhou_pred, E_pred = infer(model, ctx)
    rho_np = rho_pred[-1].cpu().numpy()
    rhou_np = rhou_pred[-1].cpu().numpy()
    E_np = E_pred[-1].cpu().numpy()
    x_np = ctx["X"][0].cpu().numpy()
    u_np = rhou_np / (rho_np + 1e-10)

    rho_ex, u_ex, p_ex = _riemann_exact(x_np, T_final)
    err = np.sqrt(np.mean((rho_np - rho_ex)**2 + (u_np - u_ex)**2))
    ref = np.sqrt(np.mean(rho_ex**2 + u_ex**2))
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
