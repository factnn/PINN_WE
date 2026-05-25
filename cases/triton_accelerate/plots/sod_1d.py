"""Plotting for Sod Shock Tube (2D grid) case.

Figures:
- Three-panel (rho/u/p) profiles at final time for each method
- Combined comparison: all methods on one plot for rho(x)
- Multi-timestep evolution of density
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from plots import (
    BACKEND_COLORS, BACKEND_LABELS, BACKEND_LINESTYLES,
    ALL_BACKENDS, MLP_BACKENDS, CNN_BACKENDS, save_fig
)


def _sod_exact_riemann(x, t, gamma=1.4):
    """Compute exact Riemann solution for Sod shock tube at time t.

    Returns rho, u, p arrays on x grid.
    Uses standard Sod IC: (rho_L, u_L, p_L) = (1, 0, 1), (rho_R, u_R, p_R) = (0.125, 0, 0.1).
    """
    if t <= 0:
        rho = np.where(x < 0.5, 1.0, 0.125)
        u = np.zeros_like(x)
        p = np.where(x < 0.5, 1.0, 0.1)
        return rho, u, p

    # Exact solution parameters for standard Sod problem
    # Post-shock state (region 3)
    p3 = 0.30313
    rho3 = 0.26557
    u3 = 0.92745
    # Rarefaction head/tail speeds, contact, shock speed
    c1 = np.sqrt(gamma * 1.0 / 1.0)  # sound speed left
    c3 = np.sqrt(gamma * p3 / 0.42632)  # sound speed in region 2 (left of contact)
    rho2 = 0.42632
    # Wave speeds
    x0 = 0.5
    s_head = x0 - c1 * t                    # rarefaction head
    s_tail = x0 + (u3 - c3) * t             # rarefaction tail
    s_contact = x0 + u3 * t                  # contact discontinuity
    s_shock = x0 + 1.7522 * t               # shock speed

    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < s_head:
            # Region 1 (undisturbed left)
            rho[i] = 1.0; u[i] = 0.0; p[i] = 1.0
        elif x[i] < s_tail:
            # Rarefaction fan
            xi = (x[i] - x0) / t
            u[i] = (2.0 / (gamma + 1)) * (c1 + xi)
            c = c1 - 0.5 * (gamma - 1) * u[i]
            rho[i] = 1.0 * (c / c1) ** (2.0 / (gamma - 1))
            p[i] = 1.0 * (c / c1) ** (2.0 * gamma / (gamma - 1))
        elif x[i] < s_contact:
            # Region 2 (between rarefaction tail and contact)
            rho[i] = rho2; u[i] = u3; p[i] = p3
        elif x[i] < s_shock:
            # Region 3 (between contact and shock)
            rho[i] = rho3; u[i] = u3; p[i] = p3
        else:
            # Region 4 (undisturbed right)
            rho[i] = 0.125; u[i] = 0.0; p[i] = 0.1

    return rho, u, p


def plot_all(physics, ctx, models, out_dir):
    """Generate all plots for sod_1d."""
    out_dir = Path(out_dir)
    x_np = ctx["X"][0].cpu().numpy()
    t_np = ctx["T"][:, 0].cpu().numpy()
    Nt = physics.Nt
    Nx = physics.Nx
    T_final = physics.T_final
    gamma = physics.gamma

    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            rho, rhou, E = physics.infer(model, ctx)
        rho_np = rho.cpu().numpy()
        rhou_np = rhou.cpu().numpy()
        E_np = E.cpu().numpy()
        u_np = rhou_np / (rho_np + 1e-10)
        p_np = (gamma - 1) * (E_np - 0.5 * rho_np * u_np**2)
        predictions[backend] = (rho_np, u_np, p_np)

    # Exact Riemann solution at final time
    rho_ex, u_ex, p_ex = _sod_exact_riemann(x_np, T_final, gamma)

    # --- Individual: three-panel (rho/u/p) at final time ---
    for backend, (rho_np, u_np, p_np) in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        label = BACKEND_LABELS.get(backend, backend)

        axes[0].plot(x_np, rho_np[-1], 'b-', lw=1.5, label='PINN')
        axes[0].plot(x_np, rho_ex, 'k--', lw=1.5, label='Exact')
        axes[0].set_xlabel('x'); axes[0].set_ylabel(r'$\rho$')
        axes[0].set_title('Density'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(x_np, u_np[-1], 'r-', lw=1.5, label='PINN')
        axes[1].plot(x_np, u_ex, 'k--', lw=1.5, label='Exact')
        axes[1].set_xlabel('x'); axes[1].set_ylabel('u')
        axes[1].set_title('Velocity'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(x_np, p_np[-1], 'g-', lw=1.5, label='PINN')
        axes[2].plot(x_np, p_ex, 'k--', lw=1.5, label='Exact')
        axes[2].set_xlabel('x'); axes[2].set_ylabel('p')
        axes[2].set_title('Pressure'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Sod Shock Tube - {label} (t={T_final})', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"solution_{backend}")

    # --- Combined comparison: all methods for rho(x) at final time ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        ax.plot(x_np, rho_ex, 'k--', lw=2.5, label='Exact (Riemann)', zorder=10)
        for backend in group:
            if backend in predictions:
                rho_np, u_np, p_np = predictions[backend]
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(x_np, rho_np[-1], color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('x'); ax.set_ylabel(r'$\rho$')
        ax.set_title(f'Density at t={T_final} - {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Sod Shock Tube - Density Comparison', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "density_comparison")

    # --- Multi-timestep evolution of density ---
    n_snapshots = min(5, Nt)
    t_indices = np.linspace(0, Nt - 1, n_snapshots, dtype=int)

    for backend, (rho_np, u_np, p_np) in predictions.items():
        fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
        label = BACKEND_LABELS.get(backend, backend)
        if n_snapshots == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
        for col, ti in enumerate(t_indices):
            # Exact at this time
            rho_ex_t, _, _ = _sod_exact_riemann(x_np, t_np[ti], gamma)
            axes[col].plot(x_np, rho_ex_t, 'k--', lw=1.5, label='Exact')
            axes[col].plot(x_np, rho_np[ti], color=colors[col], lw=1.5, label='PINN')
            axes[col].set_xlabel('x'); axes[col].set_ylabel(r'$\rho$')
            axes[col].set_title(f't={t_np[ti]:.3f}')
            axes[col].legend(fontsize=7); axes[col].grid(True, alpha=0.3)
            axes[col].set_ylim(0, 1.2)

        plt.suptitle(f'Sod Shock Tube Density Evolution - {label}', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"evolution_{backend}")


def plot_from_checkpoints(physics, out_dir, model_dir=None):
    """Load checkpoints and generate plots."""
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.utils import BACKENDS

    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "output" / physics.CASE_NAME
    model_dir = Path(model_dir)

    ctx = physics.make_context("cuda")
    models = {}

    for backend in BACKENDS:
        ckpt = model_dir / f"model_{backend}.pt"
        if ckpt.exists():
            is_cnn = "cnn" in backend
            model = physics.make_cnn() if is_cnn else physics.make_mlp()
            model.load_state_dict(torch.load(ckpt, map_location="cuda"))
            model = model.cuda().eval()
            models[backend] = model

    if models:
        plot_all(physics, ctx, models, out_dir)
    else:
        print(f"  No checkpoints found in {model_dir}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    import cases.sod_1d.physics as physics
    plot_from_checkpoints(physics, "output/sod_1d/figures")
