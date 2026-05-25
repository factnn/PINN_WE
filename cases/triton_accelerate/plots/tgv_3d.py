"""Plotting for 3D Taylor-Green Vortex case.

Figures:
- Kinetic energy decay KE(t) = 0.5*mean(u^2+v^2+w^2) (all methods on one plot)
- dKE/dt dissipation rate curve
- Q-criterion on z-mid plane (contourf) at multiple timesteps
- z-mid speed contourf at multiple timesteps
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


def _compute_q_criterion_zmid(U, V, W, dx, dy, zi):
    """Compute Q-criterion on the z-mid plane: Q = 0.5*(||Omega||^2 - ||S||^2).

    Simplified 2D version using du/dx, du/dy, dv/dx, dv/dy at z=zi.

    Args:
        U, V, W: [Nt, Nx, Ny, Nz]
        dx, dy: grid spacing
        zi: z-index for midplane

    Returns:
        Q: [Nt, Nx-2, Ny-2]
    """
    u = U[:, :, :, zi]  # [Nt, Nx, Ny]
    v = V[:, :, :, zi]
    u_x = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
    u_y = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dy)
    v_x = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2 * dx)
    v_y = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * dy)
    # S = 0.5*(grad u + grad u^T), Omega = 0.5*(grad u - grad u^T)
    # Q = 0.5*(Omega_ij*Omega_ij - S_ij*S_ij) in 2D slice
    S11 = u_x; S22 = v_y; S12 = 0.5 * (u_y + v_x)
    O12 = 0.5 * (v_x - u_y)
    Q = O12**2 - (S11**2 + S22**2 + 2 * S12**2)
    return Q


def plot_all(physics, ctx, models, out_dir):
    """Generate all plots for tgv_3d."""
    out_dir = Path(out_dir)
    x_np = ctx["X"][0, :, 0, 0].cpu().numpy()
    y_np = ctx["Y"][0, 0, :, 0].cpu().numpy()
    t_np = ctx["T"][:, 0, 0, 0].cpu().numpy()
    dx = ctx["dx"]
    dy = ctx["dy"]
    Nt = physics.Nt
    Nx = physics.Nx
    Ny = physics.Ny
    Nz = physics.Nz
    zi = Nz // 2

    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            U, V, W, P = physics.infer(model, ctx)
        predictions[backend] = (U.cpu().numpy(), V.cpu().numpy(),
                                W.cpu().numpy(), P.cpu().numpy())

    # --- Kinetic Energy Decay (CORE PLOT) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        for backend in group:
            if backend in predictions:
                U, V, W, P = predictions[backend]
                KE = 0.5 * np.mean(U**2 + V**2 + W**2, axis=(1, 2, 3))
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(t_np, KE, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('t'); ax.set_ylabel('KE(t)')
        ax.set_title(f'Kinetic Energy Decay - {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('TGV 3D - Kinetic Energy Decay', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "kinetic_energy_decay")

    # --- dKE/dt Dissipation Rate ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        for backend in group:
            if backend in predictions:
                U, V, W, P = predictions[backend]
                KE = 0.5 * np.mean(U**2 + V**2 + W**2, axis=(1, 2, 3))
                # Central difference for dKE/dt
                dKE_dt = np.gradient(KE, t_np)
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(t_np, -dKE_dt, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('t'); ax.set_ylabel('-dKE/dt (dissipation)')
        ax.set_title(f'Energy Dissipation Rate - {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('TGV 3D - Energy Dissipation Rate', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "dissipation_rate")

    # --- Q-criterion on z-mid plane at multiple timesteps ---
    time_indices = [0, Nt // 2, Nt - 1]
    time_labels = ['t=0', f't={t_np[Nt//2]:.2f}', f't={t_np[-1]:.2f}']
    x_int = x_np[1:-1]
    y_int = y_np[1:-1]

    for backend, (U, V, W, P) in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        label = BACKEND_LABELS.get(backend, backend)
        Q = _compute_q_criterion_zmid(U, V, W, dx, dy, zi)

        for col, (ti, tl) in enumerate(zip(time_indices, time_labels)):
            im = axes[col].contourf(x_int, y_int, Q[ti].T, levels=30, cmap='RdBu_r')
            plt.colorbar(im, ax=axes[col])
            axes[col].set_xlabel('x'); axes[col].set_ylabel('y')
            axes[col].set_title(f'Q-criterion at {tl}')
            axes[col].set_aspect('equal')

        plt.suptitle(f'TGV 3D Q-criterion (z-mid) - {label}', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"q_criterion_{backend}")

    # --- z-mid speed contourf at multiple timesteps ---
    for backend, (U, V, W, P) in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        label = BACKEND_LABELS.get(backend, backend)

        for col, (ti, tl) in enumerate(zip(time_indices, time_labels)):
            speed = np.sqrt(U[ti, :, :, zi]**2 + V[ti, :, :, zi]**2 + W[ti, :, :, zi]**2)
            im = axes[col].contourf(x_np, y_np, speed.T, levels=20, cmap='viridis')
            plt.colorbar(im, ax=axes[col])
            axes[col].set_xlabel('x'); axes[col].set_ylabel('y')
            axes[col].set_title(f'Speed at {tl}')
            axes[col].set_aspect('equal')

        plt.suptitle(f'TGV 3D Speed (z-mid) - {label}', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"speed_{backend}")


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
    import cases.tgv_3d.physics as physics
    plot_from_checkpoints(physics, "output/tgv_3d/figures")
