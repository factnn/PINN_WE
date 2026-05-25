"""Plotting for 2D Taylor-Green Vortex case.

Figures:
- Vorticity contourf at t=0, t_mid, t_final (for each method)
- Kinetic energy KE(t) decay curve (all methods + exact on one plot)
- Combined velocity/vorticity comparison
- Error field |u_pred - u_exact| contourf
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


def _compute_vorticity(U, V, dx, dy):
    """Compute vorticity omega = dv/dx - du/dy via central differences.

    Args:
        U, V: [Nt, Nx, Ny] velocity fields
        dx, dy: grid spacing

    Returns:
        omega: [Nt, Nx-2, Ny-2] vorticity on interior
    """
    v_x = (V[:, 2:, 1:-1] - V[:, :-2, 1:-1]) / (2 * dx)
    u_y = (U[:, 1:-1, 2:] - U[:, 1:-1, :-2]) / (2 * dy)
    return v_x - u_y


def plot_all(physics, ctx, models, out_dir):
    """Generate all plots for tgv_2d."""
    out_dir = Path(out_dir)
    x_np = ctx["X"][0, :, 0].cpu().numpy()
    y_np = ctx["Y"][0, 0, :].cpu().numpy()
    t_np = ctx["T"][:, 0, 0].cpu().numpy()
    dx = ctx["dx"]
    dy = ctx["dy"]
    Nt = physics.Nt
    Nx = physics.Nx
    Ny = physics.Ny

    U_exact = ctx["U_exact"].cpu().numpy()
    V_exact = ctx["V_exact"].cpu().numpy()

    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            U, V, P = physics.infer(model, ctx)
        predictions[backend] = (U.cpu().numpy(), V.cpu().numpy(), P.cpu().numpy())

    # --- Kinetic energy decay (MOST IMPORTANT) ---
    # Exact KE
    KE_exact = 0.5 * np.mean(U_exact**2 + V_exact**2, axis=(1, 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        ax.plot(t_np, KE_exact, 'k--', lw=2.5, label='Exact', zorder=10)
        for backend in group:
            if backend in predictions:
                U, V, P = predictions[backend]
                KE = 0.5 * np.mean(U**2 + V**2, axis=(1, 2))
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(t_np, KE, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('t'); ax.set_ylabel('KE(t)')
        ax.set_title(f'Kinetic Energy Decay - {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('TGV 2D - Kinetic Energy Decay', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "kinetic_energy_decay")

    # --- Vorticity contourf at t=0, t_mid, t_final (per method) ---
    time_indices = [0, Nt // 2, Nt - 1]
    time_labels = ['t=0', f't={t_np[Nt//2]:.2f}', f't={t_np[-1]:.2f}']
    x_int = x_np[1:-1]
    y_int = y_np[1:-1]

    for backend, (U, V, P) in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        label = BACKEND_LABELS.get(backend, backend)
        omega = _compute_vorticity(U, V, dx, dy)

        for col, (ti, tl) in enumerate(zip(time_indices, time_labels)):
            im = axes[col].contourf(x_int, y_int, omega[ti].T, levels=30, cmap='RdBu_r')
            plt.colorbar(im, ax=axes[col])
            axes[col].set_xlabel('x'); axes[col].set_ylabel('y')
            axes[col].set_title(f'Vorticity at {tl}')
            axes[col].set_aspect('equal')

        plt.suptitle(f'TGV 2D Vorticity - {label}', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"vorticity_{backend}")

    # --- Combined velocity/vorticity comparison at t_mid ---
    ti_mid = Nt // 2
    n_methods = min(len(predictions), 7)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for idx, (backend, (U, V, P)) in enumerate(predictions.items()):
        if idx >= 7:
            break
        ax = axes_flat[idx]
        omega = _compute_vorticity(U, V, dx, dy)
        im = ax.contourf(x_int, y_int, omega[ti_mid].T, levels=20, cmap='RdBu_r')
        ax.set_title(BACKEND_LABELS.get(backend, backend)[:15], fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    if n_methods <= 7:
        axes_flat[7].axis('off')

    plt.suptitle(f'TGV 2D - Vorticity Comparison at t={t_np[ti_mid]:.2f}', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "vorticity_comparison")

    # --- Error field |u_pred - u_exact| contourf ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    ti_mid = Nt // 2

    for idx, (backend, (U, V, P)) in enumerate(predictions.items()):
        if idx >= 7:
            break
        ax = axes_flat[idx]
        err = np.sqrt((U[ti_mid] - U_exact[ti_mid])**2 + (V[ti_mid] - V_exact[ti_mid])**2)
        im = ax.contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(BACKEND_LABELS.get(backend, backend)[:15], fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    if len(predictions) <= 7:
        axes_flat[7].axis('off')

    plt.suptitle(f'TGV 2D - Velocity Error at t={t_np[ti_mid]:.2f}', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "error_field")


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
    import cases.tgv_2d.physics as physics
    plot_from_checkpoints(physics, "output/tgv_2d/figures")
