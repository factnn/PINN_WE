"""Plotting for 2D LDC case.

Figures:
- 7 individual: speed contourf + velocity contourf
- 1 combined: centerline u(y) vs Ghia data (all methods)
- Pointwise error contourf
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


def plot_all(physics, ctx, models, out_dir):
    """Generate all plots for ldc_2d."""
    out_dir = Path(out_dir)
    x_np = ctx["X"][:, 0].cpu().numpy()
    y_np = ctx["Y"][0, :].cpu().numpy()
    Nx = len(x_np)

    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            U, V, P = physics.infer(model, ctx)
        predictions[backend] = (U.cpu().numpy(), V.cpu().numpy(), P.cpu().numpy())

    # --- Individual plots (7) ---
    for backend, (U, V, P) in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        label = BACKEND_LABELS.get(backend, backend)
        speed = np.sqrt(U**2 + V**2)

        # Speed contourf
        im = axes[0].contourf(x_np, y_np, speed.T, levels=20, cmap='viridis')
        plt.colorbar(im, ax=axes[0])
        axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
        axes[0].set_title(f'Speed — {label}')
        axes[0].set_aspect('equal')

        # U velocity contourf
        im2 = axes[1].contourf(x_np, y_np, U.T, levels=20, cmap='RdBu_r')
        plt.colorbar(im2, ax=axes[1])
        axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
        axes[1].set_title('u velocity')
        axes[1].set_aspect('equal')

        # Centerline u(y) vs Ghia
        u_center = U[Nx // 2, :]
        axes[2].plot(u_center, y_np, 'b-', lw=2, label='PINN')
        axes[2].plot(physics.GHIA_U, physics.GHIA_Y, 'ro', ms=5, label='Ghia Re=100')
        axes[2].set_xlabel('u'); axes[2].set_ylabel('y')
        axes[2].set_title('Centerline u (x=0.5)')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'LDC 2D — {label}', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"solution_{backend}")

    # --- Combined: Centerline comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        ax.plot(physics.GHIA_U, physics.GHIA_Y, 'ko', ms=6, label='Ghia Re=100', zorder=10)
        for backend in group:
            if backend in predictions:
                U, V, P = predictions[backend]
                u_center = U[Nx // 2, :]
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(u_center, y_np, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('u'); ax.set_ylabel('y')
        ax.set_title(f'Centerline u(y) — {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('LDC 2D — Centerline Velocity vs Ghia', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "centerline_comparison")

    # --- Speed field comparison (total) ---
    n_methods = len(predictions)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for idx, (backend, (U, V, P)) in enumerate(predictions.items()):
        if idx >= 7:
            break
        ax = axes_flat[idx]
        speed = np.sqrt(U**2 + V**2)
        im = ax.contourf(x_np, y_np, speed.T, levels=20, cmap='viridis')
        ax.set_title(BACKEND_LABELS.get(backend, backend)[:15], fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    # Last subplot: empty or legend
    if len(predictions) <= 7:
        axes_flat[7].axis('off')

    plt.suptitle('LDC 2D — Speed Field Comparison', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "speed_comparison")


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
    import cases.ldc_2d.physics as physics
    plot_from_checkpoints(physics, "output/ldc_2d/figures")
