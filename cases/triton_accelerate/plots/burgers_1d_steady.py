"""Plotting for 1D Burgers Steady case.

Figures:
- 7 individual solution plots (each method vs exact)
- 1 combined comparison plot (all methods on one figure)
- Pointwise error
- Loss curves (from common)
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from plots import (
    BACKEND_COLORS, BACKEND_LABELS, BACKEND_LINESTYLES,
    ALL_BACKENDS, MLP_BACKENDS, CNN_BACKENDS, save_fig,
    plot_accuracy_preservation_1d
)


def plot_all(physics, ctx, models, out_dir):
    """Generate all plots for burgers_1d_steady.

    Args:
        physics: the physics module
        ctx: context dict
        models: dict[backend_name] -> loaded model (on device)
        out_dir: output directory for figures
    """
    out_dir = Path(out_dir)
    x_np = ctx["X"].cpu().numpy()
    u_exact = physics._exact_u(ctx["X"]).cpu().numpy()

    # Collect predictions
    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            u_pred = physics.infer(model, ctx)
        predictions[backend] = u_pred.cpu().numpy()

    # --- Individual plots (7) ---
    for backend, u_pred in predictions.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(x_np, u_exact, 'k--', lw=2, label='Exact')
        color = BACKEND_COLORS.get(backend, 'blue')
        label = BACKEND_LABELS.get(backend, backend)
        ax.plot(x_np, u_pred, color=color, lw=2, label=label)

        l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-12)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title(f'1D Steady Burgers — {label} (L2={l2:.2e})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, out_dir, f"solution_{backend}")

    # --- Combined comparison plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MLP group
    axes[0].plot(x_np, u_exact, 'k--', lw=2.5, label='Exact', zorder=10)
    for backend in MLP_BACKENDS:
        if backend in predictions:
            color = BACKEND_COLORS[backend]
            ls = BACKEND_LINESTYLES[backend]
            label = BACKEND_LABELS[backend]
            axes[0].plot(x_np, predictions[backend], color=color, ls=ls, lw=1.5, label=label)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('u')
    axes[0].set_title('MLP Methods')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Right: CNN group
    axes[1].plot(x_np, u_exact, 'k--', lw=2.5, label='Exact', zorder=10)
    for backend in CNN_BACKENDS:
        if backend in predictions:
            color = BACKEND_COLORS[backend]
            ls = BACKEND_LINESTYLES[backend]
            label = BACKEND_LABELS[backend]
            axes[1].plot(x_np, predictions[backend], color=color, ls=ls, lw=1.5, label=label)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('u')
    axes[1].set_title('CNN Methods')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.suptitle('1D Steady Burgers — Solution Comparison', fontsize=14)
    plt.tight_layout()
    save_fig(fig, out_dir, "solution_comparison")

    # --- Pointwise error ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        for backend in group:
            if backend in predictions:
                err = np.abs(predictions[backend] - u_exact)
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(x_np, err, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('x'); ax.set_ylabel('|u_pred - u_exact|')
        ax.set_title(f'Pointwise Error — {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    save_fig(fig, out_dir, "pointwise_error")

    # --- Zero accuracy loss: FD methods should overlap ---
    plot_accuracy_preservation_1d(x_np, predictions, u_exact, out_dir,
                                   title="1D Steady Burgers")


def plot_from_checkpoints(physics, out_dir, model_dir=None):
    """Load all checkpoints and generate plots.

    Args:
        physics: physics module
        out_dir: where to save figures
        model_dir: directory containing model_<backend>.pt files (default: out_dir)
    """
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.utils import BACKENDS

    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "output" / physics.CASE_NAME
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)

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
            print(f"  Loaded: {backend}")

    if models:
        plot_all(physics, ctx, models, out_dir)
    else:
        print(f"  No checkpoints found in {model_dir}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    from cases.burgers_1d_steady.physics import *
    import cases.burgers_1d_steady.physics as physics
    plot_from_checkpoints(physics, "output/burgers_1d_steady/figures")
