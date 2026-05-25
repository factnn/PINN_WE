"""Plotting for 1D Burgers Unsteady case.

Figures:
- 7 individual spacetime contourf + multi-slice plots
- 1 combined multi-slice comparison
- Pointwise error at final time
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
    ALL_BACKENDS, MLP_BACKENDS, CNN_BACKENDS, save_fig
)

SLICES = [(0, 't=0'), (25, 't=0.25'), (50, 't=0.5'), (75, 't=0.75'), (99, 't=1.0')]


def plot_all(physics, ctx, models, u_exact, out_dir):
    """Generate all plots for burgers_1d_unsteady.

    Args:
        physics: physics module
        ctx: context dict
        models: dict[backend] -> model
        u_exact: [Nt, Nx] exact solution array
        out_dir: output directory
    """
    out_dir = Path(out_dir)
    x_np = ctx["X"][0].cpu().numpy()
    t_np = ctx["T"][:, 0].cpu().numpy()

    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            u_pred = physics.infer(model, ctx)
        predictions[backend] = u_pred.cpu().numpy()

    # --- Individual plots (7): spacetime + slices ---
    for backend, u_pred in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        label = BACKEND_LABELS.get(backend, backend)

        # Spacetime contourf
        im = axes[0].contourf(x_np, t_np, u_pred, levels=50, cmap='RdBu_r')
        plt.colorbar(im, ax=axes[0])
        axes[0].set_xlabel('x'); axes[0].set_ylabel('t')
        axes[0].set_title(f'{label}')

        # Exact spacetime
        im2 = axes[1].contourf(x_np, t_np, u_exact, levels=50, cmap='RdBu_r')
        plt.colorbar(im2, ax=axes[1])
        axes[1].set_xlabel('x'); axes[1].set_ylabel('t')
        axes[1].set_title('Exact')

        # Time slices
        colors = plt.cm.viridis(np.linspace(0, 1, len(SLICES)))
        for (ti, tlabel), c in zip(SLICES, colors):
            if ti < len(u_pred):
                axes[2].plot(x_np, u_pred[ti], color=c, lw=2, label=f'Pred {tlabel}')
                axes[2].plot(x_np, u_exact[ti], color=c, lw=1.5, ls='--')
        l2 = np.linalg.norm(u_pred - u_exact) / (np.linalg.norm(u_exact) + 1e-12)
        axes[2].set_title(f'Slices (L2={l2:.2e})')
        axes[2].legend(fontsize=7, ncol=2); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'1D Unsteady Burgers — {label}', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"solution_{backend}")

    # --- Combined slice comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ti_compare = 50  # t=0.5

    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        ax.plot(x_np, u_exact[ti_compare], 'k--', lw=2.5, label='Exact', zorder=10)
        for backend in group:
            if backend in predictions:
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(x_np, predictions[backend][ti_compare], color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('x'); ax.set_ylabel('u')
        ax.set_title(f'{title} at t=0.5')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('1D Unsteady Burgers — Solution at t=0.5', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "solution_comparison")

    # --- Pointwise error at final time ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        for backend in group:
            if backend in predictions:
                err = np.abs(predictions[backend][-1] - u_exact[-1])
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(x_np, err, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('x'); ax.set_ylabel('|error| at t=1.0')
        ax.set_title(f'Pointwise Error — {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    save_fig(fig, out_dir, "pointwise_error")


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
        x_np = ctx["X"][0].cpu().numpy()
        t_np = ctx["T"][:, 0].cpu().numpy()
        u_exact = physics._exact_solution(x_np, t_np)
        plot_all(physics, ctx, models, u_exact, out_dir)
    else:
        print(f"  No checkpoints found in {model_dir}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
    import cases.burgers_1d_unsteady.physics as physics
    plot_from_checkpoints(physics, "output/burgers_1d_unsteady/figures")
