"""Plotting for 2D Transport (Advection-Diffusion) case.

Figures:
- Concentration C(x,y) contourf at t=0, t_mid, t_final (pred vs exact side by side)
- 1D slice: C along y=mid at final time (all methods + exact)
- Error field |C_pred - C_exact| contourf
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
    """Generate all plots for transport_2d."""
    out_dir = Path(out_dir)
    x_np = ctx["X"][0, :, 0].cpu().numpy()
    y_np = ctx["Y"][0, 0, :].cpu().numpy()
    t_np = ctx["T"][:, 0, 0].cpu().numpy()
    Nt = physics.Nt
    Nx = physics.Nx
    Ny = physics.Ny
    C_exact = ctx["C_exact"].cpu().numpy()

    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            C_pred = physics.infer(model, ctx)
        predictions[backend] = C_pred.cpu().numpy()

    # --- Individual: C(x,y) pred vs exact at t=0, t_mid, t_final ---
    time_indices = [0, Nt // 2, Nt - 1]
    time_labels = ['t=0', f't={t_np[Nt//2]:.2f}', f't={t_np[-1]:.2f}']

    for backend, C_pred in predictions.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        label = BACKEND_LABELS.get(backend, backend)

        for col, (ti, tl) in enumerate(zip(time_indices, time_labels)):
            # Predicted
            vmin = min(C_pred[ti].min(), C_exact[ti].min())
            vmax = max(C_pred[ti].max(), C_exact[ti].max())
            im0 = axes[0, col].contourf(x_np, y_np, C_pred[ti].T, levels=20,
                                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=axes[0, col])
            axes[0, col].set_title(f'Pred at {tl}')
            axes[0, col].set_aspect('equal')
            axes[0, col].set_xlabel('x'); axes[0, col].set_ylabel('y')

            # Exact
            im1 = axes[1, col].contourf(x_np, y_np, C_exact[ti].T, levels=20,
                                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=axes[1, col])
            axes[1, col].set_title(f'Exact at {tl}')
            axes[1, col].set_aspect('equal')
            axes[1, col].set_xlabel('x'); axes[1, col].set_ylabel('y')

        plt.suptitle(f'Transport 2D - {label}', fontsize=12)
        plt.tight_layout()
        save_fig(fig, out_dir, f"solution_{backend}")

    # --- 1D slice: C along y=mid at final time (all methods + exact) ---
    yi_mid = Ny // 2
    ti_final = Nt - 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, group, title in [(axes[0], MLP_BACKENDS, 'MLP'), (axes[1], CNN_BACKENDS, 'CNN')]:
        ax.plot(x_np, C_exact[ti_final, :, yi_mid], 'k--', lw=2.5, label='Exact', zorder=10)
        for backend in group:
            if backend in predictions:
                color = BACKEND_COLORS[backend]
                ls = BACKEND_LINESTYLES[backend]
                label = BACKEND_LABELS[backend]
                ax.plot(x_np, predictions[backend][ti_final, :, yi_mid],
                        color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('x'); ax.set_ylabel('C')
        ax.set_title(f'C(x, y=mid) at t={t_np[ti_final]:.2f} - {title}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('Transport 2D - Concentration Slice Comparison', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "slice_comparison")

    # --- Error field |C_pred - C_exact| contourf at t_final ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()

    for idx, (backend, C_pred) in enumerate(predictions.items()):
        if idx >= 7:
            break
        ax = axes_flat[idx]
        err = np.abs(C_pred[ti_final] - C_exact[ti_final])
        im = ax.contourf(x_np, y_np, err.T, levels=20, cmap='hot_r')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(BACKEND_LABELS.get(backend, backend)[:15], fontsize=9)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    if len(predictions) <= 7:
        axes_flat[7].axis('off')

    plt.suptitle(f'Transport 2D - |C_pred - C_exact| at t={t_np[ti_final]:.2f}', fontsize=13)
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
    import cases.transport_2d.physics as physics
    plot_from_checkpoints(physics, "output/transport_2d/figures")
