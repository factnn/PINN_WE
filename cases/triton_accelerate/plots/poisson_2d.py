"""Plotting for poisson_2d (contourf pred vs exact + error)."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
from plots.ldc_2d import plot_from_checkpoints as _plot_ldc

# Poisson 2D plots similar to LDC but simpler (single field u)
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plots import BACKEND_COLORS, BACKEND_LABELS, ALL_BACKENDS, MLP_BACKENDS, CNN_BACKENDS, save_fig
from engine.utils import BACKENDS


def plot_from_checkpoints(physics, out_dir, model_dir=None):
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

    if not models:
        print(f"  No checkpoints in {model_dir}")
        return

    x_np = ctx["X"][:, 0].cpu().numpy()
    y_np = ctx["Y"][0, :].cpu().numpy()
    U_exact = physics._exact_u(ctx["X"], ctx["Y"]).cpu().numpy()

    predictions = {}
    for backend, model in models.items():
        with torch.no_grad():
            predictions[backend] = physics.infer(model, ctx).cpu().numpy()

    # Combined comparison: 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (backend, U_pred) in enumerate(predictions.items()):
        if idx >= 7: break
        ax = axes[idx // 4, idx % 4]
        im = ax.contourf(x_np, y_np, U_pred.T, levels=20, cmap='RdBu_r')
        l2 = np.linalg.norm(U_pred - U_exact) / (np.linalg.norm(U_exact) + 1e-12)
        ax.set_title(f'{BACKEND_LABELS.get(backend, backend)[:12]}\nL2={l2:.1e}', fontsize=8)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    # Last: exact
    ax = axes[1, 3]
    ax.contourf(x_np, y_np, U_exact.T, levels=20, cmap='RdBu_r')
    ax.set_title('Exact', fontsize=8); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle('2D Poisson — All Methods', fontsize=13)
    plt.tight_layout()
    save_fig(fig, out_dir, "solution_comparison")


if __name__ == "__main__":
    import cases.poisson_2d.physics as physics
    plot_from_checkpoints(physics, "output/poisson_2d/figures")
