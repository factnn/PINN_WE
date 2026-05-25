"""Common plotting utilities for all cases.

Provides:
- Consistent color scheme for 7 backends
- Loss curve plotting (loss vs epoch)
- Training time bar chart
- Pointwise error distribution
- Figure saving with consistent style
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Consistent styling ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# 7 backends with distinct colors
BACKEND_COLORS = {
    'mlp_vanilla': '#d62728',   # red
    'mlp_canpinn': '#1f77b4',   # blue
    'mlp_compile': '#9467bd',   # purple
    'mlp_triton':  '#2ca02c',   # green (highlight)
    'cnn_canpinn': '#ff7f0e',   # orange
    'cnn_compile': '#8c564b',   # brown
    'cnn_triton':  '#17becf',   # cyan (highlight)
}

BACKEND_LINESTYLES = {
    'mlp_vanilla': '-',
    'mlp_canpinn': '--',
    'mlp_compile': ':',
    'mlp_triton':  '-',
    'cnn_canpinn': '--',
    'cnn_compile': ':',
    'cnn_triton':  '-',
}

BACKEND_LABELS = {
    'mlp_vanilla': 'MLP Vanilla (autograd)',
    'mlp_canpinn': 'MLP CAN-PINN (FD)',
    'mlp_compile': 'MLP Compile (FD)',
    'mlp_triton':  'MLP Triton (ours)',
    'cnn_canpinn': 'CNN CAN-PINN (FD)',
    'cnn_compile': 'CNN Compile (FD)',
    'cnn_triton':  'CNN Triton (ours)',
}

MLP_BACKENDS = ['mlp_vanilla', 'mlp_canpinn', 'mlp_compile', 'mlp_triton']
CNN_BACKENDS = ['cnn_canpinn', 'cnn_compile', 'cnn_triton']
ALL_BACKENDS = MLP_BACKENDS + CNN_BACKENDS


def save_fig(fig, out_dir, name):
    """Save figure to out_dir/name.png"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── Loss Curves ─────────────────────────────────────────────────────────────
def plot_loss_curves(histories, out_dir, title="Loss vs Epoch"):
    """Plot loss convergence curves for multiple backends.

    Args:
        histories: dict[backend_name] -> list of (wall_time, epoch, loss)
        out_dir: output directory
        title: plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for backend, history in histories.items():
        if not history:
            continue
        epochs = [h[1] for h in history]
        losses = [h[2] for h in history]
        times = [h[0] for h in history]
        color = BACKEND_COLORS.get(backend, 'gray')
        ls = BACKEND_LINESTYLES.get(backend, '-')
        label = BACKEND_LABELS.get(backend, backend)

        # Loss vs Epoch
        ax1.plot(epochs, losses, color=color, ls=ls, lw=1.5, label=label)
        # Loss vs Wall Time
        ax2.plot(times, losses, color=color, ls=ls, lw=1.5, label=label)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.set_title(f'{title} (vs Epoch)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Wall Time (s)')
    ax2.set_ylabel('Loss')
    ax2.set_yscale('log')
    ax2.set_title(f'{title} (vs Time)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir, "loss_curves")


# ─── Training Time Bar Chart ─────────────────────────────────────────────────
def plot_time_comparison(results, out_dir, title="Training Step Time"):
    """Bar chart comparing avg_ms across backends.

    Args:
        results: dict[backend] -> dict with 'avg_ms', 'mem_gb'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    backends = [b for b in ALL_BACKENDS if b in results]
    times = [results[b]['avg_ms'] for b in backends]
    mems = [results[b].get('mem_gb', 0) for b in backends]
    colors = [BACKEND_COLORS.get(b, 'gray') for b in backends]
    labels = [BACKEND_LABELS.get(b, b) for b in backends]

    # Time bar
    bars = ax1.bar(range(len(backends)), times, color=colors)
    ax1.set_xticks(range(len(backends)))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('ms / step')
    ax1.set_title(f'{title} (ms/step)')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{t:.2f}', ha='center', va='bottom', fontsize=8)

    # Memory bar
    bars2 = ax2.bar(range(len(backends)), mems, color=colors)
    ax2.set_xticks(range(len(backends)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.set_title('Peak GPU Memory')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, m in zip(bars2, mems):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_fig(fig, out_dir, "time_comparison")


# ─── Pointwise Error ─────────────────────────────────────────────────────────
def plot_pointwise_error_1d(x, errors, out_dir, title="Pointwise Error"):
    """Plot |pred - exact| for 1D problems.

    Args:
        x: spatial coordinates (1D array)
        errors: dict[backend] -> 1D array of |pred - exact|
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for backend, err in errors.items():
        color = BACKEND_COLORS.get(backend, 'gray')
        ls = BACKEND_LINESTYLES.get(backend, '-')
        label = BACKEND_LABELS.get(backend, backend)
        ax.plot(x, err, color=color, ls=ls, lw=1.5, label=label)

    ax.set_xlabel('x')
    ax.set_ylabel('|u_pred - u_exact|')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    save_fig(fig, out_dir, "pointwise_error")
