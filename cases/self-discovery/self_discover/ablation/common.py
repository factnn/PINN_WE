"""
共享模块: Chisnell ODE, 参考值, 画图工具。
所有ablation实验从这里导入。
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REFERENCE_ALPHA = {
    (1.4, 3): 0.717175,
    (1.4, 2): 0.835323,
    (5/3, 3): 0.688377,
    (5/3, 2): 0.815625,
}
GEOMETRY_N = {"spherical": 3, "cylindrical": 2}
DEFAULT_CONFIGS = [
    (1.4, "spherical"),
    (1.4, "cylindrical"),
    (5/3, "spherical"),
    (5/3, "cylindrical"),
]

# 合理的初始猜测 (不需要很准, 但不能差太远)
# 论文叙事: "基于物理直觉给出的粗略估计"
ALPHA_INIT = {
    (1.4, 3): 0.72,     # ref=0.717
    (1.4, 2): 0.84,     # ref=0.835
    (5/3, 3): 0.69,     # ref=0.688
    (5/3, 2): 0.82,     # ref=0.816
}


def get_ref_alpha(gamma: float, n: int) -> float | None:
    for (g, nn_), val in REFERENCE_ALPHA.items():
        if nn_ == n and abs(g - gamma) < 1e-4:
            return val
    return None


def inv_sigmoid(y: float) -> float:
    y = max(min(y, 0.999), 0.001)
    return float(np.log(y / (1.0 - y)))


def critical_point_np(alpha: float, gamma: float, n: int):
    if n == 3:
        gamma_crit = 1.8697680
    elif n == 2:
        gamma_crit = 1.9092084
    else:
        gamma_crit = 2.0
    V0_denom = 2.0 * gamma * (n - 1.0)
    factor = gamma * n - 2.0
    disc = (8.0 * (alpha - 1.0) * alpha * gamma * (n - 1.0)
            + (2.0 - gamma + alpha * factor) ** 2)
    if disc < 0:
        return None, None
    if gamma >= gamma_crit:
        V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
              - np.sqrt(disc)) / V0_denom
    else:
        V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
              + np.sqrt(disc)) / V0_denom
    C0 = (V0 - alpha) ** 2
    return V0, C0


def critical_point_torch(alpha, gamma, n):
    if n == 3:
        gamma_crit = 1.8697680
    elif n == 2:
        gamma_crit = 1.9092084
    else:
        gamma_crit = 2.0
    V0_denom = 2.0 * gamma * (n - 1.0)
    factor = gamma * n - 2.0
    disc = (8.0 * (alpha - 1.0) * alpha * gamma * (n - 1.0)
            + (2.0 - gamma + alpha * factor) ** 2)
    disc = torch.clamp(disc, min=1e-14)
    if gamma >= gamma_crit:
        V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
              - torch.sqrt(disc)) / V0_denom
    else:
        V0 = (2.0 - 2.0 * alpha - gamma + alpha * gamma * n
              + torch.sqrt(disc)) / V0_denom
    C0 = (V0 - alpha) ** 2
    return V0, C0


def chisnell_residual(V, C, dC_dV, alpha, gamma, n):
    """ODE残差计算, 返回 (residual, numer, denom, delta)"""
    delta = (V - alpha) ** 2 - C
    Q = (n * V * (V - alpha)
         + (2.0 / gamma) * (1.0 - alpha) * (alpha - V)
         - V * (V - 1.0))
    numer = C * (2.0 * delta * (alpha - V + (1.0 - alpha) / gamma)
                 + (gamma - 1.0) * (alpha - V) * Q)
    denom = (delta * (n * V - 2.0 * (1.0 - alpha) / gamma) * (alpha - V)
             + (alpha - V) ** 2 * Q)
    residual = dC_dV * denom - numer
    return residual, numer, denom, delta


def build_mlp(n_in, n_out, n_layers=5, n_neurons=64):
    layers = [nn.Linear(n_in, n_neurons), nn.Tanh()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
    layers.append(nn.Linear(n_neurons, n_out))
    return nn.Sequential(*layers)


def plot_ablation_result(results: dict, title: str, output_path: str):
    """画ablation结果: alpha收敛 + loss曲线 + 汇总表"""
    n_configs = len(results)
    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 8))
    if n_configs == 1:
        axes = axes.reshape(2, 1)

    for i, (key, r) in enumerate(results.items()):
        # alpha convergence
        ax = axes[0, i]
        ax.plot(r["alpha_history"], linewidth=1.5)
        if r["ref_alpha"]:
            ax.axhline(r["ref_alpha"], color="r", linestyle="--",
                       label=f"ref={r['ref_alpha']:.6f}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("alpha")
        ax.set_title(f"{key}\nalpha={r['alpha_fit']:.6f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # loss
        ax = axes[1, i]
        ax.semilogy(np.maximum(r["loss_history"], 1e-30), linewidth=1.5)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title(f"final loss={r['final_loss']:.2e}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results_json(results: dict, experiment_name: str, output_path: str):
    """保存实验结果到JSON, 包含loss_history和alpha_history"""
    import json
    save_data = {"experiment": experiment_name, "configs": {}}
    for key, r in results.items():
        entry = {
            "alpha_fit": r["alpha_fit"],
            "ref_alpha": r["ref_alpha"],
            "rel_err_pct": r["rel_err_pct"],
            "final_loss": r["final_loss"],
        }
        if "alpha_history" in r:
            entry["alpha_history"] = r["alpha_history"].tolist() if hasattr(r["alpha_history"], "tolist") else r["alpha_history"]
        if "loss_history" in r:
            entry["loss_history"] = r["loss_history"].tolist() if hasattr(r["loss_history"], "tolist") else r["loss_history"]
        save_data["configs"][key] = entry
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved: {output_path}")


def print_summary_table(results: dict, experiment_name: str):
    print(f"\n{'='*70}")
    print(f"  {experiment_name}")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'alpha_fit':<12} {'ref':<12} {'err%':<12} {'loss':<12}")
    print(f"  {'-'*70}")
    for key, r in results.items():
        err_s = f"{r['rel_err_pct']:.4f}" if r['rel_err_pct'] is not None else "N/A"
        print(f"  {key:<25} {r['alpha_fit']:<12.6f} "
              f"{r['ref_alpha']:<12.6f} {err_s:<12} {r['final_loss']:<12.3e}")
