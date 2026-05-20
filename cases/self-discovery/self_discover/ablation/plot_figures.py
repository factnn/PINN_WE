#!/usr/bin/env python3
"""
论文级别Figure生成脚本

生成4张图:
1. 漏斗扫描2×2合成图
2. 消融对比α收敛图
3. Taylor-Maccoll β收敛图
4. 噪声鲁棒性汇总图
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "output"
TM_OUT = Path(__file__).parent.parent.parent / "taylor_maccoll" / "output"

# 统一样式
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

REF_ALPHA = {
    "g1.40_spherical": 0.717174501487,
    "g1.40_cylindrical": 0.835323191951,
    "g1.67_spherical": 0.688376822922,
    "g1.67_cylindrical": 0.815624901431,
}
LABELS = {
    "g1.40_spherical": "γ=1.4, spherical",
    "g1.40_cylindrical": "γ=1.4, cylindrical",
    "g1.67_spherical": "γ=5/3, spherical",
    "g1.67_cylindrical": "γ=5/3, cylindrical",
}
CONFIGS = ["g1.40_spherical", "g1.40_cylindrical", "g1.67_spherical", "g1.67_cylindrical"]


# ═══════════════════════════════════════════════════
# Figure 1: 漏斗扫描 2×2
# ═══════════════════════════════════════════════════
def plot_funnel_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    files = [
        ("funnel_spherical_g1.40.json", "γ=1.4, spherical"),
        ("funnel_cylindrical_g1.40.json", "γ=1.4, cylindrical"),
        ("funnel_spherical_g1.67.json", "γ=5/3, spherical"),
        ("funnel_cylindrical_g1.67.json", "γ=5/3, cylindrical"),
    ]
    for idx, (fname, title) in enumerate(files):
        ax = axes[idx // 2][idx % 2]
        fpath = OUT / fname
        if not fpath.exists():
            ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        with open(fpath) as f:
            d = json.load(f)

        alphas = np.array(d["alphas"])
        losses = np.array(d["losses"])
        ref = d["ref_alpha"]

        ax.semilogy(alphas, losses, "b.-", markersize=5, linewidth=1.5)
        ax.axvline(ref, color="r", linestyle="--", linewidth=2, label=f"α_ref={ref:.6f}")

        # 标注漏斗
        best_idx = np.argmin(losses)
        ax.annotate(f"min={losses[best_idx]:.1e}",
                    xy=(alphas[best_idx], losses[best_idx]),
                    xytext=(alphas[best_idx] + 0.03, losses[best_idx] * 5),
                    fontsize=8, color="blue",
                    arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))

        ax.set_xlabel("α (fixed)")
        ax.set_ylabel("min loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Loss Landscape: Funnel Structure around True α", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_funnel_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'fig_funnel_2x2.png'}")


# ═══════════════════════════════════════════════════
# Figure 2: 消融对比α收敛图
# ═══════════════════════════════════════════════════
def plot_ablation_alpha():
    """把关键方案的α收敛轨迹叠在一起"""
    # 加载各实验的JSON (含alpha_history)
    experiments = [
        ("exp7_success_baseline.json", "1234 (full)", "C0", "-"),
        ("exp4_no_warmup.json", "123 (final)", "C1", "-"),
        ("exp1_soft_constraint.json", "12(3-soft)4", "C2", "--"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ci, config in enumerate(CONFIGS):
        ax = axes[ci // 2][ci % 2]
        ref = REF_ALPHA[config]

        for fname, label, color, ls in experiments:
            fpath = OUT / fname
            if not fpath.exists():
                continue
            with open(fpath) as f:
                d = json.load(f)

            # 处理可能的nested structure
            data = d.get("configs", d)
            if config not in data:
                continue

            entry = data[config]
            if "alpha_history" in entry:
                hist = np.array(entry["alpha_history"])
                ax.plot(hist, label=label, linewidth=1.5, linestyle=ls)

        ax.axhline(ref, color="red", linestyle=":", linewidth=2, label=f"ref={ref:.6f}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("α")
        ax.set_title(LABELS[config])
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Ablation: α Convergence Trajectories", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation_alpha.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'fig_ablation_alpha.png'}")


# ═══════════════════════════════════════════════════
# Figure 3: Taylor-Maccoll β收敛图
# ═══════════════════════════════════════════════════
def plot_tm_beta():
    fpath = TM_OUT / "tm_pinn.json"
    if not fpath.exists():
        print(f"  SKIP: {fpath} not found")
        return

    with open(fpath) as f:
        d = json.load(f)

    # tm_pinn.json只有最终值，没有history
    # 从log文件提取β轨迹
    log_path = TM_OUT / "tm_pinn.log"
    if not log_path.exists():
        # 画bar chart代替
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(d.keys())
        refs = [d[k]["ref_beta_deg"] for k in labels]
        pinns = [d[k]["pinn_beta_deg"] for k in labels]
        errs = [d[k]["err_pct"] for k in labels]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, refs, w, label="Shooting (ref)", color="steelblue")
        ax.bar(x + w/2, pinns, w, label="PINN 123", color="coral")

        for i, e in enumerate(errs):
            ax.text(i + w/2, pinns[i] + 0.3, f"{e:.3f}%", ha="center", fontsize=8, color="red")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("β (degrees)")
        ax.set_title("Taylor-Maccoll: PINN vs Shooting Method", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(TM_OUT / "fig_tm_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {TM_OUT / 'fig_tm_comparison.png'}")
        return

    # 如果有log，从中解析β轨迹
    plot_tm_from_log(log_path, d)


def plot_tm_from_log(log_path, results):
    """从训练log解析β轨迹"""
    import re

    with open(log_path) as f:
        lines = f.readlines()

    # 按case分段
    cases = {}
    current_case = None
    for line in lines:
        m = re.match(r"\s+(\w+_g[\d.]+_tc\d+):", line)
        if m:
            current_case = m.group(1)
            cases[current_case] = {"betas": [], "epochs": []}
            continue

        if current_case and "β=" in line:
            bm = re.search(r"β=([\d.]+)°", line)
            if bm:
                cases[current_case]["betas"].append(float(bm.group(1)))

    n_cases = len(cases)
    if n_cases == 0:
        print("  SKIP: no β data in log")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, (case, data) in enumerate(cases.items()):
        if idx >= 4:
            break
        ax = axes[idx // 2][idx % 2]
        betas = data["betas"]
        ref = results.get(case, {}).get("ref_beta_deg", None)

        ax.plot(betas, linewidth=1.5, color="C0")
        if ref:
            ax.axhline(ref, color="red", linestyle="--", linewidth=2, label=f"ref={ref:.4f}°")
        ax.set_xlabel("training step (logged)")
        ax.set_ylabel("β (degrees)")
        ax.set_title(case)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Taylor-Maccoll: β Convergence (PINN 123 Framework Transfer)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(TM_OUT / "fig_tm_beta_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {TM_OUT / 'fig_tm_beta_convergence.png'}")


# ═══════════════════════════════════════════════════
# Figure 4: 噪声鲁棒性汇总图
# ═══════════════════════════════════════════════════
def plot_noise_robustness():
    fpath = OUT / "exp_noisy_robustness.json"
    if not fpath.exists():
        print(f"  SKIP: {fpath} not found")
        return

    with open(fpath) as f:
        d = json.load(f)

    # 按噪声级别和config整理
    noise_levels = [0, 1, 5, 10]
    seeds = [42, 123, 999]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 各config的 err vs noise (3-seed平均)
    ax = axes[0]
    for config in CONFIGS:
        means = []
        stds = []
        for noise in noise_levels:
            errs = []
            for seed in seeds:
                key = f"noise{noise}pct_seed{seed}"
                if key in d and config in d[key]:
                    errs.append(d[key][config]["err_pct"])
            if errs:
                means.append(np.mean(errs))
                stds.append(np.std(errs))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)
        ax.semilogy(noise_levels, means, "o-", label=LABELS[config], linewidth=1.5, markersize=5)
        ax.fill_between(noise_levels, means - stds, means + stds, alpha=0.15)

    ax.set_xlabel("Noise level σ (%)")
    ax.set_ylabel("α relative error (%)")
    ax.set_title("Error vs Noise (3-seed mean±std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(noise_levels)

    # 右图: 所有48个点的散点
    ax = axes[1]
    all_noise = []
    all_err = []
    all_config = []
    colors = {"g1.40_spherical": "C0", "g1.40_cylindrical": "C1",
              "g1.67_spherical": "C2", "g1.67_cylindrical": "C3"}

    for noise in noise_levels:
        for seed in seeds:
            key = f"noise{noise}pct_seed{seed}"
            if key not in d:
                continue
            for config in CONFIGS:
                if config in d[key]:
                    all_noise.append(noise + np.random.uniform(-0.3, 0.3))  # jitter
                    all_err.append(d[key][config]["err_pct"])
                    all_config.append(config)

    for config in CONFIGS:
        mask = [c == config for c in all_config]
        ns = [all_noise[i] for i in range(len(mask)) if mask[i]]
        es = [all_err[i] for i in range(len(mask)) if mask[i]]
        ax.semilogy(ns, es, "o", color=colors[config], alpha=0.6, markersize=4, label=LABELS[config])

    ax.set_xlabel("Noise level σ (%)")
    ax.set_ylabel("α relative error (%)")
    ax.set_title("All 48 Runs (scatter)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(noise_levels)

    fig.suptitle("Noise Robustness: α Inversion with Noisy Observations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_noise_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT / 'fig_noise_robustness.png'}")


# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  生成论文Figure")
    print("=" * 60)

    print("\n[1] 漏斗扫描 2×2")
    plot_funnel_2x2()

    print("\n[2] 消融对比α收敛")
    plot_ablation_alpha()

    print("\n[3] Taylor-Maccoll β收敛")
    plot_tm_beta()

    print("\n[4] 噪声鲁棒性")
    plot_noise_robustness()

    print("\nDone!")
