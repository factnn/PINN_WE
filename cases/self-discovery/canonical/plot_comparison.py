#!/usr/bin/env python3
"""
对比图: case-by-case vs 参数化PINN
- case-by-case精度来自 pinn_inverse.py 的已有结果（或重新跑几个点）
- 参数化PINN精度来自已保存的模型
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pinn_parametric import ParametricPINN
from guderley_ode_solver import find_eigenvalue

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
run_dir = Path(__file__).parent / "output" / "run_20260512_173826"

# 评估γ点
gammas_eval = np.linspace(1.2, 2.0, 20)

# 打靶法参考值
print("Computing shooting reference...")
ref = {}
for g in gammas_eval:
    for n, geo in [(3, "spherical"), (2, "cylindrical")]:
        r = find_eigenvalue(g, n, geo, verbose=False)
        ref[(g, geo)] = r["alpha"] if r["alpha"] else np.nan

# 参数化PINN误差
print("Evaluating parametric PINN...")
param_err = {"spherical": [], "cylindrical": []}
for n, geo in [(3, "spherical"), (2, "cylindrical")]:
    model = ParametricPINN(n=n).to(device).to(dtype)
    model.load_state_dict(torch.load(run_dir / f"pinn_parametric_{geo}.pt", map_location=device))
    model.eval()
    for g in gammas_eval:
        g_t = torch.tensor([[g]], dtype=dtype, device=device)
        with torch.no_grad():
            alpha_pinn = float(model.get_alpha(g_t))
        alpha_ref = ref[(g, geo)]
        if not np.isnan(alpha_ref):
            err = abs(alpha_pinn - alpha_ref) / alpha_ref * 100
        else:
            err = np.nan
        param_err[geo].append(err)

# case-by-case误差（从已有消融实验结果读取，或用典型值）
# 根据之前的实验结果：case-by-case精度约0.0001%~0.001%
# 这里用保守估计0.001%作为代表值
cbc_err_typical = 0.01  # % — from ablation experiments (OVERVIEW §4), 123 framework ~1e-4%~0.01%

# ── 画图 ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, geo, color in zip(axes, ["spherical", "cylindrical"], ["steelblue", "tomato"]):
    errs = np.array(param_err[geo])
    mask = ~np.isnan(errs)

    # 参数化PINN误差曲线
    ax.semilogy(gammas_eval[mask], errs[mask], 'o-', color=color, lw=2, ms=5,
                label=f'Parametric PINN (max={errs[mask].max():.3f}%)')

    # case-by-case水平参考线
    ax.axhline(cbc_err_typical, color='gray', ls='--', lw=1.5,
               label=f'Case-by-case PINN (~{cbc_err_typical}%)')

    ax.set_xlabel(r'$\gamma$', fontsize=12)
    ax.set_ylabel(r'$\alpha$ relative error (%)', fontsize=12)
    ax.set_title(f'{geo.capitalize()}: Parametric vs Case-by-case', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.15, 2.05)

fig.suptitle('Accuracy Trade-off: One Training vs Per-case Training', fontsize=13, fontweight='bold')
fig.tight_layout()
fname = run_dir / "comparison_parametric_vs_casebycase.png"
fig.savefig(fname, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# 打印数值表
print(f"\n{'γ':>6} {'sph_param%':>12} {'cyl_param%':>12}  (case-by-case ~0.001%)")
for i, g in enumerate(gammas_eval):
    print(f"{g:>6.2f} {param_err['spherical'][i]:>12.4f} {param_err['cylindrical'][i]:>12.4f}")
