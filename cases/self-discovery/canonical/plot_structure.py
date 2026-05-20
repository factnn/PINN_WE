#!/usr/bin/env python3
"""画解的结构图：加载已保存的参数化PINN模型，画C(V)曲线族"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pinn_parametric import ParametricPINN, shock_conditions, critical_point

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
out = Path(__file__).parent / "output"

gammas_plot = [1.2, 1.4, 1.6, 1.8, 2.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(gammas_plot)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for n, geometry, ax_idx in [(3, "spherical", 0), (2, "cylindrical", 1)]:
    model = ParametricPINN(n=n).to(device).to(dtype)
    model.load_state_dict(torch.load(out / f"pinn_parametric_{geometry}.pt",
                                      map_location=device))
    model.eval()

    ax = axes[ax_idx]
    for gamma_val, color in zip(gammas_plot, colors):
        g_t = torch.tensor([[gamma_val]], dtype=dtype, device=device)
        with torch.no_grad():
            alpha = model.get_alpha(g_t)
            Vs, Cs = shock_conditions(alpha, g_t)
            V0, C0 = critical_point(alpha, g_t, n)

            s_dense = torch.linspace(0, 1, 500, dtype=dtype, device=device).unsqueeze(1)
            g_dense = g_t.expand(500, 1)
            C, _, V, _, _, _, _ = model(s_dense, g_dense)

        V_np = V.cpu().numpy().flatten()
        C_np = C.cpu().numpy().flatten()
        a_val = float(alpha)
        Vs_v, Cs_v = float(Vs), float(Cs)
        V0_v, C0_v = float(V0), float(C0)

        ax.plot(V_np, C_np, color=color, lw=2,
                label=f'γ={gamma_val:.1f}, α={a_val:.4f}')
        ax.plot(Vs_v, Cs_v, 'o', color=color, ms=8, zorder=5)   # 激波端
        ax.plot(V0_v, C0_v, 's', color=color, ms=8, zorder=5)   # 音速端

    ax.set_xlabel('V', fontsize=12)
    ax.set_ylabel('C', fontsize=12)
    ax.set_title(f'C(V) phase-plane — {geometry}\n(○=shock, □=sonic)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 右图：两种几何的α(γ)对比
ax = axes[2]
gammas_dense = np.linspace(1.2, 2.0, 200)
g_t = torch.tensor(gammas_dense, dtype=dtype, device=device).unsqueeze(1)

for n, geometry, color, ls in [(3, "spherical", "blue", "-"),
                                 (2, "cylindrical", "red", "--")]:
    model = ParametricPINN(n=n).to(device).to(dtype)
    model.load_state_dict(torch.load(out / f"pinn_parametric_{geometry}.pt",
                                      map_location=device))
    model.eval()
    with torch.no_grad():
        alpha_dense = model.get_alpha(g_t).cpu().numpy().flatten()
    ax.plot(gammas_dense, alpha_dense, color=color, ls=ls, lw=2, label=geometry)

ax.set_xlabel('γ', fontsize=12)
ax.set_ylabel('α', fontsize=12)
ax.set_title('Eigenvalue α(γ)\nspherical vs cylindrical', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Parametric PINN: Solution Structure', fontsize=14, fontweight='bold')
fig.tight_layout()
fname = out / "solution_structure_combined.png"
fig.savefig(fname, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")

# 打印各γ的α值
print("\nα values:")
print(f"{'γ':>6} {'α_spherical':>14} {'α_cylindrical':>14}")
for gv in gammas_plot:
    g_t2 = torch.tensor([[gv]], dtype=dtype, device=device)
    results = {}
    for n, geo in [(3,"spherical"),(2,"cylindrical")]:
        m = ParametricPINN(n=n).to(device).to(dtype)
        m.load_state_dict(torch.load(out/f"pinn_parametric_{geo}.pt", map_location=device))
        m.eval()
        with torch.no_grad():
            results[geo] = float(m.get_alpha(g_t2))
    print(f"{gv:>6.1f} {results['spherical']:>14.6f} {results['cylindrical']:>14.6f}")
