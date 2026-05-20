#!/usr/bin/env python3
"""
Case-by-case PINN: 对固定(gamma, n)训练，得到精确alpha。
用于和参数化PINN做精度对比。
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from pinn_parametric import (ParametricPINN, shock_conditions, critical_point,
                              chisnell_rhs)
from guderley_ode_solver import find_eigenvalue

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
run_dir = Path(__file__).parent / "output" / "run_20260512_173826"

SONIC_THRESH = 1e-3


class CasePINN(nn.Module):
    """单个(gamma, n)的case-by-case PINN，123框架"""
    def __init__(self, n: int, gamma: float, alpha_init: float = 0.75):
        super().__init__()
        self.n = n
        self.gamma_val = gamma
        raw = float(np.log((alpha_init - 0.5) / (1.0 - (alpha_init - 0.5))))
        self.raw_alpha = nn.Parameter(torch.tensor(raw, dtype=torch.float64))
        layers = [nn.Linear(1, 64), nn.Tanh()]
        for _ in range(3):
            layers += [nn.Linear(64, 64), nn.Tanh()]
        layers.append(nn.Linear(64, 1))
        self.C_net = nn.Sequential(*layers)
        self.double()  # ensure float64

    def get_alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def forward(self, s):
        alpha = self.get_alpha()
        gamma = torch.tensor([[self.gamma_val]], dtype=torch.float64, device=s.device)
        Vs, Cs = shock_conditions(alpha.unsqueeze(0), gamma)
        V0, C0 = critical_point(alpha.unsqueeze(0), gamma, self.n)
        V = Vs + s * (V0 - Vs)
        correction = self.C_net(s)
        C = (1 - s) * Cs + s * C0 + s * (1 - s) * correction
        return C, alpha, V, Vs, Cs, V0, C0


def train_case(n, gamma, epochs=10000, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    ref = find_eigenvalue(gamma, n, "spherical" if n == 3 else "cylindrical", verbose=False)
    alpha_ref = ref["alpha"]
    alpha_init = alpha_ref * 1.05 if alpha_ref else 0.75  # 偏离5%作为初始值

    model = CasePINN(n=n, gamma=gamma, alpha_init=alpha_init).to(device).double()
    opt = torch.optim.Adam([
        {"params": model.C_net.parameters(), "lr": lr},
        {"params": [model.raw_alpha], "lr": lr * 0.2},
    ])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        s = torch.tensor(rng.uniform(0.01, 0.99, (200, 1)), dtype=dtype, device=device,
                         requires_grad=True)
        C, alpha, V, Vs, Cs, V0, C0 = model(s)
        dC_ds = torch.autograd.grad(C, s, grad_outputs=torch.ones_like(C),
                                     create_graph=True)[0]
        dV_ds = (V0 - Vs)
        dC_dV = dC_ds / (dV_ds + 1e-30)
        g_t = torch.full_like(V, gamma)
        N, D = chisnell_rhs(V, C, alpha, g_t, n)
        abs_D = torch.abs(D)
        w = torch.clamp(abs_D / SONIC_THRESH - 1.0, 0.0, 1.0)
        res = w * (dC_dV - N / (D + 1e-30)) + (1 - w) * (dC_dV * D - N)
        loss = res.pow(2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()

    alpha_pinn = float(model.get_alpha().detach())
    err = abs(alpha_pinn - alpha_ref) / alpha_ref * 100 if alpha_ref else np.nan
    return alpha_pinn, alpha_ref, err


# 测试点：5个γ值，两种几何
test_gammas = [1.2, 1.4, 1.6, 1.8, 2.0]
results = {}

print(f"{'γ':>5} {'geo':>12} {'α_ref':>10} {'α_pinn':>10} {'err%':>8}")
print("-" * 50)
for gamma in test_gammas:
    for n, geo in [(3, "spherical"), (2, "cylindrical")]:
        alpha_pinn, alpha_ref, err = train_case(n, gamma, epochs=10000)
        results[(gamma, geo)] = err
        print(f"{gamma:>5.1f} {geo:>12} {alpha_ref:>10.6f} {alpha_pinn:>10.6f} {err:>8.4f}%")

# 保存结果
np.save(run_dir / "casebycase_errors.npy", results)

# 更新对比图（用真实case-by-case数据）
param_errs = {"spherical": [], "cylindrical": []}
gammas_eval = np.linspace(1.2, 2.0, 20)
for n, geo in [(3, "spherical"), (2, "cylindrical")]:
    model = ParametricPINN(n=n).to(device).to(dtype)
    model.load_state_dict(torch.load(run_dir / f"pinn_parametric_{geo}.pt", map_location=device))
    model.eval()
    for g in gammas_eval:
        ref = find_eigenvalue(g, n, geo, verbose=False)
        alpha_ref = ref["alpha"]
        g_t = torch.tensor([[g]], dtype=dtype, device=device)
        with torch.no_grad():
            alpha_pinn = float(model.get_alpha(g_t))
        err = abs(alpha_pinn - alpha_ref) / alpha_ref * 100 if alpha_ref else np.nan
        param_errs[geo].append(err)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, geo, color in zip(axes, ["spherical", "cylindrical"], ["steelblue", "tomato"]):
    errs = np.array(param_errs[geo])
    mask = ~np.isnan(errs)
    ax.semilogy(gammas_eval[mask], errs[mask], 'o-', color=color, lw=2, ms=5,
                label=f'Parametric PINN (1 training)')

    # 真实case-by-case数据点
    cbc_gammas = test_gammas
    cbc_errs = [results[(g, geo)] for g in cbc_gammas]
    ax.semilogy(cbc_gammas, cbc_errs, 's--', color='gray', lw=1.5, ms=7,
                label='Case-by-case PINN (per γ)')

    ax.set_xlabel(r'$\gamma$', fontsize=12)
    ax.set_ylabel(r'$\alpha$ relative error (%)', fontsize=12)
    ax.set_title(f'{geo.capitalize()}', fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

fig.suptitle('Accuracy Trade-off: Parametric vs Case-by-case PINN', fontsize=13, fontweight='bold')
fig.tight_layout()
fname = run_dir / "comparison_real_casebycase.png"
fig.savefig(fname, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {fname}")
