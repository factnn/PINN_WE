#!/usr/bin/env python3
"""
扩展性验证3: 带噪声观测数据的α反演

打靶法假设精确的ODE结构, 无法处理含噪声的观测数据。
PINN天然支持: L = L_ode + λ * MSE(C_pred, C_obs_noisy)

实验设计: 对123单参数反演, 在观测C(V)上加不同级别噪声(1%, 5%, 10%),
测试PINN的鲁棒性。每个噪声级别跑3个seed。
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
from common import (build_mlp, inv_sigmoid, chisnell_residual,
                    critical_point_np, get_ref_alpha, GEOMETRY_N)

REF_ALPHA = {
    (1.4, 3): 0.717174501487,
    (1.4, 2): 0.835323191951,
    (5/3, 3): 0.688376822922,
    (5/3, 2): 0.815624901431,
}


def generate_noisy_data(gamma, n, n_obs=20, noise_pct=5.0, seed=42):
    """Generate C(V) observations with Gaussian noise."""
    np.random.seed(seed)
    alpha = REF_ALPHA[(gamma, n)]
    def rhs(V, C_vec):
        C = C_vec[0]
        delta = (V - alpha)**2 - C
        Q = n*V*(V-alpha) + (2.0/gamma)*(1.0-alpha)*(alpha-V) - V*(V-1.0)
        numer = C*(2.0*delta*(alpha-V+(1.0-alpha)/gamma) + (gamma-1.0)*(alpha-V)*Q)
        denom = delta*(n*V - 2.0*(1.0-alpha)/gamma)*(alpha-V) + (alpha-V)**2*Q
        return [numer/(denom+1e-30)]

    Vs = 2.0*alpha/(gamma+1.0)
    Cs = 2.0*gamma*(gamma-1.0)*alpha**2/(gamma+1.0)**2
    V0, C0 = critical_point_np(alpha, gamma, n)
    sol = solve_ivp(rhs, [Vs, V0], [Cs], method='DOP853',
                    rtol=1e-12, atol=1e-14, dense_output=True)
    V_obs = np.linspace(Vs + 0.05*(V0-Vs), V0 - 0.05*(V0-Vs), n_obs)
    C_clean = sol.sol(V_obs)[0]
    noise = np.random.normal(0, 1, size=C_clean.shape) * (noise_pct/100.0) * np.abs(C_clean)
    C_noisy = C_clean + noise
    return V_obs, C_noisy, C_clean


class InversePINN(nn.Module):
    def __init__(self, gamma, n, alpha_init=0.75, n_layers=5, n_neurons=64):
        super().__init__()
        self.gamma = gamma
        self.n = n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init-0.5)/0.5), dtype=torch.float64))
        self.net = build_mlp(1, 1, n_layers, n_neurons).double()

    @property
    def alpha(self):
        return 0.5 + 0.5*torch.sigmoid(self.raw_alpha)

    def shock_cond(self):
        a, g = self.alpha, self.gamma
        Vs = 2.0*a/(g+1.0)
        Cs = 2.0*g*(g-1.0)*a**2/(g+1.0)**2
        return Vs, Cs

    def sonic_cond(self):
        from common import critical_point_torch
        return critical_point_torch(self.alpha, self.gamma, self.n)

    def forward(self, V):
        Vs, Cs = self.shock_cond()
        V0, C0 = self.sonic_cond()
        s = (V - Vs)/(V0 - Vs + 1e-14)
        return (1.0-s)*Cs + s*C0 + s*(1.0-s)*self.net(s)


def train_noisy_inverse(gamma, n, geometry, V_obs, C_obs, alpha_init,
                        lambda_data=10.0, epochs=10000, device="cuda"):
    ref_alpha = REF_ALPHA[(gamma, n)]
    model = InversePINN(gamma, n, alpha_init).double().to(device)
    V_obs_t = torch.tensor(V_obs, dtype=torch.float64, device=device).reshape(-1,1)
    C_obs_t = torch.tensor(C_obs, dtype=torch.float64, device=device).reshape(-1,1)

    def compute_loss():
        alpha = model.alpha
        Vs, Cs = model.shock_cond()
        V0, C0 = model.sonic_cond()
        Vs_d, V0_d = Vs.detach(), V0.detach()
        V_colloc = (Vs_d + (V0_d - Vs_d) *
                    torch.linspace(0.01, 0.99, 400, dtype=torch.float64, device=device)
                    ).reshape(-1, 1)
        V_colloc.requires_grad_(True)
        C_pred = model(V_colloc)
        dC_dV = torch.autograd.grad(C_pred, V_colloc,
                    grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
        residual, numer, denom, delta = chisnell_residual(
            V_colloc, C_pred, dC_dV, alpha, gamma, n)
        abs_denom = torch.abs(denom)
        w_far = torch.clamp(abs_denom/1e-6 - 1.0, min=0.0, max=1.0)
        R_divided = dC_dV - numer/(denom + 1e-30*torch.sign(denom))
        w_sonic = 1.0 - w_far
        loss_ode = (torch.mean(w_far*R_divided**2)
                    + torch.mean(w_sonic*residual**2)
                    + 10.0*torch.mean(w_sonic*numer**2))
        loss_data = torch.mean((model(V_obs_t) - C_obs_t)**2)
        return loss_ode, loss_data

    opt = optim.Adam([
        {"params": model.net.parameters(), "lr": 1e-3},
        {"params": [model.raw_alpha], "lr": 2e-4},
    ])
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        lo, ld = compute_loss()
        loss = lo + lambda_data*ld
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

    af = float(model.alpha.detach())
    ae = abs(af - ref_alpha)/ref_alpha*100
    return {"alpha_fit": af, "ref_alpha": ref_alpha, "err_pct": ae}


CONFIGS = [
    (1.4, 3, "spherical", 0.72),
    (1.4, 2, "cylindrical", 0.84),
    (5/3, 3, "spherical", 0.69),
    (5/3, 2, "cylindrical", 0.82),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = [0.0, 1.0, 5.0, 10.0]
    seeds = [42, 123, 999]

    print("="*70)
    print("  噪声鲁棒性测试: α反演 + 不同噪声级别 + 多seed")
    print("="*70)

    all_results = {}
    for noise_pct in noise_levels:
        for seed in seeds:
            tag = f"noise{noise_pct:.0f}pct_seed{seed}"
            print(f"\n--- {tag} ---")
            results = {}
            for gamma, n, geo, a_init in CONFIGS:
                key = f"g{gamma:.2f}_{geo}"
                V_obs, C_noisy, C_clean = generate_noisy_data(
                    gamma, n, n_obs=20, noise_pct=noise_pct, seed=seed)
                r = train_noisy_inverse(
                    gamma, n, geo, V_obs, C_noisy, a_init, device=args.device)
                results[key] = r
                status = "OK" if r["err_pct"] < 0.1 else "FAIL"
                print(f"  {key}: α={r['alpha_fit']:.8f} err={r['err_pct']:.6f}% [{status}]")
            all_results[tag] = results

    # Summary table
    print(f"\n{'='*70}")
    print("  NOISE ROBUSTNESS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'noise%':<10} {'seed':<6} {'g1.4球':<12} {'g1.4柱':<12} {'g5/3球':<12} {'g5/3柱':<12} {'avg_err':<10}")
    print(f"  {'-'*72}")
    for noise_pct in noise_levels:
        for seed in seeds:
            tag = f"noise{noise_pct:.0f}pct_seed{seed}"
            r = all_results[tag]
            errs = [r[k]["err_pct"] for k in r]
            avg = np.mean(errs)
            vals = "  ".join(f"{r[k]['err_pct']:<10.6f}" for k in
                           ["g1.40_spherical","g1.40_cylindrical","g1.67_spherical","g1.67_cylindrical"])
            print(f"  {noise_pct:<10.0f} {seed:<6} {vals} {avg:<10.6f}")

    with open(out_dir / "exp_noisy_robustness.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'exp_noisy_robustness.json'}")
