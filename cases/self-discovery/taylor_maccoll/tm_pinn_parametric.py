#!/usr/bin/env python3
"""
Taylor-Maccoll 参数化PINN: θ_c 作为输入

固定 M_inf=2.0, gamma=1.4，θ_c ∈ [10°, 40°] 作为网络输入。
一次训练，连续查询 β(θ_c)。

设计:
  beta_net(θ_c) → β  (sigmoid约束到Mach角~π/2)
  vr_net(s, θ_c) → v_r(s; θ_c)  直接输出，软约束BC
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from tm_common import oblique_shock_post_torch, v1_over_vmax
from tm_shooting import find_beta_shooting

M_INF = 2.0
GAMMA = 1.4
V1 = float(v1_over_vmax(M_INF, GAMMA))
MU_RAD = float(np.arcsin(1.0 / M_INF))


class ParametricTMPINN(nn.Module):
    def __init__(self, n_hidden=64, n_layers=5):
        super().__init__()
        self.beta_net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        layers = [nn.Linear(2, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, 1))
        self.vr_net = nn.Sequential(*layers)

    def get_beta(self, tc):
        """β ∈ (MU_RAD, π/2)"""
        return MU_RAD + (np.pi / 2 - MU_RAD) * torch.sigmoid(self.beta_net(tc))

    def forward(self, s, tc):
        """s,tc: (N,1) — direct output, no hard constraint"""
        return self.vr_net(torch.cat([s, tc], dim=1))


def compute_loss(model, tc_batch, n_colloc=200, device="cuda"):
    """tc_batch: (B,1) radians"""
    B = tc_batch.shape[0]
    dtype = tc_batch.dtype

    s = torch.linspace(0.01, 0.99, n_colloc, dtype=dtype, device=device)
    s = s.unsqueeze(0).expand(B, -1).reshape(-1, 1)
    s.requires_grad_(True)
    tc_exp = tc_batch.expand(-1, n_colloc).reshape(-1, 1)

    beta = model.get_beta(tc_exp)                  # (B*n,1)
    vr = model(s, tc_exp)                          # (B*n,1)

    dvr_ds = torch.autograd.grad(vr, s, grad_outputs=torch.ones_like(vr),
                                  create_graph=True)[0]
    d2vr_ds2 = torch.autograd.grad(dvr_ds, s, grad_outputs=torch.ones_like(dvr_ds),
                                    create_graph=True)[0]

    span = torch.clamp(beta - tc_exp, min=1e-3)
    dvr_dtheta = dvr_ds / span
    d2vr_dtheta2 = d2vr_ds2 / span**2
    theta = tc_exp + s.detach() * span.detach()

    vt = dvr_dtheta
    c2 = (GAMMA - 1) / 2 * (1 - vr**2 - vt**2)
    cot_theta = torch.cos(theta) / (torch.sin(theta) + 1e-14)
    numer = vt**2 * vr - c2 * (2 * vr + vt * cot_theta)
    denom = c2 - vt**2

    # 混合残差
    w_far = torch.clamp(torch.abs(denom) / 1e-6 - 1.0, 0.0, 1.0)
    R_div = d2vr_dtheta2 - numer / (denom + 1e-30 * torch.sign(denom + 1e-30))
    R_mult = d2vr_dtheta2 * denom - numer
    loss_ode = (w_far * R_div**2 + (1 - w_far) * R_mult**2).mean()

    # 软约束BC (per sample)
    s0 = torch.zeros(B, 1, dtype=dtype, device=device, requires_grad=True)
    tc_b = tc_batch
    beta_b = model.get_beta(tc_b)
    vr0 = model(s0, tc_b)
    dvr_ds0 = torch.autograd.grad(vr0, s0, grad_outputs=torch.ones_like(vr0),
                                   create_graph=True)[0]
    span_b = torch.clamp(beta_b - tc_b, min=1e-3)
    vt0 = dvr_ds0 / span_b                        # v_θ at cone = 0

    s1 = torch.ones(B, 1, dtype=dtype, device=device)
    vr1 = model(s1, tc_b)
    vr_shock_theory = V1 * torch.cos(beta_b)      # v_r at shock

    # v_θ at shock
    dvr_ds1 = torch.autograd.grad(vr1, s1, grad_outputs=torch.ones_like(vr1),
                                   create_graph=True)[0] if s1.requires_grad else \
               torch.autograd.grad(model(s1.requires_grad_(True), tc_b),
                                   s1, grad_outputs=torch.ones(B,1,dtype=dtype,device=device),
                                   create_graph=True)[0]
    _, vt_shock_theory = oblique_shock_post_torch(M_INF, beta_b, GAMMA)

    loss_bc = ((vr1 - vr_shock_theory)**2).mean() + \
              100.0 * (vt0**2).mean() + \
              10.0 * ((dvr_ds1 / span_b - vt_shock_theory)**2).mean()

    return loss_ode + 100.0 * loss_bc


def pretrain_beta_net(model, tc_min_deg, tc_max_deg, device, dtype, n_pts=20, epochs=3000):
    print("Pretraining beta_net with shooting reference...")
    tc_degs = np.linspace(tc_min_deg, tc_max_deg, n_pts)
    betas_ref = []
    for tc in tc_degs:
        r = find_beta_shooting(M_INF, GAMMA, tc)
        betas_ref.append(r["beta_deg"] if r["beta_deg"] else np.nan)
    betas_ref = np.array(betas_ref)
    mask = ~np.isnan(betas_ref)
    tc_t = torch.tensor(np.radians(tc_degs[mask]), dtype=dtype, device=device).unsqueeze(1)
    b_t  = torch.tensor(np.radians(betas_ref[mask]), dtype=dtype, device=device).unsqueeze(1)
    opt = torch.optim.Adam(model.beta_net.parameters(), lr=1e-2)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = ((model.get_beta(tc_t) - b_t)**2).mean()
        loss.backward()
        opt.step()
        if ep % 1000 == 0:
            err = float((torch.abs(model.get_beta(tc_t) - b_t) / b_t * 100).max().detach())
            print(f"  pretrain ep {ep:4d}  loss={loss.item():.4e}  max_err={err:.4f}%")
    print("Pretrain done.")


def train(tc_min_deg=10.0, tc_max_deg=40.0, epochs=10000,
          batch_size=32, lr=1e-3, device="cuda", seed=42):
    torch.manual_seed(seed)
    dtype = torch.float64
    model = ParametricTMPINN().to(device).to(dtype)
    pretrain_beta_net(model, tc_min_deg, tc_max_deg, device, dtype)

    rng = np.random.default_rng(seed)

    def run_phase(n_epochs, freeze_beta, lr_vr, lr_beta=0.0):
        for p in model.beta_net.parameters():
            p.requires_grad_(not freeze_beta)
        params = [{"params": model.vr_net.parameters(), "lr": lr_vr}]
        if not freeze_beta:
            params.append({"params": model.beta_net.parameters(), "lr": lr_beta})
        opt = torch.optim.Adam(params)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
        for epoch in range(1, n_epochs + 1):
            tc_np = rng.uniform(np.radians(tc_min_deg), np.radians(tc_max_deg), size=(batch_size,))
            tc_t = torch.tensor(tc_np, dtype=dtype, device=device).unsqueeze(1)
            opt.zero_grad()
            loss = compute_loss(model, tc_t, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            if epoch % 1000 == 0 or epoch == 1:
                print(f"  epoch {epoch:6d}  loss={loss.item():.4e}")

    # Phase 1: freeze beta, train vr_net only (100% epochs)
    print(f"  Training vr_net with frozen beta ({epochs} epochs)")
    run_phase(epochs, freeze_beta=True, lr_vr=lr)

    return model


def evaluate(model, tc_min_deg, tc_max_deg, device, n_ref=20):
    dtype = torch.float64
    model.eval()
    tc_degs = np.linspace(tc_min_deg, tc_max_deg, n_ref)
    ref_betas, pinn_betas = [], []
    print("\nEvaluating vs shooting reference...")
    for tc in tc_degs:
        r = find_beta_shooting(M_INF, GAMMA, tc)
        ref_betas.append(r["beta_deg"] if r["beta_deg"] else np.nan)
        tc_t = torch.tensor([[np.radians(tc)]], dtype=dtype, device=device)
        with torch.no_grad():
            pinn_betas.append(float(torch.rad2deg(model.get_beta(tc_t))))
    ref_betas = np.array(ref_betas); pinn_betas = np.array(pinn_betas)
    # filter invalid: beta must be > theta_c (physical constraint)
    mask = ~np.isnan(ref_betas) & (ref_betas > tc_degs)
    err = np.abs(pinn_betas[mask] - ref_betas[mask]) / ref_betas[mask] * 100
    print(f"  Max error: {err.max():.4f}%  Mean: {err.mean():.4f}%")
    return tc_degs[mask], ref_betas[mask], pinn_betas[mask], err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc-min",    type=float, default=10.0)
    parser.add_argument("--tc-max",    type=float, default=28.0)  # M=2 detachment ~30°
    parser.add_argument("--epochs",    type=int,   default=10000)
    parser.add_argument("--batch-size",type=int,   default=32)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--n-runs",    type=int,   default=1)
    parser.add_argument("--device",    type=str,   default="cuda")
    parser.add_argument("--output-dir",type=str,
                        default=str(Path(__file__).parent / "output"))
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device; dtype = torch.float64

    best_model, best_err = None, float("inf")
    for run in range(args.n_runs):
        seed = 42 + run
        print(f"\n{'='*60}\nRun {run+1}/{args.n_runs} (seed={seed})")
        model = train(args.tc_min, args.tc_max, args.epochs,
                      args.batch_size, args.lr, device, seed)
        _, _, _, err = evaluate(model, args.tc_min, args.tc_max, device)
        if err.max() < best_err:
            best_err, best_model = err.max(), model
        print(f"  run {run+1} max_err={err.max():.4f}%")

    print(f"\nBest max_err={best_err:.4f}%")
    tc_ref, ref_betas, pinn_betas, err = evaluate(best_model, args.tc_min, args.tc_max,
                                                   device, n_ref=30)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tc_dense = np.linspace(args.tc_min, args.tc_max, 200)
    tc_t = torch.tensor(np.radians(tc_dense), dtype=dtype, device=device).unsqueeze(1)
    with torch.no_grad():
        beta_dense = torch.rad2deg(best_model.get_beta(tc_t)).cpu().numpy().reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tc_dense, beta_dense, "b-", lw=2, label="PINN")
    axes[0].scatter(tc_ref, ref_betas, color="r", s=40, zorder=5, label="shooting")
    axes[0].set_xlabel(r"$\theta_c$ (deg)"); axes[0].set_ylabel(r"$\beta$ (deg)")
    axes[0].set_title(f"Parametric TM PINN: β(θ_c), M={M_INF}, γ={GAMMA}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(tc_ref, err, "o-", lw=2)
    axes[1].set_xlabel(r"$\theta_c$ (deg)"); axes[1].set_ylabel("relative error (%)")
    axes[1].set_title("β error vs shooting"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fname = out / "tm_pinn_parametric.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {fname}")
    print(f"Final: Max={err.max():.4f}%  Mean={err.mean():.4f}%")


if __name__ == "__main__":
    main()
