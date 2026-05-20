#!/usr/bin/env python3
"""
Taylor-Maccoll PINN自发现: 123框架迁移

给定 (M∞, γ, θ_c), 发现激波角 β。
纯物理，零数据。

框架迁移 (Guderley → Taylor-Maccoll):
  1. 化简: 3D轴对称Euler PDE → Taylor-Maccoll ODE
  2. 混合残差: 远离sonic用除形式, 靠近sonic用乘形式
  3. 硬约束:
     - θ=β (激波): v_r = v₁cosβ (由β确定)
     - θ=θ_c (锥面): dv_r/dθ = 0 (切向条件)
     - 构造: v_r(s) = A + (B-A)s² + s²(1-s)net(s)
       保证 v_r(0)=A, v_r'(0)=0, v_r(1)=B
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tm_common import (oblique_shock_post_torch, oblique_shock_post,
                        v1_over_vmax, TEST_CASES)


class TaylorMaccollPINN(nn.Module):
    """
    Taylor-Maccoll PINN: 123框架

    可训练参数: β (激波角), v_rc (锥面径向速度)
    网络: s ∈ [0,1] → v_r(s)
    硬约束:
      s=0 (锥面): v_r(0) = v_rc, dv_r/ds(0) = 0
      s=1 (激波): v_r(1) = v₁·cos(β)
    """
    def __init__(self, M_inf, gamma, theta_c_deg,
                 beta_init_deg=None, n_layers=5, n_neurons=64):
        super().__init__()
        self.M_inf = M_inf
        self.gamma = gamma
        self.theta_c_rad = np.radians(theta_c_deg)

        # β初始猜测: 用2D wedge的θ-β-M近似 (cone shock比wedge shock更弱, β更小)
        if beta_init_deg is None:
            mu_deg = np.degrees(np.arcsin(1.0 / M_inf))
            # 粗略: β ≈ θ_c + mu_deg (在weak shock branch)
            beta_init_deg = theta_c_deg + mu_deg * 0.6
            beta_init_deg = max(beta_init_deg, mu_deg + 2.0)
            beta_init_deg = min(beta_init_deg, 70.0)

        # β参数化: β ∈ (Mach角, 90°)
        mu_rad = np.arcsin(1.0 / M_inf)
        self.mu_rad = mu_rad
        # β = mu + (π/2 - mu) * sigmoid(raw_beta)
        beta_init_rad = np.radians(beta_init_deg)
        sigmoid_val = (beta_init_rad - mu_rad) / (np.pi / 2 - mu_rad)
        sigmoid_val = max(min(sigmoid_val, 0.999), 0.001)
        raw_init = float(np.log(sigmoid_val / (1 - sigmoid_val)))
        self.raw_beta = nn.Parameter(torch.tensor(raw_init, dtype=torch.float64))

        # v_rc参数 (锥面径向速度, 范围约0.3~0.9)
        # 初始猜测: v1 * cos(beta_init)  (激波处的值，实际锥面值更大)
        v1 = v1_over_vmax(M_inf, gamma)
        vrc_init = v1 * np.cos(np.radians(beta_init_deg)) * 1.05
        self.raw_vrc = nn.Parameter(torch.tensor(
            float(np.log(vrc_init / (1 - vrc_init + 1e-10))), dtype=torch.float64))

        # MLP: s → raw(1)
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    @property
    def beta(self):
        """β ∈ (Mach角, 90°)"""
        return self.mu_rad + (np.pi / 2 - self.mu_rad) * torch.sigmoid(self.raw_beta)

    @property
    def v_rc(self):
        """v_r at cone, ∈ (0, 1)"""
        return torch.sigmoid(self.raw_vrc)

    def shock_vr(self):
        """v_r at shock = v₁·cos(β)"""
        v1_sq_num = self.M_inf**2 * (self.gamma - 1)
        v1 = torch.sqrt(torch.tensor(v1_sq_num, dtype=torch.float64)
                         / (v1_sq_num + 2))
        return v1 * torch.cos(self.beta)

    def forward(self, s):
        """
        硬约束构造:
          v_r(s) = A + (B - A)·s² + s²·(1-s)·net(s)

        保证:
          v_r(0) = A = v_rc     (锥面)
          v_r'(0) = 0           (切向条件)
          v_r(1) = B = v₁cosβ   (激波)
        """
        A = self.v_rc
        B = self.shock_vr()

        linear = A + (B - A) * s**2
        correction = s**2 * (1 - s) * self.net(s)
        return linear + correction

    def compute_ode_residual(self, theta, vr, dvr_dtheta, d2vr_dtheta2):
        """
        Taylor-Maccoll ODE残差

        ODE: v_r'' = [v_θ²·v_r - c̃²·(2v_r + v_θ·cotθ)] / [c̃² - v_θ²]
        乘形式: v_r''·D - N = 0
        除形式: v_r'' - N/D = 0

        v_θ = dv_r/dθ
        """
        vt = dvr_dtheta  # v_θ = dv_r/dθ
        c2 = (self.gamma - 1) / 2 * (1 - vr**2 - vt**2)

        cot_theta = torch.cos(theta) / (torch.sin(theta) + 1e-14)

        numer = vt**2 * vr - c2 * (2 * vr + vt * cot_theta)
        denom = c2 - vt**2

        residual_mult = d2vr_dtheta2 * denom - numer  # 乘形式

        return residual_mult, numer, denom


def train_pinn(M_inf, gamma, theta_c_deg,
               beta_init_deg=None, epochs=10000, lr=1e-3,
               n_colloc=300, device="cuda"):
    """
    PINN训练: 自发现激波角β

    Loss = L_ode + λ·L_tangency_regularity
    纯物理，无数据。
    """
    model = TaylorMaccollPINN(
        M_inf, gamma, theta_c_deg, beta_init_deg
    ).double().to(device)

    history = {"beta_deg": [], "loss": [], "vrc": []}

    def compute_loss():
        beta = model.beta
        theta_c = model.theta_c_rad

        # θ域: [θ_c, β]
        # s = (θ - θ_c) / (β - θ_c) ∈ [0, 1]
        beta_d = beta.detach()

        s_colloc = torch.linspace(0.01, 0.99, n_colloc,
                                  dtype=torch.float64, device=device).reshape(-1, 1)
        s_colloc.requires_grad_(True)

        vr = model(s_colloc)

        # dv_r/ds
        dvr_ds = torch.autograd.grad(
            vr, s_colloc,
            grad_outputs=torch.ones_like(vr),
            create_graph=True,
        )[0]

        # d²v_r/ds²
        d2vr_ds2 = torch.autograd.grad(
            dvr_ds, s_colloc,
            grad_outputs=torch.ones_like(dvr_ds),
            create_graph=True,
        )[0]

        # 转换到θ域
        span = beta - theta_c  # β - θ_c (可微)
        span_d = span.detach()  # 用于配点位置（不需要对β求导）
        dvr_dtheta = dvr_ds / span
        d2vr_dtheta2 = d2vr_ds2 / span**2

        # θ值 (detach: 配点位置不需要对β求导)
        theta = theta_c + s_colloc * span_d

        # ODE残差
        residual_mult, numer, denom = model.compute_ode_residual(
            theta, vr, dvr_dtheta, d2vr_dtheta2)

        # ── 混合残差 (Component 2) ──
        abs_denom = torch.abs(denom)
        denom_thr = 1e-6

        # 远离sonic: 除形式
        w_far = torch.clamp(abs_denom / denom_thr - 1.0, min=0.0, max=1.0)
        R_divided = d2vr_dtheta2 - numer / (denom + 1e-30 * torch.sign(denom))
        loss_ode_far = torch.mean(w_far * R_divided**2)

        # 靠近sonic: 乘形式
        w_sonic = 1.0 - w_far
        loss_ode_sonic = torch.mean(w_sonic * residual_mult**2)

        loss_ode = loss_ode_far + loss_ode_sonic

        # ── 激波v_θ条件 (软约束加项) ──
        # 在s=1 (激波处), v_θ应该等于斜激波关系给出的值
        s_shock = torch.tensor([[0.99]], dtype=torch.float64, device=device,
                               requires_grad=True)
        vr_shock = model(s_shock)
        dvr_ds_shock = torch.autograd.grad(
            vr_shock, s_shock,
            grad_outputs=torch.ones_like(vr_shock),
            create_graph=True,
        )[0]
        vt_shock_pred = dvr_ds_shock / span  # v_θ at shock

        # 斜激波理论值
        _, vt_shock_theory = oblique_shock_post_torch(M_inf, beta, gamma)

        loss_shock_vt = (vt_shock_pred - vt_shock_theory)**2

        return loss_ode, loss_shock_vt.squeeze()

    # ── Phase 1: warmup (固定β, v_rc可学) ──
    warmup = epochs // 3
    model.raw_beta.requires_grad_(False)
    # v_rc可以在warmup中学习

    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    beta_init = float(torch.rad2deg(model.beta.detach()))
    print(f"  Phase 1: warmup {warmup} ep, β_init={beta_init:.2f}°")

    for epoch in range(1, warmup + 1):
        opt1.zero_grad()
        loss_ode, loss_svt = compute_loss()
        loss = loss_ode + 10.0 * loss_svt
        loss.backward()
        opt1.step()

        b = float(torch.rad2deg(model.beta.detach()))
        history["beta_deg"].append(b)
        history["loss"].append(float(loss.detach()))
        history["vrc"].append(float(model.v_rc.detach()))

        if epoch % 1000 == 0 or epoch == 1:
            print(f"    warmup {epoch:5d}  loss={loss.item():.3e}  "
                  f"ode={loss_ode.item():.3e}  svt={loss_svt.item():.3e}  "
                  f"β={b:.4f}°  v_rc={float(model.v_rc.detach()):.6f}")

    # ── Phase 2: 解冻β和v_rc ──
    model.raw_beta.requires_grad_(True)
    model.raw_vrc.requires_grad_(True)
    main_epochs = epochs - warmup

    opt2 = optim.Adam([
        {"params": model.net.parameters(), "lr": lr * 0.3},
        {"params": [model.raw_beta, model.raw_vrc], "lr": lr * 0.3},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=main_epochs, eta_min=1e-6)

    print(f"  Phase 2: joint {main_epochs} ep (β, v_rc free)")

    for epoch in range(1, main_epochs + 1):
        opt2.zero_grad()
        loss_ode, loss_svt = compute_loss()
        loss = loss_ode + 100.0 * loss_svt
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt2.step()
        scheduler.step()

        b = float(torch.rad2deg(model.beta.detach()))
        history["beta_deg"].append(b)
        history["loss"].append(float(loss.detach()))
        history["vrc"].append(float(model.v_rc.detach()))

        if epoch % 1000 == 0 or epoch == 1:
            print(f"    joint {epoch:5d}  loss={loss.item():.3e}  "
                  f"ode={loss_ode.item():.3e}  svt={loss_svt.item():.3e}  "
                  f"β={b:.4f}°  v_rc={float(model.v_rc.detach()):.6f}")

    beta_fit = float(torch.rad2deg(model.beta.detach()))
    vrc_fit = float(model.v_rc.detach())

    return {
        "beta_deg": beta_fit,
        "v_rc": vrc_fit,
        "final_loss": history["loss"][-1],
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=12000)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 先运行打靶法获取参考值
    from tm_shooting import find_beta_shooting

    print("=" * 70)
    print("  Taylor-Maccoll: 123框架PINN自发现")
    print("=" * 70)

    all_results = {}
    for tc in TEST_CASES:
        M, gamma, tc_deg, label = tc["M_inf"], tc["gamma"], tc["theta_c_deg"], tc["label"]

        print(f"\n{'='*60}")
        print(f"  {label}: M={M}, γ={gamma}, θ_c={tc_deg}°")
        print(f"{'='*60}")

        # 打靶法参考
        print(f"\n  --- 打靶法参考 ---")
        ref = find_beta_shooting(M, gamma, tc_deg)
        ref_beta = ref["beta_deg"]
        print(f"    β_ref = {ref_beta:.10f}°")

        # PINN
        print(f"\n  --- PINN 123框架 ---")
        pinn_result = train_pinn(
            M, gamma, tc_deg,
            epochs=args.epochs,
            device=args.device,
        )

        err_pct = abs(pinn_result["beta_deg"] - ref_beta) / ref_beta * 100
        print(f"\n  β_pinn = {pinn_result['beta_deg']:.6f}°")
        print(f"  β_ref  = {ref_beta:.10f}°")
        print(f"  err    = {err_pct:.6f}%")

        all_results[label] = {
            "ref_beta_deg": ref_beta,
            "pinn_beta_deg": pinn_result["beta_deg"],
            "err_pct": err_pct,
            "pinn_vrc": pinn_result["v_rc"],
            "pinn_loss": pinn_result["final_loss"],
        }

    # 汇总
    print(f"\n{'='*70}")
    print("  汇总: PINN vs 打靶法")
    print(f"{'='*70}")
    print(f"  {'Label':<20} {'β_ref':>12} {'β_pinn':>12} {'err%':>12}")
    print(f"  {'-'*58}")
    for label, r in all_results.items():
        print(f"  {label:<20} {r['ref_beta_deg']:>12.6f} {r['pinn_beta_deg']:>12.6f} {r['err_pct']:>12.6f}")

    with open(out_dir / "tm_pinn.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_dir / 'tm_pinn.json'}")
