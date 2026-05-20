#!/usr/bin/env python3
"""
扩展性验证4: 多几何统一自发现 — 一个网络同时发现球面+柱面α

打靶法只能逐个几何单独求解，无法利用几何间的物理关联。
PINN天然支持: 一个共享网络 + 两组ODE残差loss + 物理一致性约束。

具体方法:
  - 网络 C_net(s, n) 接受归一化坐标s和几何参数n作为输入
  - α_sph, α_cyl 分别为两个可训练参数
  - L = L_ode_sph + L_ode_cyl + λ_order * L_ordering

  加项优势:
  1. L_ordering: α_cyl > α_sph (物理上柱面衰减更慢, α更大)
     — 这是已知的物理先验，打靶法无法利用
  2. 共享网络: 两个几何的ODE共享权重，mutual regularization
     — 打靶法两次独立积分，无法共享信息

预期: 两个α同时收敛，且利用ordering约束提高鲁棒性。
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common import (inv_sigmoid, chisnell_residual,
                    critical_point_np, critical_point_torch,
                    build_mlp, ALPHA_INIT)

REF_ALPHA = {
    (1.4, 3): 0.717174501487,
    (1.4, 2): 0.835323191951,
    (5/3, 3): 0.688376822922,
    (5/3, 2): 0.815624901431,
}


class MultiGeometryPINN(nn.Module):
    """
    共享网络 + 两组α参数，同时发现球面和柱面的Guderley特征值。

    输入: (s, n_norm) 其中 s∈[0,1] 是归一化坐标, n_norm=(n-2.5)/0.5 ∈ {-1, +1}
    输出: C(s, n)

    硬约束: C(0, n) = Cs(α_n), C(1, n) = C0(α_n)
    """
    def __init__(self, gamma, alpha_init_sph=0.72, alpha_init_cyl=0.84,
                 n_layers=5, n_neurons=64):
        super().__init__()
        self.gamma = gamma

        # 两个独立的α参数
        self.raw_alpha_sph = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init_sph - 0.5) / 0.5), dtype=torch.float64))
        self.raw_alpha_cyl = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init_cyl - 0.5) / 0.5), dtype=torch.float64))

        # 共享网络: 输入 (s, n_norm) -> 输出 1
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    def alpha(self, n):
        """返回对应几何的α"""
        if n == 3:
            return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha_sph)
        else:
            return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha_cyl)

    def shock_cond(self, alpha):
        gamma = self.gamma
        Vs = 2.0 * alpha / (gamma + 1.0)
        Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
        return Vs, Cs

    def sonic_cond(self, alpha, n):
        V0, C0 = critical_point_torch(alpha, self.gamma, n)
        return V0, C0

    def forward(self, V, n):
        """双端硬约束，几何n作为网络输入"""
        alpha = self.alpha(n)
        Vs, Cs = self.shock_cond(alpha)
        V0, C0 = self.sonic_cond(alpha, n)

        s = (V - Vs) / (V0 - Vs + 1e-14)
        n_norm = torch.full_like(s, (n - 2.5) / 0.5)  # n=3 -> +1, n=2 -> -1

        s_input = torch.cat([s, n_norm], dim=-1)

        C_linear = (1.0 - s) * Cs + s * C0
        C_correction = s * (1.0 - s) * self.net(s_input)
        return C_linear + C_correction


def compute_ode_loss(model, n, n_colloc, device):
    """计算单个几何的ODE残差loss"""
    gamma = model.gamma
    alpha = model.alpha(n)
    Vs, Cs = model.shock_cond(alpha)
    V0, C0 = model.sonic_cond(alpha, n)

    Vs_d = Vs.detach()
    V0_d = V0.detach()
    V_colloc = (Vs_d + (V0_d - Vs_d) *
                torch.linspace(0.01, 0.99, n_colloc, dtype=torch.float64, device=device)
                ).reshape(-1, 1)
    V_colloc.requires_grad_(True)

    C_pred = model(V_colloc, n)

    dC_dV = torch.autograd.grad(
        C_pred, V_colloc,
        grad_outputs=torch.ones_like(C_pred),
        create_graph=True,
    )[0]

    residual, numer, denom, delta = chisnell_residual(
        V_colloc, C_pred, dC_dV, alpha, gamma, n)

    # 混合残差: 远离奇点用除形式，靠近用乘形式
    abs_denom = torch.abs(denom)
    denom_thr = 1e-6
    w_far = torch.clamp(abs_denom / denom_thr - 1.0, min=0.0, max=1.0)
    R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
    loss_ode_far = torch.mean(w_far * R_divided ** 2)

    w_sonic = 1.0 - w_far
    loss_ode_sonic = torch.mean(w_sonic * residual ** 2)
    loss_numer = torch.mean(w_sonic * numer ** 2)

    return loss_ode_far + loss_ode_sonic, loss_numer


def train_multi_geometry(gamma, alpha_init_sph, alpha_init_cyl,
                         epochs=12000, lr=1e-3, n_colloc=400,
                         lambda_order=10.0, device="cuda"):
    """
    同时训练球面和柱面的α。

    Loss = L_ode_sph + L_ode_cyl + λ_reg * (L_reg_sph + L_reg_cyl)
         + λ_order * L_ordering

    其中 L_ordering = max(0, α_sph - α_cyl + margin)²
    物理约束: α_cyl > α_sph (柱面α更大)
    """
    ref_sph = REF_ALPHA[(gamma, 3)]
    ref_cyl = REF_ALPHA[(gamma, 2)]

    model = MultiGeometryPINN(
        gamma, alpha_init_sph, alpha_init_cyl
    ).double().to(device)

    history = {"alpha_sph": [], "alpha_cyl": [], "loss": []}

    # Phase 1: warmup (固定两个α)
    warmup = epochs // 3
    model.raw_alpha_sph.requires_grad_(False)
    model.raw_alpha_cyl.requires_grad_(False)

    opt1 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    print(f"  Phase 1: warmup {warmup} epochs")
    print(f"    α_sph_init={float(model.alpha(3)):.4f}, α_cyl_init={float(model.alpha(2)):.4f}")

    for epoch in range(1, warmup + 1):
        opt1.zero_grad()
        loss_ode_sph, loss_reg_sph = compute_ode_loss(model, 3, n_colloc, device)
        loss_ode_cyl, loss_reg_cyl = compute_ode_loss(model, 2, n_colloc, device)
        loss = (loss_ode_sph + loss_ode_cyl) + 10.0 * (loss_reg_sph + loss_reg_cyl)
        loss.backward()
        opt1.step()

        a_s = float(model.alpha(3).detach())
        a_c = float(model.alpha(2).detach())
        history["alpha_sph"].append(a_s)
        history["alpha_cyl"].append(a_c)
        history["loss"].append(float(loss.detach()))

        if epoch % 1000 == 0 or epoch == 1:
            print(f"    warmup {epoch:5d}  loss={loss.item():.3e}  "
                  f"α_sph={a_s:.6f}(ref={ref_sph:.6f})  α_cyl={a_c:.6f}(ref={ref_cyl:.6f})")

    # Phase 2: 解冻α，联合训练
    model.raw_alpha_sph.requires_grad_(True)
    model.raw_alpha_cyl.requires_grad_(True)
    main_epochs = epochs - warmup

    opt2 = optim.Adam([
        {"params": model.net.parameters(), "lr": lr * 0.3},
        {"params": [model.raw_alpha_sph, model.raw_alpha_cyl], "lr": lr * 0.2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=main_epochs, eta_min=1e-6)

    print(f"  Phase 2: joint training {main_epochs} epochs (α free, ordering constraint)")

    for epoch in range(1, main_epochs + 1):
        opt2.zero_grad()
        loss_ode_sph, loss_reg_sph = compute_ode_loss(model, 3, n_colloc, device)
        loss_ode_cyl, loss_reg_cyl = compute_ode_loss(model, 2, n_colloc, device)

        # 物理排序约束: α_cyl > α_sph (margin=0.01)
        alpha_sph = model.alpha(3)
        alpha_cyl = model.alpha(2)
        ordering_violation = torch.clamp(alpha_sph - alpha_cyl + 0.01, min=0.0)
        loss_order = ordering_violation ** 2

        loss = ((loss_ode_sph + loss_ode_cyl)
                + 10.0 * (loss_reg_sph + loss_reg_cyl)
                + lambda_order * loss_order)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt2.step()
        scheduler.step()

        a_s = float(model.alpha(3).detach())
        a_c = float(model.alpha(2).detach())
        history["alpha_sph"].append(a_s)
        history["alpha_cyl"].append(a_c)
        history["loss"].append(float(loss.detach()))

        if epoch % 1000 == 0 or epoch == 1:
            print(f"    joint {epoch:5d}  loss={loss.item():.3e}  "
                  f"α_sph={a_s:.6f}(ref={ref_sph:.6f})  α_cyl={a_c:.6f}(ref={ref_cyl:.6f})  "
                  f"order_viol={float(ordering_violation):.1e}")

    # 结果
    a_s = float(model.alpha(3).detach())
    a_c = float(model.alpha(2).detach())
    err_s = abs(a_s - ref_sph) / ref_sph * 100
    err_c = abs(a_c - ref_cyl) / ref_cyl * 100

    return {
        "alpha_sph": a_s, "alpha_cyl": a_c,
        "ref_sph": ref_sph, "ref_cyl": ref_cyl,
        "err_sph_pct": err_s, "err_cyl_pct": err_c,
        "history": history,
        "gamma": gamma,
    }


def train_single_geometry_baseline(gamma, n, alpha_init,
                                   epochs=12000, lr=1e-3, n_colloc=400, device="cuda"):
    """单几何基线（对照组），用同样的训练参数"""
    from pinn_self_discover import train
    geo = "spherical" if n == 3 else "cylindrical"
    result = train(gamma, n, geo, alpha_init=alpha_init, epochs=epochs, lr=lr,
                   n_colloc=n_colloc, device=device)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=12000)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  扩展验证4: 多几何统一自发现")
    print("  一个网络同时发现球面+柱面α, 加物理排序约束")
    print("=" * 70)

    all_results = {}

    for gamma in [1.4, 5/3]:
        gamma_key = f"g{gamma:.2f}"
        print(f"\n{'='*60}")
        print(f"  gamma={gamma:.4f}")
        print(f"{'='*60}")

        # 多几何统一训练
        print(f"\n--- 多几何统一训练 (加ordering约束) ---")
        result_multi = train_multi_geometry(
            gamma,
            alpha_init_sph=ALPHA_INIT[(gamma, 3)],
            alpha_init_cyl=ALPHA_INIT[(gamma, 2)],
            epochs=args.epochs,
            device=args.device,
        )

        print(f"\n  结果: α_sph={result_multi['alpha_sph']:.6f} (err={result_multi['err_sph_pct']:.6f}%)")
        print(f"        α_cyl={result_multi['alpha_cyl']:.6f} (err={result_multi['err_cyl_pct']:.6f}%)")

        # 对照: 单独训练 (同样epochs)
        print(f"\n--- 对照: 单独训练球面 ---")
        result_sph = train_single_geometry_baseline(
            gamma, 3, ALPHA_INIT[(gamma, 3)],
            epochs=args.epochs, device=args.device)

        print(f"\n--- 对照: 单独训练柱面 ---")
        result_cyl = train_single_geometry_baseline(
            gamma, 2, ALPHA_INIT[(gamma, 2)],
            epochs=args.epochs, device=args.device)

        all_results[gamma_key] = {
            "multi": {
                "alpha_sph": result_multi["alpha_sph"],
                "alpha_cyl": result_multi["alpha_cyl"],
                "err_sph_pct": result_multi["err_sph_pct"],
                "err_cyl_pct": result_multi["err_cyl_pct"],
            },
            "single_sph": {
                "alpha": result_sph["alpha_fit"],
                "err_pct": result_sph["rel_err_pct"],
            },
            "single_cyl": {
                "alpha": result_cyl["alpha_fit"],
                "err_pct": result_cyl["rel_err_pct"],
            },
        }

    # 汇总表
    print(f"\n{'='*70}")
    print("  汇总: 多几何统一 vs 单独训练")
    print(f"{'='*70}")
    print(f"  {'γ':<8} {'几何':<8} {'统一α':<14} {'统一err%':<14} {'单独α':<14} {'单独err%':<14}")
    print(f"  {'-'*70}")
    for gamma_key, r in all_results.items():
        print(f"  {gamma_key:<8} {'球面':<8} {r['multi']['alpha_sph']:<14.8f} "
              f"{r['multi']['err_sph_pct']:<14.6f} {r['single_sph']['alpha']:<14.8f} "
              f"{r['single_sph']['err_pct']:<14.6f}")
        print(f"  {'':<8} {'柱面':<8} {r['multi']['alpha_cyl']:<14.8f} "
              f"{r['multi']['err_cyl_pct']:<14.6f} {r['single_cyl']['alpha']:<14.8f} "
              f"{r['single_cyl']['err_pct']:<14.6f}")

    # 保存
    with open(out_dir / "exp_multi_geometry.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_dir / 'exp_multi_geometry.json'}")
