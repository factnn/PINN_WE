#!/usr/bin/env python3
"""
Guderley eigenvalue self-discovery via PINN — Chisnell phase-plane, 双端约束。

核心思路:
    Chisnell ODE: dC/dV = F(V, C; alpha, gamma, n)
    网络: V -> C(V), alpha 为可训练参数

    关键约束 (都依赖alpha):
    1. 激波端: C(Vs) = Cs, 其中 Vs = 2α/(γ+1), Cs = 2γ(γ-1)α²/(γ+1)²
    2. 奇点端: C(V0) = C0 = (V0-α)², V0 由代数公式给出
    3. ODE残差: dC/dV * denom = numer (乘denom避免除以零)

    只有正确的alpha才能同时满足三个条件。
    错误的alpha: 激波BC和奇点BC不兼容 → 高ODE残差。

用法:
    python pinn_self_discover.py --geometry spherical --gamma 1.4
    python pinn_self_discover.py --geometry spherical --gamma 1.4 --funnel-scan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── 参考特征值 ──────────────────────────────────────────
REFERENCE_ALPHA = {
    (1.4, 3): 0.717175,
    (1.4, 2): 0.835323,
    (5/3, 3): 0.688377,
    (5/3, 2): 0.815625,
}
GEOMETRY_N = {"spherical": 3, "cylindrical": 2}


def _get_ref_alpha(gamma: float, n: int) -> float | None:
    for (g, nn_), val in REFERENCE_ALPHA.items():
        if nn_ == n and abs(g - gamma) < 1e-4:
            return val
    return None


def _inv_sigmoid(y: float) -> float:
    y = max(min(y, 0.999), 0.001)
    return float(np.log(y / (1.0 - y)))


def _critical_point_np(alpha: float, gamma: float, n: int):
    """代数求解奇点 (V0, C0)。"""
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


def _critical_point_torch(alpha, gamma, n):
    """PyTorch可微版本。"""
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


# ── 网络 ────────────────────────────────────────────────
class ChisnellSelfDiscoverPINN(nn.Module):
    """
    双端硬约束 PINN:
        C(Vs) = Cs  (激波)
        C(V0) = C0  (奇点)
        V域内满足 ODE
    """

    def __init__(self, gamma: float, n: int, alpha_init: float = 0.75,
                 n_layers: int = 5, n_neurons: int = 64):
        super().__init__()
        self.gamma = gamma
        self.n = n

        self.raw_alpha = nn.Parameter(
            torch.tensor(_inv_sigmoid((alpha_init - 0.5) / 0.5), dtype=torch.float64)
        )

        # MLP: s ∈ [0,1] -> raw(1)
        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

    @property
    def alpha(self) -> torch.Tensor:
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def shock_cond(self):
        alpha = self.alpha
        gamma = self.gamma
        Vs = 2.0 * alpha / (gamma + 1.0)
        Cs = 2.0 * gamma * (gamma - 1.0) * alpha ** 2 / (gamma + 1.0) ** 2
        return Vs, Cs

    def sonic_cond(self):
        alpha = self.alpha
        V0, C0 = _critical_point_torch(alpha, self.gamma, self.n)
        return V0, C0

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """
        双端硬约束:
            s = (V - Vs) / (V0 - Vs) ∈ [0, 1]  (参数化)
            C(V) = (1-s)*Cs + s*C0 + s*(1-s)*net(s)

        保证:
            C(Vs) = Cs (s=0)
            C(V0) = C0 (s=1)
            中间由网络自由调节

        注意: Vs, Cs, V0, C0 都依赖 alpha，所以梯度可以流回 alpha。
        """
        Vs, Cs = self.shock_cond()
        V0, C0 = self.sonic_cond()

        # 归一化到 s ∈ [0, 1]
        s = (V - Vs) / (V0 - Vs + 1e-14)

        # 线性插值 + 泡泡修正
        C_linear = (1.0 - s) * Cs + s * C0
        C_correction = s * (1.0 - s) * self.net(s)
        return C_linear + C_correction

    def chisnell_residual(self, V: torch.Tensor, C: torch.Tensor, dC_dV: torch.Tensor):
        """ODE残差: dC/dV * denom - numer。"""
        alpha = self.alpha
        gamma = self.gamma
        n = self.n

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


# ── 训练 ────────────────────────────────────────────────
def train(
    gamma: float, n: int, geometry: str,
    alpha_init: float = 0.75,
    epochs: int = 10000,
    lr: float = 1e-3,
    n_colloc: int = 400,
    device: str = "cpu",
    fix_alpha: float | None = None,
) -> dict:
    ref_alpha = _get_ref_alpha(gamma, n)

    model = ChisnellSelfDiscoverPINN(
        gamma=gamma, n=n, alpha_init=alpha_init,
        n_layers=5, n_neurons=64,
    ).double().to(device)

    if fix_alpha is not None:
        with torch.no_grad():
            model.raw_alpha.data = torch.tensor(
                _inv_sigmoid((fix_alpha - 0.5) / 0.5), dtype=torch.float64
            )
        model.raw_alpha.requires_grad_(False)

    history_alpha = []
    history_loss = []

    def compute_loss():
        alpha = model.alpha
        Vs, Cs = model.shock_cond()
        V0, C0 = model.sonic_cond()

        # 配点在 [Vs, V0] 内部 (避开端点)
        Vs_d = Vs.detach()
        V0_d = V0.detach()
        V_colloc = (Vs_d + (V0_d - Vs_d) *
                    torch.linspace(0.01, 0.99, n_colloc, dtype=torch.float64, device=device)
                    ).reshape(-1, 1)
        V_colloc.requires_grad_(True)

        C_pred = model(V_colloc)

        # dC/dV
        dC_dV = torch.autograd.grad(
            C_pred, V_colloc,
            grad_outputs=torch.ones_like(C_pred),
            create_graph=True,
        )[0]

        residual, numer, denom, delta = model.chisnell_residual(V_colloc, C_pred, dC_dV)

        # 两种残差形式:
        # 1. 乘形式 (奇点附近安全): dC/dV * denom - numer
        # 2. 除形式 (远离奇点，梯度更强): dC/dV - numer/denom
        abs_denom = torch.abs(denom)
        denom_thr = 1e-6

        # 远离奇点: 除以denom，保留更强的梯度信号
        w_far = torch.clamp(abs_denom / denom_thr - 1.0, min=0.0, max=1.0)
        R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        loss_ode_far = torch.mean(w_far * R_divided ** 2)

        # 靠近奇点: 乘形式，同时约束 numer→0
        w_sonic = 1.0 - w_far
        loss_ode_sonic = torch.mean(w_sonic * residual ** 2)
        loss_numer = torch.mean(w_sonic * numer ** 2)

        loss_ode = loss_ode_far + loss_ode_sonic
        loss_regularity = loss_numer

        return loss_ode, loss_regularity

    # ── 阶段1: 固定alpha，训练网络 ──
    warmup = epochs // 3
    model.raw_alpha.requires_grad_(False)
    optimizer1 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )
    print(f"  [{geometry}] Phase 1: warmup {warmup} epochs (alpha fixed={float(model.alpha):.4f})")

    for epoch in range(1, warmup + 1):
        optimizer1.zero_grad()
        loss_ode, loss_reg = compute_loss()
        loss = loss_ode + 10.0 * loss_reg
        loss.backward()
        optimizer1.step()

        alpha_val = float(model.alpha.detach().cpu())
        history_alpha.append(alpha_val)
        history_loss.append(float(loss.detach().cpu()))

        if epoch % 1000 == 0 or epoch == 1:
            ref_s = f"{ref_alpha:.6f}" if ref_alpha else "N/A"
            print(f"  [{geometry}] warmup {epoch:5d}  loss={loss.item():.6e}  "
                  f"ode={loss_ode.item():.3e}  reg={loss_reg.item():.3e}  "
                  f"alpha={alpha_val:.6f}  (ref={ref_s})")

    # ── 阶段2: 解冻alpha，联合训练 ──
    if fix_alpha is None:
        model.raw_alpha.requires_grad_(True)
    main_epochs = epochs - warmup

    optimizer2 = optim.Adam([
        {"params": model.net.parameters(), "lr": lr * 0.3},
        {"params": [model.raw_alpha], "lr": lr * 0.2},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=main_epochs, eta_min=1e-6)
    print(f"  [{geometry}] Phase 2: joint training {main_epochs} epochs (alpha free)")

    for epoch in range(1, main_epochs + 1):
        optimizer2.zero_grad()
        loss_ode, loss_reg = compute_loss()
        loss = loss_ode + 10.0 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer2.step()
        scheduler.step()

        alpha_val = float(model.alpha.detach().cpu())
        history_alpha.append(alpha_val)
        history_loss.append(float(loss.detach().cpu()))

        if epoch % 1000 == 0 or epoch == 1:
            ref_s = f"{ref_alpha:.6f}" if ref_alpha else "N/A"
            print(f"  [{geometry}] joint {epoch:5d}  loss={loss.item():.6e}  "
                  f"ode={loss_ode.item():.3e}  reg={loss_reg.item():.3e}  "
                  f"alpha={alpha_val:.6f}  (ref={ref_s})")

    # ── 结果 ──
    alpha_fit = float(model.alpha.detach().cpu())
    final_loss = history_loss[-1]
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100 if ref_alpha else None

    ref_s = f"{ref_alpha:.6f}" if ref_alpha else "N/A"
    err_s = f"{rel_err:.4f}%" if rel_err is not None else "N/A"
    print(f"  [{geometry}] DONE  alpha={alpha_fit:.6f}  ref={ref_s}  err={err_s}  loss={final_loss:.3e}")

    # 导出
    with torch.no_grad():
        Vs_v = float(model.shock_cond()[0].detach().cpu())
        V0_v, C0_v = _critical_point_np(alpha_fit, gamma, n)
        if V0_v is None:
            V0_v = Vs_v + 0.1
        V_plot = torch.linspace(Vs_v, V0_v, 500,
                                dtype=torch.float64, device=device).reshape(-1, 1)
        C_plot = model(V_plot)

    return {
        "alpha_fit": alpha_fit,
        "ref_alpha": ref_alpha,
        "rel_err_pct": rel_err,
        "final_loss": final_loss,
        "geometry": geometry,
        "gamma": gamma, "n": n,
        "alpha_history": np.array(history_alpha),
        "loss_history": np.array(history_loss),
        "V": V_plot.detach().cpu().numpy().reshape(-1),
        "C": C_plot.detach().cpu().numpy().reshape(-1),
        "model": model,
    }


# ── 漏斗扫描 ───────────────────────────────────────────
def funnel_scan(gamma: float, n: int, geometry: str,
                alpha_range: tuple = (0.60, 0.90), n_scan: int = 30,
                epochs_per: int = 5000, device: str = "cpu",
                output_dir: str = ".") -> None:
    """固定alpha扫描loss landscape。"""
    ref_alpha = _get_ref_alpha(gamma, n)
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_scan)
    losses = []

    print(f"漏斗扫描: gamma={gamma}, {geometry}, {n_scan} points in [{alpha_range[0]:.2f}, {alpha_range[1]:.2f}]")
    for i, alpha_fix in enumerate(alphas):
        print(f"\n--- scan {i+1}/{n_scan}: alpha={alpha_fix:.4f} ---")
        result = train(
            gamma, n, geometry,
            alpha_init=alpha_fix, fix_alpha=alpha_fix,
            epochs=epochs_per, device=device,
        )
        losses.append(result["final_loss"])

    losses = np.array(losses)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(alphas, losses, "bo-", markersize=5, linewidth=2)
    if ref_alpha:
        ax.axvline(ref_alpha, color="r", linestyle="--", linewidth=2, label=f"ref alpha={ref_alpha:.6f}")
    ax.set_xlabel("alpha (fixed)", fontsize=13)
    ax.set_ylabel("min loss (log)", fontsize=13)
    ax.set_title(f"Funnel scan — gamma={gamma}, {geometry}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"funnel_{geometry}_g{gamma:.2f}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    best_idx = np.argmin(losses)
    summary = {
        "geometry": geometry, "gamma": gamma, "n": n,
        "ref_alpha": ref_alpha,
        "best_alpha": float(alphas[best_idx]),
        "best_loss": float(losses[best_idx]),
        "alphas": alphas.tolist(),
        "losses": losses.tolist(),
    }
    with open(out / f"funnel_{geometry}_g{gamma:.2f}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n漏斗扫描结果:")
    print(f"  最优 alpha = {alphas[best_idx]:.6f}  (loss={losses[best_idx]:.3e})")
    print(f"  参考 alpha = {ref_alpha:.6f}")


# ── 画图 ───────────────────────────────────────────────
def plot_self_discover(result: dict, output_dir: str = ".") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    geo = result["geometry"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(result["V"], result["C"], "b-", linewidth=2, label="C(V) PINN")
    ax.set_xlabel("V")
    ax.set_ylabel("C")
    ax.set_title(f"Chisnell phase portrait ({geo})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(result["alpha_history"], linewidth=2)
    if result["ref_alpha"]:
        ax.axhline(result["ref_alpha"], color="r", linestyle="--", linewidth=2,
                    label=f"ref={result['ref_alpha']:.6f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("alpha")
    ax.set_title(f"alpha convergence -> {result['alpha_fit']:.6f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogy(np.maximum(result["loss_history"], 1e-30), linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("training loss (pure physics)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Self-Discovery: gamma={result['gamma']}, {geo}, alpha={result['alpha_fit']:.6f}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"self_discover_{geo}_g{result['gamma']:.2f}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ── main ────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Guderley self-discovery PINN (Chisnell)")
    parser.add_argument("--geometry", choices=["spherical", "cylindrical"], default="spherical")
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-colloc", type=int, default=400)
    parser.add_argument("--alpha-init", type=float, default=0.75)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str,
                        default="/share/project/zpy/PINN_WE/cases/self-discovery/self_discover/output")
    parser.add_argument("--funnel-scan", action="store_true")
    parser.add_argument("--funnel-n", type=int, default=25)
    parser.add_argument("--funnel-lo", type=float, default=0.60)
    parser.add_argument("--funnel-hi", type=float, default=0.90)
    args = parser.parse_args()

    n = GEOMETRY_N[args.geometry]

    if args.funnel_scan:
        funnel_scan(
            args.gamma, n, args.geometry,
            alpha_range=(args.funnel_lo, args.funnel_hi),
            n_scan=args.funnel_n,
            epochs_per=args.epochs,
            device=args.device,
            output_dir=args.output_dir,
        )
    else:
        result = train(
            args.gamma, n, args.geometry,
            alpha_init=args.alpha_init,
            epochs=args.epochs, lr=args.lr,
            n_colloc=args.n_colloc,
            device=args.device,
        )
        plot_self_discover(result, args.output_dir)

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "geometry": args.geometry, "gamma": args.gamma,
            "alpha_fit": result["alpha_fit"],
            "ref_alpha": result["ref_alpha"],
            "rel_err_pct": result["rel_err_pct"],
            "final_loss": result["final_loss"],
        }
        with open(out / f"self_discover_{args.geometry}_g{args.gamma:.2f}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[{args.geometry}] 自发现结果")
        print(f"  alpha (学到) = {result['alpha_fit']:.6f}")
        print(f"  alpha (参考) = {result['ref_alpha']:.6f}")
        err_s = f"{result['rel_err_pct']:.4f}%" if result['rel_err_pct'] is not None else "N/A"
        print(f"  相对误差     = {err_s}")


if __name__ == "__main__":
    main()
