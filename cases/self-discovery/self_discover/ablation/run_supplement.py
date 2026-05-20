#!/usr/bin/env python3
"""
补充实验runner:
  1. Exp5b/Exp3a/Exp3b 无warmup版本 (补齐结论1和2的对照组)
  2. Exp1 差init多seed复现性测试

用法:
  python run_supplement.py --task exp5b_nowarmup --device cuda
  python run_supplement.py --task exp3a_nowarmup --device cuda
  python run_supplement.py --task exp3b_nowarmup --device cuda
  python run_supplement.py --task exp1_badinit_seed --seed 42 --device cuda
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (ALPHA_INIT, DEFAULT_CONFIGS, GEOMETRY_N,
                     get_ref_alpha, save_results_json)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_no_warmup(train_fn, name, device, alpha_init_override=None):
    """Run a training function with warmup disabled via monkey-patching epochs."""
    import types
    results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        ai = alpha_init_override or ALPHA_INIT[(gamma, n)]
        print(f"\n--- {name}: gamma={gamma}, {geo}, alpha_init={ai} ---")
        # All train functions have signature: (gamma, n, geometry, alpha_init, epochs, ...)
        # We need to disable warmup. Since warmup = epochs//3 is hardcoded,
        # we pass a wrapper that skips warmup phase.
        results[key] = train_fn(gamma, n, geo, alpha_init=ai, epochs=10000, device=device)
    return results


def train_exp5b_nowarmup(gamma, n, geometry, alpha_init, epochs=10000, device="cuda"):
    """Exp5b (ξ+hard) without warmup."""
    from exp5b_xi_hard_constraint import XiHardPINN, shock_conditions
    from common import critical_point_torch
    import torch.optim as optim

    ref_alpha = get_ref_alpha(gamma, n)
    model = XiHardPINN(gamma, n, alpha_init).to(device)
    history_alpha, history_loss = [], []

    def compute_loss():
        alpha = model.alpha
        xi_star = model.xi_star
        t_colloc = torch.linspace(0.01, 0.99, 400, dtype=torch.float64, device=device).reshape(-1, 1)
        t_colloc.requires_grad_(True)
        V, C, xi = model(t_colloc)
        dV_dt = torch.autograd.grad(V, t_colloc, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        dC_dt = torch.autograd.grad(C, t_colloc, grad_outputs=torch.ones_like(C), create_graph=True)[0]
        dxi_dt = 1.0 - xi_star
        dV_dxi = dV_dt / (dxi_dt + 1e-30)
        delta = (V - alpha) ** 2 - C
        Q = (n * V * (V - alpha) + (2.0 / gamma) * (1.0 - alpha) * (alpha - V) - V * (V - 1.0))
        numer = C * (2.0 * delta * (alpha - V + (1.0 - alpha) / gamma) + (gamma - 1.0) * (alpha - V) * Q)
        denom = (delta * (n * V - 2.0 * (1.0 - alpha) / gamma) * (alpha - V) + (alpha - V) ** 2 * Q)
        res_V = xi * dV_dxi * delta - V * Q
        dC_dV = dC_dt / (dV_dt + 1e-30 * torch.sign(dV_dt))
        abs_denom = torch.abs(denom)
        w_far = torch.clamp(abs_denom / 1e-6 - 1.0, min=0.0, max=1.0)
        R_divided = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        residual_mult = dC_dV * denom - numer
        w_sonic = 1.0 - w_far
        return (torch.mean(res_V ** 2) + torch.mean(w_far * R_divided ** 2)
                + torch.mean(w_sonic * residual_mult ** 2) + 10.0 * torch.mean(w_sonic * numer ** 2))

    # No warmup: all parameters free from start
    opt = optim.Adam([
        {"params": model.net.parameters(), "lr": 1e-3},
        {"params": [model.raw_alpha], "lr": 2e-4},
        {"params": [model.raw_xi_star], "lr": 1e-3},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = compute_loss()
        if torch.isnan(loss) or torch.isinf(loss):
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] {ep:5d}  loss={loss.item():.4e}  alpha={float(model.alpha.detach()):.6f}  (ref={ref_alpha})")

    alpha_fit = float(model.alpha.detach())
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100
    print(f"  [{geometry}] DONE  alpha={alpha_fit:.6f}  ref={ref_alpha}  err={rel_err:.4f}%")
    return {"alpha_fit": alpha_fit, "ref_alpha": ref_alpha, "rel_err_pct": rel_err,
            "final_loss": history_loss[-1],
            "alpha_history": np.array(history_alpha), "loss_history": np.array(history_loss)}


def train_divided_nowarmup(gamma, n, geometry, alpha_init, epochs=10000, device="cuda"):
    """Exp3a (pure divided residual + hard constraint) without warmup."""
    from exp3_divided_form_only import DualEndpointPINN as DividedOnlyPINN
    from common import critical_point_torch, chisnell_residual
    import torch.optim as optim

    ref_alpha = get_ref_alpha(gamma, n)
    model = DividedOnlyPINN(gamma, n, alpha_init).to(device)
    history_alpha, history_loss = [], []

    def compute_loss():
        alpha = model.alpha
        Vs, Cs, V0, C0, s_colloc, C_pred, dC_dV = _forward_chisnell(model, gamma, n, device)
        residual, numer, denom, delta = chisnell_residual(
            Vs + s_colloc * (V0 - Vs), C_pred, dC_dV * (V0 - Vs).detach(),
            alpha, gamma, n)
        # Pure divided
        R = dC_dV * (V0 - Vs).detach() - numer / (denom + 1e-30 * torch.sign(denom))
        return torch.mean(R ** 2)

    # Actually, easier to just inline the full training from exp3
    # Let me use a simpler approach - replicate the core logic
    from exp3_divided_form_only import train_divided_only as _orig
    # Can't easily strip warmup from existing function, so inline it

    def shock_cond(alpha, gamma):
        Vs = 2.0 / ((gamma + 1.0) * alpha)
        Cs = (2.0 * gamma * (gamma - 1.0)) / ((gamma + 1.0) ** 2 * alpha ** 2)
        return Vs, Cs

    history_alpha, history_loss = [], []

    def compute_loss_full():
        alpha = model.alpha
        Vs, Cs = shock_cond(alpha, gamma)
        V0, C0 = critical_point_torch(alpha, gamma, n)
        s = torch.linspace(0.01, 0.99, 400, dtype=torch.float64, device=device).reshape(-1, 1)
        s.requires_grad_(True)
        C_pred = model(s)
        V_colloc = Vs.detach() + s * (V0 - Vs).detach()
        dC_ds = torch.autograd.grad(C_pred, s, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
        dV_ds = (V0 - Vs).detach()
        dC_dV = dC_ds / (dV_ds + 1e-30)
        residual, numer, denom, delta = chisnell_residual(V_colloc, C_pred, dC_dV, alpha, gamma, n)
        R = dC_dV - numer / (denom + 1e-30 * torch.sign(denom))
        return torch.mean(R ** 2)

    opt = optim.Adam([
        {"params": model.net.parameters(), "lr": 1e-3},
        {"params": [model.raw_alpha], "lr": 2e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = compute_loss_full()
        if torch.isnan(loss) or torch.isinf(loss):
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] {ep:5d}  loss={loss.item():.4e}  alpha={float(model.alpha.detach()):.6f}  (ref={ref_alpha})")

    alpha_fit = float(model.alpha.detach())
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100
    print(f"  [{geometry}] DONE  alpha={alpha_fit:.6f}  ref={ref_alpha}  err={rel_err:.4f}%")
    return {"alpha_fit": alpha_fit, "ref_alpha": ref_alpha, "rel_err_pct": rel_err,
            "final_loss": history_loss[-1],
            "alpha_history": np.array(history_alpha), "loss_history": np.array(history_loss)}


def train_multiplied_nowarmup(gamma, n, geometry, alpha_init, epochs=10000, device="cuda"):
    """Exp3b (pure multiplied residual + hard constraint) without warmup."""
    from exp3b_multiplied_form_only import DualEndpointPINN as MultipliedOnlyPINN
    from common import critical_point_torch, chisnell_residual
    import torch.optim as optim

    ref_alpha = get_ref_alpha(gamma, n)
    model = MultipliedOnlyPINN(gamma, n, alpha_init).to(device)
    history_alpha, history_loss = [], []

    def shock_cond(alpha, gamma):
        Vs = 2.0 / ((gamma + 1.0) * alpha)
        Cs = (2.0 * gamma * (gamma - 1.0)) / ((gamma + 1.0) ** 2 * alpha ** 2)
        return Vs, Cs

    def compute_loss():
        alpha = model.alpha
        Vs, Cs = shock_cond(alpha, gamma)
        V0, C0 = critical_point_torch(alpha, gamma, n)
        s = torch.linspace(0.01, 0.99, 400, dtype=torch.float64, device=device).reshape(-1, 1)
        s.requires_grad_(True)
        C_pred = model(s)
        V_colloc = Vs.detach() + s * (V0 - Vs).detach()
        dC_ds = torch.autograd.grad(C_pred, s, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
        dV_ds = (V0 - Vs).detach()
        dC_dV = dC_ds / (dV_ds + 1e-30)
        residual, numer, denom, delta = chisnell_residual(V_colloc, C_pred, dC_dV, alpha, gamma, n)
        # Pure multiplied: R = dC/dV * D - N
        R = dC_dV * denom - numer
        return torch.mean(R ** 2)

    opt = optim.Adam([
        {"params": model.net.parameters(), "lr": 1e-3},
        {"params": [model.raw_alpha], "lr": 2e-4},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = compute_loss()
        if torch.isnan(loss) or torch.isinf(loss):
            history_alpha.append(float(model.alpha.detach()))
            history_loss.append(1e10)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        history_alpha.append(float(model.alpha.detach()))
        history_loss.append(float(loss.detach()))
        if ep % 2000 == 0 or ep == 1:
            print(f"  [{geometry}] {ep:5d}  loss={loss.item():.4e}  alpha={float(model.alpha.detach()):.6f}  (ref={ref_alpha})")

    alpha_fit = float(model.alpha.detach())
    rel_err = abs(alpha_fit - ref_alpha) / ref_alpha * 100
    print(f"  [{geometry}] DONE  alpha={alpha_fit:.6f}  ref={ref_alpha}  err={rel_err:.4f}%")
    return {"alpha_fit": alpha_fit, "ref_alpha": ref_alpha, "rel_err_pct": rel_err,
            "final_loss": history_loss[-1],
            "alpha_history": np.array(history_alpha), "loss_history": np.array(history_loss)}


def train_exp1_badinit(gamma, n, geometry, alpha_init, epochs=10000, device="cuda"):
    """Exp1 (soft constraint + warmup) with bad init. Imported from exp1."""
    from exp1_soft_constraint import train_soft
    return train_soft(gamma, n, geometry, alpha_init=alpha_init, epochs=epochs, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=["exp5b_nowarmup", "exp3a_nowarmup", "exp3b_nowarmup", "exp1_badinit_seed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed > 0:
        set_seed(args.seed)

    task_map = {
        "exp5b_nowarmup": ("Exp5b no warmup (23: ξ+hard, no warmup)", train_exp5b_nowarmup, None),
        "exp3a_nowarmup": ("Exp3a no warmup (1div23: divided+hard, no warmup)", train_divided_nowarmup, None),
        "exp3b_nowarmup": ("Exp3b no warmup (1mult23: multiplied+hard, no warmup)", train_multiplied_nowarmup, None),
        "exp1_badinit_seed": ("Exp1 bad init reproducibility (soft+warmup, init=0.75)", train_exp1_badinit, 0.75),
    }

    desc, train_fn, alpha_override = task_map[args.task]

    print(f"\n{'='*70}")
    print(f"  {desc}  (seed={args.seed})")
    print(f"{'='*70}")

    results = {}
    for gamma, geo in DEFAULT_CONFIGS:
        n = GEOMETRY_N[geo]
        key = f"g{gamma:.2f}_{geo}"
        ai = alpha_override if alpha_override else ALPHA_INIT[(gamma, n)]
        print(f"\n--- gamma={gamma}, {geo}, alpha_init={ai} ---")
        results[key] = train_fn(gamma, n, geo, alpha_init=ai, epochs=10000, device=args.device)

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary: {desc}")
    print(f"{'='*70}")
    print(f"  {'Config':<25} {'alpha':<14} {'ref':<14} {'err%':<12} {'loss':<12}")
    print(f"  {'-'*75}")
    for key, r in results.items():
        print(f"  {key:<25} {r['alpha_fit']:<14.6f} {r['ref_alpha']:<14.6f} "
              f"{r['rel_err_pct']:<12.4f} {r['final_loss']:<12.3e}")

    # Save
    seed_tag = f"_seed{args.seed}" if args.seed > 0 else ""
    fname = f"{args.task}{seed_tag}.json"
    save_results_json(results, desc, str(out_dir / fname))
    print(f"\n  Saved: {out_dir / fname}")
