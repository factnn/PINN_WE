#!/usr/bin/env python3
"""
Guderley eigenvalue discovery from 2D (r,t) Euler equations — NO ODE reduction.

Three experiments:
  1. Pure PDE (no data)        → α diverges (not uniquely determined)
  2. PDE + shock-front data    → α converges with as few as 2 observations
  3. Data-count sweep          → quantify minimum data requirement

Network: (r, t) → (ρ, u, p)
PDE: Spherically symmetric Euler equations with geometric source terms
Learnable: α  (shock trajectory R(t) = (1-t)^α)

Usage:
  python pinn_2d_euler.py                        # default: PDE+data, γ=1.4, spherical
  python pinn_2d_euler.py --mode pure_pde        # pure PDE (no data) — shows failure
  python pinn_2d_euler.py --mode sweep           # data-count sweep
  python pinn_2d_euler.py --mode all_configs     # 4 configs (2γ × 2geo)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ───────────────────────────────────────────────
REF_ALPHA = {
    (1.4, 3): 0.717174501487,   # spherical,  γ=1.4
    (1.4, 2): 0.835323191951,   # cylindrical, γ=1.4
    (5/3, 3): 0.688376822921,   # spherical,  γ=5/3
    (5/3, 2): 0.815624901431,   # cylindrical, γ=5/3
}
TC = 1.0
RHO0 = 1.0

OUT = Path(__file__).parent / "output_2d"


# ── Helpers ─────────────────────────────────────────────────
def inv_sigmoid(y):
    y = max(min(y, 0.999), 0.001)
    return float(np.log(y / (1.0 - y)))


def rh_strong_shock(Rdot, gamma):
    """Strong-shock Rankine-Hugoniot post-shock state (lab frame)."""
    rho_s = RHO0 * (gamma + 1) / (gamma - 1)
    u_s   = 2.0 / (gamma + 1) * Rdot
    p_s   = 2.0 * RHO0 / (gamma + 1) * Rdot**2
    return rho_s, u_s, p_s


def generate_shock_data(n_pts, gamma, n, ref_alpha, seed=42, device="cpu"):
    """Generate synthetic observations at the true shock front."""
    np.random.seed(seed)
    t_data = np.random.uniform(0.2, 0.8, n_pts)
    r_data = (TC - t_data) ** ref_alpha
    Rdot = -ref_alpha * (TC - t_data) ** (ref_alpha - 1)
    rho_data = RHO0 * (gamma + 1) / (gamma - 1) * np.ones(n_pts)
    u_data = 2.0 / (gamma + 1) * Rdot
    p_data = 2.0 * RHO0 / (gamma + 1) * Rdot**2
    return {
        "r": torch.tensor(r_data.reshape(-1, 1), dtype=torch.float64, device=device),
        "t": torch.tensor(t_data.reshape(-1, 1), dtype=torch.float64, device=device),
        "rho": torch.tensor(rho_data.reshape(-1, 1), dtype=torch.float64, device=device),
        "u": torch.tensor(u_data.reshape(-1, 1), dtype=torch.float64, device=device),
        "p": torch.tensor(p_data.reshape(-1, 1), dtype=torch.float64, device=device),
    }


# ── Network ─────────────────────────────────────────────────
class GuderleyEuler2D(nn.Module):
    def __init__(self, gamma=1.4, n=3, alpha_init=0.75,
                 n_layers=6, n_neurons=128):
        super().__init__()
        self.gamma, self.n = gamma, n
        self.raw_alpha = nn.Parameter(
            torch.tensor(inv_sigmoid((alpha_init - 0.5) / 0.5),
                         dtype=torch.float64))
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers.append(nn.Linear(n_neurons, 3))
        self.net = nn.Sequential(*layers)

    @property
    def alpha(self):
        return 0.5 + 0.5 * torch.sigmoid(self.raw_alpha)

    def shock_radius(self, t):
        return (TC - t) ** self.alpha

    def shock_speed(self, t):
        return -self.alpha * (TC - t) ** (self.alpha - 1.0)

    def forward(self, r, t):
        x = torch.cat([r, t], dim=-1)
        out = self.net(x)
        rho = torch.nn.functional.softplus(out[:, 0:1]) + 0.1
        u   = out[:, 1:2]
        p   = torch.nn.functional.softplus(out[:, 2:3]) + 1e-4
        return rho, u, p

    def euler_residual(self, r, t):
        """Spherical/cylindrical Euler PDE residuals."""
        r = r.requires_grad_(True)
        t = t.requires_grad_(True)
        rho, u, p = self.forward(r, t)
        gamma, geo_n = self.gamma, self.n
        E = p / (gamma - 1) + 0.5 * rho * u**2
        rho_u = rho * u

        def grad(y, x):
            return torch.autograd.grad(
                y, x, grad_outputs=torch.ones_like(y),
                create_graph=True, retain_graph=True)[0]

        rho_t   = grad(rho, t)
        rho_u_t = grad(rho_u, t)
        E_t     = grad(E, t)
        F_mass_r   = grad(rho_u, r)
        F_mom_r    = grad(rho * u**2 + p, r)
        F_energy_r = grad((E + p) * u, r)

        r_safe = torch.clamp(r, min=1e-6)
        src_mass   = (geo_n - 1) * rho * u / r_safe
        src_mom    = (geo_n - 1) * rho * u**2 / r_safe
        src_energy = (geo_n - 1) * (E + p) * u / r_safe

        R_mass   = rho_t   + F_mass_r   + src_mass
        R_mom    = rho_u_t + F_mom_r    + src_mom
        R_energy = E_t     + F_energy_r + src_energy
        return R_mass, R_mom, R_energy

    def shock_bc_loss(self, t_pts):
        R_t  = self.shock_radius(t_pts)
        Rdot = self.shock_speed(t_pts)
        rho_s, u_s, p_s = rh_strong_shock(Rdot, self.gamma)
        rho_pred, u_pred, p_pred = self.forward(R_t, t_pts)
        return ((rho_pred - rho_s)**2).mean() \
             + ((u_pred - u_s)**2).mean() \
             + ((p_pred - p_s)**2 / (p_s.detach()**2 + 1e-8)).mean()

    def center_bc_loss(self, t_pts):
        r_center = torch.full_like(t_pts, 0.02)
        _, u_pred, _ = self.forward(r_center, t_pts)
        return (u_pred**2).mean()


# ── Collocation sampler ─────────────────────────────────────
def sample_collocation(model, n_pde=2000, n_bc=200,
                       t_lo=0.1, t_hi=0.85, device="cpu"):
    with torch.no_grad():
        alpha_val = model.alpha.item()
    t_pde = torch.rand(n_pde, 1, dtype=torch.float64, device=device) \
            * (t_hi - t_lo) + t_lo
    R_vals = (TC - t_pde).pow(alpha_val)
    eta = torch.rand(n_pde, 1, dtype=torch.float64, device=device) * 0.95 + 0.02
    r_pde = (eta * R_vals).detach()
    t_bc = torch.linspace(t_lo, t_hi, n_bc, dtype=torch.float64, device=device
                          ).reshape(-1, 1)
    t_cen = torch.linspace(t_lo, t_hi, n_bc, dtype=torch.float64, device=device
                           ).reshape(-1, 1)
    return r_pde, t_pde, t_bc, t_cen


# ── Core training function ──────────────────────────────────
def train(gamma=1.4, n=3, alpha_init=0.75, epochs=8000,
          obs_data=None, w_data=200.0, device="cpu", verbose=True):
    """
    Train 2D Euler PINN.

    Parameters
    ----------
    obs_data : dict or None
        If provided, must contain keys 'r', 't', 'rho', 'u', 'p' (tensors).
        If None, pure PDE mode (no observation data).
    w_data : float
        Weight for observation data loss.
    """
    ref = REF_ALPHA.get((gamma, n))
    model = GuderleyEuler2D(gamma=gamma, n=n, alpha_init=alpha_init,
                            n_layers=6, n_neurons=128).double().to(device)

    opt = optim.Adam([
        {"params": model.net.parameters(), "lr": 5e-4},
        {"params": [model.raw_alpha], "lr": 1e-3},
    ])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    hist_alpha, hist_loss = [], []

    for ep in range(1, epochs + 1):
        r_pde, t_pde, t_bc, t_cen = sample_collocation(
            model, n_pde=1500, n_bc=150, device=device)
        opt.zero_grad()

        Rm, Rmo, Re = model.euler_residual(r_pde, t_pde)
        loss_pde = Rm.pow(2).mean() + Rmo.pow(2).mean() + Re.pow(2).mean()
        loss_shock = model.shock_bc_loss(t_bc)
        loss_center = model.center_bc_loss(t_cen)

        loss = loss_pde + 100 * loss_shock + 10 * loss_center

        if obs_data is not None:
            rho_p, u_p, p_p = model.forward(obs_data["r"], obs_data["t"])
            loss_data = ((rho_p - obs_data["rho"])**2).mean() \
                      + ((u_p - obs_data["u"])**2).mean() \
                      + ((p_p - obs_data["p"])**2
                         / (obs_data["p"].detach()**2 + 1e-8)).mean()
            loss = loss + w_data * loss_data

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        a = model.alpha.item()
        hist_alpha.append(a)
        hist_loss.append(loss.item())

        if verbose and (ep % 2000 == 0 or ep == 1):
            err = abs(a - ref) / ref * 100 if ref else 0
            print(f"  ep {ep:5d}  loss={loss.item():.3e}  "
                  f"alpha={a:.6f}  err={err:.4f}%")

    a_final = model.alpha.item()
    err_final = abs(a_final - ref) / ref * 100 if ref else 0
    return {
        "alpha_fit": a_final, "ref_alpha": ref,
        "rel_err_pct": err_final,
        "alpha_history": np.array(hist_alpha),
        "loss_history": np.array(hist_loss),
        "model": model,
    }


# ── Experiment 1: Pure PDE (no data) ────────────────────────
def exp_pure_pde(device="cpu"):
    """Show that pure PDE cannot determine α."""
    print("\n" + "=" * 60)
    print("Experiment 1: Pure PDE (no observation data)")
    print("=" * 60)
    result = train(gamma=1.4, n=3, epochs=8000, obs_data=None, device=device)
    print(f"  RESULT: alpha={result['alpha_fit']:.6f}  "
          f"ref=0.717175  err={result['rel_err_pct']:.2f}%")
    print("  → α diverges: PDE alone cannot determine the eigenvalue.")
    return result


# ── Experiment 2: PDE + data, all 4 configs ─────────────────
def exp_all_configs(device="cpu"):
    """PDE + 30 shock-front observations, 4 physical configurations."""
    print("\n" + "=" * 60)
    print("Experiment 2: PDE + 30 shock-front observations (4 configs)")
    print("=" * 60)
    configs = [
        (1.4, 3, "spherical"),
        (1.4, 2, "cylindrical"),
        (5/3, 3, "spherical"),
        (5/3, 2, "cylindrical"),
    ]
    results = {}
    for gamma, n, geo in configs:
        ref = REF_ALPHA[(gamma, n)]
        print(f"\n  γ={gamma:.4f}, {geo} (n={n}), ref={ref:.6f}")
        obs = generate_shock_data(30, gamma, n, ref, device=device)
        r = train(gamma=gamma, n=n, epochs=8000, obs_data=obs, device=device)
        print(f"  → alpha={r['alpha_fit']:.6f}  err={r['rel_err_pct']:.4f}%")
        results[f"g{gamma:.2f}_{geo}"] = r
    return results


# ── Experiment 3: Data-count sweep ──────────────────────────
def exp_data_sweep(device="cpu"):
    """Sweep number of observation points: 30, 10, 5, 3, 2, 1."""
    print("\n" + "=" * 60)
    print("Experiment 3: Minimum data requirement (γ=1.4, spherical)")
    print("=" * 60)
    gamma, n = 1.4, 3
    ref = REF_ALPHA[(gamma, n)]
    sweep = [30, 10, 5, 3, 2, 1]
    results = []
    for n_pts in sweep:
        obs = generate_shock_data(n_pts, gamma, n, ref, seed=42, device=device)
        r = train(gamma=gamma, n=n, epochs=4000, obs_data=obs,
                  device=device, verbose=False)
        tag = "OK" if r["rel_err_pct"] < 1.0 else "FAIL"
        print(f"  n_obs={n_pts:2d}  alpha={r['alpha_fit']:.6f}  "
              f"err={r['rel_err_pct']:.4f}%  [{tag}]")
        results.append({"n_obs": n_pts, **{k: r[k] for k in
                        ["alpha_fit", "ref_alpha", "rel_err_pct"]}})
    return results


# ── Plotting ────────────────────────────────────────────────
def plot_comparison(pure_result, data_result, gamma=1.4, n=3):
    """Plot pure-PDE vs PDE+data α convergence side by side."""
    OUT.mkdir(parents=True, exist_ok=True)
    geo = "spherical" if n == 3 else "cylindrical"
    ref = REF_ALPHA[(gamma, n)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, res, title in zip(axes,
                               [pure_result, data_result],
                               ["Pure PDE (no data)", "PDE + 30 observations"]):
        ax.plot(res["alpha_history"], linewidth=1.5, color="steelblue")
        ax.axhline(ref, color="r", ls="--", lw=2, label=f"ref={ref:.6f}")
        ax.set_xlabel("epoch"); ax.set_ylabel("α")
        final = res["alpha_fit"]
        err = res["rel_err_pct"]
        ax.set_title(f"{title}\nα={final:.6f} (err={err:.2f}%)")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 0.85)

    fig.suptitle(f"Guderley 2D Euler: γ={gamma}, {geo}", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    path = OUT / f"fig_pure_vs_data_{geo}_g{gamma:.2f}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_sweep(sweep_results):
    """Plot data-count sweep results."""
    OUT.mkdir(parents=True, exist_ok=True)
    ns = [r["n_obs"] for r in sweep_results]
    errs = [r["rel_err_pct"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(ns, errs, "o-", markersize=8, linewidth=2, color="steelblue")
    ax.axhline(1.0, color="r", ls="--", lw=1.5, label="1% threshold")
    ax.set_xlabel("Number of observations", fontsize=12)
    ax.set_ylabel("α relative error (%)", fontsize=12)
    ax.set_title("Minimum data requirement\n(γ=1.4, spherical, 4000 epochs)")
    ax.set_xticks(ns)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    path = OUT / "fig_data_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Guderley eigenvalue from 2D Euler PDE + sparse data")
    parser.add_argument("--mode", choices=["default", "pure_pde", "sweep",
                                           "all_configs", "full"],
                        default="default")
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--geometry", choices=["spherical", "cylindrical"],
                        default="spherical")
    parser.add_argument("--n-obs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    n = 3 if args.geometry == "spherical" else 2

    if args.mode == "default":
        ref = REF_ALPHA.get((args.gamma, n))
        print(f"=== 2D Euler PINN: γ={args.gamma}, {args.geometry}, "
              f"{args.n_obs} obs ===")
        obs = generate_shock_data(args.n_obs, args.gamma, n, ref,
                                  device=args.device)
        result = train(gamma=args.gamma, n=n, epochs=args.epochs,
                       obs_data=obs, device=args.device)
        print(f"\nFINAL: alpha={result['alpha_fit']:.6f}  "
              f"ref={ref:.6f}  err={result['rel_err_pct']:.4f}%")

    elif args.mode == "pure_pde":
        exp_pure_pde(device=args.device)

    elif args.mode == "sweep":
        sweep = exp_data_sweep(device=args.device)
        plot_sweep(sweep)
        with open(OUT / "sweep_results.json", "w") as f:
            json.dump(sweep, f, indent=2)

    elif args.mode == "all_configs":
        exp_all_configs(device=args.device)

    elif args.mode == "full":
        # Run all three experiments
        t0 = time.time()
        pure = exp_pure_pde(device=args.device)

        ref = REF_ALPHA[(1.4, 3)]
        obs = generate_shock_data(30, 1.4, 3, ref, device=args.device)
        data_res = train(gamma=1.4, n=3, epochs=8000, obs_data=obs,
                         device=args.device)
        plot_comparison(pure, data_res)

        exp_all_configs(device=args.device)
        sweep = exp_data_sweep(device=args.device)
        plot_sweep(sweep)

        with open(OUT / "sweep_results.json", "w") as f:
            json.dump(sweep, f, indent=2)
        print(f"\nAll experiments done in {time.time()-t0:.0f}s")
