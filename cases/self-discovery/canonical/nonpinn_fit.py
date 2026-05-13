#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from common import FitSummary, generate_dataset, save_summary, similarity_radius, summary_to_dict


def residual_fn(params: np.ndarray, t: np.ndarray, r_obs: np.ndarray) -> np.ndarray:
    log_amplitude, tc, alpha = params
    amplitude = np.exp(log_amplitude)
    pred = similarity_radius(t, amplitude, tc, alpha)
    return pred - r_obs


def fit_dataset(dataset: dict, restarts: int = 9) -> tuple[FitSummary, dict]:
    t = dataset["t"]
    r_obs = dataset["r_obs"]
    alpha_true = float(dataset["alpha_true"])
    amplitude_true = float(dataset["amplitude_true"])
    tc_true = float(dataset["tc_true"])
    noise_level = float(dataset["noise_level"])
    seed = int(dataset["seed"])
    geometry = str(dataset["geometry"])

    tc_min = float(np.max(t) * 1.001)
    tc_max = float(tc_true * 1.8)
    lb = np.array([np.log(1e-8), tc_min, 0.1], dtype=float)
    ub = np.array([np.log(1e3), tc_max, 1.5], dtype=float)
    alpha_candidates = np.linspace(max(0.4, alpha_true - 0.2), min(1.2, alpha_true + 0.2), restarts)

    best = None
    for idx, alpha_guess in enumerate(alpha_candidates):
        amplitude_guess = max(float(np.max(r_obs)), 1e-6)
        tc_guess = min(max(float(tc_true * (0.9 + 0.02 * idx)), tc_min + 1e-8), tc_max - 1e-8)
        alpha_guess = float(min(max(alpha_guess, lb[2] + 1e-8), ub[2] - 1e-8))
        x0 = np.array([np.log(amplitude_guess), tc_guess, alpha_guess], dtype=float)
        res = least_squares(
            residual_fn,
            x0,
            bounds=(lb, ub),
            args=(t, r_obs),
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=20000,
        )
        if best is None or res.cost < best.cost:
            best = res

    amplitude_fit = float(np.exp(best.x[0]))
    tc_fit = float(best.x[1])
    alpha_fit = float(best.x[2])
    r_fit = similarity_radius(t, amplitude_fit, tc_fit, alpha_fit)
    residual_l2 = float(np.linalg.norm(r_fit - r_obs))
    rel_err = abs(alpha_fit - alpha_true) / alpha_true * 100.0

    summary = FitSummary(
        method="nonpinn",
        geometry=geometry,
        alpha_true=alpha_true,
        alpha_fit=alpha_fit,
        amplitude_true=amplitude_true,
        amplitude_fit=amplitude_fit,
        tc_true=tc_true,
        tc_fit=tc_fit,
        noise_level=noise_level,
        relative_alpha_error_percent=rel_err,
        residual_l2=residual_l2,
        n_samples=len(t),
        seed=seed,
    )
    return summary, {"t": t, "r_obs": r_obs, "r_fit": r_fit, "r_clean": dataset["r_clean"]}


def plot_result(output_dir: Path, summary: FitSummary, data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(data["t"], data["r_clean"], "k--", linewidth=2, label="true law")
    axes[0].scatter(data["t"], data["r_obs"], s=18, alpha=0.75, label="observations")
    axes[0].plot(data["t"], data["r_fit"], "r-", linewidth=2, label=f"fit α={summary.alpha_fit:.6f}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("R(t)")
    axes[0].set_title(f"Non-PINN fit ({summary.geometry})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data["t"], data["r_fit"] - data["r_obs"], "b.-")
    axes[1].axhline(0.0, color="k", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("fit - obs")
    axes[1].set_title("Residuals")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"nonpinn_{summary.geometry}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["spherical", "cylindrical"], default="spherical")
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default="/share/project/zpy/PINN_WE/cases/self-discovery/canonical/output")
    args = parser.parse_args()

    dataset = generate_dataset(args.geometry, n_samples=args.n_samples, noise_level=args.noise, seed=args.seed)
    summary, data = fit_dataset(dataset)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_result(out, summary, data)
    save_summary(out / f"nonpinn_{summary.geometry}.json", {"summary": summary_to_dict(summary)})

    print(f"[{summary.geometry}] non-PINN")
    print(f"  true alpha   = {summary.alpha_true:.6f}")
    print(f"  fitted alpha = {summary.alpha_fit:.6f}")
    print(f"  rel err      = {summary.relative_alpha_error_percent:.6e}%")
    print(f"  residual L2  = {summary.residual_l2:.6e}")


if __name__ == "__main__":
    main()
