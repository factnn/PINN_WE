#!/usr/bin/env python3
"""Non-PINN baseline for recovering the similarity exponent alpha.

This baseline intentionally assumes the user-selected Guderley convention:

    R(t) = A * (t_c - t)^alpha

where for gamma=1.4:
  - spherical converging shock  -> alpha ~= 0.717
  - cylindrical converging shock -> alpha ~= 0.800

The script generates synthetic shock-radius samples and uses classical nonlinear
least squares (SciPy) to recover alpha without any neural network machinery.

This is a convention-consistent, non-PINN validation baseline for the adopted
alpha definition. It validates the inverse estimation pipeline, not the full
Euler/Guderley BVP derivation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


REFERENCE_ALPHA = {
    "spherical": 0.717,
    "cylindrical": 0.800,
}


@dataclass
class FitResult:
    geometry: str
    alpha_true: float
    alpha_fit: float
    amplitude_true: float
    amplitude_fit: float
    tc_true: float
    tc_fit: float
    noise_level: float
    relative_alpha_error_percent: float
    residual_l2: float
    n_samples: int
    seed: int


def similarity_radius(t: np.ndarray, amplitude: float, tc: float, alpha: float) -> np.ndarray:
    gap = np.maximum(tc - t, 1e-14)
    return amplitude * gap ** alpha


def make_dataset(
    alpha_true: float,
    amplitude_true: float,
    tc_true: float,
    n_samples: int,
    noise_level: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, 0.95 * tc_true, n_samples)
    r_clean = similarity_radius(t, amplitude_true, tc_true, alpha_true)

    if noise_level > 0.0:
        noise = noise_level * np.max(r_clean) * rng.normal(size=r_clean.shape)
        r_obs = np.clip(r_clean + noise, 1e-12, None)
    else:
        r_obs = r_clean.copy()

    return t, r_clean, r_obs


def residual_fn(params: np.ndarray, t: np.ndarray, r_obs: np.ndarray) -> np.ndarray:
    log_amplitude, tc, alpha = params
    amplitude = np.exp(log_amplitude)
    pred = similarity_radius(t, amplitude, tc, alpha)
    return pred - r_obs


def fit_alpha_from_radius(
    t: np.ndarray,
    r_obs: np.ndarray,
    alpha_guess: float,
    tc_guess: float,
    amplitude_guess: float,
) -> FitResult:
    raise NotImplementedError("Use fit_case() instead")


def fit_case(
    geometry: str,
    alpha_true: float,
    amplitude_true: float,
    tc_true: float,
    n_samples: int,
    noise_level: float,
    seed: int,
    restarts: int,
) -> Tuple[FitResult, Dict[str, np.ndarray]]:
    t, r_clean, r_obs = make_dataset(
        alpha_true=alpha_true,
        amplitude_true=amplitude_true,
        tc_true=tc_true,
        n_samples=n_samples,
        noise_level=noise_level,
        seed=seed,
    )

    best = None
    tc_min = float(np.max(t) * 1.001)
    tc_max = float(tc_true * 1.8)

    alpha_candidates = np.linspace(max(0.4, alpha_true - 0.2), min(1.2, alpha_true + 0.2), restarts)
    for idx, alpha_guess in enumerate(alpha_candidates):
        amplitude_guess = max(float(np.max(r_obs)), 1e-6)
        lb = np.array([np.log(1e-8), tc_min, 0.1], dtype=float)
        ub = np.array([np.log(1e3), tc_max, 1.5], dtype=float)
        tc_guess = float(tc_true * (0.9 + 0.02 * idx))
        tc_guess = min(max(tc_guess, tc_min + 1e-8), tc_max - 1e-8)
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

    assert best is not None
    amplitude_fit = float(np.exp(best.x[0]))
    tc_fit = float(best.x[1])
    alpha_fit = float(best.x[2])
    rel_err = abs(alpha_fit - alpha_true) / alpha_true * 100.0

    result = FitResult(
        geometry=geometry,
        alpha_true=alpha_true,
        alpha_fit=alpha_fit,
        amplitude_true=amplitude_true,
        amplitude_fit=amplitude_fit,
        tc_true=tc_true,
        tc_fit=tc_fit,
        noise_level=noise_level,
        relative_alpha_error_percent=rel_err,
        residual_l2=float(np.linalg.norm(best.fun)),
        n_samples=n_samples,
        seed=seed,
    )

    pred = similarity_radius(t, amplitude_fit, tc_fit, alpha_fit)
    data = {
        "t": t,
        "r_clean": r_clean,
        "r_obs": r_obs,
        "r_fit": pred,
    }
    return result, data


def save_plot(output_dir: Path, geometry: str, data: Dict[str, np.ndarray], alpha_true: float, alpha_fit: float) -> None:
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(data["t"], data["r_clean"], "k--", linewidth=2, label="true law")
    axes[0].scatter(data["t"], data["r_obs"], s=18, alpha=0.7, label="observations")
    axes[0].plot(data["t"], data["r_fit"], "r-", linewidth=2, label=f"fit α={alpha_fit:.6f}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("R(t)")
    axes[0].set_title(f"{geometry} shock-radius fit")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data["t"], data["r_fit"] - data["r_obs"], "b.-", linewidth=1)
    axes[1].axhline(0.0, color="k", linewidth=1)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("fit - obs")
    axes[1].set_title(f"Residuals (true α={alpha_true:.6f})")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"{geometry}_fit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> List[FitResult]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometries = [args.geometry] if args.geometry != "both" else ["spherical", "cylindrical"]
    results: List[FitResult] = []

    print("=" * 72)
    print("Non-PINN baseline: recover similarity exponent alpha from R(t)")
    print("=" * 72)
    print(f"n_samples={args.n_samples}, noise={args.noise}, restarts={args.restarts}, seed={args.seed}")
    print()

    for geometry in geometries:
        alpha_true = REFERENCE_ALPHA[geometry]
        result, data = fit_case(
            geometry=geometry,
            alpha_true=alpha_true,
            amplitude_true=args.amplitude,
            tc_true=args.tc,
            n_samples=args.n_samples,
            noise_level=args.noise,
            seed=args.seed,
            restarts=args.restarts,
        )
        results.append(result)

        np.savez(
            output_dir / f"{geometry}_data.npz",
            **data,
        )
        save_plot(output_dir, geometry, data, result.alpha_true, result.alpha_fit)

        print(f"[{geometry}]")
        print(f"  true alpha = {result.alpha_true:.6f}")
        print(f"  fitted alpha = {result.alpha_fit:.6f}")
        print(f"  relative error = {result.relative_alpha_error_percent:.6e}%")
        print(f"  fitted A = {result.amplitude_fit:.6f}, fitted t_c = {result.tc_fit:.6f}")
        print(f"  residual L2 = {result.residual_l2:.6e}")
        print()

    summary = {
        "assumed_convention": {
            "spherical": 0.717,
            "cylindrical": 0.800,
            "law": "R(t) = A * (t_c - t)^alpha",
            "note": "This validates non-PINN inverse recovery under the adopted alpha convention.",
        },
        "results": [asdict(item) for item in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to {output_dir / 'summary.json'}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover alpha from synthetic shock-radius trajectory")
    parser.add_argument("--geometry", choices=["spherical", "cylindrical", "both"], default="spherical")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--noise", type=float, default=1e-3, help="relative noise scale against max radius")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=9)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--tc", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/share/project/zpy/PINN_WE/cases/self-discovery/nonpinn_baseline/output",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()