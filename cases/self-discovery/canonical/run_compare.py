#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import generate_dataset, save_summary, summary_to_dict
from nonpinn_fit import fit_dataset as fit_nonpinn
from pinn_inverse import train_pinn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=["spherical", "cylindrical"], default="spherical")
    parser.add_argument("--noise", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--pinn-epochs", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="/share/project/zpy/PINN_WE/cases/self-discovery/canonical/output")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(args.geometry, n_samples=args.n_samples, noise_level=args.noise, seed=args.seed)
    nonpinn_summary, _ = fit_nonpinn(dataset)
    pinn_summary, _ = train_pinn(dataset, epochs=args.pinn_epochs, device=args.device)

    payload = {
        "geometry": args.geometry,
        "noise": args.noise,
        "seed": args.seed,
        "nonpinn": summary_to_dict(nonpinn_summary),
        "pinn": summary_to_dict(pinn_summary),
    }
    save_summary(out / f"compare_{args.geometry}.json", payload)

    print(f"[{args.geometry}] comparison")
    print(f"  non-PINN alpha = {nonpinn_summary.alpha_fit:.6f}")
    print(f"  PINN alpha     = {pinn_summary.alpha_fit:.6f}")
    print(f"  target alpha   = {nonpinn_summary.alpha_true:.6f}")


if __name__ == "__main__":
    main()
